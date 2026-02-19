import os
from dotenv import load_dotenv
load_dotenv()

from llm.hiaku import claude_haiku

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class CoverageStatus(str, Enum):
    FULFILLED   = "FULFILLED"
    PARTIAL     = "PARTIAL"
    UNFULFILLED = "UNFULFILLED"


# ─────────────────────────────────────────────
# OUTPUT MODELS
# ─────────────────────────────────────────────

class NoteItem(BaseModel):
    topic:       str           = Field(description="'<Entity> — <Dimension>' format. E.g. 'Zomato — FY2024 Revenue'. Never generic headings.")
    description: str           = Field(description="4–6 sentences. Must contain: core fact + specific number/date/name + trend/magnitude + comparison + implication. Zero vague sentences.")
    source:      Optional[str] = Field(default=None, description="Direct URL from collected_data only. Never a field path. Null if no URL available.")


class ResearchAnalysisOutput(BaseModel):
    primary_status:                     CoverageStatus
    secondary_status:                   CoverageStatus
    remaining_primary_research_purpose:   List
    remaining_secondary_research_purpose: List
    notes: Optional[List[NoteItem]] = Field(
        default=None,
        description=(
            "Populated when primary_status is FULFILLED or PARTIAL. "
            "Minimum 5 notes if FULFILLED, minimum 3 if PARTIAL. "
            "Each note covers exactly ONE distinct fact — never merge two facts."
        )
    )
    search_queries: Optional[List[str]] = Field(
        default=None,
        description=(
            "Exactly 2 web search queries. "
            "Populated when primary_status is UNFULFILLED or PARTIAL. "
            "Must target domains with ZERO overlap with already_used_search_queries."
        ),
        min_length=2,
        max_length=2
    )


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precision research analysis agent. Your job is to evaluate collected research data \
against two research purposes and produce either structured intelligence notes, \
targeted gap-fill search queries, or both — depending on coverage.

You operate in a multi-step research pipeline. Every output feeds the next step, \
so accuracy, specificity, and zero hallucination are non-negotiable.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — SCORE COVERAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Always evaluate `primary_research_purpose` first. Only evaluate `secondary_research_purpose` \
after primary is scored.

Coverage definitions:
  FULFILLED    — Data clearly and completely answers the purpose. No meaningful gaps.
  PARTIAL      — Data partially answers but at least one significant gap remains.
  UNFULFILLED  — Data contains no meaningful answer to the purpose.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — SELECT OUTPUT MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PRIMARY = FULFILLED  → populate `notes` only        (search_queries = null)
  PRIMARY = PARTIAL    → populate `notes` AND `search_queries`
  PRIMARY = UNFULFILLED → populate `search_queries` only  (notes = null)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTES — WRITING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOPIC FORMAT
  Pattern: "<Entity> — <Dimension>"
  ✅ "Zomato — FY2024 Revenue"
  ✅ "Priya Nair — Current Role"
  ✅ "Blinkit — Unit Economics"
  ❌ "Overview" | "Background" | "Summary" | "Key Facts"

SPLITTING — one note = one fact
  Never merge two distinct facts into a single note. Split them.
  ❌ WRONG: "Revenue grew 40% and the company hired a new CFO."
  ✅ RIGHT: Two separate notes — one for revenue, one for CFO hire.

QUANTITY
  FULFILLED → minimum 5 notes
  PARTIAL   → minimum 3 notes
  Produce the maximum number of notes the data can support. Never truncate.
  Cover BOTH primary and secondary purposes across the note set.

DESCRIPTION DENSITY — each description must contain ALL of:
  1. Core fact (what happened / what is true)
  2. At least one specific number, date, or named person/product/company
  3. Trend or magnitude (direction + size of change)
  4. Comparison (vs. competitor, prior period, or industry benchmark)
  5. Implication (what this means for the research purpose)
  Length: 4–6 sentences. Every sentence must carry unique information.

  ✅ CORRECT:
    "Raised $120M Series C at a $1.4B valuation in March 2024, led by Sequoia, \
bringing total funding to $210M — a 3× step-up from the $400M Series B valuation in 2022. \
The round was oversubscribed by 2×, signaling strong investor confidence despite the broader \
market correction. Competitors Razorpay and BharatPe raised at flat valuations in the same period, \
making this round a meaningful outlier. The capital is earmarked for Southeast Asia expansion and \
LLM infrastructure, directly relevant to the primary research purpose of assessing international \
growth readiness."

  ❌ WRONG (vague, no numbers, no names):
    "The company has been growing rapidly and has raised several rounds of funding."

SOURCE RULE
  `source` must be a direct URL extracted verbatim from `collected_data`.
  ❌ Never use field paths like "web_results[1]", "linkedin_data", or "collected_data.news[0]".
  If no direct URL exists for a note, set source = null.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEARCH QUERIES — WRITING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return exactly 2 strings.
Each query must target a domain with ZERO semantic overlap with `already_used_search_queries`.

Test before writing: "Does any already-used query touch this domain?"
  If YES  → pick a completely different angle (filetype, portal, geography, time period).
  If NO   → proceed.

  ✅ CORRECT (different domains):
    ["Hindustan Unilever supply chain tender CPPP GeM portal 2024",
     "HUL annual report 2024 supply chain capex filetype:pdf"]

  ❌ WRONG (same domain as used queries):
    ["Priya Nair HUL LinkedIn",         ← LinkedIn already used
     "Priya Nair supply chain HUL news"] ← news already used


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Every note must be directly traceable to `collected_data`. Zero hallucination.
2. `source` must be a verbatim URL from `collected_data`, or null. Never a field path.
3. `search_queries` must have zero domain overlap with `already_used_search_queries`.
4. `notes` = null when primary_status = UNFULFILLED.
5. `search_queries` = null when primary_status = FULFILLED AND secondary_status = FULFILLED.
6. One note = one fact. Split every distinct fact into its own note.
7. Minimum note counts: 5 if FULFILLED, 3 if PARTIAL.
8. Every description must contain at least one specific number, date, or named entity.
9. Never produce generic topic headings (Overview, Summary, Background, Key Findings).
"""


def build_user_prompt(research: dict) -> str:
    return f"""\
Evaluate the research data below against the two stated purposes and produce the correct output.

────────────────────────────────────────
PRIMARY RESEARCH PURPOSE
────────────────────────────────────────
{research["user_intent"]["primary_research_purpose"]}

────────────────────────────────────────
SECONDARY RESEARCH PURPOSE
────────────────────────────────────────
{research["user_intent"]["secondary_research_purpose"]}

────────────────────────────────────────
ALREADY USED SEARCH QUERIES
────────────────────────────────────────
{research.get("used_queries", [])}

────────────────────────────────────────
COLLECTED DATA
────────────────────────────────────────
{research["research_data"]}

────────────────────────────────────────
OUTPUT CHECKLIST (verify before responding)
────────────────────────────────────────
[ ] primary_status   scored: FULFILLED / PARTIAL / UNFULFILLED
[ ] secondary_status scored: FULFILLED / PARTIAL / UNFULFILLED
[ ] Output mode selected based on primary_status (see STEP 2)
[ ] notes populated if primary = FULFILLED or PARTIAL
    [ ] Minimum 5 notes if FULFILLED, 3 if PARTIAL
    [ ] Each note covers exactly ONE distinct fact
    [ ] Each topic follows "<Entity> — <Dimension>" format
    [ ] Each description is 4–6 sentences with number + trend + comparison + implication
    [ ] Each source is a verbatim URL from collected_data, or null
[ ] search_queries populated if primary = UNFULFILLED or PARTIAL
    [ ] Exactly 2 queries
    [ ] Zero domain overlap with already_used_search_queries
[ ] Zero hallucination — every note traceable to collected_data
"""


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

async def shallow_research_prompt(research: dict) -> ResearchAnalysisOutput:
    try:
        result = await claude_haiku(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(research),
            user_context=None,
            pydantic_model=ResearchAnalysisOutput
        )
        return result
    except Exception as e:
        return {"error": f"Error in research analysis: {str(e)}"}


# ─────────────────────────────────────────────
# PRINT HELPER
# ─────────────────────────────────────────────

def print_analysis(output: ResearchAnalysisOutput):
    print("\n" + "=" * 70)
    print("RESEARCH ANALYSIS")
    print("=" * 70)

    print(f"\n📊 Coverage:")
    print(f"   Primary:   {output.primary_status.value}")
    print(f"   Secondary: {output.secondary_status.value}")

    if output.notes:
        print(f"\n📋 NOTES  ({len(output.notes)} items):")
        for i, note in enumerate(output.notes, 1):
            src = f"\n     🔗 {note.source}" if note.source else ""
            print(f"\n   {i}. [{note.topic}]\n     {note.description}{src}")

    if output.search_queries:
        print(f"\n🔍 SEARCH QUERIES:")
        for i, q in enumerate(output.search_queries, 1):
            print(f"   {i}. {q}")

    print("\n" + "=" * 70)
