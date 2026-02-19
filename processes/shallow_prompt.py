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
    primary_status:   CoverageStatus
    secondary_status: CoverageStatus

    remaining_primary_research_purpose:   List[str] = Field(
        description=(
            "Ordered list of specific primary sub-questions or facts that are still unanswered or incomplete "
            "after evaluating collected_data. Empty list [] if primary_status = FULFILLED. "
            "Each item must be a concrete answerable question, not a vague category. "
            "Order by importance — most critical gap first."
        )
    )
    remaining_secondary_research_purpose: List[str] = Field(
        description=(
            "Ordered list of specific secondary sub-questions or facts still unanswered. "
            "Empty list [] if secondary_status = FULFILLED. "
            "Only populated after primary gaps are identified. "
            "Each item must be a concrete answerable question. "
            "Order by importance — most critical gap first."
        )
    )

    notes: Optional[List[NoteItem]] = Field(
        default=None,
        description=(
            "Populated when primary_status is FULFILLED or PARTIAL. "
            "Minimum 5 notes if FULFILLED, minimum 3 if PARTIAL. "
            "Each note covers exactly ONE distinct fact — never merge two facts. "
            "Cover both primary and secondary purposes across the note set."
        )
    )
    search_queries: Optional[List[str]] = Field(
        default=None,
        description=(
            "Exactly 2 web search queries derived from remaining gaps. "
            "Populated when primary_status is UNFULFILLED or PARTIAL. "
            "PRIORITY: exhaust remaining_primary_research_purpose first. "
            "Only use remaining_secondary_research_purpose if no primary gaps remain. "
            "Must target domains with ZERO overlap with already_used_search_queries."
        ),
        min_length=2,
        max_length=2
    )


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precision research analysis agent operating inside a multi-step research pipeline. \
Your job has four sequential stages. Complete them in strict order.

Every output feeds the next iteration of the pipeline. \
Accuracy, specificity, and zero hallucination are non-negotiable.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1 — EXTRACT NOTES FROM COLLECTED DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before scoring anything, read all of `collected_data` and extract every relevant fact \
as a structured note. This is your evidence base for all subsequent stages.

TOPIC FORMAT
  Pattern: "<Entity> — <Dimension>"
  ✅ "Zomato — FY2024 Revenue"
  ✅ "Priya Nair — Current Role"
  ✅ "Blinkit — Unit Economics"
  ❌ "Overview" | "Background" | "Summary" | "Key Facts" | "General Info"

SPLITTING — one note = one fact
  Never merge two distinct facts into a single note. Split them.
  ❌ WRONG: "Revenue grew 40% and the company hired a new CFO."
  ✅ RIGHT: Two separate notes — one for revenue growth, one for CFO hire.

DESCRIPTION DENSITY — each description must contain ALL five:
  1. Core fact  (what happened / what is true)
  2. Specific number, date, or named person/product/company
  3. Trend or magnitude  (direction + size of change)
  4. Comparison  (vs. competitor, prior period, or benchmark)
  5. Implication  (what this means for the research purpose)
  Length: 4–6 sentences. Every sentence must carry unique, non-redundant information.

  ✅ CORRECT:
    "Raised $120M Series C at a $1.4B valuation in March 2024, led by Sequoia, bringing total \
funding to $210M — a 3× step-up from the $400M Series B valuation in 2022. The round was \
oversubscribed by 2×, signaling strong investor confidence despite the broader market correction. \
Competitors Razorpay and BharatPe raised at flat valuations in the same period, making this a \
meaningful outlier. The capital is earmarked for Southeast Asia expansion and LLM infrastructure, \
directly relevant to assessing international growth readiness."

  ❌ WRONG (vague, no numbers, no names):
    "The company has been growing and has raised several rounds of funding."

  ❌ WRONG (two facts merged):
    "Revenue grew 40% and the company also hired a new CFO."

SOURCE RULE
  `source` must be a URL extracted verbatim from `collected_data`.
  ❌ Never: "web_results[1]" | "linkedin_data" | "collected_data.news[0]"
  If no direct URL exists for a note, set source = null.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2 — SCORE COVERAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Using the notes you just extracted, score both purposes:

  FULFILLED    — Notes clearly and completely answer the purpose. No meaningful gaps.
  PARTIAL      — Notes partially answer but at least one significant gap remains.
  UNFULFILLED  — Notes contain no meaningful answer to the purpose.

Score `primary_research_purpose` first.
Score `secondary_research_purpose` independently after.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3 — IDENTIFY REMAINING GAPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After scoring, list every sub-question or fact within each purpose that is still unanswered \
or only partially answered by the extracted notes.

`remaining_primary_research_purpose`
  - List every primary sub-question not yet answered by the notes.
  - Each item must be a concrete, specific, answerable question — not a vague category.
  - Order by criticality: most important gap first.
  - Set to [] if primary_status = FULFILLED.

  ✅ CORRECT items:
    "What is HUL's total supply chain capex for FY2024?"
    "Which logistics vendors did HUL onboard in Q3 FY2024?"
  ❌ WRONG items:
    "Supply chain details"    ← too vague, not answerable
    "More information needed" ← useless

`remaining_secondary_research_purpose`
  - Same rules as above, applied to secondary purpose.
  - Set to [] if secondary_status = FULFILLED.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 4 — GENERATE SEARCH QUERIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate exactly 2 search queries only when primary_status = PARTIAL or UNFULFILLED.
Derive queries directly from the remaining gaps identified in Stage 3.

PRIORITY ORDER — strictly enforced:
  1. Pick queries from `remaining_primary_research_purpose` first.
     Fill both query slots with primary gaps if 2+ primary gaps exist.
  2. Only use `remaining_secondary_research_purpose` for a query slot if
     `remaining_primary_research_purpose` is fully exhausted (empty []).
  3. Never generate a secondary-gap query when a primary gap is still unresolved.

DOMAIN FRESHNESS RULE
  Each query must target a domain with ZERO semantic overlap with `already_used_search_queries`.
  Before writing a query, ask: "Does any already-used query touch this domain?"
    YES → pick a different angle (different portal, filetype, geography, time period, data type).
    NO  → proceed.

  ✅ CORRECT (new domains):
    ["HUL supply chain technology tender CPPP GeM portal 2024",
     "Hindustan Unilever annual report 2024 logistics capex filetype:pdf"]

  ❌ WRONG (same domain as used queries):
    ["Priya Nair HUL LinkedIn",          ← LinkedIn already used
     "Priya Nair supply chain HUL news"] ← news already used

Set search_queries = null when both primary_status AND secondary_status = FULFILLED.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT MODE DECISION TABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PRIMARY = FULFILLED  →  notes only          (search_queries = null)
  PRIMARY = PARTIAL    →  notes + queries     (both populated)
  PRIMARY = UNFULFILLED → queries only        (notes = null)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Complete all 4 stages in order: Extract → Score → Gaps → Queries.
2. Every note must be directly traceable to `collected_data`. Zero hallucination.
3. `source` must be a verbatim URL from `collected_data`, or null. Never a field path.
4. Remaining gap items must be concrete answerable questions, not vague categories.
5. Search queries are derived from remaining gaps — primary gaps take absolute priority.
6. Never generate a secondary-gap query while primary gaps remain unresolved.
7. `search_queries` must have zero domain overlap with `already_used_search_queries`.
8. Minimum notes: 5 if FULFILLED, 3 if PARTIAL. Produce maximum notes data supports.
9. One note = one fact. Never merge distinct facts.
10. Every description must contain at least one specific number, date, or named entity.
"""


def build_user_prompt(research: dict) -> str:
    return f"""\
Execute all 4 stages in order for the research data below.

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
EXECUTION CHECKLIST — complete in order
────────────────────────────────────────

STAGE 1 — EXTRACT NOTES
[ ] Read all of collected_data
[ ] Extract every relevant fact as a separate note
[ ] Each topic follows "<Entity> — <Dimension>" format
[ ] Each description: 4–6 sentences with core fact + number + trend + comparison + implication
[ ] Each source is a verbatim URL or null — never a field path
[ ] No two distinct facts merged into one note

STAGE 2 — SCORE COVERAGE
[ ] primary_status   scored: FULFILLED / PARTIAL / UNFULFILLED
[ ] secondary_status scored: FULFILLED / PARTIAL / UNFULFILLED

STAGE 3 — IDENTIFY GAPS
[ ] remaining_primary_research_purpose  — specific answerable questions, ordered by criticality
[ ] remaining_secondary_research_purpose — specific answerable questions, ordered by criticality
[ ] Both set to [] if their respective status = FULFILLED

STAGE 4 — GENERATE QUERIES (only if primary = PARTIAL or UNFULFILLED)
[ ] Exactly 2 queries
[ ] Both slots filled from remaining_primary gaps if 2+ primary gaps exist
[ ] Secondary gap used for a slot ONLY if remaining_primary is fully empty []
[ ] Zero domain overlap with already_used_search_queries
[ ] search_queries = null if both statuses = FULFILLED
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

    if output.remaining_primary_research_purpose:
        print(f"\n🔴 REMAINING PRIMARY GAPS ({len(output.remaining_primary_research_purpose)}):")
        for i, gap in enumerate(output.remaining_primary_research_purpose, 1):
            print(f"   {i}. {gap}")

    if output.remaining_secondary_research_purpose:
        print(f"\n🟡 REMAINING SECONDARY GAPS ({len(output.remaining_secondary_research_purpose)}):")
        for i, gap in enumerate(output.remaining_secondary_research_purpose, 1):
            print(f"   {i}. {gap}")

    if output.notes:
        print(f"\n📋 NOTES  ({len(output.notes)} items):")
        for i, note in enumerate(output.notes, 1):
            src = f"\n     🔗 {note.source}" if note.source else ""
            print(f"\n   {i}. [{note.topic}]\n     {note.description}{src}")

    if output.search_queries:
        print(f"\n🔍 NEXT SEARCH QUERIES (targeting remaining gaps):")
        for i, q in enumerate(output.search_queries, 1):
            print(f"   {i}. {q}")

    print("\n" + "=" * 70)
