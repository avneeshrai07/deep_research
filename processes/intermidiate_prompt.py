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
    source:      Optional[str] = Field(default=None, description="Direct URL from new_collected_data only. Never a field path. Null if no URL available.")


class ResearchAnalysisOutput(BaseModel):
    primary_status:   CoverageStatus = Field(
        description="Coverage of remaining_primary_research_purpose by new_collected_data only."
    )
    secondary_status: CoverageStatus = Field(
        description="Coverage of remaining_secondary_research_purpose by new_collected_data only."
    )

    remaining_primary_research_purpose: List[str] = Field(
        description=(
            "Sub-questions from the input remaining_primary list that are STILL unanswered "
            "after evaluating new_collected_data. "
            "Empty list [] if primary_status = FULFILLED. "
            "Each item must be a concrete answerable question — not a vague category. "
            "Order by criticality — most important gap first."
        )
    )
    remaining_secondary_research_purpose: List[str] = Field(
        description=(
            "Sub-questions from the input remaining_secondary list that are STILL unanswered "
            "after evaluating new_collected_data. "
            "Empty list [] if secondary_status = FULFILLED. "
            "Each item must be a concrete answerable question. "
            "Order by criticality — most important gap first."
        )
    )

    notes: Optional[List[NoteItem]] = Field(
        default=None,
        description=(
            "Notes extracted ONLY from new_collected_data that answer items in "
            "remaining_primary or remaining_secondary lists. "
            "Do NOT re-extract facts already covered in already_formatted_notes. "
            "Minimum 3 notes if PARTIAL, minimum 5 if FULFILLED. "
            "One note = one fact. Never merge two distinct facts."
        )
    )
    search_queries: Optional[List[str]] = Field(
        default=None,
        description=(
            "Exactly 5 web search queries derived from still-remaining gaps. "
            "Populated when primary_status is UNFULFILLED or PARTIAL. "
            "PRIORITY: fill all 5 slots from remaining_primary gaps if 5+ primary gaps exist. "
            "Use remaining_secondary gaps for slots ONLY after remaining_primary is exhausted. "
            "If remaining_primary has fewer than 5 gaps, fill leftover slots from remaining_secondary. "
            "If combined gaps < 5, generate angle variants per gap (portal, filetype, geography, time period). "
            "Zero domain overlap with already_used_search_queries."
        ),
        min_length=5,
        max_length=5
    )


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

INTERMEDIATE_SYSTEM_PROMPT = """\
You are a gap-filling research analysis agent operating inside a multi-step research pipeline.

A previous step already extracted notes from an initial data batch and identified specific \
unanswered questions (remaining gaps). New search data has now been collected specifically \
to answer those gaps. Your job is to evaluate only the new data against only the remaining gaps, \
extract new notes that fill those gaps, re-score coverage, update the remaining gap lists, \
and generate new search queries for anything still unresolved.

You do NOT re-analyze the original research purposes from scratch. \
You do NOT re-extract facts already present in already_formatted_notes. \
You only work with: new_collected_data → remaining gap lists.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1 — EXTRACT NOTES FROM NEW DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read `new_collected_data` and extract every fact that answers any item in \
`remaining_primary_research_purpose` or `remaining_secondary_research_purpose`.

Strict scope: Only extract facts relevant to the remaining gap questions. \
Do not extract facts already present in `already_formatted_notes`. \
Do not extract tangentially related facts that don't address a gap.

TOPIC FORMAT
  Pattern: "<Entity> — <Dimension>"
  ✅ "HUL — FY2024 Supply Chain Capex"
  ✅ "Priya Nair — Board Appointment Date"
  ❌ "Overview" | "Background" | "New Information" | "Additional Data"

ONE NOTE = ONE FACT
  Each note covers exactly one distinct, atomic fact.
  ❌ WRONG: "Capex grew 30% and two new vendors were onboarded."
  ✅ RIGHT: Split into two notes — one for capex, one for vendor onboarding.

DESCRIPTION DENSITY — every description must contain ALL five:
  1. Core fact  (what happened / what is true)
  2. Specific number, date, or named person/product/company
  3. Trend or magnitude  (direction + size of change, if applicable)
  4. Comparison  (vs. competitor, prior period, or benchmark)
  5. Implication  (why this matters for the research gap being filled)
  Length: 4–6 sentences. Every sentence must carry unique, non-redundant information.

  ✅ CORRECT:
    "HUL allocated ₹850 Cr to supply chain infrastructure in FY2024, a 34% increase from \
₹635 Cr in FY2023, as disclosed in the Q4 FY2024 earnings call. The increase was driven \
primarily by cold-chain expansion into Tier-2 cities and automated warehouse deployments \
in Maharashtra and UP. Competitor P&G India spent an estimated ₹520 Cr on equivalent \
infrastructure, making HUL's investment 63% larger. This directly answers the gap on \
FY2024 supply chain capex and signals continued prioritization of distribution as a moat."

  ❌ WRONG (vague, no numbers):
    "HUL has been investing in supply chain. The company seems to be expanding."

SOURCE RULE
  `source` must be a URL extracted verbatim from `new_collected_data`.
  ❌ Never: "web_results[1]" | "new_data[0]" | "search_result.url"
  If no direct URL exists for a note, set source = null.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2 — RE-SCORE COVERAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Score coverage of each remaining gap list using the new notes just extracted.

  FULFILLED    — New notes clearly and completely answer all remaining gap questions.
  PARTIAL      — New notes answer some gaps but at least one significant gap remains.
  UNFULFILLED  — New notes contain no meaningful answer to any gap question.

Score `remaining_primary_research_purpose` → set primary_status.
Score `remaining_secondary_research_purpose` → set secondary_status.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3 — UPDATE REMAINING GAP LISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each input gap list, remove questions answered by new notes. Carry forward only those \
still unanswered or only partially answered.

`remaining_primary_research_purpose` output:
  - Items from input primary gap list NOT answered by new notes.
  - Set to [] if primary_status = FULFILLED.
  - Each item remains a concrete answerable question.
  - Order by criticality — most important gap first.

`remaining_secondary_research_purpose` output:
  - Items from input secondary gap list NOT answered by new notes.
  - Set to [] if secondary_status = FULFILLED.
  - Same rules as above.

Never add new gap questions that weren't in the input lists. \
Only carry forward or remove — do not invent.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 4 — GENERATE SEARCH QUERIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate exactly 5 queries only when primary_status = PARTIAL or UNFULFILLED.
Derive queries directly from the still-remaining gaps after Stage 3.

PRIORITY — strictly enforced:
  1. Fill query slots from remaining_primary gaps first — up to all 5 if enough gaps exist.
  2. Once remaining_primary is exhausted, fill leftover slots from remaining_secondary.
  3. Never generate a secondary-gap query while a primary gap is still unresolved.
  4. If combined remaining gaps < 5, generate angle variants per gap item using
     different domains (portal, filetype, geography, time period) to reach exactly 5
     without domain overlap with already_used_search_queries.

SLOT-FILL REFERENCE TABLE
  Primary gaps ≥ 5          → all 5 from primary
  Primary = 3, Secondary ≥ 2 → 3 from primary + 2 from secondary
  Primary = 3, Secondary = 1 → 3 from primary + 1 from secondary + 1 angle variant
  Primary = 0, Secondary ≥ 5 → all 5 from secondary
  Primary = 0, Secondary < 5 → all secondary + angle variants to reach 5

DOMAIN FRESHNESS RULE
  Every query must target a domain with ZERO semantic overlap with `already_used_search_queries`.
  Before writing each query, check: "Does any already-used query touch this domain?"
    YES → different angle (different portal, filetype, geography, time period, data type).
    NO  → proceed.

  ✅ CORRECT (5 queries, primary-first, all different domains):
    [
      "HUL supply chain vendor contracts procurement GeM CPPP portal 2024",
      "Hindustan Unilever cold chain expansion Tier-2 cities FY2024 filetype:pdf",
      "HUL logistics automation warehouse Maharashtra UP annual report",
      "Hindustan Unilever distribution capex investor presentation 2024 site:hul.co.in",
      "HUL supply chain technology RFP tender 2024 site:eprocure.gov.in"
    ]

  ❌ WRONG (fewer than 5, reusing same domain):
    [
      "HUL supply chain LinkedIn",
      "Priya Nair HUL news 2024"
    ]

Set search_queries = null when both statuses = FULFILLED.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT MODE DECISION TABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PRIMARY = FULFILLED   →  notes only           (search_queries = null)
  PRIMARY = PARTIAL     →  notes + queries      (both populated)
  PRIMARY = UNFULFILLED →  queries only         (notes = null)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.  Complete all 4 stages in strict order: Extract → Score → Update Gaps → Queries.
2.  Only extract facts from new_collected_data. Never hallucinate.
3.  Never re-extract facts already present in already_formatted_notes.
4.  Remaining gap lists: only remove answered items — never add new gap questions.
5.  source must be a verbatim URL from new_collected_data, or null. Never a field path.
6.  Search queries derived from remaining gaps only — primary gaps have absolute priority.
7.  Never generate a secondary-gap query while primary gaps remain unresolved.
8.  search_queries must have zero domain overlap with already_used_search_queries.
9.  Minimum notes: 5 if FULFILLED, 3 if PARTIAL. Produce maximum notes data supports.
10. One note = one fact. Never merge distinct facts.
11. Every description must contain at least one specific number, date, or named entity.
12. Always return exactly 5 search_queries when queries are required — never fewer.
"""


def build_intermediate_user_prompt(
    research: dict,
    new_research_data: list,
    remaining_primary: list,
    remaining_secondary: list,
    already_formatted_notes: list,
) -> str:
    return f"""\
Execute all 4 stages in strict order for the data below.

────────────────────────────────────────
REMAINING PRIMARY GAPS  (fill these first — highest priority)
────────────────────────────────────────
{remaining_primary}

────────────────────────────────────────
REMAINING SECONDARY GAPS  (fill only after primary slots are exhausted)
────────────────────────────────────────
{remaining_secondary}

────────────────────────────────────────
ALREADY USED SEARCH QUERIES
────────────────────────────────────────
{research.get("used_queries", [])}

────────────────────────────────────────
ALREADY FORMATTED NOTES  (do NOT re-extract these facts)
────────────────────────────────────────
{already_formatted_notes}

────────────────────────────────────────
NEW COLLECTED DATA  (extract notes only from this)
────────────────────────────────────────
{new_research_data}

────────────────────────────────────────
EXECUTION CHECKLIST — complete in strict order
────────────────────────────────────────

STAGE 1 — EXTRACT NOTES FROM NEW DATA
[ ] Read new_collected_data fully
[ ] Extract only facts that answer items in remaining_primary or remaining_secondary
[ ] Skip any fact already present in already_formatted_notes
[ ] Each topic: "<Entity> — <Dimension>" — no generic headings
[ ] Each description: 4–6 sentences with core fact + number + trend + comparison + implication
[ ] Each source: verbatim URL from new_collected_data, or null — never a field path
[ ] One note = one fact — no merging

STAGE 2 — RE-SCORE COVERAGE
[ ] primary_status   scored against remaining_primary_research_purpose
[ ] secondary_status scored against remaining_secondary_research_purpose

STAGE 3 — UPDATE REMAINING GAP LISTS
[ ] Remove answered questions from remaining_primary — carry forward only unresolved ones
[ ] Remove answered questions from remaining_secondary — carry forward only unresolved ones
[ ] Set both to [] if their respective status = FULFILLED
[ ] Do NOT add new gap questions not in input lists

STAGE 4 — GENERATE QUERIES (only if primary = PARTIAL or UNFULFILLED)
[ ] Exactly 5 queries derived from still-remaining gaps
[ ] Fill all 5 slots from remaining_primary first
[ ] Use remaining_secondary for leftover slots only after primary is exhausted
[ ] If combined gaps < 5, generate angle variants (portal, filetype, geography, time period) to reach 5
[ ] Every query has zero domain overlap with already_used_search_queries
[ ] search_queries = null if both statuses = FULFILLED
"""


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

async def intermediate_research_prompt(
    research: dict,
    new_research_data: list,
    remaining_primary_research_purpose: list,
    remaining_secondary_research_purpose: list,
    already_formatted_notes: list,
) -> ResearchAnalysisOutput:
    try:
        result = await claude_haiku(
            system_prompt=INTERMEDIATE_SYSTEM_PROMPT,
            user_prompt=build_intermediate_user_prompt(
                research=research,
                new_research_data=new_research_data,
                remaining_primary=remaining_primary_research_purpose,
                remaining_secondary=remaining_secondary_research_purpose,
                already_formatted_notes=already_formatted_notes,
            ),
            user_context=None,
            pydantic_model=ResearchAnalysisOutput,
        )
        return result
    except Exception as e:
        return {"error": f"Error in intermediate research analysis: {str(e)}"}


# ─────────────────────────────────────────────
# PRINT HELPER
# ─────────────────────────────────────────────

def print_analysis(output: ResearchAnalysisOutput):
    print("\n" + "=" * 70)
    print("INTERMEDIATE RESEARCH ANALYSIS")
    print("=" * 70)

    print(f"\n📊 Coverage (after new data):")
    print(f"   Primary:   {output.primary_status.value}")
    print(f"   Secondary: {output.secondary_status.value}")

    if output.remaining_primary_research_purpose:
        print(f"\n🔴 STILL UNRESOLVED — PRIMARY ({len(output.remaining_primary_research_purpose)}):")
        for i, gap in enumerate(output.remaining_primary_research_purpose, 1):
            print(f"   {i}. {gap}")

    if output.remaining_secondary_research_purpose:
        print(f"\n🟡 STILL UNRESOLVED — SECONDARY ({len(output.remaining_secondary_research_purpose)}):")
        for i, gap in enumerate(output.remaining_secondary_research_purpose, 1):
            print(f"   {i}. {gap}")

    if output.notes:
        print(f"\n📋 NEW NOTES FROM THIS ITERATION ({len(output.notes)} items):")
        for i, note in enumerate(output.notes, 1):
            src = f"\n     🔗 {note.source}" if note.source else ""
            print(f"\n   {i}. [{note.topic}]\n     {note.description}{src}")

    if output.search_queries:
        print(f"\n🔍 NEXT SEARCH QUERIES (targeting still-remaining gaps):")
        for i, q in enumerate(output.search_queries, 1):
            print(f"   {i}. {q}")

    print("\n" + "=" * 70)
