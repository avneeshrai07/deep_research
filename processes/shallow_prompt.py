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


class SearchQuery(BaseModel):
    type: str = Field(
        description="Entity type: person, company, product, place, event, concept, sports, statistics, route, etc."
    )
    name: str = Field(
        description="Full canonical name of the entity exactly as it appears on the web."
    )
    primary_identifier: str = Field(
        description=(
            "The single most identifying attribute that distinguishes this entity from all others "
            "with the same name. Pick the ONE fact that eliminates ambiguity on the web.\n"
            "Examples by type:\n"
            "  person  → current employer or last known role  ('CEO, Hindustan Unilever')\n"
            "  company → industry + country                   ('FMCG, India')\n"
            "  product → parent company or category           ('Unilever, soap')\n"
            "  place   → state/country                        ('Maharashtra, India')\n"
            "  event   → year + organizer                     ('2024, ICC')\n"
            "Never use vague values like 'unknown', 'N/A', or research directives."
        )
    )
    secondary_identifier: Optional[str] = Field(
        default=None,
        description=(
            "A second grounding fact — only include if primary_identifier alone is still ambiguous. "
            "Leave null if primary is sufficient. Null is better than a weak second identifier.\n"
            "Examples:\n"
            "  person  → city or alma mater  ('Mumbai' / 'IIT Bombay')\n"
            "  company → founding year or HQ ('1933' / 'Mumbai')\n"
            "  product → product line        ('Dove Beauty Bar')\n"
            "  place   → district or region  ('Konkan coast')\n"
            "Never populate just to add detail."
        )
    )
    query: str = Field(
        description=(
            "Exactly 1 web search query string targeting a gap domain. "
            "Must incorporate name + primary_identifier to stay anchored. "
            "ZERO domain overlap with already_used_search_queries."
        )
    )


class ResearchAnalysisOutput(BaseModel):
    primary_status:   CoverageStatus = Field(
        description="Coverage of primary_research_purpose by collected_data."
    )
    secondary_status: CoverageStatus = Field(
        description="Coverage of secondary_research_purpose by collected_data."
    )

    remaining_primary_research_purpose: List[str] = Field(
        description=(
            "Ordered list of specific primary sub-questions or facts still unanswered "
            "after evaluating collected_data. Empty list [] if primary_status = FULFILLED. "
            "Each item must be a concrete answerable question — not a vague category. "
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

    notes: List[NoteItem] = Field(
        default_factory=list,
        description=(
            "Populated whenever the agent is confident about a fact, "
            "regardless of primary_status or secondary_status. "
            "Each note covers exactly ONE distinct fact — never merge two facts. "
            "Cover both primary and secondary purposes across the note set. "
            "Minimum 5 notes if primary_status is FULFILLED, "
            "minimum 3 if PARTIAL, and as many as confidently known if UNFULFILLED."
        )
    )
    search_queries: List[SearchQuery] = Field(
        default_factory=list,
        description=(
            "List of SearchQuery objects, each targeting one remaining gap. "
            "Populated when primary_status is UNFULFILLED or PARTIAL. "
            "PRIORITY: fill slots from remaining_primary gaps first. "
            "Use remaining_secondary gaps for slots ONLY after remaining_primary is exhausted. "
            "Each SearchQuery: type, name, primary_identifier (required), "
            "secondary_identifier (only if needed), and 1 query string anchored to name + primary_identifier. "
        ),
    )



# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────


def build_system_prompt(num_queries: int) -> str:
    return f"""\
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


ONE NOTE = ONE FACT
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


  ✅ CORRECT: "What is HUL's total supply chain capex for FY2024?"
  ❌ WRONG:   "Supply chain details" | "More information needed"


`remaining_secondary_research_purpose`
  - Same rules applied to secondary purpose.
  - Set to [] if secondary_status = FULFILLED.



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 4 — GENERATE SEARCH QUERIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


Generate exactly {num_queries} SearchQuery objects only when primary_status = PARTIAL or UNFULFILLED.
Each SearchQuery targets one remaining gap and must contain:
  - `type`                 — entity type (person, company, product, place, event, etc.)
  - `name`                 — full canonical name as it appears on the web
  - `primary_identifier`   — the ONE fact that eliminates ambiguity for this entity on the web
                             person  → current employer/role  ('CEO, Hindustan Unilever')
                             company → industry + country     ('FMCG, India')
                             product → parent company/category ('Unilever, soap')
                             place   → state/country          ('Maharashtra, India')
                             event   → year + organizer       ('2024, ICC')
  - `secondary_identifier` — second grounding fact ONLY if primary alone is still ambiguous.
                             Set to null if primary is sufficient. Null > weak second identifier.
  - `query`                — exactly 1 search string incorporating name + primary_identifier.
                             ZERO domain overlap with already_used_search_queries.


PRIORITY — strictly enforced:
  1. Fill slots from remaining_primary gaps first — up to all {num_queries} if enough primary gaps exist.
  2. Use remaining_secondary gaps for leftover slots ONLY after remaining_primary is exhausted.
  3. Never generate a secondary-gap query when a primary gap is still unresolved.
  4. If combined remaining gaps < {num_queries}, generate angle variants per gap
     (different portal, filetype, geography, time period) to reach exactly {num_queries}.


SLOT-FILL REFERENCE  (N = {num_queries})
  Primary gaps ≥ N   → all N slots from primary
  Primary gaps < N   → fill primary first, use secondary for remaining slots
  Primary gaps = 0   → all N slots from secondary
  Combined gaps < N  → cover each gap from multiple angles to reach N


DOMAIN FRESHNESS RULE
  Each query string must target a domain with ZERO semantic overlap with `already_used_search_queries`.
  Before writing each query: "Does any already-used query touch this domain?"
    YES → different angle (portal, filetype, geography, time period, data type).
    NO  → proceed.


  ✅ CORRECT SearchQuery objects:
    {{
      "type": "company",
      "name": "Hindustan Unilever",
      "primary_identifier": "FMCG, India",
      "secondary_identifier": null,
      "query": "Hindustan Unilever FMCG India supply chain vendor contracts GeM portal 2024"
    }},
    {{
      "type": "company",
      "name": "Hindustan Unilever",
      "primary_identifier": "FMCG, India",
      "secondary_identifier": "Mumbai HQ",
      "query": "Hindustan Unilever annual report 2024 logistics capex filetype:pdf"
    }}


  ❌ WRONG (primary_identifier is vague / query reuses used domain):
    {{
      "type": "person",
      "name": "Priya Nair",
      "primary_identifier": "unknown",
      "secondary_identifier": "HUL",
      "query": "Priya Nair HUL LinkedIn"
    }}


Set search_queries = [] when both primary_status AND secondary_status = FULFILLED.



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMPTY DATA HANDLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


If collected_data is absent, null, or contains the string "NO DATA COLLECTED YET", you MUST:
  - Set primary_status   = UNFULFILLED
  - Set secondary_status = UNFULFILLED
  - Set notes            = []            ← empty list, never omitted
  - Set search_queries   = exactly {num_queries} SearchQuery objects derived from
                           primary_research_purpose and secondary_research_purpose
  - remaining_primary_research_purpose  = decompose primary_research_purpose
                                          into concrete answerable sub-questions
  - remaining_secondary_research_purpose = decompose secondary_research_purpose
                                           into concrete answerable sub-questions



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT MODE DECISION TABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


  PRIMARY = FULFILLED   →  notes populated,  search_queries = []
  PRIMARY = PARTIAL     →  notes populated,  search_queries populated
  PRIMARY = UNFULFILLED →  notes = [],       search_queries populated



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


1.  Complete all 4 stages in order: Extract → Score → Gaps → Queries.
2.  Every note must be directly traceable to `collected_data`. Zero hallucination.
3.  `source` must be a verbatim URL from `collected_data`, or null. Never a field path.
4.  Remaining gap items must be concrete answerable questions, not vague categories.
5.  Search queries derived from remaining gaps — primary gaps take absolute priority.
6.  Never generate a secondary-gap query while primary gaps remain unresolved.
7.  Each SearchQuery.query must have zero domain overlap with `already_used_search_queries`.
8.  SearchQuery.primary_identifier must be a known, confident fact — never 'unknown' or 'N/A'.
9.  SearchQuery.secondary_identifier must be null unless primary alone is genuinely ambiguous.
10. SearchQuery.query must incorporate name + primary_identifier.
11. Minimum notes: 5 if FULFILLED, 3 if PARTIAL. Produce maximum notes data supports.
12. One note = one fact. Never merge distinct facts.
13. Every description must contain at least one specific number, date, or named entity.
14. Always return exactly {num_queries} SearchQuery objects when queries are required — never fewer.
15. notes must be [] (empty list) when primary_status = UNFULFILLED — never omitted.
16. search_queries must be [] (empty list) when both statuses = FULFILLED — never omitted.
"""


def build_user_prompt(research: dict, num_queries: int) -> str:
    raw_data = research.get("research_data")
    if not raw_data or (isinstance(raw_data, (list, dict)) and len(raw_data) == 0):
        collected_data_block = "NO DATA COLLECTED YET. Treat all statuses as UNFULFILLED."
    else:
        collected_data_block = raw_data

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
{collected_data_block}


────────────────────────────────────────
EXECUTION CHECKLIST — complete in order
────────────────────────────────────────


STAGE 1 — EXTRACT NOTES
[ ] Read all of collected_data
[ ] If collected_data = "NO DATA COLLECTED YET" → skip to STAGE 2 with notes = []
[ ] Extract every relevant fact as a separate note
[ ] Each topic: "<Entity> — <Dimension>" — no generic headings
[ ] Each description: 4–6 sentences with core fact + number + trend + comparison + implication
[ ] Each source: verbatim URL or null — never a field path
[ ] One note = one fact — no merging


STAGE 2 — SCORE COVERAGE
[ ] primary_status   scored: FULFILLED / PARTIAL / UNFULFILLED
[ ] secondary_status scored: FULFILLED / PARTIAL / UNFULFILLED
[ ] Both = UNFULFILLED when collected_data was empty


STAGE 3 — IDENTIFY GAPS
[ ] remaining_primary_research_purpose  — concrete answerable questions, ordered by criticality
[ ] remaining_secondary_research_purpose — concrete answerable questions, ordered by criticality
[ ] Both set to [] if their respective status = FULFILLED
[ ] Decompose raw purpose strings into sub-questions when collected_data was empty


STAGE 4 — GENERATE QUERIES (only if primary = PARTIAL or UNFULFILLED)
[ ] Exactly {num_queries} SearchQuery objects
[ ] Each has: type, name, primary_identifier (confident fact only), secondary_identifier (null if not needed), query
[ ] primary_identifier must be a known confident fact — never 'unknown' or 'N/A'
[ ] secondary_identifier = null unless primary alone is genuinely ambiguous
[ ] query incorporates name + primary_identifier
[ ] Fill all {num_queries} slots from remaining_primary first
[ ] Use remaining_secondary for leftover slots ONLY after primary is exhausted
[ ] If combined gaps < {num_queries}, generate angle variants to reach exactly {num_queries}
[ ] Every query string has zero domain overlap with already_used_search_queries
[ ] search_queries = [] if both statuses = FULFILLED
[ ] notes = [] if primary_status = UNFULFILLED
"""



# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────


async def shallow_research_prompt(
    research: dict,
    num_queries: int = 2,
) -> ResearchAnalysisOutput:
    result = await claude_haiku(
        system_prompt=build_system_prompt(num_queries),
        user_prompt=build_user_prompt(research, num_queries),
        user_context=None,
        pydantic_model=ResearchAnalysisOutput,
    )
    return result



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

    gaps_p = output.remaining_primary_research_purpose or []
    gaps_s = output.remaining_secondary_research_purpose or []

    if gaps_p:
        print(f"\n🔴 REMAINING PRIMARY GAPS ({len(gaps_p)}):")
        for i, gap in enumerate(gaps_p, 1):
            print(f"   {i}. {gap}")

    if gaps_s:
        print(f"\n🟡 REMAINING SECONDARY GAPS ({len(gaps_s)}):")
        for i, gap in enumerate(gaps_s, 1):
            print(f"   {i}. {gap}")

    if output.notes:
        print(f"\n📋 NOTES  ({len(output.notes)} items):")
        for i, note in enumerate(output.notes, 1):
            src = f"\n     🔗 {note.source}" if note.source else ""
            print(f"\n   {i}. [{note.topic}]\n     {note.description}{src}")

    if output.search_queries:
        print(f"\n🔍 NEXT SEARCH QUERIES ({len(output.search_queries)} targeting remaining gaps):")
        for i, sq in enumerate(output.search_queries, 1):
            sec = f" / {sq.secondary_identifier}" if sq.secondary_identifier else ""
            print(f"\n   {i}. [{sq.type.upper()}] {sq.name}")
            print(f"      id    : {sq.primary_identifier}{sec}")
            print(f"      query : {sq.query}")

    print("\n" + "=" * 70)
