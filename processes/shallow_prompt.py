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
# OUTPUT MODELS  (unchanged)
# ─────────────────────────────────────────────

class NoteItem(BaseModel):
    topic:       str            = Field(description="Topic heading for this note")
    description: str            = Field(description="Single concise factual finding")
    source:      Optional[str]  = Field(default=None, description="Direct URL this note was derived from")


class ResearchAnalysisOutput(BaseModel):
    primary_status:   CoverageStatus
    secondary_status: CoverageStatus
    notes:            Optional[List[NoteItem]] = Field(
        default=None,
        description="Populated when primary_status is FULFILLED or PARTIAL."
    )
    search_queries:   Optional[List[str]] = Field(
        default=None,
        description="Exactly 2 web search queries. Populated when primary_status is UNFULFILLED or PARTIAL.",
        min_length=2,
        max_length=2
    )


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

async def shallow_research_prompt(research: dict) -> ResearchAnalysisOutput:

    system_prompt = """
You are a research analysis agent. Evaluate collected research data against two research purposes
and produce structured intelligence notes or targeted gap search queries.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- `primary_research_purpose`    — Must-have. Always evaluate this first.
- `secondary_research_purpose`  — Enrichment layer. Evaluate only if primary is covered.
- `collected_data`              — All gathered data (LinkedIn, web results, company data, etc.)
- `already_used_search_queries` — Already executed queries. New queries MUST come from completely different domains.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — SCORE COVERAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- `FULFILLED`    — Data clearly and completely answers the purpose
- `PARTIAL`      — Data partially answers but meaningful gaps remain
- `UNFULFILLED`  — Data has no meaningful answer to the purpose


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — BRANCH OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIMARY = FULFILLED or PARTIAL  → populate `notes`
PRIMARY = UNFULFILLED           → populate `search_queries` only
PRIMARY = PARTIAL               → populate BOTH `notes` AND `search_queries`


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTES — QUANTITY & GRANULARITY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUANTITY
- Produce the MAXIMUM number of notes the data supports.
- Do NOT merge two distinct facts into one note. Split them.
- Cover BOTH primary and secondary purposes across the note set.

DESCRIPTION DENSITY
- Each description must pack in: the core fact + a key number/date/name + context or comparison.
- Write 4–6 sentences. Never vague. Never generic.
- If the data gives a trend, include the direction and magnitude.
- If the data names a person, role, company, or product — include it explicitly.

✅ CORRECT (dense, specific, standalone):
  "Raised $120M Series C at $1.4B valuation in March 2024 led by Sequoia, bringing total funding to $210M — a 3× step-up from the $400M Series B valuation in 2022."

✅ CORRECT (role + context):
  "VP of Engineering Arjun Mehta joined in Jan 2024 from Google DeepMind, where he led the Gemini infrastructure team; his hire signals a shift toward in-house LLM capability."

❌ WRONG (too vague, no numbers, no names):
  "The company has been growing rapidly and has raised several rounds of funding."

❌ WRONG (two facts merged, should be two notes):
  "Revenue grew 40% and the company also hired a new CFO."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOPIC FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use the pattern:  "<Entity> — <Dimension>"
Examples:
  "Zomato — FY2024 Revenue"
  "Priya Nair — Current Role"
  "HUL — Supply Chain Capex"
  "Blinkit — Unit Economics"

Never use generic headings like "Overview", "Background", "Summary".


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- `source` must be a real URL directly from `collected_data`.
- Never use field paths like "web_results[1]" or "linkedin_data".
- If no URL is available for a note, set source = null.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ CORRECT — 5 split, dense notes:
[
  {
    "topic": "Zomato — FY2024 Revenue",
    "description": "Total consolidated revenue reached ₹14,112 Cr in FY2024, a 73% YoY increase from ₹8,152 Cr in FY2023. Growth was broad-based, with food delivery, Hyperpure B2B supplies, and Blinkit quick-commerce all posting triple-digit or near-triple-digit gains. The primary growth driver was Blinkit's rapid dark-store expansion, which added 200+ stores to reach 639 total by March 2024. This growth rate outpaces competitors Swiggy (est. 45% YoY) and Zepto, positioning Zomato as the clear revenue leader in Indian food-tech. Sustained top-line momentum at this scale significantly de-risks the path to consistent profitability.",
    "source": "https://ir.zomato.com/press-release/fy2024-results"
  },
  {
    "topic": "Zomato — FY2024 Profitability",
    "description": "Zomato reported its first-ever full-year net profit of ₹351 Cr in FY2024, reversing a net loss of ₹971 Cr in FY2023 — a ₹1,322 Cr swing in a single year. EBITDA turned positive at the consolidated level for the first time, driven by improved take-rate and cost discipline in the core food-delivery segment. Customer delivery costs fell 8% per order YoY as density in metro geographies improved. This profitability inflection was achieved while simultaneously investing heavily in Blinkit expansion, demonstrating operating leverage in the base business. Continued profitability at scale will be the key metric investors and analysts watch through FY2025.",
    "source": "https://economictimes.com/zomato-profit-fy2024"
  },
  {
    "topic": "Blinkit — Order Volume",
    "description": "Blinkit crossed 1M daily orders in Q3 FY2024, up from 650K in Q1, with average order value at ₹625.",
    "source": "https://ir.zomato.com/blinkit-q3-fy2024"
  },
  {
    "topic": "Zomato — Hyperpure Revenue",
    "description": "Hyperpure B2B supplies revenue grew 97% YoY to ₹3,258 Cr in FY2024, now contributing 23% of total consolidated revenue.",
    "source": "https://ir.zomato.com/press-release/fy2024-results"
  },
  {
    "topic": "Zomato — Headcount",
    "description": "Total employee count reached 6,800 as of March 2024, up 12% YoY, with 40% of new hires in engineering and product roles.",
    "source": "https://zomato.com/annual-report-2024"
  }
]

❌ WRONG:
[
  {
    "topic": "Overview",
    "description": "Zomato is a food delivery company.",
    "source": "web_results[0]"
  }
]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEARCH QUERIES FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return exactly 2 strings targeting domains completely different from already_used_search_queries.

✅ CORRECT:
[
  "Hindustan Unilever supply chain technology tender CPPP GeM portal 2024",
  "HUL annual report 2024 supply chain capital expenditure filetype:pdf"
]

❌ WRONG (same domain as already-used queries):
[
  "Priya Nair HUL LinkedIn profile",
  "Priya Nair supply chain HUL news"
]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Every note must be traceable to `collected_data`. Never hallucinate.
2. `source` must be a real URL — never a field path.
3. `search_queries` must target a domain with ZERO overlap with `already_used_search_queries`.
4. `notes` = null when primary is UNFULFILLED.
5. `search_queries` = null when both primary and secondary are FULFILLED.
6. Split every distinct fact into its own note — never merge.
7. Minimum 5 notes when FULFILLED, minimum 3 when PARTIAL.
8. Every description must contain at least one specific number, name, date, or named entity.
"""

    user_prompt = f"""
Analyze the following research data and produce the appropriate structured output.

PRIMARY RESEARCH PURPOSE
{research["user_intent"]["primary_research_purpose"]}

SECONDARY RESEARCH PURPOSE
{research["user_intent"]["secondary_research_purpose"]}

ALREADY USED SEARCH QUERIES
{research.get("used_queries", [])}

COLLECTED DATA
{research["research_data"]}

INSTRUCTIONS
1. Score primary_status:   FULFILLED / PARTIAL / UNFULFILLED
2. Score secondary_status: FULFILLED / PARTIAL / UNFULFILLED
3. Based on scores:
   - PRIMARY FULFILLED or PARTIAL → populate notes
   - PRIMARY UNFULFILLED          → populate search_queries only
   - PRIMARY PARTIAL              → populate both notes AND search_queries
4. Every note source must be a direct URL from collected_data.
5. Search queries must come from a completely different domain than already_used_search_queries.
6. Split every distinct fact into its own note. Minimum 5 notes if FULFILLED, 3 if PARTIAL.
7. Each description must be 4–6 sentences containing: core fact → trend/magnitude → driver → comparison → implication. No vague sentences.
"""

    try:
        result = await claude_haiku(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
