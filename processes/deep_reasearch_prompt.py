import os
from dotenv import load_dotenv
load_dotenv()

from llm.hiaku import claude_haiku

from pydantic import BaseModel, Field
from typing import List, Optional


# ─────────────────────────────────────────────
# OUTPUT MODELS
# ─────────────────────────────────────────────

class NoteItem(BaseModel):
    topic:       str           = Field(description="'<Entity> — <Dimension>' format. E.g. 'Zomato — FY2024 Revenue'. Never generic headings.")
    description: str           = Field(description="4–6 sentences. Must contain: core fact + specific number/date/name + trend/magnitude + comparison + implication. Zero vague sentences.")
    source:      Optional[str] = Field(default=None, description="Direct URL from new_collected_data only. Never a field path. Null if no URL available.")


class DeepResearchOutput(BaseModel):
    notes: List[NoteItem] = Field(
        description=(
            "All facts extracted from new_collected_data that are relevant to the research purposes. "
            "Do NOT re-extract facts already present in already_formatted_topics. "
            "Produce the MAXIMUM number of notes the data supports. "
            "Minimum 5 notes. One note = one fact. Never merge two distinct facts."
        )
    )


# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────

DEEP_SYSTEM_PROMPT = """\
You are a research extraction agent operating inside a multi-step research pipeline.

A previous step already extracted notes from earlier data batches. New search data has now \
been collected. Your only job is to extract every new relevant fact from the new data as \
structured notes. Nothing else — no scoring, no gap lists, no search queries.

You do NOT re-analyze purposes from scratch. \
You do NOT re-extract facts already present in already_formatted_topics. \
You only extract: new_collected_data → notes.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACT NOTES FROM NEW DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read `new_collected_data` and extract every fact relevant to the research purposes,
with priority on facts that answer items in the remaining gap lists.

Strict scope:
  - Prioritize facts that answer remaining_primary gaps first, then remaining_secondary.
  - Also extract any other fact relevant to the primary or secondary research purpose.
  - Never re-extract facts already present in `already_formatted_topics`.
  - Do not extract tangentially related facts that don't serve the research purpose.

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
  5. Implication  (why this matters for the research purpose)
  Length: 4–6 sentences. Every sentence must carry unique, non-redundant information.

  ✅ CORRECT:
    "HUL allocated ₹850 Cr to supply chain infrastructure in FY2024, a 34% increase from \
₹635 Cr in FY2023, as disclosed in the Q4 FY2024 earnings call. The increase was driven \
primarily by cold-chain expansion into Tier-2 cities and automated warehouse deployments \
in Maharashtra and UP. Competitor P&G India spent an estimated ₹520 Cr on equivalent \
infrastructure, making HUL's investment 63% larger. This directly answers the research \
purpose and signals continued prioritization of distribution as a competitive moat."

  ❌ WRONG (vague, no numbers):
    "HUL has been investing in supply chain. The company seems to be expanding."

SOURCE RULE
  `source` must be a URL extracted verbatim from `new_collected_data`.
  ❌ Never: "web_results[1]" | "new_data[0]" | "search_result.url"
  If no direct URL exists for a note, set source = null.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Only extract facts from new_collected_data. Never hallucinate.
2. Never re-extract facts already present in already_formatted_topics.
3. source must be a verbatim URL from new_collected_data, or null. Never a field path.
4. Produce the MAXIMUM number of notes the data supports. Minimum 5.
5. One note = one fact. Never merge distinct facts.
6. Every description must contain at least one specific number, date, or named entity.
7. Every topic must follow "<Entity> — <Dimension>" — never generic headings.
"""


def build_deep_user_prompt(
    research: dict,
    new_research_data: list,
    remaining_primary: list,
    remaining_secondary: list,
    already_formatted_topics: list,
) -> str:
    return f"""\
Extract all relevant notes from the new data below.

────────────────────────────────────────
PRIMARY RESEARCH PURPOSE
────────────────────────────────────────
{research["user_intent"]["primary_research_purpose"]}

────────────────────────────────────────
SECONDARY RESEARCH PURPOSE
────────────────────────────────────────
{research["user_intent"]["secondary_research_purpose"]}

────────────────────────────────────────
REMAINING PRIMARY GAPS  (prioritize facts that answer these)
────────────────────────────────────────
{remaining_primary}

────────────────────────────────────────
REMAINING SECONDARY GAPS  (fill after primary gaps are addressed)
────────────────────────────────────────
{remaining_secondary}

────────────────────────────────────────
ALREADY FORMATTED TOPIC  (do NOT re-extract facts on these topics)
────────────────────────────────────────
{already_formatted_topics}

────────────────────────────────────────
NEW COLLECTED DATA  (extract notes only from this)
────────────────────────────────────────
{new_research_data}

────────────────────────────────────────
EXTRACTION CHECKLIST
────────────────────────────────────────

[ ] Read new_collected_data fully
[ ] Prioritize facts that answer remaining_primary gaps first
[ ] Then extract facts that answer remaining_secondary gaps
[ ] Also extract any other fact relevant to the research purposes
[ ] Skip any fact already present in already_formatted_topics
[ ] Each topic: "<Entity> — <Dimension>" — no generic headings
[ ] Each description: 4–6 sentences with core fact + number + trend + comparison + implication
[ ] Each source: verbatim URL from new_collected_data, or null — never a field path
[ ] One note = one fact — no merging
[ ] Minimum 5 notes — produce maximum the data supports
"""


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

async def deep_research_prompt(
    research: dict,
    new_research_data: list,
    remaining_primary: list,
    remaining_secondary: list,
    already_formatted_topics: list,
) -> DeepResearchOutput:
    try:
        result = await claude_haiku(
            system_prompt=DEEP_SYSTEM_PROMPT,
            user_prompt=build_deep_user_prompt(
                research=research,
                new_research_data=new_research_data,
                remaining_primary=remaining_primary,
                remaining_secondary=remaining_secondary,
                already_formatted_topics=already_formatted_topics,
            ),
            user_context=None,
            pydantic_model=DeepResearchOutput,
        )
        return result
    except Exception as e:
        return {"error": f"Error in deep research extraction: {str(e)}"}


# ─────────────────────────────────────────────
# PRINT HELPER
# ─────────────────────────────────────────────

def print_deep_analysis(output: DeepResearchOutput):
    print("\n" + "=" * 70)
    print("DEEP RESEARCH EXTRACTION")
    print("=" * 70)

    if output.notes:
        print(f"\n📋 EXTRACTED NOTES ({len(output.notes)} items):")
        for i, note in enumerate(output.notes, 1):
            src = f"\n     🔗 {note.source}" if note.source else ""
            print(f"\n   {i}. [{note.topic}]\n     {note.description}{src}")

    print("\n" + "=" * 70)
