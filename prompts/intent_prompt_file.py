from langchain_core.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

from llm.hiaku import claude_haiku

# AWS credentials
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class IntentType(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    UPDATE = "update"
    EXPLANATION = "explanation"
    RECOMMENDATION = "recommendation"
    OUTREACH = "outreach"
    NAVIGATION = "navigation"
    STATISTICS = "statistics"


class EntityPriority(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class ResearchTarget(BaseModel):
    type: str = Field(description="Entity type: person, company, product, place, event, concept, sports, statistics, route, etc.")
    name: str = Field(description="Entity name")
    priority: EntityPriority = Field(description="primary or secondary")
    purpose: Optional[str] = Field(
        default=None,
        description="Why this target is being researched (e.g., 'sales outreach', 'comparison'). Omit if not applicable."
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Known factual properties of this entity only (e.g., company, sport, metric, category, industry, domain). Never include research directives, focus areas, scope descriptions, or unknown values here."
    )


class ResearchIntent(BaseModel):
    targets: List[ResearchTarget] = Field(
        description="All research targets with their flexible attributes"
    )
    primary_research_purpose: str = Field(
        description="DIRECT and EXPLICIT goal — exactly what the user asked for. MUST-HAVE during research. Concise but complete."
    )
    secondary_research_purpose: str = Field(
        description="COMPREHENSIVE and EXPLORATORY extension beyond the explicit ask. Cherry-on-top insights, contextual depth, comparisons, breakdowns, and related metrics that enrich the research but are not strictly required."
    )
    confidence: str = Field(
        description="Confidence: 'high', 'medium', 'low'"
    )

    def get_primary_targets(self) -> List[ResearchTarget]:
        return [t for t in self.targets if t.priority == EntityPriority.PRIMARY]

    def get_secondary_targets(self) -> List[ResearchTarget]:
        return [t for t in self.targets if t.priority == EntityPriority.SECONDARY]


async def intent_prompt(user_query: str) -> ResearchIntent:
    """
    Classifies user's research intent - GENERAL PURPOSE for any query type
    """

    system_prompt = """
You are a research planning system. Your job is to create a COMPREHENSIVE research plan
from the user's query.

You must output TWO research purposes:

1. primary_research_purpose — EXACTLY what the user asked for. Short, direct, must-have.
   This is the NON-NEGOTIABLE core goal of the research. Always resolved first.

2. secondary_research_purpose — The EXPLORATORY extension. What a thorough research report
   would ALSO include: breakdowns, comparisons, historical context, related metrics,
   subcategories, etc. This is the cherry on top — enriches output but not blocking.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES FOR targets
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"purpose" — TOP-LEVEL field, NOT inside attributes.
  Set only when there is a clear research intent (e.g., "sales outreach", "comparison").
  Omit entirely if not applicable.

"attributes" — ONLY known factual properties of the entity:
  ✅ Allowed: company, sport, metric, category, industry, domain, channels, relation
  ❌ Forbidden — never add these keys under any name variation:
       research_scope, research_focus, research_areas, focus, scope,
       what_to_find, study_areas, investigation_points
  ❌ Forbidden — never use placeholder values:
       "<UNKNOWN>", "N/A", "TBD", "unknown", null, empty string ""
  If a value is not explicitly known from the user's query — omit that key entirely.

RULE OF THUMB:
  If the key answers "what should I research?" → FORBIDDEN in attributes
  If the key answers "what do I already know about this entity?" → ALLOWED in attributes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ CORRECT:
{
    "type": "person",
    "name": "Hemant Gadodia",
    "priority": "primary",
    "purpose": "sales outreach for B2B AI marketing software",
    "attributes": {
        "company": "Epack"
    }
}

❌ WRONG — research directives inside attributes, unknowns padded:
{
    "type": "person",
    "name": "Hemant Gadodia",
    "priority": "primary",
    "attributes": {
        "company": "Epack",
        "purpose": "sales outreach",
        "research_scope": "professional background, values",
        "current_role": "<UNKNOWN>"
    }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "Count the number of centuries of Virat Kohli"
→ targets: [
    {
        "type": "person",
        "name": "Virat Kohli",
        "priority": "primary",
        "attributes": {
            "sport": "cricket",
            "metric": "centuries"
        }
    }
]
→ primary_research_purpose: "Find the total number of centuries scored by Virat Kohli
   across all cricket formats: Test, ODI, T20I, IPL, and domestic cricket."
→ secondary_research_purpose: "For each format, gather: list of all centuries with dates
   and venues, opponents, year-wise breakdown, balls faced, strike rates, century conversion
   rate, double/triple centuries, fastest centuries, centuries in winning vs losing causes,
   home vs overseas breakdown, and comparison with top contemporaries (Sachin, Root, Babar)."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "Show me the way to California"
→ targets: [
    {
        "type": "place",
        "name": "California",
        "priority": "primary",
        "attributes": {
            "query_type": "directions"
        }
    }
]
→ primary_research_purpose: "Provide directions to California — fastest route, major
   highways and interstate numbers, estimated travel time and distance."
→ secondary_research_purpose: "Include alternate routes (scenic, toll-free), traffic
   conditions and best travel times, rest stops, gas stations, key landmarks along the way,
   points of interest en route, and weather considerations for the journey."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "What is quantum computing?"
→ targets: [
    {
        "type": "concept",
        "name": "quantum computing",
        "priority": "primary",
        "attributes": {
            "domain": "technology"
        }
    }
]
→ primary_research_purpose: "Explain what quantum computing is — core concepts, how it
   works, and how it fundamentally differs from classical computing."
→ secondary_research_purpose: "Cover: qubits, superposition, entanglement, quantum gates,
   hardware requirements, current major players (IBM, Google, Microsoft, IonQ), real-world
   applications (cryptography, drug discovery, optimization, ML), limitations (decoherence,
   error correction), development timeline, and future industry impact."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "Research Hemant Gadodia from Epack to pitch B2B AI marketing software"
→ targets: [
    {
        "type": "person",
        "name": "Hemant Gadodia",
        "priority": "primary",
        "purpose": "sales outreach for B2B AI marketing software",
        "attributes": {
            "company": "Epack"
        }
    },
    {
        "type": "company",
        "name": "Epack",
        "priority": "secondary",
        "attributes": {
            "relation": "employer of Hemant Gadodia"
        }
    }
]
→ primary_research_purpose: "Find professional background, current role, and
   responsibilities of Hemant Gadodia at Epack. Understand Epack's business model,
   products, and current marketing approach to identify pain points for AI marketing tools."
→ secondary_research_purpose: "Deep dive into: Hemant's LinkedIn, thought leadership,
   speaking engagements, personal interests. Epack's tech stack, competitors, recent news,
   funding, and decision-making stakeholders. Best cold email/LinkedIn outreach templates
   for B2B SaaS, personalization opportunities, objection handling, and case studies of
   similar AI tool adoption."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "Compare iPhone 15 vs Samsung Galaxy S24"
→ targets: [
    {
        "type": "product",
        "name": "iPhone 15",
        "priority": "primary",
        "purpose": "comparison",
        "attributes": { "category": "smartphone" }
    },
    {
        "type": "product",
        "name": "Samsung Galaxy S24",
        "priority": "primary",
        "purpose": "comparison",
        "attributes": { "category": "smartphone" }
    }
]
→ primary_research_purpose: "Direct spec-for-spec comparison of iPhone 15 and Samsung
   Galaxy S24: processor, RAM, display, camera, battery, price, and OS."
→ secondary_research_purpose: "Extended comparison: benchmark scores, camera samples,
   ecosystem lock-in, software update lifespan, 5G performance, accessories, resale value,
   carrier deals, user reviews, expert ratings, and recommendation by use case
   (photography, gaming, business)."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "Tesla stock price"
→ targets: [
    {
        "type": "company",
        "name": "Tesla",
        "priority": "primary",
        "attributes": {
            "metric": "stock price",
            "domain": "finance"
        }
    }
]
→ primary_research_purpose: "Get Tesla's current stock price, today's change (amount
   and percentage), and after-hours trading price."
→ secondary_research_purpose: "Include: 52-week high/low, YTD performance, market cap,
   P/E ratio, trading volume vs average, analyst ratings and price targets, recent news
   affecting the stock, quarterly revenue and earnings, comparison with competitors
   (Ford, GM, Rivian, Lucid), institutional holders, insider trading, technical analysis
   (support/resistance, moving averages), and analyst outlook for next quarter."
"""

    user_prompt = f"Create a comprehensive research plan for this query: {user_query}"

    try:
        results = await claude_haiku(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            user_context=None,
            pydantic_model=ResearchIntent
        )

        return results

    except Exception as e:
        return {"error": f"Error classifying intent: {str(e)}"}


# Helper to print results
def print_intent(intent: ResearchIntent):
    print("\n" + "=" * 70)
    print("RESEARCH PLAN")
    print("=" * 70)

    print("\n📌 PRIMARY TARGETS:")
    for target in intent.get_primary_targets():
        print(f"   • {target.type}: {target.name}")
        if target.purpose:
            print(f"     purpose: {target.purpose}")
        if target.attributes:
            for key, value in target.attributes.items():
                print(f"     {key}: {value}")

    if intent.get_secondary_targets():
        print("\n📎 SECONDARY TARGETS:")
        for target in intent.get_secondary_targets():
            print(f"   • {target.type}: {target.name}")
            if target.purpose:
                print(f"     purpose: {target.purpose}")
            if target.attributes:
                for key, value in target.attributes.items():
                    print(f"     {key}: {value}")

    print(f"\n🎯 PRIMARY PURPOSE (Must-Have):")
    print(f"   {intent.primary_research_purpose}")

    print(f"\n✨ SECONDARY PURPOSE (Cherry on Top):")
    print(f"   {intent.secondary_research_purpose}")

    print(f"\n✅ Confidence: {intent.confidence}")


# Usage
async def main():
    test_queries = [
        "Count the number of centuries of Virat Kohli",
        "What is quantum computing?",
        "Research Hemant Gadodia from Epack to pitch B2B AI marketing software"
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"USER QUERY: {query}")
        print('=' * 70)

        intent = await intent_prompt(query)
        print_intent(intent)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
