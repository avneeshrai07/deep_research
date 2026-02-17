from langchain_core.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

from llm.hiaku import claude_haiku

# AWS credentials
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


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
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible key-value attributes relevant to this entity"
    )


class ResearchIntent(BaseModel):
    targets: List[ResearchTarget] = Field(
        description="All research targets with their flexible attributes"
    )
    research_purpose: str = Field(
        description="COMPREHENSIVE and EXPLORATORY description of what to research. This should guide the LLM to gather EXTENSIVE information, covering all dimensions, categories, subcategories, and related metrics. Think: 'What would a thorough research report include?' NOT a short summary."
    )
    confidence: str = Field(
        description="Confidence: 'high', 'medium', 'low'"
    )
    
    def get_primary_targets(self) -> List[ResearchTarget]:
        return [t for t in self.targets if t.priority == EntityPriority.PRIMARY]
    
    def get_secondary_targets(self) -> List[ResearchTarget]:
        return [t for t in self.targets if t.priority == EntityPriority.SECONDARY]


async def shallow_prompt(user_query: str) -> ResearchIntent:
    """
    Classifies user's research intent - GENERAL PURPOSE for any query type
    """
    
    system_prompt = """
You are a research planning system. Your job is to create a COMPREHENSIVE research plan 
from the user's query.

CRITICAL: The research_purpose should be EXPLORATORY and EXPANSIVE.
- NOT: "get century statistics" ❌
- YES: "Find all centuries of Virat Kohli across all formats (Test, ODI, T20I, IPL, domestic cricket including Ranji Trophy, Syed Mushtaq Ali, Vijay Hazare). For each format, get the total count, highest scores, venues, opponents, year-wise breakdown, and balls faced. Also include century conversion rate, number of double centuries, and comparison with other top players." ✅

Think: What would a COMPLETE research report include? Cover all angles, formats, categories, metrics, and comparisons.

Examples:

Query: "Count the number of centuries of Virat Kohli"
→ Targets: [
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
→ Research Purpose: "Find all centuries scored by Virat Kohli across ALL cricket formats: Test matches, One Day Internationals (ODIs), T20 Internationals, Indian Premier League (IPL), domestic tournaments (Ranji Trophy, Syed Mushtaq Ali Trophy, Vijay Hazare Trophy, Duleep Trophy). For each format, gather: total century count, list of all centuries with dates and venues, highest scores in each format, opponents against whom centuries were scored, year-wise breakdown, innings-wise breakdown (1st innings, 2nd innings), balls faced for each century, strike rates, century conversion rate (centuries per innings played), number of double centuries and triple centuries, fastest centuries, centuries in winning vs losing causes, overseas vs home centuries, and comparison with other top batsmen in the same era."

Query: "Show me the way to California"
→ Targets: [
    {
        "type": "place",
        "name": "California",
        "priority": "primary",
        "attributes": {
            "query_type": "directions"
        }
    }
]
→ Research Purpose: "Provide comprehensive directions to California including multiple route options (fastest route, scenic route, toll-free route), estimated travel time for each route, distance in miles/kilometers, major highways and interstate numbers, key cities and landmarks along the way, rest stops and gas stations, traffic conditions and best travel times, alternate routes in case of traffic, points of interest worth visiting en route, and weather considerations for the journey."

Query: "What is quantum computing?"
→ Targets: [
    {
        "type": "concept",
        "name": "quantum computing",
        "priority": "primary",
        "attributes": {
            "domain": "technology"
        }
    }
]
→ Research Purpose: "Provide a comprehensive explanation of quantum computing covering: fundamental principles and how it differs from classical computing, key concepts (qubits, superposition, entanglement, quantum gates), how quantum computers are built and their physical requirements, current state of quantum computing technology and major players (IBM, Google, Microsoft, IonQ), real-world applications and use cases (cryptography, drug discovery, optimization, machine learning), limitations and challenges (decoherence, error correction, scalability), comparison with classical computing performance, timeline of quantum computing development, future predictions and potential impact on industries, and resources for learning more."

Query: "Research Hemant Gadodia from Epack to pitch B2B AI marketing software"
→ Targets: [
    {
        "type": "person",
        "name": "Hemant Gadodia",
        "priority": "primary",
        "attributes": {
            "company": "Epack",
            "purpose": "sales outreach"
        }
    },
    {
        "type": "company",
        "name": "Epack",
        "priority": "secondary",
        "attributes": {
            "relation": "employer"
        }
    }
]
→ Research Purpose: "Conduct deep research on Hemant Gadodia including: professional background and career history, current role and responsibilities at Epack, LinkedIn profile and professional network, educational background, leadership philosophy and values, public statements or interviews, speaking engagements and thought leadership content, personal interests and causes. For Epack, research: company overview and business model, products and services offered, target market and customers, company size and revenue, recent news and announcements, technology stack and tools currently used, marketing strategies and channels, pain points in current marketing approach, competitors and market position, company culture and values, recent funding or growth, decision-making process and key stakeholders. Additionally, research: best practices for reaching executives at similar companies, effective cold email templates for B2B SaaS, LinkedIn outreach strategies, common objections and how to address them, case studies of similar companies using AI marketing tools, and personalization opportunities based on Hemant's interests and Epack's needs."

Query: "Compare iPhone 15 vs Samsung Galaxy S24"
→ Targets: [
    {
        "type": "product",
        "name": "iPhone 15",
        "priority": "primary",
        "attributes": {
            "category": "smartphone"
        }
    },
    {
        "type": "product",
        "name": "Samsung Galaxy S24",
        "priority": "primary",
        "attributes": {
            "category": "smartphone"
        }
    }
]
→ Research Purpose: "Conduct comprehensive comparison of iPhone 15 and Samsung Galaxy S24 covering: detailed specifications (processor, RAM, storage options, display size and quality, resolution, refresh rate), camera systems (megapixels, number of lenses, night mode, video capabilities, zoom quality, AI features), battery life and charging speeds (wired and wireless), design and build quality (materials, weight, dimensions, colors available), operating systems and software features (iOS vs Android, exclusive features, update support), performance benchmarks (gaming, multitasking, thermal management), pricing across different storage tiers and carrier deals, ecosystem integration (smartwatch, earbuds, tablets), 5G and connectivity features, durability and water resistance ratings, expert reviews and user ratings, pros and cons of each, best use cases for each device, resale value, and final recommendation based on different user priorities."

Query: "Tesla stock price"
→ Targets: [
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
→ Research Purpose: "Provide comprehensive Tesla stock analysis including: current stock price and today's change (dollar amount and percentage), after-hours trading price, 52-week high and low prices, year-to-date performance, market capitalization, price-to-earnings ratio, trading volume compared to average, analyst ratings and price targets (consensus, highest, lowest), recent news affecting the stock price, quarterly and annual revenue and earnings, earnings call highlights, comparison with competitors (Ford, GM, Rivian, Lucid), major institutional holders, insider trading activity, dividend information (if any), technical analysis (support and resistance levels, moving averages), historical price chart (1 month, 6 months, 1 year, 5 years), factors affecting future price (production numbers, new model launches, regulatory changes), and analyst outlooks for next quarter and year."


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
        if target.attributes:
            for key, value in target.attributes.items():
                print(f"     {key}: {value}")
    
    if intent.get_secondary_targets():
        print("\n📎 SECONDARY TARGETS:")
        for target in intent.get_secondary_targets():
            print(f"   • {target.type}: {target.name}")
            if target.attributes:
                for key, value in target.attributes.items():
                    print(f"     {key}: {value}")
    
    print(f"\n🎯 RESEARCH PURPOSE (Comprehensive):")
    print(f"   {intent.research_purpose}")
    
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
        
        intent = await shallow_prompt(query)
        print_intent(intent)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
