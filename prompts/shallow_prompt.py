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

class IntentType(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    UPDATE = "update"
    EXPLANATION = "explanation"
    RECOMMENDATION = "recommendation"

class ResearchIntent(BaseModel):
    domain_entity: Dict[str, List[str]] = Field(
        description="Mapping of research domains to their corresponding entities. Keys are domains (person, company, place, product, event, concept, sports, statistics, comparison, news, historical, technical), values are lists of entity names"
    )
    intent_type: IntentType = Field(
        description="What type of information the user is seeking"
    )
    temporal_focus: Optional[str] = Field(
        default="current",
        description="Time focus: 'current', 'historical', 'recent', 'future', or specific period"
    )
    specific_aspects: List[str] = Field(
        default=[],
        description="Specific aspects or dimensions the user wants to know about"
    )
    confidence: str = Field(
        description="Confidence in intent classification: 'high', 'medium', 'low'"
    )

async def shallow_prompt(user_query: str) -> ResearchIntent:
    """
    Classifies user's research intent before actual research
    """
    
    
    system_prompt = """
You are an intent classification system. Analyze the user's research query and 
identify what they're trying to learn about.

Your task is to:
1. Identify the primary research domains (person, company, place, product, event, concept, sports, statistics, comparison, news, historical, technical)
2. Determine the intent type (what kind of information they want)
3. Extract key entities mentioned
4. Identify temporal focus (current, historical, recent developments, etc.)
5. Note any specific aspects they care about

Be precise but flexible - a query can belong to multiple domains.

Examples:

Query: "Nvidia's AI chip market share"
→ Domains: [company, statistics, technical]
→ Intent: analysis
→ Entities: ["Nvidia", "AI chips"]
→ Aspects: ["market share"]

Query: "Tell me about Cristiano Ronaldo's current team"
→ Domains: [person, sports]
→ Intent: factual_lookup
→ Entities: ["Cristiano Ronaldo"]
→ Temporal: current
→ Aspects: ["team", "current status"]

Query: "Compare iPhone 15 vs Samsung Galaxy S24"
→ Domains: [product, comparison, technical]
→ Intent: comparison
→ Entities: ["iPhone 15", "Samsung Galaxy S24"]

Query: "What happened in the 2024 US elections"
→ Domains: [event, news, historical]
→ Intent: factual_lookup
→ Entities: ["2024 US elections"]
→ Temporal: historical

Query: "Explain quantum computing"
→ Domains: [concept, technical]
→ Intent: explanation
→ Entities: ["quantum computing"]

{format_instructions}
"""

    user_prompt = f"Classify this research query: {user_query}"
    
    try:
        results = await claude_haiku(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            user_context=None,
            pydantic_model=ResearchIntent
        )
        
        return results
        
    except Exception as e:
        return {f"Error classifying intent: {str(e)}"}


# Usage example
async def main():
    user_query = "Tesla's stock performance in 2025"
    
    # Step 1: Classify intent
    intent = await shallow_prompt(user_query)
    
    print(f"Domains: {intent.primary_domains}")
    print(f"Intent Type: {intent.intent_type}")
    print(f"Entities: {intent.entities}")
    print(f"Temporal Focus: {intent.temporal_focus}")
    print(f"Specific Aspects: {intent.specific_aspects}")
    
    # Step 2: Use this intent to guide your research
    # Now you know it's about: company + statistics + financial
    # So you can tailor your search and response structure accordingly
