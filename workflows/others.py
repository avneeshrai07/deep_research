from api.person_details import fetch_person_details
from tools.tavily import tavily_web_search_function
from helper.websearch_filter import weak_filter_results
from helper.mpnet_keyword_extractor import MPNetExtractor
import asyncio


async def other_workflow_function(user_intent: dict, research: dict) -> dict:
    try:
        primary_research_purpose = user_intent.get("primary_research_purpose")
        primary_research_purpose = primary_research_purpose[:400]
        data            = await tavily_web_search_function(primary_research_purpose)
        research["used_queries"].append(primary_research_purpose)
        filtered_data   = await weak_filter_results(data=data)
        print("filtered_data other_workflow_function:   ",filtered_data)
        return filtered_data

    except Exception as e:
        print(f"❌ other_workflow_function failed for primary_research_purpose '{primary_research_purpose}': {str(e)}")
        return {"error": str(e)}
