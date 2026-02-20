from workflows.person import person_workflow_function
from workflows.company import company_workflow_function
from workflows.others import other_workflow_function
import asyncio
from tools.tavily import tavily_web_search_function
from helper.websearch_filter import filter_results
async def step0_function(research: dict) -> dict:
    user_intent = research.get("user_intent")
    targets = user_intent.get("targets", [])

    # Build coroutines for all primary targets in parallel
    tasks = []
    for single_target in targets:
        if single_target["priority"] == "primary":
            if single_target["type"] == "person":
                print("perosn")
                tasks.append(person_workflow_function(single_target, research))
            elif single_target["type"] == "company":
                tasks.append(company_workflow_function(single_target, research))
            else:
                tasks.append(other_workflow_function(user_intent, research))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    

        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Workflow failed: {result}")
            else:
                research["research_data"].append(result)

    return research
