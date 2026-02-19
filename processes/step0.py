from workflows.person import person_workflow_function
from workflows.company import company_workflow_function
import asyncio

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

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Workflow failed: {result}")
            else:
                research["research_data"].append(result)

    return research
