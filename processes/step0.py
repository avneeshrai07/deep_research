
from workflows.person import person_workflow_function

async def step0_function(research: dict) -> dict:
    user_intent = research.get("user_intent")
    targets = user_intent.get("targets", [])

    for single_target in targets:
        if single_target["type"] == "person" and single_target["priority"] == "primary":
            research["person"] = await person_workflow_function(single_target)

    return research
