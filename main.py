from prompts.intent_prompt_file import intent_prompt
from processes.step0 import step0_function




async def main_function(research_type: str, query: str):

    #  research_documantation
    research = {}
    
    user_intent = await intent_prompt(query)
    research["user_intent"] = user_intent

    research = await step0_function(research)

    
    if research_type=="Shallow":
        return research
    



    if research_type=="Intermediate":
        return research
    




    return research