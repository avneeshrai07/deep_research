from prompts.intent_prompt_file import intent_prompt
from processes.step0 import step0_function
from processes.shallow_prompt import shallow_research_prompt



async def main_function(research_type: str, query: str):

    #  research_documantation
    research = {}
    
    user_intent = await intent_prompt(query)
    research["user_intent"] = user_intent

    research["used_queries"] = []
    research["research_data"] = []
    research = await step0_function(research)

    print("research:    ",research)
    if research_type=="Shallow":
        shallow_reasearch = await shallow_research_prompt(research=research)
        research["shallow_reasearch"] = shallow_reasearch
        return research
    



    if research_type=="Intermediate":
        return research
    




    return research