from prompts.shallow_prompt import shallow_prompt


async def main_function(research_type: str, query: str):

    #  shallow
    research = []
    
    shallow_reasearch = await shallow_prompt(query)
    research.append(shallow_reasearch)
    if research_type=="Shallow":
        return research
    



    if research_type=="Intermediate":
        return research
    




    return research