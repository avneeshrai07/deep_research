from prompts.intent_prompt_file import intent_prompt
from processes.step0 import step0_function
from processes.shallow_prompt import shallow_research_prompt
from tools.tavily import tavily_web_search_function
from processes.intermidiate_prompt import intermediate_research_prompt




async def main_function(research_type: str, query: str):

    #  research_documantation
    research = {}
    
    user_intent = await intent_prompt(query)
    research["user_intent"] = user_intent
    research["used_queries"] = []
    research["research_data"] = []
    research["DeepResearch"] = []
    research = await step0_function(research)

    shallow_reasearch = await shallow_research_prompt(research=research)
    remaining_primary_research_purpose = shallow_reasearch.get("remaining_primary_research_purpose",[])
    remaining_secondary_research_purpose = shallow_reasearch.get("remaining_primary_research_purpose",[])
    search_queries = shallow_reasearch.get("search_queries",[])
    notes = shallow_reasearch.get("notes",[])
    print("search_queries:  ",search_queries)
    research["DeepResearch"].extend(notes)
    if research_type=="Shallow":
        return research["DeepResearch"]
    

    step_2_research_data = []
    for single_query in search_queries:
        data = await tavily_web_search_function(single_query)
        step_2_research_data.append(data)
        research["used_queries"].append(single_query)



    intermidiate_reasearch = await intermediate_research_prompt(research,step_2_research_data, remaining_primary_research_purpose, remaining_secondary_research_purpose, notes)
    remaining_primary_research_purpose = intermidiate_reasearch.get("remaining_primary_research_purpose",[])
    remaining_secondary_research_purpose = intermidiate_reasearch.get("remaining_primary_research_purpose",[])
    search_queries = intermidiate_reasearch.get("search_queries",[])
    notes = intermidiate_reasearch.get("notes",[])
    print("search_queries:  ",search_queries)
    research["DeepResearch"].extend(notes)
    if research_type=="Intermediate":
        return research["DeepResearch"]
    




    return research["DeepResearch"]