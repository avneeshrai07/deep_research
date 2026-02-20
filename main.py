from prompts.intent_prompt_file import intent_prompt
from processes.step0 import step0_function
from processes.shallow_prompt import shallow_research_prompt
from tools.tavily import tavily_web_search_function
from processes.intermidiate_prompt import intermediate_research_prompt
from processes.deep_reasearch_prompt import deep_research_prompt
from helper.query_creator import query_creator_function
from helper.websearch_filter import filter_results
from helper.websearch_filter import update_completed_topics

async def main_function(research_type: str, query: str):

    #  research_documantation
    research = {}
    
    user_intent = await intent_prompt(query)
    print("user_intent: ", user_intent)
    research["user_intent"] = user_intent
    research["used_queries"] = []
    research["research_data"] = []
    research["DeepResearch"] = []
    completed_topics = []
    research = await step0_function(research)
    print("research step0:  ",research)
    shallow_reasearch = await shallow_research_prompt(research=research)
    print("shallow_reasearch:   ",shallow_reasearch)
    remaining_primary_research_purpose = shallow_reasearch.get("remaining_primary_research_purpose",[])
    remaining_secondary_research_purpose = shallow_reasearch.get("remaining_primary_research_purpose",[])
    search_queries = shallow_reasearch.get("search_queries",[])
    notes = shallow_reasearch.get("notes",[])
    completed_topics = await update_completed_topics(completed_topics, notes)
    print("search_queries:  ",search_queries)
    research["DeepResearch"].extend(notes)
    if research_type=="Shallow":
        return research["DeepResearch"]
    

    step_2_research_data = []
    for single_query in search_queries:
        query = await query_creator_function(single_query)
        data = await tavily_web_search_function(query)
        filtered_data = await filter_results(data=data, keyword=single_query["name"])
        print("filtered_data:   ",filtered_data)
        step_2_research_data.extend(filtered_data)
        research["used_queries"].append(single_query)


    print("type completed_topics 1: ", type(completed_topics))
    intermidiate_reasearch = await intermediate_research_prompt(research,step_2_research_data, remaining_primary_research_purpose, remaining_secondary_research_purpose, completed_topics)
    remaining_primary_research_purpose = intermidiate_reasearch.get("remaining_primary_research_purpose",[])
    remaining_secondary_research_purpose = intermidiate_reasearch.get("remaining_primary_research_purpose",[])
    search_queries = intermidiate_reasearch.get("search_queries",[])
    notes = intermidiate_reasearch.get("notes",[])
    completed_topics = await update_completed_topics(completed_topics, notes)
    print("search_queries:  ",search_queries)
    research["DeepResearch"].extend(notes)
    if research_type=="Intermediate":
        return research["DeepResearch"]
    

    step_3_research_data = []
    for single_query in search_queries:
        query = await query_creator_function(single_query)
        data = await tavily_web_search_function(query)
        filtered_data = await filter_results(data=data, keyword=single_query["name"])
        print("filtered_data:   ",filtered_data)
        step_3_research_data.extend(filtered_data)
        research["used_queries"].append(single_query)


    deep_reasearch = await deep_research_prompt(research,step_3_research_data, remaining_primary_research_purpose, remaining_secondary_research_purpose, completed_topics)

    notes = deep_reasearch.get("notes",[])
    research["DeepResearch"].extend(notes)

    if research_type=="Deep":
        return research["DeepResearch"]
