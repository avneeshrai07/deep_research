from api.person_details import fetch_person_details
from tools.tavily import tavily_web_search_function
from helper.pattern_match import match_pattern
from api.person_post import get_all_posts
from helper.extractor import extract_linkedin_username
from helper.mpnet_keyword_extractor import MPNetExtractor
import asyncio

async def linkedin_post_extractor(
    linkedin_person_details
    ):
        try:
            data = linkedin_person_details.get("data",{})
            user_linkedin = data.get("profile_link","")
            print("user_linkedin:   ", user_linkedin)

            user_id = extract_linkedin_username(user_linkedin)
            print("user_id", user_id)


            user_data, user_post = get_all_posts(user_id)
            print("user_data in PersonColdEmail:   \n\n", user_data, "\n\n")
            print("user_post RAW:   \n\n", user_post, "\n\n")

            if user_data is None:
                user_data = []
            if user_post is None:
                return {
                "user_data": {},
                "keyword_posts": [],
                "cluster_posts": []
            }

            # **FILTER OUT EMPTY POSTS**
            valid_posts = [
                post for post in user_post 
                if post.get('title', '').strip() or post.get('text', '').strip()
            ]
            
            print(f"Valid posts after filtering: {len(valid_posts)} out of {len(user_post)}")

            # **CHECK IF NO VALID POSTS**
            if not valid_posts:
                return {
                    "user_data": {},
                    "keyword_posts": [],
                    "cluster_posts": []
                }


            # Initialize extractor
            extractor = MPNetExtractor(user_intent="user_post")
            results = extractor.extract(valid_posts)


            # Run both extraction methods in parallel
            loop = asyncio.get_event_loop()
            
            keyword_posts, cluster_posts = await asyncio.gather(
                loop.run_in_executor(
                    None,
                    extractor.extract_top_n,
                    valid_posts,  # Use filtered posts
                    3,
                    0.3,
                    True,
                    True,
                    32
                ),
                loop.run_in_executor(
                    None,
                    extractor.extract_top_cluster,
                    valid_posts,  # Use filtered posts
                    5,
                    True,
                    32,
                    0.3,
                    3
                ),
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(keyword_posts, Exception):
                print(f"Keyword extraction failed: {keyword_posts}")
                keyword_posts = []
            
            if isinstance(cluster_posts, Exception):
                print(f"Cluster extraction failed: {cluster_posts}")
                cluster_posts = []

            print("keyword_posts (top 3):   \n\n", keyword_posts, "\n\n")
            print("cluster_posts (0-5):   \n\n", cluster_posts, "\n\n")

            

            return {
                "user_data": user_data,
                "keyword_posts": keyword_posts,
                "cluster_posts": cluster_posts
            }
            
        except Exception as e:
            print(f"Error in linkedin_post_extractor: {e}")

            return {
                "user_data": {},
                "keyword_posts": [],
                "cluster_posts": []
            }





async def person_workflow_function(target: dict) -> dict:
    try:
        name = target.get("name", "")
        attributes = target.get("attributes", {})

        # Build basic_details string from attributes
        basic_details = " | ".join(
            f"{key}={value}" for key, value in attributes.items()
        )
        print("basic_details:   ", basic_details)

        # Fetch LinkedIn person details
        try:
            linkedin_person_details = await fetch_person_details(
                user_name=name,
                basic_details=basic_details
            )
        except Exception as e:
            print(f"❌ fetch_person_details failed for '{name}': {str(e)}")
            linkedin_person_details = {}

        # Tavily web search
        try:
            query = f"About ('{name}' '{basic_details}') -inurl:hiring -inurl:jobs -inurl:careers -inurl:activity"
            query_result = await tavily_web_search_function(query=query)
            print("query_result:    ",query_result)
            results = query_result.get("results", [])
        except Exception as e:
            print(f"❌ tavily_web_search_function failed for '{name}': {str(e)}")
            results = []

        # Filter results by regex match on person name
        filtered_query_result = []
        for single_result in results:
            try:
                content = single_result.get("content", "")
                if await match_pattern(content, name):
                    filtered_query_result.append(single_result)
            except Exception as e:
                print(f"❌ regex match failed for result: {str(e)}")
                continue

        print("filtered_query_result:    ", filtered_query_result)

        return {
            "linkedin_about": linkedin_person_details,
            "web_results_about": filtered_query_result
        }

    except Exception as e:
        print(f"❌ person_workflow_function failed for target '{target}': {str(e)}")
        return {"error": str(e)}
