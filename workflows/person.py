from api.person_details import fetch_person_details
from tools.tavily import tavily_web_search_function
from helper.pattern_match import match_pattern
from api.person_post import get_all_posts
from helper.extractor import extract_linkedin_username
from helper.mpnet_keyword_extractor import MPNetExtractor
import asyncio


async def linkedin_post_extractor(
    target, research: dict
):

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

    try:
        data = linkedin_person_details.get("data", {})
        user_linkedin = data.get("profile_link", "")
        print("user_linkedin:   ", user_linkedin)

        user_id = extract_linkedin_username(user_linkedin)
        print("user_id", user_id)

        # FIX 1: Offload blocking sync call to thread pool
        user_data, user_post = await asyncio.to_thread(get_all_posts, user_id)
        print("user_data in PersonColdEmail:   \n\n", user_data, "\n\n")


        if user_data is None:
            user_data = []
        if user_post is None:
            return {
                "user_data": {},
                "keyword_posts": [],
                "cluster_posts": []
            }

        # Filter out empty posts
        valid_posts = [
            post for post in user_post
            if post.get('title', '').strip() or post.get('text', '').strip()
        ]

        print(f"Valid posts after filtering: {len(valid_posts)} out of {len(user_post)}")

        if not valid_posts:
            return {
                "user_data": {},
                "keyword_posts": [],
                "cluster_posts": []
            }

        # FIX 2: Removed unused `results = extractor.extract(valid_posts)` call
        extractor = MPNetExtractor(user_intent="user_post")

        # FIX 3: Replaced deprecated get_event_loop().run_in_executor with asyncio.to_thread
        keyword_posts, cluster_posts = await asyncio.gather(
            asyncio.to_thread(
                extractor.extract_top_n,
                valid_posts,
                3,
                0.3,
                True,
                True,
                32
            ),
            asyncio.to_thread(
                extractor.extract_top_cluster,
                valid_posts,
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


async def web_search_function(
    target: dict, research: dict
):
    try:
        name = target.get("name", "")
        attributes = target.get("attributes", {})

        # Build basic_details string from attributes
        basic_details = " | ".join(
            f"{key}={value}" for key, value in attributes.items()
        )
        print("basic_details:   ", basic_details)

        try:
            query = f"About ('{name}' '{basic_details}') -inurl:hiring -inurl:jobs -inurl:careers -inurl:activity"

            query_result = await tavily_web_search_function(query=query)
            print("query_result:    ", query_result)
            research["used_queries"].append(query)
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
            "web_results_about": filtered_query_result
        }

    except Exception as e:
        print(f"❌ person_workflow_function failed for target '{target}': {str(e)}")
        return {"error": str(e)}


async def person_workflow_function(target: dict, research: dict) -> dict:
    try:
        # FIX 4: Added missing `await` and missing comma between coroutines
        # FIX 5: Added return_exceptions=True for safe parallel execution
        linkedin_posts, web_results_about = await asyncio.gather(
            linkedin_post_extractor(target, research),
            web_search_function(target, research),
            return_exceptions=True
        )

        # Handle top-level exceptions from gather
        if isinstance(linkedin_posts, Exception):
            print(f"❌ linkedin_post_extractor raised: {linkedin_posts}")
            linkedin_posts = {"user_data": {}, "keyword_posts": [], "cluster_posts": []}

        if isinstance(web_results_about, Exception):
            print(f"❌ web_search_function raised: {web_results_about}")
            web_results_about = {"web_results_about": []}

        return {
            "linkedin_about": linkedin_posts,
            "web_results_about": web_results_about
        }

    except Exception as e:
        print(f"❌ person_workflow_function failed for target '{target}': {str(e)}")
        return {"error": str(e)}
