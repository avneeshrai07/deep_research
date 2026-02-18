from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json
import traceback
import asyncio
from typing import List

load_dotenv()

# -------------------- Tavily API Keys --------------------
TEST_TAVILY_KEYS = [
    os.getenv(f"TAVILY_API_KEY{'' if i == 1 else i}")
    for i in range(1, 13)
]


from datetime import date
from dateutil.relativedelta import relativedelta

end_date = date.today().isoformat()
start_date = (date.today() - relativedelta(years=1)).isoformat()


# -------------------- Initialize clients safely --------------------
def _initialize_clients(api_keys: List[str]):
    clients = []
    for key in api_keys:
        if key:
            try:
                clients.append(TavilyClient(api_key=key))
            except Exception:
                pass
    return clients


# -------------------- Function Definition --------------------
async def tavily_web_search_function(query: str, tavily_api_keys: List[str] = TEST_TAVILY_KEYS) -> dict:
    clients = _initialize_clients(tavily_api_keys)

    if not clients:
        return {"error": "No valid Tavily API clients were initialized."}

    for client in clients:
        try:
            response = await asyncio.to_thread(
                client.search,
                query=query,
                include_answer="advanced",
                search_depth="advanced",
                max_results=20,
                start_date=start_date,
                end_date=end_date,
            )
            return response  # ✅ raw dict, not json.dumps

        except Exception as e:
            tb = traceback.format_exc()
            print(f"❌ Tavily client failed: {e}\n{tb}")
            continue

    return {"error": "All Tavily clients failed to retrieve web search results."}
