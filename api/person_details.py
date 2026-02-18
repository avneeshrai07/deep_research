import aiohttp
import asyncio
from typing import Optional


async def fetch_person_details(
    user_name: str,
    basic_details: str,
    timeout: int = 120
) -> dict:
    url = "https://onboardsapi.miatibro.art/person_details"
    payload = {
        "user_name": user_name,
        "basic_details": basic_details
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return {"success": True, "data": data}

    except aiohttp.ClientResponseError as e:
        return {"success": False, "error": f"HTTP {e.status}: {e.message}", "status_code": e.status}

    except aiohttp.ClientConnectionError as e:
        return {"success": False, "error": f"Connection failed: {str(e)}"}

    except asyncio.TimeoutError:
        return {"success": False, "error": f"Request timed out after {timeout}s"}

    except aiohttp.ContentTypeError as e:
        return {"success": False, "error": f"Invalid response format (expected JSON): {str(e)}"}

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


# Usage
async def main():
    result = await fetch_person_details(
        user_name="Punya Modi",
        basic_details="IIIT Bhopal"
    )

    if result["success"]:
        print("✅ Success:", result["data"])
    else:
        print("❌ Error:", result["error"])


if __name__ == "__main__":
    asyncio.run(main())
