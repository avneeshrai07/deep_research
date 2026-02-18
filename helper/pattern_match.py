import re


async def match_pattern(content: str, pattern: str) -> bool:
    try:
        return bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL))
    except re.error:
        return False