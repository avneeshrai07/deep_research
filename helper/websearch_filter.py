import re

async def filter_results(data: dict, keyword: str) -> list[dict]:
    pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    overall = []
    results = data.get("results",[])
    for item in results:
        if item.get("score", 0) > 0.90 and pattern.search(item.get("content", "")):
            overall.append({
                "url":     item.get("url"),
                "title":   item.get("title"),
                "content": item.get("content"),
                "score":   item.get("score"),
            })

    return overall



async def update_completed_topics(
    completed_topics: list[str],
    new_notes: list[dict],
) -> list[str]:
    existing = set(completed_topics)
    if not new_notes:
        return completed_topics
    for note in new_notes:
        topic = note.get("topic")
        if topic and topic not in existing:
            completed_topics.append(topic)
            existing.add(topic)

    return completed_topics