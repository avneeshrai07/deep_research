async def query_creator_function(single_query: dict) -> str:
    name = single_query["name"]
    qtype = single_query["type"]
    pid = single_query["primary_identifier"]
    q = single_query["query"]

    full = f"{qtype} ('{name}' '{pid}') {q}"
    if len(full) < 400:
        return full

    without_pid = f"{qtype} ('{name}') {q}"
    if len(without_pid) < 400:
        return without_pid

    without_type = f"('{name}') {q}"
    if len(without_type) < 400:
        return without_type

    without_name = f"{qtype} {q}"
    if len(without_name) < 400:
        return without_name

    return q
