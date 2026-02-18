def extract_linkedin_username(url: str) -> str:
    """
    Extracts the username from a LinkedIn profile or company URL.
    Example:
        https://www.linkedin.com/in/rkts7258/  -> rkts7258
        https://www.linkedin.com/company/openai/ -> openai
    """
    if not url:
        return ""
    url = url.rstrip("/")   # remove trailing slash
    return url.split("/")[-1]
