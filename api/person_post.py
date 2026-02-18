from dotenv import load_dotenv
load_dotenv()
import requests
import re
import os
base_url = os.getenv("LINKEDIN_API","https://linkedin.miatibro.art/api/v1")
def get_all_posts(user_name):
    api_base = base_url + "/unipile/user/"
    
    # Step 1: Get provider_id from user_name
    user_url = api_base + user_name
    print("Fetching user data:", user_url)
    response = requests.get(user_url)

    if response.status_code != 200:
        print("Error fetching user data:", response.status_code, response.text)
        return None, None
    
    user_data_raw = response.json()
    user = user_data_raw.get("user", {})
    provider_id = user.get("provider_id")

    if not provider_id:
        print("No provider_id found for user:", user_name)
        print("Full JSON response:", user_data_raw)  # debug
        return None, None
    
    print("user:    ",user)
    print("provider_id:    ",provider_id)

    # Extract useful user data
    user_data = {
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "headline": user.get("headline"),
        "location": user.get("location"),
        # "follower_count": user.get("follower_count"),
        # "connections_count": user.get("connections_count"),
        "websites": user.get("websites", []),
        "emails": user.get("contact_info", {}).get("emails", [])
        # "profile_picture_url": user.get("profile_picture_url"),
        # "provider_id": provider_id,
        # "public_identifier": user.get("public_identifier")
    }
    
    # Step 2: Fetch posts using provider_id
                # https://linkedin.miatibro.art
    post_base = base_url + "/users/"
    posts_url = f"{post_base}{provider_id}/posts"
    print("Fetching posts from:", posts_url)
    response = requests.get(posts_url)

    if response.status_code != 200:
        print("Error fetching posts:", response.status_code, response.text)
        return user_data, None
    
    posts_data = response.json()
    posts_list = posts_data.get("posts", {}).get("items", [])
    
    if not posts_list:
        print("No posts found for provider:", provider_id)
        return user_data, []
    
    # Step 3: Take top N posts
    top_posts = []
    for post in posts_list:
        text = post.get("text", "").strip()
        match = re.match(r'^[^\n]*', text)
        title = match.group(0)
        share_url = post.get("share_url")
        author = post.get("author", {}).get("name", "")
        
        attachments = post.get("attachments", [])
        attachment_urls = [a.get("url") for a in attachments if "url" in a]
        
        top_posts.append({
            "title": title,
            # "author": author,
            "text": text,
            # "share_url": share_url,
            # "attachments": attachment_urls
        })
    return user_data, top_posts


# Example usage
if __name__ == "__main__":
    user_data, top_posts = get_all_posts("sa2003hil")
    
    print("\n=== User Data ===")
    print(user_data)
    
    print("\n=== Top Posts ===")
    for i, post in enumerate(top_posts, start=1):
        print(f"\nPost {i}:")
        print(post)

