from dotenv import load_dotenv
import os
import json
import requests
from urllib.parse import quote_plus
from snapshot_operations import download_snapshot,poll_snapshot_status

dataset_id="gd_lvz8ah06191smkebj4"

load_dotenv()

def _make_api_request(url,**kwargs):
    api_key= os.getenv("BRIGHTDATA_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url,headers=headers,**kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except Exception as e:
        print(f"Unknown error: {e}")
        return None

def serp_search(query,engine="google"):
    if engine == "google":
        base_url= "https://www.google.com/search"
    elif engine == "bing":
        base_url = "https://www.bing.com/search"
    else:
        raise ValueError(f"Unknown engine {engine}")
    
    url = "https://api.brightdata.com/request"

    payload ={
        "zone": "agent",
        "url": f"{base_url}?q={quote_plus(query)}&brd_json=1",
        "format":"raw"
    }

    full_response = _make_api_request(url,json=payload)
    if not full_response:
        return None
    
    extracted_data = {
        "knowledge": full_response.get("knowledge", {}),
        "organic": full_response.get("organic",[])
    }
    return extracted_data

def _trigger_and_download_snapshot(trigger_url,params,data,operation_name="operation"):
    trigger_result = _make_api_request(trigger_url,params=params,json=data)
    if not trigger_result:
        return None
    
    snapshot_id = trigger_result.get("snapshot_id")
    if not snapshot_id:
        return None
    
    if not poll_snapshot_status(snapshot_id):
        return None
    raw_data = download_snapshot(snapshot_id)
    return raw_data



def reddit_search_api(keyword, date="All time", sort_by="Hot", num_of_posts=75):
    """
    Finds relevant Reddit posts AND retrieves comments for the top results.
    """
    # --- STEP 1: Find relevant posts (existing logic) ---
    trigger_url = "https://api.brightdata.com/datasets/v3/trigger"
    params = {
        "dataset_id": "gd_lvz8ah06191smkebj4", # Dataset for post searching
        "include_errors": "true",
        "type": "discover_new",
        "discover_by": "keyword"
    }
    data = [{"keyword": keyword, "date": date, "sort_by": sort_by, "num_of_posts": num_of_posts}]

    raw_post_data = _trigger_and_download_snapshot(trigger_url, params, data, operation_name="reddit search")
    if not raw_post_data:
        return {"parsed_posts": [], "total_found": 0, "comments_retrieved": 0}

    parsed_posts = []
    for item in raw_post_data:
        post = None
        try:
            post = json.loads(item) if isinstance(item, str) else item
            if post:
                parsed_posts.append({"title": post.get("title"), "url": post.get("url")})
        except json.JSONDecodeError:
            print(f"Skipping an item that was not valid JSON: {item}")
            continue
    
    if not parsed_posts:
        return {"parsed_posts": [], "total_found": 0, "comments_retrieved": 0}

    # --- STEP 2: Extract URLs and retrieve comments for top posts (NEW LOGIC) ---
    # To manage cost and time, let's only get comments for the top 5 posts.
    top_post_urls = [post["url"] for post in parsed_posts[:5] if post.get("url")]
    
    print(f"Found {len(parsed_posts)} posts. Retrieving comments for top {len(top_post_urls)}...")
    
    # Call the existing comment retrieval function
    comment_data = reddit_post_retrieval(urls=top_post_urls, comment_limit=20) # Limit to 20 comments per post
    
    # --- STEP 3: Combine posts and their comments (NEW LOGIC) ---
    final_results = []
    if comment_data and comment_data.get("comments"):
        # Create a dictionary to map comments to their parent post URL for easy lookup
        comments_by_url = {}
        for comment in comment_data["comments"]:
            # This assumes your comment data includes the source post URL. 
            # If not, the structure needs adjustment, but this is a common pattern.
            post_url = comment.get("post_url") # You might need to add 'post_url' to your comment scraper
            if post_url not in comments_by_url:
                comments_by_url[post_url] = []
            comments_by_url[post_url].append(comment['content'])

        # Merge posts with their comments
        for post in parsed_posts:
            post_url = post.get("url")
            post['comments'] = comments_by_url.get(post_url, []) # Add comments list to each post object
            final_results.append(post)
    else:
        # If no comments are found, just return the posts
        final_results = parsed_posts

    return {
        "results": final_results,
        "total_posts_found": len(parsed_posts),
        "total_comments_retrieved": len(comment_data.get("comments", [])) if comment_data else 0
    }

def reddit_post_retrieval(urls,days_back=10,load_all_replies=False,comment_limit=""):
    if not urls:
        return None
    
    trigger_url = "https://api.brightdata.com/datasets/v3/trigger"

    params = {
        "dataset_id": "gd_lvzdpsdlw09j6t702",
        "include_errors": "true"
    }

    data = [
        {
            "url": url,
            "days_back": days_back,
            "load_all_replies": load_all_replies,
            "comment_limit": comment_limit
        }
        for url in urls
    ]
    raw_data= _trigger_and_download_snapshot(
        trigger_url,params,data,operation_name="reddit comments"
    )
    if not raw_data:
        return None
    
    parsed_comments = []
    for comment in raw_data:
        parsed_comment = {
            "comment_id": comment.get("comment_id"),
            "content": comment.get("comment"),
            "date": comment.get("date_posted"),
        }
        parsed_comments.append(parsed_comment)
    return {"comments": parsed_comments,"total_retrieved": len(parsed_comments)}