import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

URLS_FILE = Path("urls.txt")
PUBLISHED_FILE = Path("published.json")

GEMINI_MODEL = "gemini-2.5-flash"

BLOG_SECTION = "/blog/"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def load_published():
    if not PUBLISHED_FILE.exists():
        return {}
    with PUBLISHED_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_published(published):
    with PUBLISHED_FILE.open("w", encoding="utf-8") as f:
        json.dump(published, f, indent=2)


def read_urls():
    if not URLS_FILE.exists():
        return []
    with URLS_FILE.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def scrape(url):
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    if soup.h1 and soup.h1.get_text(strip=True):
        title = soup.h1.get_text(strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)
    else:
        title = ""

    content = "\n\n".join(
        p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)
    )

    og_image = soup.find("meta", attrs={"property": "og:image"})
    image_url = og_image["content"].strip() if og_image and og_image.get("content") else ""

    return {"title": title, "content": content, "image_url": image_url}


def discover_post_urls(page_url):
    """Scan a listing/root page and return individual blog post URLs (newest first).

    Posts linked directly on the page come first; if the page also links to a
    separate blog index (e.g. the homepage's "Blog" nav link), that index is
    followed once and its posts are merged in so the full archive is covered.
    """
    resp = requests.get(page_url, headers=REQUEST_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    posts = []
    blog_index = None
    for a in soup.find_all("a", href=True):
        absolute = urljoin(page_url, a["href"]).split("#")[0].split("?")[0]
        if urlparse(absolute).netloc != urlparse(page_url).netloc:
            continue
        norm = urlparse(absolute).path.rstrip("/")
        if BLOG_SECTION in urlparse(absolute).path and not norm.endswith("/blog"):
            if absolute not in posts:
                posts.append(absolute)
        elif norm.endswith("/blog"):
            blog_index = absolute

    if blog_index and blog_index.rstrip("/") != page_url.rstrip("/"):
        for post_url in discover_post_urls(blog_index):
            if post_url not in posts:
                posts.append(post_url)
    return posts


def _gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def generate_linkedin_post(title, content, url):
    cta = f"Read the full article here: {url}"

    prompt = f"""You are an experienced software developer sharing practical, battle-tested engineering insights with your peers on LinkedIn. Write in the first person, with the credibility of someone who has actually shipped this in production — not a marketer.

Write a highly SEO-optimized, 100% LinkedIn-native post based on the article below. Follow these rules exactly:

1. HOOK: Open with a single scroll-stopping line — make it contrarian, surprising, or a sharp question that stops an engineer mid-scroll.

2. FORMAT (LinkedIn-native): Every paragraph must be 1-2 sentences MAX for mobile readability. Put a blank line between every paragraph so the post is full of whitespace. No walls of text.

3. SEO & KEYWORDS: Analyze the article, identify the core technical concepts, and naturally weave high-intent SEO keywords (the specific technologies, patterns, and problems engineers actually search for) into the body. Keep it natural — never keyword-stuff.

4. VALUE: Extract exactly 3 specific, actionable technical takeaways from the article and present them as a cleanly spaced bulleted list (one blank line is fine between bullets if it aids readability).

5. HASHTAGS: End the body with exactly 5 to 8 highly targeted, SEO-friendly hashtags on their own line — mix a few broad tech tags with niche, specific ones relevant to the article.

6. CTA: After the hashtags, append this exact Call-to-Action as the final line, unchanged and pointing to the exact URL provided:
{cta}

Article title: {title}

Article content:
{content}
"""

    model = _gemini_model()
    response = model.generate_content(prompt)
    post = (response.text or "").strip()

    if cta not in post:
        post = f"{post}\n\n{cta}"

    return post


LINKEDIN_API_BASE = "https://api.linkedin.com"


def _linkedin_author_urn(token):
    resp = requests.get(
        f"{LINKEDIN_API_BASE}/v2/userinfo",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return f"urn:li:person:{resp.json()['sub']}"


def publish_to_linkedin(text):
    token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("LINKEDIN_ACCESS_TOKEN is not set. Add it to your .env file.")

    author = _linkedin_author_urn(token)
    payload = {
        "author": author,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }
    resp = requests.post(
        f"{LINKEDIN_API_BASE}/v2/ugcPosts",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.headers.get("x-restli-id") or resp.json().get("id")


def next_unpublished_post(roots, published):
    for root in roots:
        for post_url in discover_post_urls(root):
            if post_url not in published:
                return post_url
    return None


def process_next_url():
    roots = read_urls()
    published = load_published()

    url = next_unpublished_post(roots, published)
    if url is None:
        return None

    article = scrape(url)
    post = generate_linkedin_post(article["title"], article["content"], url)
    post_id = publish_to_linkedin(post)
    print(f"Successfully published to LinkedIn (post id: {post_id})")

    published[url] = {
        "status": "published",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": article["title"],
        "image_url": article["image_url"],
        "post": post,
        "linkedin_post_id": post_id,
    }
    save_published(published)

    return {"url": url, "post": post, "post_id": post_id, **article}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = process_next_url()
    if result is None:
        print("No unpublished URLs to process.")
    else:
        print(f"URL:    {result['url']}")
        print(f"Title:  {result['title']}")
        print(f"Image:  {result['image_url']}")
        print("\n----- Generated LinkedIn Post -----\n")
        print(result["post"])
