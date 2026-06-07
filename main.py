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


def _strip_json_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()


def generate_linkedin_content(title, content, url):
    """Generate either a standard LinkedIn post or a poll for the article.

    Returns a dict shaped as either:
      {"type": "post", "text": "..."}
      {"type": "poll", "commentary": "...", "question": "...", "options": [...]}
    The model decides which format fits; polls are produced only occasionally
    when the article naturally raises a sharp technical trade-off worth a vote.
    """
    cta = f"Read the full article here: {url}"

    prompt = f"""You are an experienced software developer sharing practical, battle-tested engineering insights with your peers on LinkedIn. Write in the first person, with the credibility of someone who has actually shipped this in production — not a marketer.

Based on the article below, decide which of TWO formats best fits, then return ONLY a single JSON object (no markdown fences, no commentary around it).

Most of the time, write a standard post. OCCASIONALLY — only when the article raises a genuine, debatable technical trade-off that engineers would have real opinions on — produce a poll instead to drive engagement. Roughly one in four articles should become a poll; otherwise prefer a post.

== FORMAT A: standard post ==
Return: {{"type": "post", "text": "<the full post>"}}
The "text" value must follow these rules exactly:
1. HOOK: Open with a single scroll-stopping line — contrarian, surprising, or a sharp question that stops an engineer mid-scroll.
2. FORMAT (LinkedIn-native): Every paragraph 1-2 sentences MAX for mobile readability. A blank line between every paragraph so the post is full of whitespace. No walls of text.
3. SEO & KEYWORDS: Identify the core technical concepts and naturally weave high-intent SEO keywords (the specific technologies, patterns, and problems engineers actually search for) into the body. Never keyword-stuff.
4. VALUE: Extract exactly 3 specific, actionable technical takeaways as a cleanly spaced bulleted list.
5. HASHTAGS: End the body with exactly 5 to 8 highly targeted, SEO-friendly hashtags on their own line — mix broad and niche tags.
6. CTA: After the hashtags, append this exact Call-to-Action as the final line, unchanged:
{cta}

== FORMAT B: poll ==
Return: {{"type": "poll", "commentary": "<intro text>", "question": "<poll question>", "options": ["<opt1>", "<opt2>", ...]}}
- "commentary": a short, LinkedIn-native intro (a hook line, 1-2 sentences of context, then 5-8 hashtags on their own line, then the CTA line "{cta}"). Same whitespace-rich, mobile-friendly style as a post.
- "question": the poll question itself, a focused technical question. MAX 140 characters.
- "options": 3 to 4 distinct, mutually exclusive answer choices. Each option short (ideally under 30 characters). Make them real, defensible positions an engineer would pick between.

Return strictly valid JSON for exactly one of the two formats.

Article title: {title}

Article content:
{content}
"""

    model = _gemini_model()
    response = model.generate_content(prompt)
    raw = _strip_json_fence(response.text or "")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict) and data.get("type") == "poll" and isinstance(data.get("options"), list):
        options = [str(o).strip() for o in data["options"] if str(o).strip()][:4]
        commentary = (data.get("commentary") or "").strip()
        if cta not in commentary:
            commentary = f"{commentary}\n\n{cta}".strip()
        return {
            "type": "poll",
            "commentary": commentary,
            "question": (data.get("question") or "").strip()[:140],
            "options": options,
        }

    if isinstance(data, dict) and isinstance(data.get("text"), str):
        post = data["text"].strip()
    else:
        post = raw

    if cta not in post:
        post = f"{post}\n\n{cta}"

    return {"type": "post", "text": post}


LINKEDIN_API_BASE = "https://api.linkedin.com"
LINKEDIN_API_VERSION = os.environ.get("LINKEDIN_API_VERSION", "202605")


def _linkedin_token():
    token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("LINKEDIN_ACCESS_TOKEN is not set. Add it to your .env file.")
    return token


def _linkedin_author_urn(token):
    resp = requests.get(
        f"{LINKEDIN_API_BASE}/v2/userinfo",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return f"urn:li:person:{resp.json()['sub']}"


def publish_to_linkedin(text):
    token = _linkedin_token()
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


def publish_poll_to_linkedin(commentary, question, options):
    if not 2 <= len(options) <= 4:
        raise ValueError(f"A LinkedIn poll needs 2 to 4 options, got {len(options)}.")

    token = _linkedin_token()
    author = _linkedin_author_urn(token)
    payload = {
        "author": author,
        "commentary": commentary,
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
        "content": {
            "poll": {
                "question": question[:140],
                "options": [{"text": option} for option in options],
                "settings": {"duration": "SEVEN_DAYS"},
            }
        },
    }
    resp = requests.post(
        f"{LINKEDIN_API_BASE}/rest/posts",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.headers.get("x-restli-id") or (resp.json().get("id") if resp.text else None)


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
    content = generate_linkedin_content(article["title"], article["content"], url)

    if content["type"] == "poll":
        post_id = publish_poll_to_linkedin(
            content["commentary"], content["question"], content["options"]
        )
        preview = (
            f"{content['commentary']}\n\nPOLL: {content['question']}\n"
            + "\n".join(f"  - {opt}" for opt in content["options"])
        )
    else:
        post_id = publish_to_linkedin(content["text"])
        preview = content["text"]

    print(f"Successfully published to LinkedIn (post id: {post_id})")

    record = {
        "status": "published",
        "type": content["type"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": article["title"],
        "image_url": article["image_url"],
        "linkedin_post_id": post_id,
    }
    if content["type"] == "poll":
        record["commentary"] = content["commentary"]
        record["question"] = content["question"]
        record["options"] = content["options"]
    else:
        record["post"] = content["text"]
    published[url] = record
    save_published(published)

    return {"url": url, "post": preview, "post_id": post_id, **article}


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
