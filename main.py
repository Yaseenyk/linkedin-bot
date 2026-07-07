import base64
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from openai import OpenAI
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

URLS_FILE = Path("urls.txt")
PUBLISHED_FILE = Path("published.json")

OPENAI_MODEL = "gpt-5.5"

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
    """Read urls.txt: one entry per line, top-down priority, # comments allowed.

    An entry may be either a listing page to scan for posts (e.g. the site
    root) or a direct blog-post URL, which is queued exactly as written.
    """
    if not URLS_FILE.exists():
        return []
    with URLS_FILE.open("r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


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

    # Resolve links against the final URL after redirects, not the requested
    # one — otherwise a site that moved domains yields URLs on the old host.
    base_url = resp.url

    posts = []
    blog_index = None
    for a in soup.find_all("a", href=True):
        absolute = urljoin(base_url, a["href"]).split("#")[0].split("?")[0]
        if urlparse(absolute).netloc != urlparse(base_url).netloc:
            continue
        norm = urlparse(absolute).path.rstrip("/")
        if BLOG_SECTION in urlparse(absolute).path and not norm.endswith("/blog"):
            if absolute not in posts:
                posts.append(absolute)
        elif norm.endswith("/blog"):
            blog_index = absolute

    if blog_index and blog_index.rstrip("/") != base_url.rstrip("/"):
        for post_url in discover_post_urls(blog_index):
            if post_url not in posts:
                posts.append(post_url)
    return posts


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


def _strip_json_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()


POLL_EVERY_N_POSTS = 4


def generate_linkedin_content(title, content, url, want_poll=False):
    """Generate either a standard LinkedIn post or a poll for the article.

    Returns a dict shaped as either:
      {"type": "post", "text": "..."}
      {"type": "poll", "commentary": "...", "question": "...", "options": [...]}
    The caller decides the format (want_poll); the cadence is enforced in code
    rather than left to the model, which otherwise over-produces polls for
    trade-off-heavy articles.
    """
    # UTM-tagged so LinkedIn traffic is distinguishable in site analytics.
    cta = (
        f"Read the full article here: "
        f"{url}?utm_source=linkedin&utm_medium=social&utm_campaign=auto-pipeline"
    )

    if want_poll:
        format_instruction = (
            "Produce a POLL using FORMAT B below, built around the sharpest "
            "technical trade-off in the article."
        )
    else:
        format_instruction = (
            "Write a STANDARD POST using FORMAT A below. Do NOT produce a poll."
        )

    prompt = f"""You are an experienced software developer sharing practical, battle-tested engineering insights with your peers on LinkedIn. Write in the first person, with the credibility of someone who has actually shipped this in production — not a marketer.

Based on the article below, return ONLY a single JSON object (no markdown fences, no commentary around it).

{format_instruction}

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

    client = _openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = _strip_json_fence(response.choices[0].message.content or "")

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


IMAGE_MODEL = "gpt-image-1"

IMAGE_STYLE_RULES = (
    "Style: bold, clickable tech-thumbnail cover (YouTube-thumbnail energy, "
    "LinkedIn-professional polish). One dominant focal subject, dramatic "
    "lighting, rich saturated colors, high contrast, instantly readable as a "
    "small feed preview. The headline text must appear exactly once, spelled "
    "exactly as quoted in the prompt, in massive bold clean sans-serif type "
    "with strong contrast against the background. No other text, no "
    "watermarks, no logos anywhere in the image."
)


def _image_brief(title, content):
    """Have the text model art-direct a bespoke image prompt for the article."""
    prompt = (
        "You are an art director for a software engineering blog. Based on the "
        "article below, write ONE vivid image-generation prompt (max 100 words) "
        "for a scroll-stopping, clickable LinkedIn cover thumbnail.\n"
        "- Distill the article into a punchy 3-6 word HEADLINE (plain ASCII, "
        "may include one number if it is the article's key result). Put the "
        "headline in double quotes in your prompt and state that it must be "
        "rendered exactly once, spelled exactly as given, in huge bold "
        "sans-serif type dominating the layout.\n"
        "- Build the rest of the scene around ONE concrete, creative visual "
        "metaphor for the article's core technical idea. Be specific to THIS "
        "article - never a generic laptop, circuit board or glowing cube.\n"
        "- High contrast, 2-3 vivid accent colors that fit the topic's mood, "
        "one clear focal subject, composition that stays readable as a small "
        "thumbnail in a busy feed.\n"
        "- Besides the quoted headline, no other text, labels or lettering "
        "anywhere - describe every object purely by shape, color and material, "
        "and keep the scene to a few strong elements rather than a busy "
        "diagram.\n"
        "Return only the prompt text, nothing else.\n\n"
        f"Article title: {title}\n\n"
        f"Article excerpt:\n{content[:2000]}"
    )
    client = _openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def _strip_image_metadata(image_bytes):
    """Re-encode the image, dropping all embedded metadata.

    gpt-image-1 embeds C2PA content credentials, which LinkedIn surfaces as a
    "Content Credentials" badge on the post. Re-encoding through Pillow drops
    every metadata chunk, and JPEG output shrinks the ~2 MB PNGs as a bonus.
    Returns the original bytes untouched if re-encoding fails.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception as exc:
        print(f"Metadata strip failed ({exc}); using the original image bytes.")
        return image_bytes


def generate_post_image(title, content=""):
    """Generate a cover image for the post. Returns PNG bytes, or None on failure.

    A text-model "art director" pass turns the article into a topic-specific
    creative brief first; the image model then renders that brief. Image
    problems must never block publishing, so all errors are swallowed and the
    caller falls back to a text-only post.
    """
    try:
        try:
            brief = _image_brief(title, content)
            print(f"Image brief: {brief}")
        except Exception as exc:
            print(f"Image brief failed ({exc}); using a generic prompt.")
            brief = (
                "A striking abstract visual metaphor for a software engineering "
                f'article titled "{title}".'
            )
        client = _openai_client()
        result = client.images.generate(
            model=IMAGE_MODEL,
            prompt=f"{brief}\n\n{IMAGE_STYLE_RULES}",
            size="1536x1024",
            quality="medium",
        )
        return _strip_image_metadata(base64.b64decode(result.data[0].b64_json))
    except Exception as exc:
        print(f"Image generation failed ({exc}); trying the article's own cover image.")
        return None


def download_image(url):
    """Download an image (e.g. the article's og:image). Returns bytes or None."""
    if not url:
        return None
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        print(f"Could not download article image ({exc}); posting text-only.")
        return None


LINKEDIN_API_BASE = "https://api.linkedin.com"
LINKEDIN_API_VERSION = os.environ.get("LINKEDIN_API_VERSION", "202605")


LITTLE_TEXT_RESERVED = "\\|{}@[]()<>*_~"


def _escape_little_text(text):
    """Escape characters reserved by LinkedIn's "little text format".

    The /rest/posts endpoint (used for image posts and polls) parses commentary
    as little text; unescaped reserved characters cause 400 errors or mangled
    text. '#' is deliberately NOT escaped so hashtags stay clickable.
    """
    for ch in LITTLE_TEXT_RESERVED:
        text = text.replace(ch, "\\" + ch)
    return text


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


def _upload_image_to_linkedin(token, author, image_bytes):
    """Upload image bytes via the LinkedIn Images API and return the image URN."""
    init = requests.post(
        f"{LINKEDIN_API_BASE}/rest/images?action=initializeUpload",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
        },
        json={"initializeUploadRequest": {"owner": author}},
        timeout=30,
    )
    init.raise_for_status()
    value = init.json()["value"]

    put = requests.put(
        value["uploadUrl"],
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        },
        data=image_bytes,
        timeout=60,
    )
    put.raise_for_status()
    return value["image"]


def publish_to_linkedin(text, image_bytes=None, image_title=""):
    token = _linkedin_token()
    author = _linkedin_author_urn(token)

    image_urn = None
    if image_bytes:
        try:
            image_urn = _upload_image_to_linkedin(token, author, image_bytes)
        except Exception as exc:
            print(f"Image upload failed ({exc}); publishing text-only instead.")

    if image_urn:
        payload = {
            "author": author,
            "commentary": _escape_little_text(text),
            "visibility": "PUBLIC",
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
            "content": {
                "media": {"id": image_urn, "altText": image_title[:300]}
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
        return resp.headers.get("x-restli-id") or (
            resp.json().get("id") if resp.text else None
        )

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
        "commentary": _escape_little_text(commentary),
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


def _is_post_url(url):
    path = urlparse(url).path
    return BLOG_SECTION in path and not path.rstrip("/").endswith("/blog")


def next_unpublished_post(roots, published):
    for root in roots:
        # A direct post URL queues that exact post, in urls.txt order;
        # anything else is a listing page to be scanned for posts.
        if _is_post_url(root):
            if root not in published:
                return root
            continue
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
    want_poll = (len(published) + 1) % POLL_EVERY_N_POSTS == 0
    content = generate_linkedin_content(
        article["title"], article["content"], url, want_poll=want_poll
    )

    if content["type"] == "poll":
        post_id = publish_poll_to_linkedin(
            content["commentary"], content["question"], content["options"]
        )
        preview = (
            f"{content['commentary']}\n\nPOLL: {content['question']}\n"
            + "\n".join(f"  - {opt}" for opt in content["options"])
        )
    else:
        image_bytes = generate_post_image(
            article["title"], article["content"]
        ) or download_image(article["image_url"])
        post_id = publish_to_linkedin(
            content["text"], image_bytes=image_bytes, image_title=article["title"]
        )
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
        record["has_image"] = bool(image_bytes)
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
