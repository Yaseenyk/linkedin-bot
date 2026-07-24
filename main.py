import base64
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

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
    # The article URL is deliberately kept OUT of the post body: LinkedIn
    # demotes posts that push people off-platform, which was capping reach at
    # roughly the size of the initial test batch. The link is posted as the
    # FIRST COMMENT instead, automatically, immediately after publishing — so
    # readers still get it and the post keeps its distribution.
    # UTM-tagged so LinkedIn traffic stays distinguishable in site analytics.
    article_link = (
        f"{url}?utm_source=linkedin&utm_medium=social&utm_campaign=auto-pipeline"
    )
    comment_text = f"Full write-up here: {article_link}"
    cta = "Full write-up in the first comment."

    if want_poll:
        format_instruction = (
            "Produce a POLL using FORMAT B below, built around the sharpest "
            "technical trade-off in the article."
        )
    else:
        format_instruction = (
            "Write a STANDARD POST using FORMAT A below. Do NOT produce a poll."
        )

    prompt = f"""You are an experienced software developer telling a peer a war story from production — the way one engineer talks to another, not the way a brand talks to an audience. Write in the first person with the credibility of someone who actually shipped this. Never sound like a marketer.

You are a storyteller first: tension, turn, resolution. A post that only lists facts has failed.

Based on the article below, return ONLY a single JSON object (no markdown fences, no commentary around it).

{format_instruction}

== FORMAT A: standard post ==
Return: {{"type": "post", "text": "<the full post>"}}
Write it as a SHORT STORY with a clear arc, not a listicle and not an announcement. The reader must FEEL the problem before they hear the solution.

1. HOOK (line 1, on its own): One scroll-stopping line, 12 words max, that creates tension or curiosity. Use one of these shapes:
   - Provocative question: "Will AI replace you?"
   - Challenged assumption: "Everyone told me to cache it. Caching was the bug."
   - Blunt verdict with a turn: "The MERN stack is not dead. Your architecture is."
   - Surprising result: "I deleted 300 lines and the page got faster."
   No hashtags, no emoji, no title case, no colon-heavy headline formatting. Then a blank line.

2. THE FRICTION (2-4 short paragraphs): Set the scene in the first person. What broke, what was slow, what silently failed, and why it mattered. Be concrete and specific to THIS article: the real technology, the real symptom, the real constraint. This is the part that earns the rest of the read.

3. THE TURN (1-2 paragraphs): The wrong assumption, the dead end, or the moment the real cause became obvious. This is the pivot of the story.

4. THE FIX (2-3 paragraphs): What actually worked, and WHY it worked. Technical and concrete, explained in plain language a busy engineer understands on first read.

5. THE LESSON: 2-3 crisp takeaways a peer could apply tomorrow, as a cleanly spaced bulleted list.

6. QUESTION: One genuine question inviting the reader's own experience or opinion. This is what earns comments. One line.

7. HASHTAGS: 5 to 8 targeted hashtags on their own line, mixing broad and niche.

8. CTA: Append this exact line, unchanged, as the final line:
{cta}

STORY RULES (never break these):
- Every paragraph is 1-2 sentences MAX with a blank line between them. Mobile-first whitespace, never a wall of text.
- The story must come from the ARTICLE'S ACTUAL TECHNICAL CONTENT. Never invent an employer, client, colleague, date, outage, deadline, or metric that is not in the article.
- If the article contains no personal incident, frame the friction as the problem engineers genuinely hit ("This breaks the moment traffic is uneven...") rather than fabricating an anecdote that never happened.
- The hook must be paid off by the post. Never promise a claim the article does not actually support, and never declare something dead or broken that the article does not argue.
- Weave in the technologies and problems engineers actually search for, naturally. Never keyword-stuff.
- No marketing voice, no "excited to share", no emoji section headers, no engagement-bait phrasing.

== FORMAT B: poll ==
Return: {{"type": "poll", "commentary": "<intro text>", "question": "<poll question>", "options": ["<opt1>", "<opt2>", ...]}}
- "commentary": a short, LinkedIn-native intro that opens with the SAME kind of scroll-stopping hook as FORMAT A (12 words max, tension or curiosity, on its own line), then 1-2 sentences of real context framing the trade-off, then 5-8 hashtags on their own line, then the CTA line "{cta}". Same whitespace-rich, mobile-friendly style, and the same honesty rules: never invent an incident or a metric.
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
            "comment_text": comment_text,
        }

    if isinstance(data, dict) and isinstance(data.get("text"), str):
        post = data["text"].strip()
    else:
        post = raw

    if cta not in post:
        post = f"{post}\n\n{cta}"

    return {"type": "post", "text": post, "comment_text": comment_text}


IMAGE_MODEL = "gpt-image-1"

# Brand-locked; kept byte-identical to the portfolio's scripts/ai_writer.py and
# scripts/generate-og-images.py. Edit all three together.
IMAGE_STYLE_RULES = (
    "Style: minimal, flat, 2D vector editorial illustration for a premium "
    "developer portfolio. Background: uniform matte near-black navy ink "
    "(#05070A), edge to edge. "
    "PALETTE - use ONLY these: deep navy blues, cyan (#22D3EE), violet "
    "(#A855F7), and muted slate grey. A thin off-white (#F4F1EA) line or "
    "shape may be used sparingly as an accent. No other colours exist. "
    "Composition: ONE clear focal geometric metaphor, clean crisp shapes, "
    "flat fills, generous negative space, instantly readable as a small "
    "thumbnail. "
    "ABSOLUTELY NO TEXT: no words, letters, numbers, captions, labels, "
    "headlines, typography, watermarks, signatures, logos, UI text, or "
    "lettering of any kind anywhere in the image. The image must be purely "
    "pictorial. "
    "STRICTLY FORBIDDEN: any warm colour (orange, red, amber, gold, yellow, "
    "brown, peach, copper), green, neon glow, bloom, light bursts, lens "
    "flares, glowing auras or halos, gradients blowing out to white, drop "
    "shadows, photorealism, 3D renders, clay or plastic textures, "
    "skeuomorphism, realistic animals or creatures, busy backgrounds, "
    "collages, and stock-photo looks."
)


def _image_brief(title, content):
    """Have the text model art-direct a bespoke image prompt for the article."""
    prompt = (
        "You are an art director for a software engineering blog. Based on the "
        "article below, write ONE vivid image-generation prompt (max 100 words) "
        "for a LinkedIn cover thumbnail.\n"
        "- The cover is WORDLESS. Never ask for a headline, caption, label, or "
        "any lettering - describe only what is drawn.\n"
        "- Build the scene around ONE concrete, creative visual metaphor for "
        "the article's core technical idea, expressed in flat geometric "
        "shapes. Be specific to THIS article - never a generic laptop, circuit "
        "board or glowing cube, and never a realistic animal or character.\n"
        "- Flat 2D vector illustration on a uniform near-black navy "
        "background, using only cyan and violet accents on navy/slate. One "
        "clear focal subject with generous negative space, readable as a small "
        "thumbnail in a busy feed. No glow, no 3D, no warm colours.\n"
        "- Describe every object purely by shape and colour, keeping the scene "
        "to a few strong elements rather than a busy diagram.\n"
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


def comment_on_post(post_urn, text):
    """Add the article link as the first comment on a freshly published post.

    Keeping the URL out of the post body and putting it here preserves reach
    (LinkedIn demotes off-platform links in the body) while still giving
    readers the article. Best-effort: a failure here must never undo a
    successful publish, so the caller treats it as non-fatal.
    """
    if not post_urn or not text:
        return None
    token = _linkedin_token()
    author = _linkedin_author_urn(token)
    # The URN is a path segment, so ':' must be percent-encoded.
    encoded_urn = quote(post_urn, safe="")
    resp = requests.post(
        f"{LINKEDIN_API_BASE}/rest/socialActions/{encoded_urn}/comments",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
        },
        json={"actor": author, "object": post_urn, "message": {"text": text}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.headers.get("x-restli-id") or (resp.json().get("id") if resp.text else None)


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

    # Drop the article link in as the first comment. Non-fatal: the post is
    # already live, so a comment failure is logged and the run still succeeds.
    comment_id = None
    try:
        comment_id = comment_on_post(post_id, content.get("comment_text", ""))
        print(f"Posted the article link as the first comment (id: {comment_id})")
    except Exception as exc:
        print(
            f"Could not post the link comment ({exc}). "
            f"The post is live; add this link manually if you want it: "
            f"{content.get('comment_text', '')}"
        )

    record = {
        "status": "published",
        "type": content["type"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": article["title"],
        "image_url": article["image_url"],
        "linkedin_post_id": post_id,
        "link_comment_id": comment_id,
        "link_comment_posted": bool(comment_id),
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
