import base64
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse

import requests
from openai import OpenAI
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

URLS_FILE = Path("urls.txt")
TOPICS_FILE = Path("topics.txt")
PUBLISHED_FILE = Path("published.json")
PREVIEW_DIR = Path("preview")

OPENAI_MODEL = "gpt-5.5"

PORTFOLIO_URL = "https://yaseenkhatib.streamerosai.com/"
BLOG_SECTION = "/blog/"

# What each run publishes, by position in the cycle. The index is the number
# of entries already in published.json, so the cadence is deterministic and
# survives re-runs. Memes sit between the technical posts on purpose: they
# travel further than articles and pull people to the profile, where the
# technical posts do the actual work.
SLOT_CYCLE = ("post", "post", "meme", "post", "poll")

# --dry-run: generate everything, publish nothing, save nothing. Output goes to
# preview/ so a draft can be checked before the next scheduled run.
DRY_RUN = "--dry-run" in sys.argv

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


# ------------------------------------------------------------------ state ---


def load_published():
    if not PUBLISHED_FILE.exists():
        return {}
    with PUBLISHED_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_published(published):
    if DRY_RUN:
        print("[dry-run] published.json left untouched")
        return
    with PUBLISHED_FILE.open("w", encoding="utf-8") as f:
        json.dump(published, f, indent=2)


def _read_lines(path):
    """One entry per line, top-down priority, # comments and blanks ignored."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def read_urls():
    """urls.txt: listing pages to scan for posts, or direct post URLs."""
    return _read_lines(URLS_FILE)


def read_topics():
    """topics.txt: standalone post topics, consumed top-down."""
    return _read_lines(TOPICS_FILE)


def remove_topic(topic):
    if DRY_RUN:
        print("[dry-run] topics.txt left untouched")
        return
    lines = TOPICS_FILE.read_text(encoding="utf-8").splitlines()
    remaining = [ln for ln in lines if ln.strip() != topic]
    TOPICS_FILE.write_text(
        "\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8"
    )


def topic_key(topic):
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:80]
    return f"topic:{slug}"


def with_utm(url, campaign="auto-pipeline"):
    """UTM-tag a link so LinkedIn traffic stays distinguishable in analytics."""
    params = {"utm_source": "linkedin", "utm_medium": "social", "utm_campaign": campaign}
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urlencode(params)}"


# -------------------------------------------------------------- scraping ---


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


# ---------------------------------------------------------------- OpenAI ---


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


def _ensure_cta(text, cta):
    """Guarantee the CTA is the final line, separated by a blank line."""
    text = text.strip()
    if cta in text:
        text = text.replace(cta, "").rstrip()
    return f"{text}\n\n{cta}"


def _chat_json(prompt):
    client = _openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = _strip_json_fence(response.choices[0].message.content or "")
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


# Verified facts about the author's own systems. A topic post has no article
# to lean on, so this is the ONLY first-person material the model may use.
# Everything here is read from the shipped code; do not add claims that are
# not in a repo.
FACTS = """VERIFIED FACTS ABOUT THE AUTHOR'S OWN SYSTEMS (the only first-person material allowed):
- Stack: MERN (MongoDB, Express, React, Node.js) with TypeScript and Next.js; Python for automation; Cloudflare Workers for serverless.
- Portfolio concierge: a single Cloudflare Worker with a grounded RAG endpoint (POST /api/chat) and an MCP server (POST /mcp). Embeddings use bge-base-en-v1.5 (768 dims), vectors live in Cloudflare Vectorize (cosine), generation uses Llama 3.3 70B on Workers AI. Retrieval takes the top 5 chunks; if the best cosine score is below 0.55 the endpoint refuses with an honest "outside my grounded context" message instead of guessing. Context is capped at 6000 characters. The response includes traces (vector search time and top score, chunk count, guardrail result) that the site's terminal UI renders step by step.
- Corpus indexing: a script fetches the site's llms-full.txt, splits each article on paragraph boundaries into chunks of about 1400 characters, embeds them in batches of 40 and upserts with stable ids (slug plus chunk index) so re-runs overwrite in place. It runs once a day on a GitHub Actions cron to stay inside the Workers AI free tier.
- MCP server: hand-written JSON-RPC over Streamable HTTP in plain TypeScript, no SDK. It implements initialize, ping, tools/list and tools/call, returns 202 for notifications, and exposes three tools: search_portfolio, get_products, get_availability. Recruiters can add the URL to Claude and interrogate the portfolio from inside their own AI.
- Outreach drafter: a passcode-gated endpoint that drafts job-application emails with OpenAI. The prompt carries an explicit allowlist of the author's real stack and a rule that a job description's technologies must never be mirrored back as experience. Optional OCR of a job-post screenshot via a vision model. Returns a draft only; the human sends it. The same propose-then-confirm boundary gates every database write in Sable, a local-first finance agent (SQLite on device, OpenAI function calling, React Native).
- LinkedIn pipeline (this bot): Python on GitHub Actions, no server. Queue and topics are text files, state is a JSON file committed back to the repo. Posts come from OpenAI in JSON mode; the post format (story, poll, meme) is chosen by code on a fixed cycle because the model over-produced polls when allowed to choose. Cover images are generated in two passes: a text model writes an art-director brief, then gpt-image-1 renders it; the result is re-encoded through Pillow to drop C2PA metadata, which LinkedIn otherwise surfaces as a badge. The article link was once posted as a first comment to dodge link demotion; the comment step was unreliable so the link went back into the body.
- AI blog writer: GitHub Actions plus OpenAI structured outputs with a strict JSON schema (title, description, tags, keywords, takeaways, body) so no SEO field can be skipped. Its schedule was disabled after bulk AI-generated posts got the site de-indexed: indexed pages fell from 88 to 25 in one day (2026-07-10 to 2026-07-11), impressions went to zero, and 72 URLs sat in "Discovered, currently not indexed". Recovery meant shrinking the corpus to first-hand work, not publishing more.
- IntegrateX: a React Flow workflow engine whose schema-aware serialization adapter cut graph payloads by 94%. This is the only performance number the author ever cites.
- streamerOS: desktop cockpit for streamers (real-time chat ingestion, telemetry, OBS automation).
- Path Saathi: a client LMS taken from a Monday MVP brief to a working platform live on a dev environment the next day. No other delivery-time claims exist."""


def generate_linkedin_content(title, content, cta, want_poll=False, source="article"):
    """Generate either a standard LinkedIn post or a poll.

    `source` is "article" (content is scraped article text) or "topic"
    (content is a topic brief plus FACTS). Returns a dict shaped as either:
      {"type": "post", "text": "..."}
      {"type": "poll", "commentary": "...", "question": "...", "options": [...]}
    The caller decides the format (want_poll); the cadence is enforced in code
    rather than left to the model, which otherwise over-produces polls.
    """
    if want_poll:
        format_instruction = (
            "Produce a POLL using FORMAT B below, built around the sharpest "
            "technical trade-off in the material."
        )
    else:
        format_instruction = (
            "Write a STANDARD POST using FORMAT A below. Do NOT produce a poll."
        )

    if source == "topic":
        source_rules = (
            "The material is a TOPIC plus a block of VERIFIED FACTS about the "
            "author's own systems. The story must be built from those facts. "
            "Where the facts do not cover the topic, write about the problem "
            "the way engineers generally hit it, in first person about design "
            "judgment, and never invent an incident, client, employer, date, "
            "outage or metric to fill the gap. The only numbers you may cite "
            "as the author's results are the 94% payload reduction and the "
            "de-indexing figures, exactly as given."
        )
        material_label = "Topic and verified facts"
    else:
        source_rules = (
            "The story must come from the ARTICLE'S ACTUAL TECHNICAL CONTENT. "
            "Never invent an employer, client, colleague, date, outage, "
            "deadline, or metric that is not in the article. If the article "
            "contains no personal incident, frame the friction as the problem "
            "engineers genuinely hit rather than fabricating an anecdote."
        )
        material_label = "Article content"

    prompt = f"""You are an experienced software developer telling a peer a war story from production — the way one engineer talks to another, not the way a brand talks to an audience. Write in the first person with the credibility of someone who actually shipped this. Never sound like a marketer.

You are a storyteller first: tension, turn, resolution. A post that only lists facts has failed.

Based on the material below, return ONLY a single JSON object (no markdown fences, no commentary around it).

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

2. THE FRICTION (2-4 short paragraphs): Set the scene in the first person. What broke, what was slow, what silently failed, and why it mattered. Be concrete and specific: the real technology, the real symptom, the real constraint. This is the part that earns the rest of the read.

3. THE TURN (1-2 paragraphs): The wrong assumption, the dead end, or the moment the real cause became obvious. This is the pivot of the story.

4. THE FIX (2-3 paragraphs): What actually worked, and WHY it worked. Technical and concrete, explained in plain language a busy engineer understands on first read.

5. THE LESSON: 2-3 crisp takeaways a peer could apply tomorrow, as a cleanly spaced bulleted list.

6. QUESTION: One genuine question inviting the reader's own experience or opinion. This is what earns comments. One line.

7. HASHTAGS: 5 to 8 targeted hashtags on their own line, mixing broad and niche.

8. CTA: Append this exact line, unchanged, as the final line:
{cta}

STORY RULES (never break these):
- Every paragraph is 1-2 sentences MAX with a blank line between them. Mobile-first whitespace, never a wall of text.
- {source_rules}
- The hook must be paid off by the post. Never promise a claim the material does not actually support, and never declare something dead or broken that the material does not argue.
- Weave in the technologies and problems engineers actually search for, naturally. Never keyword-stuff.
- No marketing voice, no "excited to share", no emoji section headers, no engagement-bait phrasing.

== FORMAT B: poll ==
Return: {{"type": "poll", "commentary": "<intro text>", "question": "<poll question>", "options": ["<opt1>", "<opt2>", ...]}}
- "commentary": a short, LinkedIn-native intro that opens with the SAME kind of scroll-stopping hook as FORMAT A (12 words max, tension or curiosity, on its own line), then 1-2 sentences of real context framing the trade-off, then 5-8 hashtags on their own line, then the CTA line "{cta}". Same whitespace-rich, mobile-friendly style, and the same honesty rules: never invent an incident or a metric.
- "question": the poll question itself, a focused technical question. MAX 140 characters.
- "options": 3 to 4 distinct, mutually exclusive answer choices. Each option short (ideally under 30 characters). Make them real, defensible positions an engineer would pick between.

Return strictly valid JSON for exactly one of the two formats.

Title: {title}

{material_label}:
{content}
"""

    data, raw = _chat_json(prompt)

    if isinstance(data, dict) and data.get("type") == "poll" and isinstance(data.get("options"), list):
        options = [str(o).strip() for o in data["options"] if str(o).strip()][:4]
        commentary = _ensure_cta(data.get("commentary") or "", cta)
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

    return {"type": "post", "text": _ensure_cta(post, cta)}


# ---------------------------------------------------------------- images ---

IMAGE_MODEL = "gpt-image-1"

# NOT brand-locked (unlike the portfolio's on-site OG covers in
# scripts/ai_writer.py + generate-og-images.py). On LinkedIn the image should be
# relevant and recognizable to each specific post — depict the actual topic —
# rather than an abstract brand mark. Do NOT re-sync this with those two scripts.
IMAGE_STYLE_RULES = (
    "Style: a polished, modern editorial tech illustration that clearly and "
    "specifically depicts the article's subject - recognizable at a glance and "
    "relevant to the actual technology or concept, not a generic abstract mark. "
    "Clean, confident and premium, aimed at senior software engineers scrolling "
    "LinkedIn. "
    "Composition: one clear focal subject with strong negative space, instantly "
    "readable as a small thumbnail in a busy feed. Use a cohesive, tasteful "
    "colour palette that genuinely fits the subject (cool blues, cyan and violet "
    "read well on a dark background, but choose what suits the topic) on an "
    "uncluttered background. Tasteful depth, soft gradients and dimensionality "
    "are welcome where they serve the idea. "
    "ABSOLUTELY NO TEXT: no words, letters, numbers, captions, labels, headlines, "
    "UI text, code, watermarks, logos, or lettering of any kind - AI-rendered "
    "text always looks broken. The image must be purely pictorial. "
    "AVOID: cheesy stock-photo scenes and posed people, generic laptops / circuit "
    "boards / glowing cubes, busy collages, cluttered fake diagrams, garbled UI "
    "and low-effort clip-art. Keep it specific, clean and high-craft."
)


def _image_brief(title, content):
    """Have the text model art-direct a bespoke image prompt for the post."""
    prompt = (
        "You are an art director for a senior software engineer's brand. Based "
        "on the material below, write ONE vivid image-generation prompt (max 100 "
        "words) for a LinkedIn cover thumbnail.\n"
        "- The cover is WORDLESS. Never ask for a headline, caption, label, code "
        "or any lettering - describe only what is drawn.\n"
        "- The image must clearly and specifically depict THIS post's core "
        "technical idea, so a scrolling engineer instantly understands what it is "
        "about. Prefer a recognizable, relevant scene or concrete metaphor over a "
        "purely abstract shape. Be specific to THIS post - never a generic "
        "laptop, circuit board or glowing cube, and never a realistic animal or "
        "character.\n"
        "- Modern, clean, premium editorial illustration on an uncluttered "
        "background. One clear focal subject with generous negative space, "
        "readable as a small thumbnail in a busy feed. Choose colours and a style "
        "that genuinely fit the subject; tasteful depth is fine.\n"
        "- Keep the scene to a few strong, purposeful elements rather than a busy "
        "diagram.\n"
        "Return only the prompt text, nothing else.\n\n"
        f"Title: {title}\n\n"
        f"Excerpt:\n{content[:2000]}"
    )
    client = _openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def _to_jpeg(image_bytes, quality=90):
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
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except Exception as exc:
        print(f"Metadata strip failed ({exc}); using the original image bytes.")
        return image_bytes


def _render_image(prompt, size="1536x1024"):
    client = _openai_client()
    result = client.images.generate(
        model=IMAGE_MODEL, prompt=prompt, size=size, quality="medium"
    )
    return base64.b64decode(result.data[0].b64_json)


def generate_post_image(title, content=""):
    """Generate a cover image for the post. Returns JPEG bytes, or None on failure.

    A text-model "art director" pass turns the material into a topic-specific
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
                f'post titled "{title}".'
            )
        return _to_jpeg(_render_image(f"{brief}\n\n{IMAGE_STYLE_RULES}"))
    except Exception as exc:
        print(f"Image generation failed ({exc}); trying a fallback image.")
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


# ----------------------------------------------------------------- memes ---

MEME_STYLE_RULES = (
    "Style: a bold, funny, flat-colour cartoon in the spirit of classic internet "
    "meme templates - exaggerated expressions, simple shapes, thick clean "
    "outlines, bright readable colours, one instantly legible scene. It must "
    "read as a joke at thumbnail size. "
    "Leave clear, uncluttered space along the TOP and BOTTOM edges (about a "
    "fifth of the height each) where captions will be added later; keep the "
    "action in the middle band. "
    "ABSOLUTELY NO TEXT of any kind: no words, letters, numbers, labels, signs, "
    "screens with writing, code, logos or watermarks - the captions are added "
    "afterwards. "
    "No real people, no real company logos, no photorealism, no stock-photo look."
)

MEME_FONT_CANDIDATES = [
    "C:/Windows/Fonts/impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _meme_font(size):
    from PIL import ImageFont

    for path in MEME_FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_caption(draw, text, font, width, y, anchor_top, stroke):
    """Draw centred, wrapped, outlined meme text at the top or bottom."""
    text = text.upper()
    # Wrap by measured width, not character count, so long words stay intact.
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= width * 0.92 or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    bbox = font.getbbox("Ag")
    line_h = (bbox[3] - bbox[1]) + int(font.size * 0.25)
    total_h = line_h * len(lines)
    start_y = y if anchor_top else y - total_h
    for i, line in enumerate(lines):
        w = draw.textlength(line, font=font)
        draw.text(
            ((width - w) / 2, start_y + i * line_h),
            line,
            font=font,
            fill="white",
            stroke_width=stroke,
            stroke_fill="black",
        )


def overlay_meme_captions(image_bytes, top, bottom):
    """Burn top/bottom captions onto the image in classic meme style (Pillow).

    Captions are rendered locally rather than by the image model: short words
    in Impact with a black stroke are deterministic, while model-rendered text
    is the one thing that reliably looks broken.
    """
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font = _meme_font(max(28, width // 13))
    stroke = max(2, width // 300)
    margin = int(height * 0.03)
    if top:
        _draw_caption(draw, top, font, width, margin, True, stroke)
    if bottom:
        _draw_caption(draw, bottom, font, width, height - margin, False, stroke)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def recent_meme_captions(published, limit=12):
    memes = [v for v in published.values() if v.get("type") == "meme"]
    memes.sort(key=lambda v: v.get("timestamp", ""))
    return [f"{m.get('top', '')} / {m.get('bottom', '')}" for m in memes[-limit:]]


def generate_meme(published, cta):
    """Write a meme (captions + scene + short post text) and render it.

    Returns {"top", "bottom", "scene", "text", "image": bytes|None}.
    """
    avoid = recent_meme_captions(published)
    avoid_block = (
        "Already used (do NOT repeat these jokes or their structure):\n"
        + "\n".join(f"- {a}" for a in avoid)
        if avoid
        else ""
    )
    prompt = f"""You write developer memes for a senior MERN + AI engineer's LinkedIn. The audience is working full-stack and AI engineers. The humour is dry, specific and knowing - the kind that makes an engineer tag a colleague - never cringe, never corporate, never cruel.

Theme pool (pick ONE angle, be specific, not generic):
- LLMs inside a MERN app: RAG that answers confidently from the wrong chunk, embeddings that match on vibes, "the model said it was done", JSON that was almost JSON, token bills, prompt tweaks as "architecture".
- React / Node / MongoDB life: useEffect dependency arrays, "works on my machine" in a Docker world, aggregation pipelines at 2am, npm install doubling the repo size, Express error handlers that swallow everything, TypeScript 'any' as a lifestyle.
- AI coding tools: the confident refactor that deleted the tests, "let me fix that" for the fifth time, reviewing 800 generated lines, agents that run the whole test suite to change one string.
- Hiring and the job market: "5 years of experience with a 2-year-old framework", take-home assignments, "we are a family", recruiter messages for the wrong stack.

Return ONLY a JSON object:
{{"top": "<top caption, 3-8 words>", "bottom": "<bottom caption, 3-8 words>", "scene": "<one-sentence wordless cartoon scene that carries the joke visually, no text in the scene>", "text": "<the LinkedIn post body>"}}

Rules:
- The top/bottom captions must make sense WITHOUT the post text and the joke must land in under two seconds.
- "scene" describes what is DRAWN: characters, expressions, objects, situation. It must never contain words that should appear in the image.
- "text": 3-6 short lines of commentary in first person, dry and relatable, each on its own line with a blank line between. End with one genuine one-line question that invites the reader's own story. Then 3-5 hashtags on one line. Then append this exact line, unchanged, as the final line:
{cta}
- No invented statistics, no named companies or real people, no punching down, no emoji.
{avoid_block}
"""
    data, raw = _chat_json(prompt)
    if not isinstance(data, dict) or not data.get("top") or not data.get("scene"):
        raise RuntimeError(f"meme generation returned unusable JSON: {raw[:200]}")

    top = str(data.get("top", "")).strip()
    bottom = str(data.get("bottom", "")).strip()
    scene = str(data.get("scene", "")).strip()
    text = _ensure_cta(str(data.get("text", "")), cta)

    image = None
    try:
        png = _render_image(f"{scene}\n\n{MEME_STYLE_RULES}", size="1024x1024")
        image = overlay_meme_captions(png, top, bottom)
    except Exception as exc:
        print(f"Meme image failed ({exc}); the meme will be skipped.")

    return {"top": top, "bottom": bottom, "scene": scene, "text": text, "image": image}


# -------------------------------------------------------------- LinkedIn ---

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
    if resp.status_code == 401:
        # LinkedIn access tokens expire after ~60 days and cannot be
        # auto-refreshed without partner-program approval, so this is routine.
        raise SystemExit(
            "LINKEDIN_ACCESS_TOKEN expired or revoked (401 from /v2/userinfo).\n"
            "Fix: LinkedIn Developer portal -> your app -> OAuth token generator\n"
            "(https://www.linkedin.com/developers/tools/oauth/token-generator),\n"
            "generate a token with the same scopes, then update the\n"
            "LINKEDIN_ACCESS_TOKEN secret in this repo's Actions settings."
        )
    resp.raise_for_status()
    return f"urn:li:person:{resp.json()['sub']}"


def _rest_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
        "LinkedIn-Version": LINKEDIN_API_VERSION,
    }


def _upload_image_to_linkedin(token, author, image_bytes):
    """Upload image bytes via the LinkedIn Images API and return the image URN."""
    init = requests.post(
        f"{LINKEDIN_API_BASE}/rest/images?action=initializeUpload",
        headers=_rest_headers(token),
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


def _post_id(resp):
    return resp.headers.get("x-restli-id") or (resp.json().get("id") if resp.text else None)


def publish_to_linkedin(text, image_bytes=None, image_title=""):
    if DRY_RUN:
        print("[dry-run] would publish a post" + (" with image" if image_bytes else " text-only"))
        return "urn:li:share:DRY-RUN"

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
            headers=_rest_headers(token),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return _post_id(resp)

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
    return _post_id(resp)


def publish_poll_to_linkedin(commentary, question, options):
    if not 2 <= len(options) <= 4:
        raise ValueError(f"A LinkedIn poll needs 2 to 4 options, got {len(options)}.")

    if DRY_RUN:
        print("[dry-run] would publish a poll")
        return "urn:li:share:DRY-RUN"

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
        headers=_rest_headers(token),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return _post_id(resp)


# ------------------------------------------------------------- selection ---


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


def next_source(published):
    """Pick what the next technical post is about.

    topics.txt wins: a topic is a standalone post that links to the portfolio.
    When no topics remain, fall back to the article queue in urls.txt.
    """
    for topic in read_topics():
        if topic_key(topic) not in published:
            return {"kind": "topic", "topic": topic}
    url = next_unpublished_post(read_urls(), published)
    if url:
        return {"kind": "article", "url": url}
    return None


def _save_preview(name, text, image):
    if not DRY_RUN:
        return
    PREVIEW_DIR.mkdir(exist_ok=True)
    (PREVIEW_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
    if image:
        (PREVIEW_DIR / f"{name}.jpg").write_bytes(image)
    print(f"[dry-run] preview written to {PREVIEW_DIR}/{name}.*")


# --------------------------------------------------------------- runners ---


def publish_meme(published):
    cta = f"More of how I actually build this: {with_utm(PORTFOLIO_URL, 'meme')}"
    meme = generate_meme(published, cta)
    if not meme["image"]:
        # A meme without its picture is just a confusing sentence. Skip the
        # slot rather than post it; the cadence index moves on next run.
        print("No meme image; skipping this slot.")
        return None

    _save_preview("meme", f"{meme['top']} / {meme['bottom']}\n\n{meme['text']}", meme["image"])
    post_id = publish_to_linkedin(
        meme["text"], image_bytes=meme["image"], image_title=f"{meme['top']} {meme['bottom']}"
    )
    print(f"Successfully published meme to LinkedIn (post id: {post_id})")

    key = f"meme:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    published[key] = {
        "status": "published",
        "type": "meme",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top": meme["top"],
        "bottom": meme["bottom"],
        "scene": meme["scene"],
        "post": meme["text"],
        "linkedin_post_id": post_id,
        "has_image": True,
    }
    save_published(published)
    return {"url": key, "title": f"{meme['top']} / {meme['bottom']}", "post": meme["text"], "post_id": post_id}


def publish_technical(published, want_poll):
    source = next_source(published)
    if source is None:
        return None

    if source["kind"] == "topic":
        title = source["topic"]
        material = f"TOPIC: {title}\n\n{FACTS}"
        cta = f"More of how I build this, with the code: {with_utm(PORTFOLIO_URL)}"
        key = topic_key(title)
        image_url = ""
    else:
        article = scrape(source["url"])
        title = article["title"]
        material = article["content"]
        cta = f"Full write-up: {with_utm(source['url'])}"
        key = source["url"]
        image_url = article["image_url"]

    content = generate_linkedin_content(
        title, material, cta, want_poll=want_poll, source=source["kind"]
    )

    image_bytes = None
    if content["type"] == "poll":
        preview = (
            f"{content['commentary']}\n\nPOLL: {content['question']}\n"
            + "\n".join(f"  - {opt}" for opt in content["options"])
        )
        _save_preview("poll", preview, None)
        post_id = publish_poll_to_linkedin(
            content["commentary"], content["question"], content["options"]
        )
    else:
        image_bytes = generate_post_image(title, content["text"]) or download_image(image_url)
        preview = content["text"]
        _save_preview("post", preview, image_bytes)
        post_id = publish_to_linkedin(content["text"], image_bytes=image_bytes, image_title=title)

    print(f"Successfully published to LinkedIn (post id: {post_id})")

    record = {
        "status": "published",
        "type": content["type"],
        "source": source["kind"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "image_url": image_url,
        "linkedin_post_id": post_id,
    }
    if source["kind"] == "topic":
        record["topic"] = title
    if content["type"] == "poll":
        record["commentary"] = content["commentary"]
        record["question"] = content["question"]
        record["options"] = content["options"]
    else:
        record["post"] = content["text"]
        record["has_image"] = bool(image_bytes)
    published[key] = record
    save_published(published)
    if source["kind"] == "topic":
        remove_topic(title)

    return {"url": key, "title": title, "post": preview, "post_id": post_id}


def process_next():
    published = load_published()
    slot = SLOT_CYCLE[len(published) % len(SLOT_CYCLE)]

    # --dry-run meme|poll|post forces a slot so a format can be previewed.
    for forced in ("meme", "poll", "post"):
        if DRY_RUN and forced in sys.argv:
            slot = forced
    print(f"Slot: {slot} (entry #{len(published) + 1})")

    if slot == "meme":
        return publish_meme(published)
    return publish_technical(published, want_poll=(slot == "poll"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = process_next()
    if result is None:
        print("Nothing to publish this run.")
    else:
        print(f"Key:    {result['url']}")
        print(f"Title:  {result['title']}")
        print("\n----- Generated LinkedIn Post -----\n")
        print(result["post"])
