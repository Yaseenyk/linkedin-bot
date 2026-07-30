# linkedin-bot

Autonomous LinkedIn publisher. Reads a queue of article URLs, drafts a post
about each one with OpenAI, generates a brand-locked cover image, publishes to
LinkedIn on a schedule, and commits its own state back to the repo.

No server, no database, no dashboard — the queue is a text file, the state is a
JSON file, and GitHub Actions is the scheduler.

## How it works

1. **Queue** — `urls.txt` holds article URLs, one per line.
2. **State** — `published.json` records what has already gone out, so a re-run
   never double-posts.
3. **Extract** — the article is fetched and parsed with BeautifulSoup to get the
   real title and body, rather than trusting metadata.
4. **Draft** — OpenAI writes the post under a voice contract that bans the usual
   LLM tells.
5. **Cover** — a wordless brand-locked image is generated to match the site's
   visual system.
6. **Publish** — posted via the LinkedIn API, then the state file is committed
   back to `main`.

## The link goes in the first comment, not the post

LinkedIn demotes posts that send people off-platform, so the article URL is
deliberately kept out of the post body and published as the first comment
instead. This is the single most important behaviour in the pipeline and it is
easy to "fix" by accident — do not move the link into the body.

## Cover image style is brand-locked

The image style rules here must stay byte-identical to the two generators in the
portfolio repo (`scripts/generate-og-images.py` and `scripts/campus_social.py`).
Flat 2D vector on near-black ink, cyan and violet only, no glow, no warm hues,
and no lettering of any kind. Editing one without the others breaks visual
consistency across every channel.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # add OPENAI_API_KEY and the LinkedIn credentials
python main.py
```

Secrets live in `.env`, which is gitignored. The same `OPENAI_API_KEY` is used
by the portfolio repo's generators.

## Schedule

Runs Tue/Wed/Thu at 10:00 UTC via GitHub Actions. Adding a URL to `urls.txt` is
the only manual step; everything after that is automatic.

---

Built by [Yaseen Khatib](https://yaseenkhatib.streamerosai.com) — Senior
Full-Stack AI Engineer, Hyderabad. This pipeline is one of five products shipped
solo; architecture teardowns of all of them are at
[yaseenkhatib.streamerosai.com/products](https://yaseenkhatib.streamerosai.com/products).

Available for [architecture reviews, AI integration and custom
systems](https://yaseenkhatib.streamerosai.com/solutions).
