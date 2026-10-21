# linkedin-bot

Autonomous LinkedIn publisher. Each scheduled run publishes one item: a
technical story post, a poll, or a meme, on a fixed cycle. Technical posts are
written by OpenAI either from a topic in `topics.txt` (grounded in a verified
facts block about the author's own systems, linking to the portfolio) or from
an article URL in `urls.txt` (grounded in the scraped article, linking to it).
State is committed back to the repo after every run.

No server, no database, no dashboard. The queue is two text files, the state
is a JSON file, and GitHub Actions is the scheduler.

## How it works

1. **Slot** — `SLOT_CYCLE` in `main.py` is `post, post, meme, post, poll`.
   The slot is picked by the number of entries already in `published.json`,
   so the cadence is deterministic and survives re-runs.
2. **Source** — for a post or poll slot, the first unused line of
   `topics.txt` wins. When no topics remain, the bot falls back to `urls.txt`
   (direct post URLs, or listing pages it scans for `/blog/` links).
3. **State** — `published.json` records every item (keyed by URL, `topic:<slug>`
   or `meme:<timestamp>`), so a re-run never double-posts. A used topic is
   also removed from `topics.txt`.
4. **Draft** — OpenAI writes the post in JSON mode under a voice contract
   that bans LLM tells and forbids invented incidents or metrics. Topic posts
   may only use the `FACTS` block in `main.py`; keep it accurate to the code.
5. **Image** — technical posts get a two-pass cover: a text model writes an
   art-director brief, `gpt-image-1` renders it, Pillow re-encodes to JPEG to
   drop the C2PA badge. Memes get a wordless cartoon scene with the captions
   burned on locally in Impact (model-rendered text is unreliable).
6. **Publish** — posted via the LinkedIn Posts API, then `published.json` and
   `topics.txt` are committed back to `main`.

## The link lives in the post body

The article (or portfolio) link is the last line of every post, UTM-tagged so
LinkedIn traffic is distinguishable in analytics. It was briefly moved to a
first comment to dodge LinkedIn's off-platform link demotion, but the comment
step was unreliable and left posts pointing at a comment that never appeared.

## Cover images are topic-specific, not brand-locked

Unlike the portfolio's on-site OG covers, LinkedIn covers should depict the
actual subject of each post so they read in a busy feed. Do not sync
`IMAGE_STYLE_RULES` here with the portfolio scripts.

## Previewing without posting

```bash
python main.py --dry-run          # next slot in the cycle
python main.py --dry-run meme     # force a slot: meme | poll | post
```

Dry runs generate everything (text and image) but publish nothing and save
nothing. Output lands in `preview/` (gitignored).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # add OPENAI_API_KEY and LINKEDIN_ACCESS_TOKEN
python main.py
```

Secrets live in `.env`, which is gitignored. The same `OPENAI_API_KEY` is used
by the portfolio repo's generators. LinkedIn tokens expire after about 60 days
and cannot be refreshed automatically; the bot exits with the fix when that
happens, but only on a run that has something to publish.

## Schedule

Runs Tue/Wed/Thu at 10:00 UTC via GitHub Actions. Adding a line to
`topics.txt` or `urls.txt` is the only manual step.

---

Built by [Yaseen Khatib](https://yaseenkhatib.streamerosai.com) — Senior
Full-Stack AI Engineer, Hyderabad. This pipeline is one of five products shipped
solo; architecture teardowns of all of them are at
[yaseenkhatib.streamerosai.com/products](https://yaseenkhatib.streamerosai.com/products).

Available for [architecture reviews, AI integration and custom
systems](https://yaseenkhatib.streamerosai.com/solutions).
