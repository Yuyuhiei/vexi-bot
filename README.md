# Vexi — AI-Powered UGC Video Review Bot

> Derived from Latin *vexillum* (flag). Vexi flags potential issues in UGC videos so human coaches can make the final call.

## What Vexi Does

Vexi is a Discord bot that provides AI-powered first-pass reviews of UGC (User-Generated Content) videos for the Manus Creator Program.

**Vexi is NOT an approver.** It is a friendly AI flagger that helps coaches by surfacing potential issues before they do a deeper manual review.

---

## Commands

| Trigger | What it does |
|---|---|
| `/vexi video1..5:` / `video_url1..5:` `[coach:@Coach]` | Review up to 5 videos, queued per-user, one review reply each |
| `/study video:` / `video_url:` `[niche:...]` | Study any creator's video → format analysis + Manus adaptation brief + copyable script |
| `@Vexi study <url> [niche:...]` | Mention form of /study, works in any channel |
| Right-click a Study result → **Apps → Revise this script** | Iterate on a generated script with a free-text instruction |
| **📋 View full analysis** button under a review | Private popup with script summary, transcript (translated), every check & measured facts |
| Auto-detect | Automatically reviews videos posted in `VEXI_CHANNELS` |

---

## Two Pipelines (`VEXI_PIPELINE`)

### `multiagent` (v3) — extract-then-adjudicate

Instead of one model watching the whole video and writing the verdict in one shot, v3 splits the work across specialists and lets code compute the facts:

```
video ──► ffmpeg: 1fps frames (pHash-deduped) + 16kHz audio
   │
   ├─► Vision extractor  (Gemini Flash-Lite) — verbatim on-screen text, Manus
   │        logo/UI visibility, websites & features shown, per second
   ├─► Transcriber       (Groq Whisper, word timestamps; Gemini fallback)
   ├─► Witness           (Gemini Flash, full video) — pacing, energy, hook,
   │        production quality: the things frame logs can't see
   ├─► Deterministic layer (pure Python — cannot hallucinate) — logo screen
   │        time, homophone corrections ("Manners"→"Manus"), website count,
   │        feature count, hook/CTA windows, spelling checks
   └─► Adjudicator "head reviewer" (Gemini 2.5 Pro, auto-fallback to Flash) —
            reads all evidence, trusts measured facts over model impressions,
            writes the final structured review
```

The reply is a conversational full-width message (kudos + only the things to fix + colored risk line) with a **📋 View full analysis** button that opens a private popup. Review state is stored invisibly on the message itself, so buttons keep working after restarts with no database.

If any single extractor fails, the review proceeds with a caveat and is flagged for a closer coach look. If the pipeline hard-fails, that video automatically falls back to the legacy pipeline — creators always get a review.

### `legacy` (v2) — single call

The original one-Gemini-call pipeline with the embed reply. Kept fully intact as the rollback path.

---

## Rollout & Rollback

```bash
# Turn v3 ON in prod (machine restarts in seconds):
fly secrets set -a vexi-bot VEXI_PIPELINE=multiagent GROQ_API_KEY=<key>

# ROLLBACK — one command, no redeploy, old pipeline + old embed reply:
fly secrets set -a vexi-bot VEXI_PIPELINE=legacy

# Nuclear option — redeploy the last v2 code entirely:
git checkout v2-stable && flyctl deploy
```

Notes:
- Any `fly secrets set` restarts the machine and drops in-flight queued reviews — flip during quiet hours.
- The v3 default head model `gemini-2.5-pro` needs a **paid-tier** Gemini key. On free tier, also set `GEMINI_MODEL_ADJUDICATOR=gemini-2.5-flash`.
- Test on staging first: see `fly.staging.toml` (self-documenting — separate Discord bot + separate Fly app, deployed manually, never touched by CI).

---

## Verdicts

| Verdict | Risk line | Meaning |
|---|---|---|
| `LOOKS GOOD` | 🟢 Low | No flags found |
| `NEEDS REVIEW` | 🟡 Medium | Minor issues flagged, coach should check |
| `COACH ATTENTION NEEDED` | 🟠 High | Significant flags — coach review required |
| `AUTO-REJECT` | 🔴 Critical | Banned language detected — coach must clear |
| `NOT MANUS CONTENT` | ⚪ | Video is not related to Manus |

In v3 the verdict is **derived in code from the structured findings** (flag/recommend + risk level), not free-form model choice.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.11+ (`python:3.11-slim` + ffmpeg) |
| Discord | discord.py ≥2.6 (Components V2, persistent dynamic buttons) |
| Vision/OCR | Gemini 2.5 Flash-Lite (batched deduped frames) |
| Transcription | Groq `whisper-large-v3-turbo` (word timestamps) → Gemini fallback |
| Witness / Legacy | Gemini 2.5 Flash (native video understanding) |
| Head reviewer | Gemini 2.5 Pro → Flash fallback |
| Frame dedupe | Pillow + ImageHash (perceptual hash) |
| HTTP | aiohttp (async) |
| Hosting | Fly.io (Paris region, 24/7, 1 GB shared VM) |

All model names are env-overridable (`GEMINI_MODEL_*`) — the announced Gemini 2.5 retirement (Oct 16, 2026) is a secrets change, not a redeploy.

~Cost per review: **$0.03–0.04** with the Pro head model, ~$0.01 on Flash.

---

## Environment Variables

See `.env.example` for the full annotated list. The important ones:

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Yes | Discord bot token from the Developer Portal |
| `GEMINI_API_KEY` | Yes | Google Gemini API key (paid tier for the Pro head model) |
| `VEXI_CHANNELS` | Yes | Comma-separated channel IDs for auto-detect |
| `VEXI_PIPELINE` | No | `legacy` (default) or `multiagent` — the kill-switch |
| `GROQ_API_KEY` | No | Free key → best transcription (word timestamps) |
| `COACH_ROLE_ID` | No | Role to ping on auto-detected videos |
| `APIFY_API_TOKEN` | No | /study fallback when yt-dlp is blocked on IG/TikTok |

---

## Local Development

```bash
git clone https://github.com/your-org/vexi-bot.git
cd vexi-bot
pip install -r requirements.txt          # needs Python 3.11+, ffmpeg on PATH
cp .env.example .env                     # fill in tokens
set -a; source .env; set +a
python bot.py
```

### Selftest CLI — run the v3 pipeline without Discord

```bash
# Any stage against a local file or URL (GDrive/TikTok/IG/YouTube/direct):
python -m vexi.selftest video.mp4 --stage media          # ffmpeg + dedupe only, no API
python -m vexi.selftest video.mp4 --stage vision         # vision log
python -m vexi.selftest video.mp4 --stage asr            # transcript + homophone fixes
python -m vexi.selftest video.mp4 --stage deterministic  # measured facts
python -m vexi.selftest video.mp4 --stage all --json-out /tmp/logs   # full review
```

### Unit tests

```bash
python -m pytest tests/ -q
```

---

## Deploying to Fly.io

Pushes to `main` auto-deploy to **production** via GitHub Actions. Manual deploy:

```bash
flyctl auth login
flyctl secrets set DISCORD_BOT_TOKEN=... GEMINI_API_KEY=... VEXI_CHANNELS=...
flyctl deploy

flyctl status   # machine status
flyctl logs     # live logs
```

Staging (never auto-deployed): `fly deploy -c fly.staging.toml -a vexi-bot-staging --remote-only`

---

## Supported Video Sources

| Source | Format | Notes |
|---|---|---|
| Discord attachment | `.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`, `.m4v` | Direct upload, ≤100MB |
| Google Drive | Sharing link | "Anyone with the link"; >25MB interstitial handled |
| Instagram / TikTok / YouTube | Post/Reel/video URL | yt-dlp, Apify fallback for IG/TikTok |
| Direct URL | Any direct `.mp4` / `.mov` link | Must be publicly accessible |

---

## Project Structure

```
vexi-bot/
├── bot.py                 # Entrypoint (thin)
├── vexi/
│   ├── config.py          # All env vars, model names, VEXI_PIPELINE flag
│   ├── prompts.py         # All Gemini prompts (v2 legacy + v3)
│   ├── gemini.py          # Gemini client layer (retry, JSON parse, File API)
│   ├── downloaders.py     # CDN / GDrive / yt-dlp / Apify + acquire_video
│   ├── media.py           # ffprobe/ffmpeg + pHash frame dedupe
│   ├── extractors.py      # Vision+OCR, ASR (Groq/Gemini), witness
│   ├── deterministic.py   # Pure-Python evidence math (unit-tested)
│   ├── multiagent.py      # v3 orchestrator + adjudicator (Pro→Flash)
│   ├── pipeline.py        # legacy/multiagent dispatch + legacy routing
│   ├── render.py          # CV2 conversational reply + legacy embeds
│   ├── commands.py        # Slash commands, queue, auto-detect, revise
│   ├── progress.py        # Shared progress animation
│   └── selftest.py        # Pipeline CLI (no Discord needed)
├── tests/                 # Unit tests (deterministic layer)
├── requirements.txt
├── Dockerfile             # python:3.11-slim + ffmpeg
├── fly.toml               # Production app
├── fly.staging.toml       # Staging app (setup steps inside)
├── .env.example
└── README.md
```

---

## Version

v3.0 — Multi-agent pipeline, conversational replies, staging + kill-switch rollback
(v2 tagged as `v2-stable`; v1.0 was the initial single-file release)
