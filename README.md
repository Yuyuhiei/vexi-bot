# Vexi — AI-Powered UGC Video Review Bot

> Derived from Latin *vexillum* (flag). Vexi flags potential issues in UGC videos so human coaches can make the final call.

## What Vexi Does

Vexi is a Discord bot that provides AI-powered first-pass reviews of UGC (User-Generated Content) videos for the Manus Creator Program. It analyzes videos across two layers:

- **Layer 1 — Legal Compliance:** Checks for copyrighted content, fake testimonials, exaggerated claims, competitor logos, unlicensed AI-generated faces/voices, cultural sensitivity issues, and more. Includes soft reminders for music licensing and ad-disclosure hashtags.
- **Layer 2 — UGC Fundamentals & Storytelling:** Reviews safe zones (IG/TikTok/YouTube), lighting, audio quality, hook strength (evaluated against 12 hook categories), and pacing.

Vexi also detects the video language and provides an English script summary for non-English videos — enabling coaches to review foreign-language creator content without needing to understand the language.

**Vexi is NOT an approver.** It is a friendly AI flagger that helps coaches by surfacing potential issues before they do a deeper manual review.

---

## Features

- `/vexi` slash command — works in any channel, accepts video file attachments or URLs
- **Auto-detect** — automatically reviews videos posted in configured channels
- **Manus Relevance Gate** — rejects non-Manus content before wasting a review
- **Google Drive & YouTube support** — accepts direct links in addition to file uploads
- **Coach tagging** — optionally ping a coach role on auto-detect
- **Multilingual** — supports Korean, Japanese, German, and 100+ other languages
- **Progress indicator** — animated progress bar shown while AI is reviewing
- **Compact output** — single embed with conversational paragraphs (no wall of bullet points)
- **Fallback upload** — if a direct URL fails, Vexi downloads and uploads the video to Gemini automatically

---

## How It Works

```
User posts video (attachment or URL)
        ↓
Vexi detects it (slash command or auto-detect)
        ↓
Video sent to Google Gemini 2.5 Flash (native video understanding)
        ↓
AI checks: Manus relevance → Legal compliance → UGC fundamentals
        ↓
Results posted as a single Discord embed with verdict + coach summary
```

---

## Verdicts

| Verdict | Meaning |
|---|---|
| `LOOKS GOOD` | No significant flags found |
| `NEEDS REVIEW` | Minor issues flagged, coach should check |
| `COACH ATTENTION NEEDED` | Significant flags — coach review required |
| `NOT MANUS CONTENT` | Video is not related to Manus, rejected |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.11+ |
| Discord | discord.py 2.3+ |
| AI | Google Gemini 2.5 Flash (native video understanding) |
| HTTP | aiohttp (async) |
| Hosting | Fly.io (Paris region, 24/7) |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Yes | Discord bot token from the Developer Portal |
| `GEMINI_API_KEY` | Yes | Google Gemini API key from AI Studio |
| `VEXI_CHANNELS` | Yes | Comma-separated channel IDs for auto-detect |
| `COACH_ROLE_ID` | No | Discord role ID to ping when a video is auto-detected |
| `GUILD_ID` | No | Discord guild ID (auto-detected if not set) |

---

## Local Development

### 1. Clone the repo

```bash
git clone https://github.com/your-org/vexi-bot.git
cd vexi-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Or export them directly:

```bash
export DISCORD_BOT_TOKEN="your-token-here"
export GEMINI_API_KEY="your-gemini-key-here"
export VEXI_CHANNELS="1234567890,9876543210"
```

### 4. Run

```bash
python bot.py
```

---

## Deploying to Fly.io

This repo includes a `Dockerfile` and `fly.toml` configured for 24/7 deployment on [Fly.io](https://fly.io).

### Prerequisites

- [flyctl](https://fly.io/docs/hands-on/install-flyctl/) installed
- A Fly.io account

### Deploy

```bash
# Authenticate
flyctl auth login

# Set secrets (environment variables)
flyctl secrets set DISCORD_BOT_TOKEN=your-token-here
flyctl secrets set GEMINI_API_KEY=your-key-here
flyctl secrets set VEXI_CHANNELS=1234567890,9876543210

# Deploy
flyctl deploy
```

### Monitor

```bash
# Check machine status
flyctl status

# Stream live logs
flyctl logs
```

The bot runs as a single machine in the `cdg` (Paris) region with auto-stop disabled, so it stays online 24/7.

---

## Supported Video Sources

| Source | Format | Notes |
|---|---|---|
| Discord attachment | `.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`, `.m4v` | Direct upload |
| Google Drive | Sharing link | Must be set to "Anyone with the link" |
| YouTube | `youtube.com` or `youtu.be` | Public videos |
| Direct URL | Any direct `.mp4` / `.mov` link | Must be publicly accessible |

---

## Adding or Removing Auto-Detect Channels

Update the `VEXI_CHANNELS` secret on Fly.io — no code changes needed:

```bash
flyctl secrets set VEXI_CHANNELS=1234567890,9876543210,1122334455
```

The `/vexi` slash command always works in every channel regardless of this setting.

---

## Project Structure

```
vexi-bot/
├── bot.py              # Main bot (single file — all logic lives here)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition for Fly.io
├── fly.toml            # Fly.io app configuration
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## Version

v1.0 — Initial release
