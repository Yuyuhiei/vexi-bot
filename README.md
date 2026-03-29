# Vexi — AI-Powered UGC Video Review Bot

> Derived from Latin *vexillum* (flag). Vexi flags potential issues in UGC videos so human coaches can make the final call.

## What Vexi Does

Vexi is a Discord bot that provides AI-powered first-pass reviews of UGC (User-Generated Content) videos for the Manus Creator Program. It analyzes videos in two layers:

- **Layer 1 — Legal Compliance:** Checks for copyrighted content, fake testimonials, exaggerated claims, competitor logos, cultural sensitivity, and more. Includes soft reminders for music licensing and ad-disclosure hashtags.
- **Layer 2 — UGC Fundamentals & Storytelling:** Reviews safe zones (IG/TikTok/YouTube), lighting, audio quality, hook strength (evaluated against 12 hook categories), and pacing.

Vexi also detects the video language and provides an English script summary for non-English videos — enabling coaches to review foreign-language creator content.

**Vexi is NOT an approver.** It is a friendly AI flagger that helps coaches by surfacing potential issues before they do a deeper manual review.

## Features

- `/vexi` slash command — works in any channel, accepts video attachments or URLs
- Auto-detect — automatically reviews videos posted in configured channels
- Coach tagging — optionally ping a coach for notification
- Multilingual — supports Korean, Japanese, German, and 100+ languages
- Manus Relevance Gate — rejects non-Manus content before wasting review time
- Progress indicator — animated progress bar while reviewing
- Compact output — single embed with conversational paragraphs

## Tech Stack

- **Runtime:** Python 3.11+
- **Discord:** discord.py 2.3+
- **AI:** Google Gemini 2.5 Flash (native video understanding)
- **Hosting:** Manus CloudHost (24/7)

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Yes | Discord bot token from Developer Portal |
| `GEMINI_API_KEY` | Yes | Google Gemini API key from AI Studio |
| `VEXI_CHANNELS` | Yes | Comma-separated channel IDs for auto-detect |
| `COACH_ROLE_ID` | No | Discord role ID to ping when videos are auto-detected |
| `GUILD_ID` | No | Discord guild ID (auto-detected if not set) |

## Setup

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

```bash
export DISCORD_BOT_TOKEN="your-token-here"
export GEMINI_API_KEY="your-gemini-key-here"
export VEXI_CHANNELS="1234567890,9876543210"
```

### 4. Run

```bash
python3 bot.py
```

## Deploy to Manus CloudHost

1. Push this repo to GitHub (private repo recommended)
2. Go to Manus CloudHost → **Import GitHub**
3. Paste the repo URL → Select **Python** runtime
4. Set the environment variables in the CloudHost settings
5. Deploy — Vexi runs 24/7

## Project Structure

```
vexi-bot/
├── bot.py              # Main bot code (single file)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Adding More Channels

Update the `VEXI_CHANNELS` environment variable in CloudHost:

```
VEXI_CHANNELS=1234567890,9876543210,1122334455
```

No code changes needed. The `/vexi` slash command always works in every channel regardless.

## Version

v1.0 — Initial release
