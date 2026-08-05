"""Configuration for Vexi. Every environment variable is read here — no other
module calls os.environ directly."""

import logging
import os

# ---------------------------------------------------------------------------
# Discord / core
# ---------------------------------------------------------------------------
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
VEXI_CHANNELS_RAW = os.environ.get("VEXI_CHANNELS", "")

VEXI_CHANNEL_IDS: list[int] = []
if VEXI_CHANNELS_RAW.strip():
    VEXI_CHANNEL_IDS = [int(ch.strip()) for ch in VEXI_CHANNELS_RAW.split(",") if ch.strip()]

# Coach role ID for auto-detect channel pings (optional)
COACH_ROLE_ID = os.environ.get("COACH_ROLE_ID", "")

# ---------------------------------------------------------------------------
# Apify fallback (used by /study only when yt-dlp fails on Instagram/TikTok)
# ---------------------------------------------------------------------------
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
APIFY_INSTAGRAM_ACTOR = os.environ.get("APIFY_INSTAGRAM_ACTOR", "apify/instagram-scraper")
APIFY_TIKTOK_ACTOR = os.environ.get("APIFY_TIKTOK_ACTOR", "clockworks/free-tiktok-scraper")
# Set APIFY_USE_PROXY=true to route the video-bytes download through Apify's
# residential proxy. Costs Apify proxy bandwidth (~$8/GB) but bypasses Instagram
# CDN's datacenter-IP block that returns 403 on raw downloads.
APIFY_USE_PROXY = os.environ.get("APIFY_USE_PROXY", "").strip().lower() in ("1", "true", "yes", "on")
INSTAGRAM_COOKIES_FILE = os.environ.get("INSTAGRAM_COOKIES_FILE", "")

# Last-resort fallback when both yt-dlp and Apify can't fetch the video
LUMISCRIPT_URL = "https://lumiscript.manus.space/"

# ---------------------------------------------------------------------------
# Pipeline selection — the rollback kill-switch.
#   'legacy'     → original single Gemini call per review (v2 behavior)
#   'multiagent' → v3 extract-then-adjudicate pipeline
# Flipping back in prod is one command, no redeploy:
#   fly secrets set -a vexi-bot VEXI_PIPELINE=legacy
# ---------------------------------------------------------------------------
VEXI_PIPELINE = os.environ.get("VEXI_PIPELINE", "legacy").strip().lower()

# ---------------------------------------------------------------------------
# Model names. All env-overridable so the Gemini 2.5 → 3.x migration
# (2.5 retires Oct 16, 2026) is a `fly secrets set`, not a redeploy.
# ---------------------------------------------------------------------------
GEMINI_MODEL_LEGACY = os.environ.get("GEMINI_MODEL_LEGACY", "gemini-2.5-flash")
GEMINI_MODEL_VISION = os.environ.get("GEMINI_MODEL_VISION", "gemini-2.5-flash-lite")
GEMINI_MODEL_WITNESS = os.environ.get("GEMINI_MODEL_WITNESS", "gemini-2.5-flash")
GEMINI_MODEL_ADJUDICATOR = os.environ.get("GEMINI_MODEL_ADJUDICATOR", "gemini-2.5-pro")
GEMINI_MODEL_ADJUDICATOR_FALLBACK = os.environ.get("GEMINI_MODEL_ADJUDICATOR_FALLBACK", "gemini-2.5-flash")
GEMINI_MODEL_TRANSLATE = os.environ.get("GEMINI_MODEL_TRANSLATE", "gemini-2.5-flash-lite")

# ---------------------------------------------------------------------------
# ASR backend: 'auto' picks Groq when GROQ_API_KEY is set, else Gemini.
# ---------------------------------------------------------------------------
VEXI_ASR = os.environ.get("VEXI_ASR", "auto").strip().lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_ASR_MODEL = os.environ.get("GROQ_ASR_MODEL", "whisper-large-v3-turbo")

# ---------------------------------------------------------------------------
# Media pipeline tuning
# ---------------------------------------------------------------------------
VEXI_FRAME_FPS = int(os.environ.get("VEXI_FRAME_FPS", "1"))
VEXI_FRAME_MAX_W = int(os.environ.get("VEXI_FRAME_MAX_W", "768"))
VEXI_PHASH_THRESHOLD = int(os.environ.get("VEXI_PHASH_THRESHOLD", "6"))
VEXI_MAX_FRAMES = int(os.environ.get("VEXI_MAX_FRAMES", "45"))
VEXI_FFMPEG_TIMEOUT_S = int(os.environ.get("VEXI_FFMPEG_TIMEOUT_S", "120"))
VEXI_MAX_CONCURRENT_PIPELINES = int(os.environ.get("VEXI_MAX_CONCURRENT_PIPELINES", "2"))

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_VIDEO_BYTES = 100 * 1024 * 1024  # 100MB cap, enforced by every downloader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vexi")
