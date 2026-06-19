"""
Vexi — AI-Powered UGC Video Review Discord Bot
Derived from Latin "vexillum" (flag). Vexi flags potential issues in UGC videos
so human coaches can make the final call.

Triggers:
  1. /vexi slash command (works in any channel)
  2. Auto-detect video posts in configured channels (VEXI_CHANNELS env var)
"""

import os
import json
import re
import shutil
import tempfile
import asyncio
import logging
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands
import aiohttp
import google.genai as genai
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
VEXI_CHANNELS_RAW = os.environ.get("VEXI_CHANNELS", "")

VEXI_CHANNEL_IDS: list[int] = []
if VEXI_CHANNELS_RAW.strip():
    VEXI_CHANNEL_IDS = [int(ch.strip()) for ch in VEXI_CHANNELS_RAW.split(",") if ch.strip()]

# Guild ID for instant slash command sync (set via env var or auto-detected)
GUILD_ID = os.environ.get("GUILD_ID", "")

# Coach role ID for auto-detect channel pings (optional)
COACH_ROLE_ID = os.environ.get("COACH_ROLE_ID", "")

# Apify fallback (used by /study only when yt-dlp fails on Instagram/TikTok)
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

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vexi")

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# The Core AI Prompt — Compact Conversational Output
# ---------------------------------------------------------------------------
REVIEW_PROMPT = r"""
You are Vexi, a friendly, non-authoritative AI assistant helping human coaches review UGC videos for Manus. Your job is to flag potential issues for the human coach to make the final decision. You are NOT the approver — you are a helpful first-pass flagger.

═══════════════════════════════════════
LAYER 0: MANUS RELEVANCE GATE (CHECK THIS FIRST)
═══════════════════════════════════════
Before doing ANY review, determine if this video is related to Manus (the AI agent product).

A Manus-related video includes ANY of the following:
- Mentions, shows, or demonstrates Manus (the AI tool/product)
- Is a UGC-style video that could be promoting or reviewing Manus
- Shows someone using an AI tool that could be Manus
- Contains Manus branding, logo, website, or app
- Is a talking-head, tutorial, reaction, or testimonial that references Manus
- Is clearly a creator's UGC content intended for the Manus Creator Program

If the video is NOT related to Manus at all (e.g., random memes, personal vlogs, unrelated content):
- Set "manus_relevant" to false
- Set "quick_verdict" to "NOT MANUS CONTENT"
- Skip all other analysis. Set "legal_paragraph" and "content_paragraph" to empty strings.
- In "overall_summary" write: "This video does not appear to be related to Manus. Please submit a Manus-related UGC video for review."

If the video IS related to Manus, set "manus_relevant" to true and proceed.

═══════════════════════════════════════
LAYER 0.5: MONEY & INCOME CLAIM SCAN (v1.3 — RUN BEFORE ALL OTHER CHECKS)
═══════════════════════════════════════
Scan ALL spoken audio, on-screen text, captions, and visuals for money and income language. Use the tiers below — not everything money-adjacent is an auto-reject.

IMPORTANT — IGNORE COMPLETELY (do not flag):
- Song lyrics or audio slang referencing money ("getting my paper wet", "money on my mind", etc.) — these are background music/culture, not claims
- Productivity or client-work framing with no income claim: "I use Manus to deliver client projects faster", "helps me take on more clients", "saves me time"
- Tool capability demos with no income angle: building a Shopify store, e-commerce site, or app to show Manus features — fine even if the tool could theoretically be used to earn money

─────────────────────────────────────
AUTO-REJECT — set "quick_verdict" to "AUTO-REJECT" if ANY of these are present:
─────────────────────────────────────
1. Explicit personal income claims (with or without numbers):
   - "I made $5k with Manus" / "I earned ₱10,000" / "I make $2,000/month"
   - "this replaced my 9-5" / "I quit my job because of Manus" / "this is my full-time income"
   - "I make a full-time living with this" / "this pays my bills"

2. Third-party earnings claims:
   - "My friend made $3,000 using Manus" / "creators are earning $X with this"

3. Cost-savings with a dollar amount:
   - "I saved $500 by using Manus instead of hiring a freelancer"
   - Any specific currency amount framed as money saved or replaced

4. Banned income phrases regardless of context:
   - "passive income" / "financial freedom" / "get rich quick" / "easy money" / "guaranteed income" / "make you rich" / "zero risk"

5. On-screen revenue proof as the focus:
   - Showing a Shopify dashboard, YouTube analytics, or any earnings screenshot where real revenue numbers are the point being made

─────────────────────────────────────
MEDIUM RISK [MEDIUM] — flag for coach review, not auto-reject:
─────────────────────────────────────
- Aspirational/goal language with numbers but no claim: "I'm working toward my $10k month" / "my goal is $5k"
- Revenue dashboards or analytics accidentally visible in the background but clearly not the focus of the video

─────────────────────────────────────
If AUTO-REJECT is triggered:
- In "legal_paragraph", name the exact phrase(s) with an [AUTO-REJECT] tag and timestamp if possible
- Still complete all remaining checks below (the coach needs the full picture)
- End "overall_summary" with: "If you think this is a mistake, please tag your coach for a manual review."

═══════════════════════════════════════
WHAT TO CHECK (Internal — use these to inform your paragraphs)
═══════════════════════════════════════

LEGAL COMPLIANCE CHECKS (v1.2 Checklist):
1. Income & Money Claims — See LAYER 0.5 above. Follow the three-tier system: AUTO-REJECT for explicit claims, third-party earnings, cost-saving amounts, banned phrases, and on-screen revenue proof; [MEDIUM] for aspirational goal language or background dashboards; ignore song lyrics, slang, and productivity framing entirely. (HIGH RISK / AUTO-REJECT)
2. Absolute Claims — Phrases like "100%", "zero errors", "fully replaces humans", "best AI" without proof. (HIGH RISK)
3. Efficiency Numbers Without Proof — Time-saved or speed claims (e.g., "build a site in 10 minutes", "save 5 hours") require real supporting data or evidence. Flag if none is visible. (MEDIUM RISK)
4. Copyrighted / Trademarked Material — ZERO TOLERANCE. UGC is a paid advertisement, so any famous brand, logo, celebrity, or copyrighted character that the creator does not own CANNOT be published as-is. This includes: brand logos (Nike, Adidas, Apple, etc.), copyrighted characters (Disney, Marvel, anime, Iron Man, etc.), celebrity names/images/likenesses (e.g. David Goggins, Robert Downey Jr.), and protected event branding (FIFA World Cup, Olympics, named teams/players/jersey numbers). Flag every instance as [HIGH] with the exact element and timestamp, and clearly tell the creator it must be REMOVED — they can regenerate any AI image with a prompt that omits the logo/character/likeness. EXCEPTIONS (do NOT flag): a copyrighted character appearing incidentally in the background or on a desktop screen for under 2 seconds; a creator simply wearing a branded jersey or shirt (e.g. an Adidas tee) as everyday clothing. (HIGH RISK)
5. Fake Reviews or Testimonials — Actors scripting fake customer stories, fake "first-time" reactions. (HIGH RISK)
6. Exaggerated or Unproven Claims — Unprovable numerical claims beyond income (e.g., "10x your revenue"). Personal honest experiences without guarantees are fine. (MEDIUM RISK)
7. People Without Permission — Identifiable bystanders, friends, or children without release. (MEDIUM RISK)
8. Competitor Logos or Products — Visible competitor logos, mocking competitors. (MEDIUM RISK)
9. AI-Generated Faces or Voices — AI-generated people used as testimonials without disclosure. (MEDIUM RISK)
10. Privacy Claims — Any "data security/privacy" statements must match Manus's privacy policy; flag vague or absolute privacy promises. (MEDIUM RISK)
11. Product Demo Accuracy — Only show Manus UI/features that exist and work in the current version. Flag if demo shows non-existent or unshippable features. (MEDIUM RISK)
12. Real-Person Likeness — AI-generated faces or voices resembling real people without authorization. (MEDIUM RISK)
13. Font Licensing — Premium fonts without commercial license. Google Fonts are fine. (LOW RISK)
14. Filming Locations — Inside recognizable branded private spaces. (LOW RISK)
15. Platform Rules — Missing branded content toggles. (LOW RISK)
16. Cultural Sensitivity — Stereotypes, accents as jokes, religious/political imagery. (LOW RISK)
17. Music — If you hear music, give a soft reminder to confirm it's from TikTok/IG library or approved royalty-free source. NEVER flag music as a risk.
18. Ad Disclosure — Remind creators to include at least one ad-disclosure hashtag (#ManusAd, #ManusPartner, #Ad, #Sponsored) in their caption when posting. Generic tags like #Manus alone are NOT enough. NEVER suggest putting hashtags on the video itself. NEVER flag as a risk.

MANUS PLUG & BRAND PRESENCE CHECKS (a weak or missing plug is a potential rejection — verdict COACH ATTENTION NEEDED):
1. Clear Manus Plug — Every video must clearly feature Manus. There should be an actual demonstration of the creator USING Manus (the interface, a real task, a workflow), not just a passing mention. (HIGH RISK if absent)
2. Manus Interface/Logo Visibility — The Manus interface, website, or logo should be clearly on screen for at least 4 seconds total. Flag if it appears only briefly (e.g. a 1-2 second flash). (HIGH RISK if under ~4s)
3. End CTA — The video should end with a call-to-action that ties back to Manus, e.g. "I made this with Manus — comment 'PROMPT' and I'll send you the exact one" or "comment 'WEBSITE' for the build". Flag if there's no closing CTA. (MEDIUM RISK if missing)
4. Low-Effort Plug — Flag low-effort plugs as a potential rejection for the coach: e.g. just a "made with Manus" text card on screen for ~2 seconds with no real demo of how Manus was used. These should be flagged [HIGH] and routed to COACH ATTENTION NEEDED.

UGC FUNDAMENTALS CHECKS:
1. Safe Zones — Critical text/face in bottom 350px (caption area) or top 250px (UI overlay)?
2. Lighting & Audio — Face visible? Audio clear? No echo/background noise?
3. Hook & Storytelling — Evaluate the first 3 seconds against these 12 hook categories:
   (1) Curiosity / "Feels Illegal to Know"  (2) Challenge / Speed Run  (3) Before & After / Transformation
   (4) Hot Take / Controversial / Pattern Interrupt  (5) Demo / How-To (Punchy Openers)
   (6) Social Proof / Flex / Authority  (7) Skits  (8) News & Presentation  (9) FOMO / Urgency
   (10) Anti-Hook / Reverse Psychology  (11) Comparison / "This vs. That"  (12) Emotional / Relatable
4. Pacing & Dead Air — Awkward silences, long pauses, or dead space?

═══════════════════════════════════════
OUTPUT FORMAT — COMPACT & CONVERSATIONAL
═══════════════════════════════════════
Return ONLY a valid JSON object (no markdown, no code fences) with this exact structure:

{
  "manus_relevant": true,
  "language_detected": "English",
  "script_summary": "2-3 sentence English summary of what the creator says and shows. If non-English, this serves as the translation for coaches.",
  "legal_paragraph": "Write a SHORT conversational paragraph (3-5 sentences max) summarizing the legal compliance findings. If AUTO-REJECT keywords were found, lead with them clearly using [AUTO-REJECT] and the exact phrase. Then naturally weave in any other flags — mention the specific issue, the risk level in brackets like [HIGH] or [MEDIUM], and the timestamp if applicable. If there are no flags, say so briefly. Always end with the music soft reminder (if music was detected) and the ad-disclosure hashtag reminder as natural sentences. Example tone for clean video: 'No major legal flags here! No income guarantees, absolute claims, or copyrighted content spotted. One soft note — at 0:15 there's a time-saved claim without visible proof [MEDIUM], so your coach might want to verify that. I hear some background music, so just confirm it's from a licensed source. And remember to pop an ad-disclosure hashtag like #ManusAd in your caption when posting!'",
  "content_paragraph": "Write a SHORT conversational paragraph (4-6 sentences max) summarizing the UGC fundamentals AND the Manus plug quality. Cover safe zones, lighting/audio, the hook (mention which of the 12 categories it fits and whether it's strong or could be improved — suggest a specific alternative if weak), and pacing. Then ALWAYS address the Manus plug: is there a clear demo of the creator using Manus, is the Manus interface/website/logo on screen for at least ~4 seconds, and does the video end with a Manus CTA? If the plug is weak, missing, or low-effort (e.g. just a 2-second 'made with Manus' text card with no real demo), flag it as [HIGH] and note it needs coach attention. Be constructive and specific. Example tone: 'Lighting and audio are solid and safe zones look good for IG and TikTok. Your hook falls into the Demo/How-To category — decent, but try opening with \"I built an entire website in 30 seconds\" for more instant curiosity. Pacing is smooth. On the Manus side, your plug is a bit light [HIGH] — the interface only flashes for about 1 second and there's no clear demo of how you used it, so a coach should take a look. Try showing the Manus workspace for at least 4 seconds and end with a CTA like \"comment PROMPT and I'll send you the exact one.\"'",
  "quick_verdict": "LOOKS GOOD / NEEDS REVIEW / COACH ATTENTION NEEDED / AUTO-REJECT / NOT MANUS CONTENT",
  "overall_summary": "One final sentence. Always include: 'A human coach will review this shortly for final approval.' If AUTO-REJECT, start with: 'This video contains auto-reject language and must be reviewed by a coach before any use.' then end with: 'If you think this is a mistake, please tag your coach for a manual review.'"
}

VERDICT ROUTING:
- Any AUTO-REJECT trigger (see LAYER 0.5) → "quick_verdict" = "AUTO-REJECT".
- Any [HIGH] flag — including copyrighted/trademarked material or a weak/missing/low-effort Manus plug → "quick_verdict" = "COACH ATTENTION NEEDED".
- Only [MEDIUM] flags and no higher → "quick_verdict" = "NEEDS REVIEW".
- No flags at all → "quick_verdict" = "LOOKS GOOD".

CRITICAL RULES FOR THE PARAGRAPHS:
- Keep the legal paragraph SHORT (3-5 sentences) and the content paragraph concise (4-6 sentences). Do NOT write essays.
- Be conversational and friendly, like a peer creator giving feedback in a chat.
- Naturally mention ALL relevant checks within the paragraph flow — don't use headers, bullet points, or field labels.
- If something is fine, you can group multiple "all good" items in one sentence (e.g., "No income guarantees, copyrighted content, or fake testimonials spotted.").
- If something needs attention, be specific but brief (mention what, where/when, and risk level).
- AUTO-REJECT findings must always be called out first and clearly, with the exact phrase detected.
- Music and ad disclosure reminders should feel like natural sentences at the end of the legal paragraph, not separate callouts.
- For hooks, always mention which of the 12 categories it falls into.
- If the video is in a foreign language, still review it fully. Use "script_summary" for the English translation.
- NEVER use markdown formatting (no bold, no headers, no bullets) inside the paragraph strings — just plain conversational text.
"""

# ---------------------------------------------------------------------------
# Study Mode Prompt — Format Analysis + Manus Adaptation Brief
# ---------------------------------------------------------------------------
STUDY_PROMPT = r"""
You are Vexi in "Study Mode" — a creative strategist helping the Manus UGC team learn from high-performing content in any niche or brand.

Your job: watch this video, break down WHY it works, then write a concise Manus adaptation brief.

═══════════════════════════════════════
WHAT TO ANALYZE
═══════════════════════════════════════

PART 1 — FORMAT ANALYSIS:
- Source context: niche, creator style, platform
- Hook: which of these 12 categories does it use?
  (1) Curiosity / "Feels Illegal to Know"  (2) Challenge / Speed Run  (3) Before & After / Transformation
  (4) Hot Take / Controversial / Pattern Interrupt  (5) Demo / How-To (Punchy Openers)
  (6) Social Proof / Flex / Authority  (7) Skits  (8) News & Presentation  (9) FOMO / Urgency
  (10) Anti-Hook / Reverse Psychology  (11) Comparison / "This vs. That"  (12) Emotional / Relatable
- Narrative structure (hook → problem → solution → CTA, etc.)
- Pacing, editing style, key visual/audio techniques
- The single core reason this format works

PART 2 — MANUS ADAPTATION:
- How to port this exact format to a Manus UGC video, beat by beat
- Which Manus features fill each narrative role (agentic tasks, browser automation, research, document generation, etc.)
- A numbered shot/beat outline a creator can follow (5-7 beats max)
- What NOT to copy — flag anything that would fail a Vexi compliance check (income claims, absolute promises, competitor mentions, fake testimonials)

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════
Return ONLY a valid JSON object (no markdown, no code fences):

{
  "source_context": "One sentence: niche, creator style, platform.",
  "format_breakdown": "2-3 sentences max. Name the hook category explicitly. Cover structure and pacing briefly.",
  "what_makes_it_work": "One sentence only.",
  "manus_adaptation": "2-3 sentences max. What to keep, what to swap, which Manus features fill each beat.",
  "suggested_outline": "Numbered plain-text outline, 5-7 beats, each on its own line. No markdown symbols.",
  "copy_guardrails": "1-2 sentences. Flag compliance risks or confirm it's clean.",
  "adaptation_difficulty": "EASY / MODERATE / COMPLEX"
}

CRITICAL RULES:
- Be concise. Every field has a strict length cap — do not exceed it.
- NEVER use markdown formatting inside string values — plain text only.
- If the video has no audio or is very short, work with what is visible.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
GDRIVE_PATTERN = re.compile(r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)")
GDRIVE_OPEN_PATTERN = re.compile(r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)")
YOUTUBE_PATTERN = re.compile(r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)")
DISCORD_CDN_PATTERN = re.compile(r"https?://cdn\.discordapp\.com/attachments/")
INSTAGRAM_PATTERN = re.compile(r"https?://(www\.)?instagram\.com/(reel|p|tv)/([a-zA-Z0-9_-]+)")
TIKTOK_PATTERN = re.compile(r"https?://(www\.|vm\.|vt\.)?tiktok\.com/")


def _is_social_media_url(url: str) -> bool:
    return bool(
        INSTAGRAM_PATTERN.search(url)
        or TIKTOK_PATTERN.search(url)
        or YOUTUBE_PATTERN.search(url)
    )


async def download_with_ytdlp(url: str) -> tuple[str | None, str | None]:
    """Download a public social media video via yt-dlp. Returns (file_path, error).

    Hardened: captures yt-dlp logs, retries once with backoff, sends desktop UA,
    optionally loads cookies from INSTAGRAM_COOKIES_FILE.
    """
    import yt_dlp

    tmp_dir = tempfile.mkdtemp(prefix="vexi_study_")
    output_template = os.path.join(tmp_dir, "video.%(ext)s")

    log_buffer: list[str] = []

    class _YtdlpLogger:
        def debug(self, msg): pass
        def info(self, msg): log_buffer.append(f"[info] {msg}")
        def warning(self, msg): log_buffer.append(f"[warn] {msg}")
        def error(self, msg): log_buffer.append(f"[err] {msg}")

    ydl_opts = {
        "outtmpl": output_template,
        "format": "best[ext=mp4][filesize<100M]/best[filesize<100M]/best",
        "max_filesize": 100 * 1024 * 1024,
        "logger": _YtdlpLogger(),
        "retries": 3,
        "extractor_retries": 2,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "sleep_interval": 1,
        "max_sleep_interval": 5,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        },
    }
    if INSTAGRAM_COOKIES_FILE and os.path.exists(INSTAGRAM_COOKIES_FILE):
        ydl_opts["cookiefile"] = INSTAGRAM_COOKIES_FILE
        log.info(f"yt-dlp using cookie file: {INSTAGRAM_COOKIES_FILE}")

    def _download():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        files = list(Path(tmp_dir).glob("video.*"))
        if files:
            return str(files[0]), None
        return None, "Download completed but output file not found."

    loop = asyncio.get_event_loop()
    last_err: str | None = None
    for attempt in range(2):
        try:
            path, err = await loop.run_in_executor(None, _download)
            if path:
                return path, None
            last_err = err
        except Exception as e:
            last_err = str(e)
            log.warning(f"yt-dlp attempt {attempt + 1}/2 failed: {e}")
            if log_buffer:
                log.warning("yt-dlp log tail: " + " | ".join(log_buffer[-5:]))
        for f in Path(tmp_dir).glob("video.*"):
            try:
                f.unlink()
            except Exception:
                pass
        if attempt == 0:
            await asyncio.sleep(8)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return None, last_err or "yt-dlp failed for unknown reason."


async def download_with_apify(url: str, session: aiohttp.ClientSession) -> tuple[str | None, str | None]:
    """Apify fallback for Instagram/TikTok. Returns (file_path, error).

    Uses run-sync-get-dataset-items so we don't have to poll. Picks an actor
    based on URL host. Downloads the resolved video URL to a tmp file.
    """
    if not APIFY_API_TOKEN:
        return None, "APIFY_API_TOKEN not configured."

    if INSTAGRAM_PATTERN.search(url):
        actor = APIFY_INSTAGRAM_ACTOR
        actor_input = {
            "directUrls": [url],
            "resultsType": "details",
            "resultsLimit": 1,
            "addParentData": False,
        }
    elif TIKTOK_PATTERN.search(url):
        actor = APIFY_TIKTOK_ACTOR
        actor_input = {
            "postURLs": [url],
            "resultsPerPage": 1,
            "shouldDownloadVideos": False,
        }
    else:
        return None, "No Apify actor configured for this URL type."

    actor_path = actor.replace("/", "~")
    api_url = f"https://api.apify.com/v2/acts/{actor_path}/run-sync-get-dataset-items?token={APIFY_API_TOKEN}"

    log.info(f"Apify fallback: actor={actor}, url={url[:80]}")
    try:
        async with session.post(
            api_url,
            json=actor_input,
            timeout=aiohttp.ClientTimeout(total=180, connect=30),
        ) as resp:
            if resp.status not in (200, 201):
                body = await resp.text()
                return None, f"Apify HTTP {resp.status}: {body[:200]}"
            items = await resp.json()
    except Exception as e:
        return None, f"Apify request failed: {e}"

    if not isinstance(items, list) or not items:
        return None, "Apify returned no items (video may be private/deleted)."

    item = items[0]
    video_url = (
        (item.get("mediaUrls") or [None])[0]
        or item.get("videoUrl")
        or item.get("video_url")
        or (item.get("videoMeta") or {}).get("downloadAddr")
    )
    if not video_url:
        return None, f"Apify item missing video URL. Available keys: {list(item.keys())[:10]}"

    log.info(f"Apify scrape OK, downloading video: {video_url[:80]}")

    # Instagram CDN returns 403 to bare datacenter requests. Mimic a real Safari
    # request initiated from instagram.com — User-Agent, Referer, Range, and the
    # Sec-Fetch-* trio together unblock most 403s when the signed URL is fresh.
    ig_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity;q=1, *;q=0",
        "Range": "bytes=0-",
        "Referer": "https://www.instagram.com/",
        "Origin": "https://www.instagram.com",
        "Sec-Fetch-Dest": "video",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
        "Connection": "keep-alive",
    }

    proxy = None
    proxy_auth = None
    if APIFY_USE_PROXY and APIFY_API_TOKEN:
        proxy = "http://proxy.apify.com:8000"
        proxy_auth = aiohttp.BasicAuth("groups-RESIDENTIAL", APIFY_API_TOKEN)
        log.info("Routing video download through Apify residential proxy")

    tmp_dir = tempfile.mkdtemp(prefix="vexi_apify_")
    tmp_path = os.path.join(tmp_dir, "video.mp4")
    try:
        async with session.get(
            video_url,
            timeout=aiohttp.ClientTimeout(total=120, connect=30),
            headers=ig_headers,
            proxy=proxy,
            proxy_auth=proxy_auth,
            allow_redirects=True,
        ) as r:
            # Range header makes IG return 206 Partial Content — accept both.
            if r.status not in (200, 206):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                proxy_note = " (via Apify proxy)" if proxy else " (direct)"
                return None, f"Apify scrape OK but CDN download blocked: HTTP {r.status}{proxy_note}"
            total = 0
            with open(tmp_path, "wb") as fh:
                async for chunk in r.content.iter_chunked(256 * 1024):
                    fh.write(chunk)
                    total += len(chunk)
                    if total > 100 * 1024 * 1024:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        return None, "Apify video exceeds 100MB."
        log.info(f"Apify download complete: {total / 1024 / 1024:.1f}MB")
        return tmp_path, None
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, f"Apify video download error: {e}"


def _parse_json_with_repair(text: str) -> dict | None:
    """Try to parse JSON. If it fails, attempt to extract the largest {...} substring."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def convert_gdrive_to_direct(url: str) -> str:
    """Convert a Google Drive sharing URL to a direct download URL."""
    m = GDRIVE_PATTERN.search(url)
    if not m:
        m = GDRIVE_OPEN_PATTERN.search(url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    return url


def extract_video_source(message: discord.Message) -> str | None:
    """Return a video URL from the message (attachment or link)."""
    for att in message.attachments:
        suffix = Path(att.filename).suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return att.url

    text = message.content or ""
    m = GDRIVE_PATTERN.search(text)
    if not m:
        m = GDRIVE_OPEN_PATTERN.search(text)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"

    m = YOUTUBE_PATTERN.search(text)
    if m:
        return text.strip()

    for word in text.split():
        if any(word.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
            return word

    return None


def guess_mime_type(url: str, filename: str = "") -> str:
    """Guess video MIME type from URL or filename."""
    check = (filename or url).lower()
    if ".mov" in check:
        return "video/quicktime"
    elif ".webm" in check:
        return "video/webm"
    elif ".mkv" in check:
        return "video/x-matroska"
    elif ".avi" in check:
        return "video/x-msvideo"
    elif ".m4v" in check:
        return "video/x-m4v"
    return "video/mp4"


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(code in msg for code in ("503", "429", "unavailable", "resource_exhausted", "resourceexhausted", "too many requests"))


async def _gemini_generate(contents: list, retries: int = 3, config=None) -> object:
    delays = [5, 15, 30]
    last_exc = None
    for attempt in range(retries):
        try:
            kwargs = {"model": "gemini-2.5-flash", "contents": contents}
            if config is not None:
                kwargs["config"] = config
            return gemini_client.models.generate_content(**kwargs)
        except Exception as e:
            last_exc = e
            if _is_retryable(e) and attempt < retries - 1:
                wait = delays[attempt]
                log.warning(f"Gemini transient error (attempt {attempt + 1}/{retries}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
    raise last_exc


async def analyze_video_with_gemini(video_url: str, mime_type: str = "video/mp4", prompt: str = None, response_json: bool = False) -> dict:
    """Send video URL directly to Gemini for review (no local download needed).

    If response_json=True, asks Gemini to enforce JSON output via response_mime_type.
    """
    if prompt is None:
        prompt = REVIEW_PROMPT
    raw_text = ""
    response = None
    try:
        log.info(f"Sending video URL to Gemini: {video_url[:120]}...")
        log.info(f"MIME type: {mime_type}")

        video_part = genai_types.Part.from_uri(
            file_uri=video_url,
            mime_type=mime_type,
        )

        config = None
        if response_json:
            config = genai_types.GenerateContentConfig(response_mime_type="application/json")

        log.info("Calling Gemini 2.5 Flash for review...")
        response = await _gemini_generate([video_part, prompt], config=config)

        raw_text = response.text.strip()
        log.info(f"Gemini response length: {len(raw_text)} chars")
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        result = _parse_json_with_repair(raw_text)
        if result is None:
            try:
                fr = response.candidates[0].finish_reason
            except Exception:
                fr = "unknown"
            log.error(f"JSON parse failed (finish_reason={fr}). Raw: {raw_text[:500]}")
            return {"error": f"AI returned invalid JSON (finish_reason={fr}). Raw excerpt: {raw_text[:300]}"}
        log.info(f"Review complete. Verdict: {result.get('quick_verdict', 'N/A')}")
        return result

    except Exception as e:
        log.error(f"Gemini error: {type(e).__name__}: {e}")
        import traceback
        log.error(traceback.format_exc())
        return {"error": str(e)}


async def _analyze_local_file_with_gemini(file_path: str, prompt: str = None, response_json: bool = False) -> dict:
    """Upload a local file to Gemini File API and analyze it.

    If response_json=True, asks Gemini to enforce JSON output via response_mime_type.
    On JSON parse failure, includes a raw excerpt and finish_reason in the error.
    """
    if prompt is None:
        prompt = REVIEW_PROMPT
    raw_text = ""
    response = None
    uploaded_file = None
    try:
        mime = guess_mime_type(file_path)
        log.info(f"Uploading local file to Gemini File API: {file_path} ({mime})")
        uploaded_file = gemini_client.files.upload(file=file_path)
        log.info(f"Upload complete: {uploaded_file.name}, state={uploaded_file.state}")

        max_wait = 120
        waited = 0
        while uploaded_file.state.name == "PROCESSING" and waited < max_wait:
            await asyncio.sleep(5)
            waited += 5
            uploaded_file = gemini_client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            return {"error": f"File processing failed. State: {uploaded_file.state.name}"}

        config = None
        if response_json:
            config = genai_types.GenerateContentConfig(response_mime_type="application/json")

        response = await _gemini_generate([uploaded_file, prompt], config=config)
        raw_text = response.text.strip()
        log.info(f"Gemini response length: {len(raw_text)} chars")
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        result = _parse_json_with_repair(raw_text)

        try:
            gemini_client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        if result is None:
            try:
                fr = response.candidates[0].finish_reason
            except Exception:
                fr = "unknown"
            log.error(f"JSON parse failed (finish_reason={fr}). Raw: {raw_text[:500]}")
            return {"error": f"AI returned invalid JSON (finish_reason={fr}). Raw excerpt: {raw_text[:300]}"}
        return result

    except Exception as e:
        log.error(f"Local file Gemini upload error: {type(e).__name__}: {e}")
        return {"error": str(e)}


async def analyze_video_with_gemini_upload(video_url: str, session: aiohttp.ClientSession, prompt: str = None, response_json: bool = False) -> dict:
    """Fallback: Download video and upload to Gemini File API."""
    if prompt is None:
        prompt = REVIEW_PROMPT
    raw_text = ""
    video_path = None
    try:
        log.info(f"Fallback: Downloading video to upload to Gemini: {video_url[:120]}...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with session.get(
            video_url,
            timeout=aiohttp.ClientTimeout(total=300, connect=30),
            headers=headers,
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                return {"error": f"Download failed: HTTP {resp.status}"}
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                return {"error": "Got HTML instead of video. Check the URL."}

            ext = ".mp4"
            if "quicktime" in content_type or ".mov" in video_url.lower():
                ext = ".mov"
            elif "webm" in content_type:
                ext = ".webm"

            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            total = 0
            async for chunk in resp.content.iter_chunked(1024 * 256):
                tmp.write(chunk)
                total += len(chunk)
                if total > 100 * 1024 * 1024:
                    tmp.close()
                    os.unlink(tmp.name)
                    return {"error": "Video exceeds 100MB limit."}
            tmp.close()
            video_path = tmp.name
            log.info(f"Downloaded {total / 1024 / 1024:.1f}MB to {video_path}")

        return await _analyze_local_file_with_gemini(video_path, prompt, response_json=response_json)

    except Exception as e:
        log.error(f"Gemini upload fallback error: {type(e).__name__}: {e}")
        return {"error": str(e)}
    finally:
        if video_path:
            try:
                os.unlink(video_path)
            except Exception:
                pass


def build_review_message(review: dict, creator: str = None) -> tuple[str | None, list[discord.Embed]]:
    """Convert the Gemini JSON review into a single compact Discord embed.
    
    Returns (content_text, embeds_list).
    """
    # --- Error ---
    if "error" in review:
        err_embed = discord.Embed(
            title="Vexi — Review Error",
            description=f"Something went wrong during the review:\n```{review['error'][:500]}```\nPlease try again or ask a coach for help.",
            color=discord.Color.red(),
        )
        return (None, [err_embed])

    # --- NOT MANUS CONTENT ---
    if not review.get("manus_relevant", True):
        gate_embed = discord.Embed(
            title="🚫 Vexi — NOT MANUS CONTENT",
            description=(
                "Hey! I took a look at this video, but it **doesn't appear to be related to Manus**.\n\n"
                "Please submit a Manus-related UGC video for review. "
                "If you think this IS a Manus video, ask a coach to review it manually.\n\n"
                "*Vexi is just an AI flagger — if I got this wrong, a coach can override.*"
            ),
            color=discord.Color.dark_grey(),
        )
        gate_embed.set_footer(text="Vexi • Derived from Latin 'vexillum' (flag) • v1.2")
        return (None, [gate_embed])

    # --- Build single compact embed ---
    verdict = review.get("quick_verdict", "NEEDS REVIEW")
    verdict_color = {
        "LOOKS GOOD": discord.Color.green(),
        "NEEDS REVIEW": discord.Color.gold(),
        "COACH ATTENTION NEEDED": discord.Color.orange(),
        "AUTO-REJECT": discord.Color.red(),
    }.get(verdict, discord.Color.gold())

    # Assemble the description as one clean message
    parts = []

    # Creator + Language + Script Summary
    if creator:
        parts.append(f"**Creator:** {creator}")
    lang = review.get("language_detected", "")
    if lang:
        parts.append(f"🌐 **Language:** {lang}")
    parts.append("🎬 **Video:** See the attached video above ⬆️")

    script = review.get("script_summary", "")
    if script:
        parts.append(f"\n📜 **Script Summary:** {script}")

    # AUTO-REJECT banner
    if verdict == "AUTO-REJECT":
        parts.append(
            "\n🚨 **AUTO-REJECT — COACH ESCALATION REQUIRED**"
            "\n⛔ *This video contains banned language per the Manus UGC Guidelines v1.2. Do NOT approve or publish until a coach reviews and clears it.*"
        )

    # Intro + Disclaimer
    parts.append(
        "\n👋 Hey! I'm **Vexi**, your AI review buddy."
        "\n⚠️ *I'm just an AI flagger — a real coach will make the final call.*"
    )

    # Layer 1 — Legal paragraph
    legal = review.get("legal_paragraph", "")
    if legal:
        parts.append(f"\n🛡️ **Legal Check:**\n{legal}")

    # Layer 2 — Content paragraph
    content = review.get("content_paragraph", "")
    if content:
        parts.append(f"\n🎬 **Content Review:**\n{content}")

    # Overall summary
    summary = review.get("overall_summary", "A human coach will review this shortly for final approval.")
    parts.append(f"\n📝 **Summary:** {summary}")

    # Risk grid — 9×9 colored squares, always at the very bottom
    _risk_grid = {
        "LOOKS GOOD":             ("🟢", "LOW RISK"),
        "NEEDS REVIEW":           ("🟡", "MEDIUM RISK"),
        "COACH ATTENTION NEEDED": ("🟠", "HIGH RISK"),
        "AUTO-REJECT":            ("🔴", "CRITICAL — AUTO-REJECT"),
    }
    grid_emoji, grid_label = _risk_grid.get(verdict, ("🟡", "MEDIUM RISK"))
    grid_row = grid_emoji * 9
    grid_block = "\n".join([grid_row] * 9)
    parts.append(f"\n**{grid_label}**\n{grid_block}")

    embed = discord.Embed(
        title=f"Vexi Review — {verdict}",
        description="\n".join(parts),
        color=verdict_color,
    )
    embed.set_footer(text="Vexi • Derived from Latin 'vexillum' (flag) • v1.2")

    return (None, [embed])


def build_study_message(study: dict, source_label: str = "Video") -> tuple[str | None, list[discord.Embed]]:
    """Convert a Gemini Study Mode JSON result into a Discord embed."""
    if "error" in study:
        err_msg = study["error"]
        is_download_fail = err_msg.startswith("DOWNLOAD_FAILED:")
        if is_download_fail:
            err_msg = err_msg[len("DOWNLOAD_FAILED:"):].strip()

        desc = f"Something went wrong:\n```{err_msg[:500]}```\n"
        if is_download_fail:
            desc += (
                f"\n💡 **Both yt-dlp and Apify couldn't fetch this video.** "
                f"Instagram or TikTok may be blocking automated access right now.\n\n"
                f"**Try this instead:** Paste the video at **{LUMISCRIPT_URL}** — "
                f"it's another Manus platform that reviews scripts and can analyze the video for you.\n\n"
                f"Or download the video manually and re-run `/study video:` with the file attached."
            )
        else:
            desc += "Make sure the video is public and in a supported format."

        err_embed = discord.Embed(
            title="Vexi Study — Error",
            description=desc,
            color=discord.Color.red(),
        )
        return (None, [err_embed])

    difficulty = study.get("adaptation_difficulty", "MODERATE")
    diff_color = {
        "EASY": discord.Color.green(),
        "MODERATE": discord.Color.gold(),
        "COMPLEX": discord.Color.orange(),
    }.get(difficulty, discord.Color.gold())

    # --- Embed 1: Format Analysis ---
    p1 = []
    p1.append(f"🎯 **Adaptation Difficulty:** {difficulty}")

    source_ctx = study.get("source_context", "")
    if source_ctx:
        p1.append(f"\n📌 **Source Context:** {source_ctx}")

    fmt = study.get("format_breakdown", "")
    if fmt:
        p1.append(f"\n🎬 **Format Breakdown:** {fmt}")

    works = study.get("what_makes_it_work", "")
    if works:
        p1.append(f"\n✨ **Why It Works:** {works}")

    embed1 = discord.Embed(
        title=f"Vexi Study — {source_label}",
        description="\n".join(p1),
        color=diff_color,
    )
    embed1.set_footer(text="Vexi Study Mode • 1 of 2 — Format Analysis")

    # --- Embed 2: Manus Adaptation Brief ---
    p2 = []

    adaptation = study.get("manus_adaptation", "")
    if adaptation:
        p2.append(f"🔄 **Manus Adaptation:** {adaptation}")

    outline = study.get("suggested_outline", "")
    if outline:
        p2.append(f"\n📋 **Suggested Outline:**\n{outline}")

    guardrails = study.get("copy_guardrails", "")
    if guardrails:
        p2.append(f"\n⚠️ **Copy Guardrails:** {guardrails}")

    embed2 = discord.Embed(
        title="Vexi Study — Manus Adaptation Brief",
        description="\n".join(p2),
        color=diff_color,
    )
    embed2.set_footer(text="Vexi Study Mode • 2 of 2 — Manus Adaptation Brief • v1.2")

    return (None, [embed1, embed2])


# ---------------------------------------------------------------------------
# Discord Bot Setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Track whether we've already synced commands (avoid re-syncing on reconnect)
_commands_synced = False


@bot.event
async def on_ready():
    global _commands_synced
    log.info(f"Vexi is online as {bot.user} (ID: {bot.user.id})")
    log.info(f"Auto-detect channels: {VEXI_CHANNEL_IDS}")

    if _commands_synced:
        log.info("Reconnected — skipping command sync (already synced).")
        return

    try:
        for g in bot.guilds:
            bot.tree.copy_global_to(guild=g)
            synced = await bot.tree.sync(guild=g)
            log.info(f"Synced {len(synced)} slash command(s) to guild {g.name} ({g.id})")

        bot.tree.clear_commands(guild=None)
        await bot.tree.sync()
        log.info("Cleared global commands to prevent duplicates")
        _commands_synced = True
    except Exception as e:
        log.error(f"Failed to sync commands: {e}")


# ---------------------------------------------------------------------------
# Slash Command: /vexi
# ---------------------------------------------------------------------------
@bot.tree.command(name="vexi", description="Submit a video for Vexi AI review")
@app_commands.describe(
    video="Attach a video file directly",
    video_url="Or paste a direct video URL (Google Drive, YouTube, or direct link)",
    coach="Tag a coach to notify them about this review",
)
async def vexi_command(
    interaction: discord.Interaction,
    video: discord.Attachment = None,
    video_url: str = None,
    coach: discord.Member = None,
):
    """Handle /vexi slash command."""
    # DEFER IMMEDIATELY
    try:
        await interaction.response.defer(thinking=True)
    except (discord.errors.NotFound, discord.errors.HTTPException) as e:
        log.warning(f"Interaction defer failed: {e}")
        return

    # Validate input
    source_url = None
    original_url = None

    if video is not None:
        suffix = Path(video.filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            await interaction.followup.send(
                f"❌ That doesn't look like a video file (`{video.filename}`).\n"
                f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}",
            )
            return
        source_url = video.url
        original_url = video.url
    elif video_url:
        original_url = video_url
        source_url = convert_gdrive_to_direct(video_url)
    else:
        await interaction.followup.send(
            "👋 Hey! Please attach a video file or provide a video URL.\n\n"
            "**Option 1:** `/vexi video:` → click to attach a file\n"
            "**Option 2:** `/vexi video_url:https://drive.google.com/file/d/abc123/view`\n"
            "**Option 3:** `/vexi video: coach:@CoachName` → notify a coach",
        )
        return

    # Post the video visibly in the channel
    coach_ping = ""
    if coach:
        coach_ping = f"🏷️ {coach.mention} — "
    intro_text = f"{coach_ping}🔍 **Vexi is reviewing a video from {interaction.user.mention}...** Hang tight!"

    if video is not None:
        try:
            video_file = await video.to_file()
            visible_msg = await interaction.followup.send(
                content=intro_text,
                file=video_file,
            )
        except Exception as e:
            log.warning(f"Could not re-attach video: {e}")
            visible_msg = await interaction.followup.send(content=intro_text)
    elif video_url:
        visible_msg = await interaction.followup.send(
            content=f"{intro_text}\n🎬 Video: {original_url}",
        )
    else:
        visible_msg = await interaction.followup.send(content=intro_text)

    # Progress indicator
    progress_msg = await visible_msg.reply(
        content="⏳ **Vexi is analyzing your video...**\n▁▁▁▁▁▁▁▁▁▁ Sending to AI..."
    )

    progress_done = asyncio.Event()

    async def update_progress():
        stages = [
            ("▓▓▁▁▁▁▁▁▁▁", "🧠 AI is watching your video..."),
            ("▓▓▓▓▁▁▁▁▁▁", "🧠 AI is watching your video..."),
            ("▓▓▓▓▓▁▁▁▁▁", "🔍 Checking legal compliance..."),
            ("▓▓▓▓▓▓▁▁▁▁", "🔍 Checking legal compliance..."),
            ("▓▓▓▓▓▓▓▁▁▁", "🎬 Reviewing hooks & storytelling..."),
            ("▓▓▓▓▓▓▓▓▁▁", "🎬 Reviewing hooks & storytelling..."),
            ("▓▓▓▓▓▓▓▓▓▁", "📝 Writing up the review..."),
            ("▓▓▓▓▓▓▓▓▓▓", "✅ Almost done..."),
        ]
        for bar, status in stages:
            if progress_done.is_set():
                return
            try:
                await progress_msg.edit(
                    content=f"⏳ **Vexi is analyzing your video...**\n{bar} {status}"
                )
            except Exception:
                return
            try:
                await asyncio.wait_for(progress_done.wait(), timeout=15)
                return
            except asyncio.TimeoutError:
                continue

    progress_task = asyncio.create_task(update_progress())

    # Analyze
    filename = video.filename if video else ""
    mime = guess_mime_type(source_url, filename)

    try:
        # Discord CDN URLs are signed/temporary — Gemini can't fetch them directly.
        # Skip straight to the upload fallback for Discord attachments.
        is_discord_cdn = bool(DISCORD_CDN_PATTERN.match(source_url))

        if is_discord_cdn:
            log.info("Discord CDN URL detected — using upload fallback directly.")
            async with aiohttp.ClientSession() as session:
                review = await analyze_video_with_gemini_upload(source_url, session)
        else:
            review = await analyze_video_with_gemini(source_url, mime_type=mime)
            if "error" in review:
                log.info(f"Direct URL failed ({review['error']}), trying upload fallback...")
                async with aiohttp.ClientSession() as session:
                    review = await analyze_video_with_gemini_upload(source_url, session)

        # Stop progress
        progress_done.set()
        await progress_task

        # Build compact review
        content_text, embeds = build_review_message(review, creator=f"{interaction.user.mention}")

        await progress_msg.edit(content=content_text, embeds=embeds)
    except Exception as e:
        progress_done.set()
        await progress_task
        log.error(f"Slash command error: {type(e).__name__}: {e}")
        await progress_msg.edit(
            content=f"❌ Something went wrong during the review. Please try again.\nError: {str(e)[:200]}"
        )


# ---------------------------------------------------------------------------
# Slash Command: /study
# ---------------------------------------------------------------------------
@bot.tree.command(name="study", description="Study any creator's video and get a Manus UGC adaptation brief")
@app_commands.describe(
    video="Attach a downloaded video file",
    video_url="Or paste an Instagram, TikTok, YouTube, or direct video URL",
    niche="Optional: describe the niche or brand for extra context (e.g. 'productivity SaaS', 'fitness app')",
)
async def study_command(
    interaction: discord.Interaction,
    video: discord.Attachment = None,
    video_url: str = None,
    niche: str = None,
):
    """Handle /study slash command — format analysis + Manus adaptation brief."""
    try:
        await interaction.response.defer(thinking=True)
    except (discord.errors.NotFound, discord.errors.HTTPException) as e:
        log.warning(f"Study interaction defer failed: {e}")
        return

    if video is None and not video_url:
        await interaction.followup.send(
            "👋 Give me a video to study!\n\n"
            "**Option 1:** `/study video:` → attach a downloaded video\n"
            "**Option 2:** `/study video_url:https://www.tiktok.com/@creator/video/123`\n"
            "**Option 3:** Add `niche:` for context — e.g. `niche:productivity SaaS`\n\n"
            "Supports Instagram Reels, TikTok, YouTube, Google Drive, and direct links."
        )
        return

    source_label = "Video"
    source_url = None
    tmp_path = None
    tmp_dir = None
    downloaded_video_path_for_discord = None

    intro_text = f"🔍 **Vexi is studying a video for {interaction.user.mention}...** Hang tight!"
    if niche:
        intro_text += f"\n📌 Niche context: *{niche}*"

    if video is not None:
        suffix = Path(video.filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            await interaction.followup.send(
                f"❌ That doesn't look like a video file (`{video.filename}`).\n"
                f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}"
            )
            return
        source_url = video.url
        source_label = f"Uploaded: {video.filename}"
        try:
            video_file = await video.to_file()
            visible_msg = await interaction.followup.send(content=intro_text, file=video_file)
        except Exception as e:
            log.warning(f"Could not re-attach video: {e}")
            visible_msg = await interaction.followup.send(content=intro_text)
    else:
        source_url = video_url
        if INSTAGRAM_PATTERN.search(video_url):
            source_label = "Instagram"
        elif TIKTOK_PATTERN.search(video_url):
            source_label = "TikTok"
        elif YOUTUBE_PATTERN.search(video_url):
            source_label = "YouTube"
        elif GDRIVE_PATTERN.search(video_url) or GDRIVE_OPEN_PATTERN.search(video_url):
            source_label = "Google Drive"
        else:
            source_label = "Direct Link"
        visible_msg = await interaction.followup.send(
            content=f"{intro_text}\n🎬 Video: {source_url}"
        )

    progress_msg = await visible_msg.reply(
        content="⏳ **Vexi is studying the format...**\n▁▁▁▁▁▁▁▁▁▁ Loading video..."
    )

    progress_done = asyncio.Event()

    async def update_progress():
        stages = [
            ("▓▓▁▁▁▁▁▁▁▁", "🎬 Watching the video..."),
            ("▓▓▓▓▁▁▁▁▁▁", "🎬 Watching the video..."),
            ("▓▓▓▓▓▁▁▁▁▁", "🔍 Breaking down the format..."),
            ("▓▓▓▓▓▓▁▁▁▁", "🔍 Analyzing hook & structure..."),
            ("▓▓▓▓▓▓▓▁▁▁", "🔄 Building Manus adaptation brief..."),
            ("▓▓▓▓▓▓▓▓▁▁", "🔄 Building Manus adaptation brief..."),
            ("▓▓▓▓▓▓▓▓▓▁", "📋 Writing the outline..."),
            ("▓▓▓▓▓▓▓▓▓▓", "✅ Almost done..."),
        ]
        for bar, status in stages:
            if progress_done.is_set():
                return
            try:
                await progress_msg.edit(
                    content=f"⏳ **Vexi is studying the format...**\n{bar} {status}"
                )
            except Exception:
                return
            try:
                await asyncio.wait_for(progress_done.wait(), timeout=15)
                return
            except asyncio.TimeoutError:
                continue

    progress_task = asyncio.create_task(update_progress())

    active_prompt = STUDY_PROMPT
    if niche:
        active_prompt = STUDY_PROMPT + f"\n\nNICHE CONTEXT PROVIDED BY SUBMITTER: {niche}\nTailor your Manus adaptation and suggested outline to this niche specifically.\n"

    try:
        is_discord_cdn = bool(DISCORD_CDN_PATTERN.match(source_url))
        is_social = (video is None) and _is_social_media_url(source_url)
        is_gdrive = bool(GDRIVE_PATTERN.search(source_url) or GDRIVE_OPEN_PATTERN.search(source_url))

        if is_social:
            log.info(f"Social media URL — downloading with yt-dlp: {source_url[:80]}")
            await progress_msg.edit(
                content="⏳ **Vexi is studying the format...**\n▓▁▁▁▁▁▁▁▁▁ 📥 Downloading from social media..."
            )
            tmp_path, dl_error = await download_with_ytdlp(source_url)

            # Fallback: Apify (Instagram/TikTok only) when yt-dlp fails
            if (not tmp_path) and (INSTAGRAM_PATTERN.search(source_url) or TIKTOK_PATTERN.search(source_url)):
                if APIFY_API_TOKEN:
                    log.warning(f"yt-dlp failed ({dl_error}). Trying Apify fallback...")
                    try:
                        await progress_msg.edit(
                            content="⏳ **Vexi is studying the format...**\n▓▓▁▁▁▁▁▁▁▁ 🔄 yt-dlp blocked, trying Apify..."
                        )
                    except Exception:
                        pass
                    async with aiohttp.ClientSession() as sess:
                        tmp_path, apify_err = await download_with_apify(source_url, sess)
                    if not tmp_path:
                        dl_error = f"yt-dlp: {dl_error} | apify: {apify_err}"
                else:
                    log.warning("yt-dlp failed and APIFY_API_TOKEN not set — skipping Apify fallback.")

            if not tmp_path:
                study = {"error": f"DOWNLOAD_FAILED: {dl_error or 'Unknown error'}"}
            else:
                tmp_dir = os.path.dirname(tmp_path)
                log.info(f"Download succeeded: {tmp_path}")
                downloaded_video_path_for_discord = tmp_path
                study = await _analyze_local_file_with_gemini(tmp_path, prompt=active_prompt, response_json=True)
        elif is_discord_cdn:
            log.info("Discord CDN URL — using upload fallback.")
            async with aiohttp.ClientSession() as sess:
                study = await analyze_video_with_gemini_upload(source_url, sess, prompt=active_prompt, response_json=True)
        elif is_gdrive:
            direct_url = convert_gdrive_to_direct(source_url)
            study = await analyze_video_with_gemini(direct_url, prompt=active_prompt, response_json=True)
            if "error" in study:
                log.info(f"Direct GDrive failed, trying upload fallback...")
                async with aiohttp.ClientSession() as sess:
                    study = await analyze_video_with_gemini_upload(direct_url, sess, prompt=active_prompt, response_json=True)
        else:
            mime = guess_mime_type(source_url, video.filename if video else "")
            study = await analyze_video_with_gemini(source_url, mime_type=mime, prompt=active_prompt, response_json=True)
            if "error" in study:
                log.info(f"Direct URL failed, trying upload fallback...")
                async with aiohttp.ClientSession() as sess:
                    study = await analyze_video_with_gemini_upload(source_url, sess, prompt=active_prompt, response_json=True)

        progress_done.set()
        await progress_task

        _, embeds = build_study_message(study, source_label=source_label)

        # Delete the progress bar so only two messages remain: intro + results
        try:
            await progress_msg.delete()
        except Exception:
            pass

        # Build the single reply: video (if downloaded) + both embeds
        send_kwargs: dict = {"embeds": embeds}
        if downloaded_video_path_for_discord and os.path.exists(downloaded_video_path_for_discord):
            ext = Path(downloaded_video_path_for_discord).suffix.lower()
            friendly_name = f"{source_label.lower().replace(' ', '_')}{ext}"
            send_kwargs["file"] = discord.File(downloaded_video_path_for_discord, filename=friendly_name)

        try:
            await visible_msg.reply(**send_kwargs)
        except discord.HTTPException as e:
            if e.status == 413 and "file" in send_kwargs:
                del send_kwargs["file"]
                await visible_msg.reply(**send_kwargs)
                await visible_msg.reply(content=f"📹 Video too large to attach — original: {source_url}")
            else:
                raise

    except Exception as e:
        progress_done.set()
        await progress_task
        log.error(f"Study command error: {type(e).__name__}: {e}")
        await progress_msg.edit(
            content=f"❌ Something went wrong. Please try again.\nError: {str(e)[:200]}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Mention Handler: @Vexi study <url>
# ---------------------------------------------------------------------------
async def handle_mention_study(message: discord.Message, content_after_mention: str):
    """Handle @Vexi study <url> [niche:...] mentions from any channel."""
    text = content_after_mention.strip()

    # No keyword at all — generic help
    if not text:
        await message.reply(
            "👋 Hey! I'm **Vexi**, your UGC review buddy.\n\n"
            "To study a video format, use:\n"
            "`@Vexi study <Instagram/TikTok/YouTube link>`\n\n"
            "**Example:** `@Vexi study https://www.instagram.com/reel/abc123`\n\n"
            "You can also add niche context:\n"
            "`@Vexi study https://www.tiktok.com/... niche:productivity SaaS`\n\n"
            "Not sure what to do? Ask your coach for help! 🙌"
        )
        return

    # Wrong keyword (mentioned bot but typed something other than "study")
    if not text.lower().startswith("study"):
        await message.reply(
            "🤔 Hmm, I didn't catch that!\n\n"
            "To study a video, type:\n"
            "`@Vexi study <Instagram/TikTok/YouTube link>`\n\n"
            "**Example:** `@Vexi study https://www.instagram.com/reel/abc123`\n\n"
            "If you need help, reach out to your coach! 🙌"
        )
        return

    # Strip "study" and parse the rest
    rest = text[len("study"):].strip()

    # Parse optional niche: param (e.g. "niche:productivity SaaS")
    niche = None
    niche_match = re.search(r'\bniche:(.+?)(?=\s+https?://|\s*$)', rest, re.IGNORECASE)
    if niche_match:
        niche = niche_match.group(1).strip()

    # Extract URL
    url_match = re.search(r'https?://\S+', rest)
    if not url_match:
        await message.reply(
            "❌ I need a link to study!\n\n"
            "**Format:** `@Vexi study <link>`\n"
            "**Example:** `@Vexi study https://www.instagram.com/reel/abc123`\n\n"
            "Supports Instagram Reels, TikTok, YouTube, Google Drive, and direct video links.\n"
            "If you're stuck, ask your coach for help! 🙌"
        )
        return

    source_url = url_match.group(0).rstrip(".,;)")  # strip trailing punctuation

    # Validate it's a URL type we can handle
    is_social = _is_social_media_url(source_url)
    is_gdrive = bool(GDRIVE_PATTERN.search(source_url) or GDRIVE_OPEN_PATTERN.search(source_url))
    has_video_ext = any(source_url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

    if not (is_social or is_gdrive or has_video_ext):
        await message.reply(
            "❌ That doesn't look like a supported video link.\n\n"
            "I can study links from:\n"
            "• **Instagram** — `instagram.com/reel/...`\n"
            "• **TikTok** — `tiktok.com/...`\n"
            "• **YouTube** — `youtube.com/watch?...` or `youtu.be/...`\n"
            "• **Google Drive** — `drive.google.com/file/d/...`\n"
            "• **Direct video links** (.mp4, .mov, etc.)\n\n"
            "If you're unsure, ask your coach for help! 🙌"
        )
        return

    # Determine source label
    if INSTAGRAM_PATTERN.search(source_url):
        source_label = "Instagram"
    elif TIKTOK_PATTERN.search(source_url):
        source_label = "TikTok"
    elif YOUTUBE_PATTERN.search(source_url):
        source_label = "YouTube"
    elif is_gdrive:
        source_label = "Google Drive"
    else:
        source_label = "Direct Link"

    # Kick off — react and show intro
    try:
        await message.add_reaction("🔍")
    except Exception:
        pass

    intro_text = f"🔍 **Vexi is studying a video for {message.author.mention}...** Hang tight!"
    if niche:
        intro_text += f"\n📌 Niche context: *{niche}*"
    intro_text += f"\n🎬 Video: {source_url}"
    visible_msg = await message.reply(content=intro_text)

    progress_msg = await visible_msg.reply(
        content="⏳ **Vexi is studying the format...**\n▁▁▁▁▁▁▁▁▁▁ Loading video..."
    )

    progress_done = asyncio.Event()

    async def update_progress():
        stages = [
            ("▓▓▁▁▁▁▁▁▁▁", "🎬 Watching the video..."),
            ("▓▓▓▓▁▁▁▁▁▁", "🎬 Watching the video..."),
            ("▓▓▓▓▓▁▁▁▁▁", "🔍 Breaking down the format..."),
            ("▓▓▓▓▓▓▁▁▁▁", "🔍 Analyzing hook & structure..."),
            ("▓▓▓▓▓▓▓▁▁▁", "🔄 Building Manus adaptation brief..."),
            ("▓▓▓▓▓▓▓▓▁▁", "🔄 Building Manus adaptation brief..."),
            ("▓▓▓▓▓▓▓▓▓▁", "📋 Writing the outline..."),
            ("▓▓▓▓▓▓▓▓▓▓", "✅ Almost done..."),
        ]
        for bar, status in stages:
            if progress_done.is_set():
                return
            try:
                await progress_msg.edit(content=f"⏳ **Vexi is studying the format...**\n{bar} {status}")
            except Exception:
                return
            try:
                await asyncio.wait_for(progress_done.wait(), timeout=15)
                return
            except asyncio.TimeoutError:
                continue

    progress_task = asyncio.create_task(update_progress())

    active_prompt = STUDY_PROMPT
    if niche:
        active_prompt = STUDY_PROMPT + f"\n\nNICHE CONTEXT PROVIDED BY SUBMITTER: {niche}\nTailor your Manus adaptation and suggested outline to this niche specifically.\n"

    tmp_path = None
    tmp_dir = None
    downloaded_video_path_for_discord = None

    try:
        if is_social:
            log.info(f"[mention] Social URL — yt-dlp: {source_url[:80]}")
            await progress_msg.edit(
                content="⏳ **Vexi is studying the format...**\n▓▁▁▁▁▁▁▁▁▁ 📥 Downloading from social media..."
            )
            tmp_path, dl_error = await download_with_ytdlp(source_url)

            if (not tmp_path) and (INSTAGRAM_PATTERN.search(source_url) or TIKTOK_PATTERN.search(source_url)):
                if APIFY_API_TOKEN:
                    log.warning(f"[mention] yt-dlp failed ({dl_error}). Trying Apify...")
                    try:
                        await progress_msg.edit(
                            content="⏳ **Vexi is studying the format...**\n▓▓▁▁▁▁▁▁▁▁ 🔄 yt-dlp blocked, trying Apify..."
                        )
                    except Exception:
                        pass
                    async with aiohttp.ClientSession() as sess:
                        tmp_path, apify_err = await download_with_apify(source_url, sess)
                    if not tmp_path:
                        dl_error = f"yt-dlp: {dl_error} | apify: {apify_err}"

            if not tmp_path:
                study = {"error": f"DOWNLOAD_FAILED: {dl_error or 'Unknown error'}"}
            else:
                tmp_dir = os.path.dirname(tmp_path)
                downloaded_video_path_for_discord = tmp_path
                study = await _analyze_local_file_with_gemini(tmp_path, prompt=active_prompt, response_json=True)
        elif is_gdrive:
            direct_url = convert_gdrive_to_direct(source_url)
            study = await analyze_video_with_gemini(direct_url, prompt=active_prompt, response_json=True)
            if "error" in study:
                async with aiohttp.ClientSession() as sess:
                    study = await analyze_video_with_gemini_upload(direct_url, sess, prompt=active_prompt, response_json=True)
        else:
            mime = guess_mime_type(source_url)
            study = await analyze_video_with_gemini(source_url, mime_type=mime, prompt=active_prompt, response_json=True)
            if "error" in study:
                async with aiohttp.ClientSession() as sess:
                    study = await analyze_video_with_gemini_upload(source_url, sess, prompt=active_prompt, response_json=True)

        progress_done.set()
        await progress_task

        _, embeds = build_study_message(study, source_label=source_label)

        try:
            await progress_msg.delete()
        except Exception:
            pass

        send_kwargs: dict = {"embeds": embeds}
        if downloaded_video_path_for_discord and os.path.exists(downloaded_video_path_for_discord):
            ext = Path(downloaded_video_path_for_discord).suffix.lower()
            friendly_name = f"{source_label.lower().replace(' ', '_')}{ext}"
            send_kwargs["file"] = discord.File(downloaded_video_path_for_discord, filename=friendly_name)

        try:
            await visible_msg.reply(**send_kwargs)
        except discord.HTTPException as e:
            if e.status == 413 and "file" in send_kwargs:
                del send_kwargs["file"]
                await visible_msg.reply(**send_kwargs)
                await visible_msg.reply(content=f"📹 Video too large to attach — original: {source_url}")
            else:
                raise

    except Exception as e:
        progress_done.set()
        await progress_task
        log.error(f"[mention] Study error: {type(e).__name__}: {e}")
        await progress_msg.edit(
            content=f"❌ Something went wrong. Please try again.\nError: {str(e)[:200]}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        await message.remove_reaction("🔍", bot.user)
        await message.add_reaction("✅")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Auto-Detect in Configured Channels
# ---------------------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # @Vexi study <url> — works in any channel
    if bot.user in message.mentions:
        # Ignore replies to Vexi's own messages (Discord auto-includes the mention on reply)
        if message.reference is not None:
            ref = message.reference.resolved
            if isinstance(ref, discord.Message) and ref.author.id == bot.user.id:
                await bot.process_commands(message)
                return
        content = message.content
        for mention_str in (f"<@{bot.user.id}>", f"<@!{bot.user.id}>"):
            content = content.replace(mention_str, "").strip()
        await handle_mention_study(message, content)
        return

    if message.channel.id not in VEXI_CHANNEL_IDS:
        await bot.process_commands(message)
        return

    video_url = extract_video_source(message)
    if not video_url:
        await bot.process_commands(message)
        return

    try:
        await message.add_reaction("🔍")
    except Exception:
        pass

    thinking_msg = await message.reply(
        "⏳ **Vexi is analyzing your video...**\n▁▁▁▁▁▁▁▁▁▁ Sending to AI..."
    )

    # Progress animation
    progress_done = asyncio.Event()

    async def update_progress():
        stages = [
            ("▓▓▁▁▁▁▁▁▁▁", "🧠 AI is watching your video..."),
            ("▓▓▓▓▁▁▁▁▁▁", "🧠 AI is watching your video..."),
            ("▓▓▓▓▓▁▁▁▁▁", "🔍 Checking legal compliance..."),
            ("▓▓▓▓▓▓▁▁▁▁", "🔍 Checking legal compliance..."),
            ("▓▓▓▓▓▓▓▁▁▁", "🎬 Reviewing hooks & storytelling..."),
            ("▓▓▓▓▓▓▓▓▁▁", "🎬 Reviewing hooks & storytelling..."),
            ("▓▓▓▓▓▓▓▓▓▁", "📝 Writing up the review..."),
            ("▓▓▓▓▓▓▓▓▓▓", "✅ Almost done..."),
        ]
        for bar, status in stages:
            if progress_done.is_set():
                return
            try:
                await thinking_msg.edit(
                    content=f"⏳ **Vexi is analyzing your video...**\n{bar} {status}"
                )
            except Exception:
                return
            try:
                await asyncio.wait_for(progress_done.wait(), timeout=15)
                return
            except asyncio.TimeoutError:
                continue

    progress_task = asyncio.create_task(update_progress())

    att_filename = ""
    for att in message.attachments:
        if Path(att.filename).suffix.lower() in VIDEO_EXTENSIONS:
            att_filename = att.filename
            break
    mime = guess_mime_type(video_url, att_filename)

    try:
        # Discord CDN URLs are signed/temporary — Gemini can't fetch them directly.
        # Skip straight to the upload fallback for Discord attachments.
        is_discord_cdn = bool(DISCORD_CDN_PATTERN.match(video_url))

        if is_discord_cdn:
            log.info("Discord CDN URL detected — using upload fallback directly.")
            async with aiohttp.ClientSession() as session:
                review = await analyze_video_with_gemini_upload(video_url, session)
        else:
            review = await analyze_video_with_gemini(video_url, mime_type=mime)
            if "error" in review:
                log.info(f"Direct URL failed ({review['error']}), trying upload fallback...")
                async with aiohttp.ClientSession() as session:
                    review = await analyze_video_with_gemini_upload(video_url, session)

        # Stop progress
        progress_done.set()
        await progress_task

        if "error" in review:
            await thinking_msg.edit(
                content=(
                    f"❌ I couldn't process that video. Error: {review['error']}\n\n"
                    "Please make sure:\n"
                    "• The file is a supported format (.mp4, .mov, .webm)\n"
                    "• Google Drive links have sharing set to 'Anyone with the link'\n"
                    "• The file is under 100MB"
                )
            )
            return

        content_text, embeds = build_review_message(review, creator=f"{message.author.mention}")

        # Coach ping
        ping_text = content_text or ""
        if COACH_ROLE_ID:
            ping_text = f"🏷️ <@&{COACH_ROLE_ID}> — new video submitted by {message.author.mention} for review!"

        await thinking_msg.edit(content=ping_text if ping_text else None, embeds=embeds)
    except Exception as e:
        progress_done.set()
        await progress_task
        log.error(f"Auto-detect error: {type(e).__name__}: {e}")
        await thinking_msg.edit(
            content=f"❌ Something went wrong during the review. Please try again.\nError: {str(e)[:200]}"
        )

    try:
        await message.remove_reaction("🔍", bot.user)
        await message.add_reaction("✅")
    except Exception:
        pass

    await bot.process_commands(message)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        log.error("DISCORD_BOT_TOKEN not set!")
        exit(1)
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set!")
        exit(1)
    bot.run(DISCORD_BOT_TOKEN)
