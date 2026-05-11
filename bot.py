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
LAYER 0.5: MONEY & INCOME CLAIM SCAN (v1.2 — RUN BEFORE ALL OTHER CHECKS)
═══════════════════════════════════════
Scan ALL audio, on-screen text, captions, and visuals for AUTO-REJECT keywords. These are absolute violations that require immediate escalation regardless of context.

AUTO-REJECT PHRASES — if ANY of these are detected, set "quick_verdict" to "AUTO-REJECT":
- "guaranteed income" / "guarantees income" / "guarantee you'll make"
- "make you rich" / "will make you rich" / "Manus will make you rich"
- "100% success" / "100% results"
- "zero risk"
- "easy money"
- "everyone earns" / "everyone will earn"
- "get rich quick"
- "passive income" (when presented as a Manus guarantee, not a personal goal)
- "financial freedom" (when presented as a Manus guarantee)
- Any phrasing that equates using Manus with a guaranteed dollar outcome (e.g., "use Manus and earn $X guaranteed")

If AUTO-REJECT keyword(s) are found:
- Set "quick_verdict" to "AUTO-REJECT"
- In "legal_paragraph", clearly name the exact phrase(s) detected with an [AUTO-REJECT] tag and timestamp if possible
- Still complete all remaining checks below (the coach needs the full picture)

MONEY CLAIM GUARDRAILS — HIGH RISK (not auto-reject, but flag as [HIGH]):
- Any claim that Manus guarantees income or financial results, even without using the exact banned phrases
- Implied "use Manus = you will make money" framing without explicit disclaimers or personal attribution
- Specific dollar amounts (e.g., "$5k/month", "$10,000") stated as achievable outcomes without attached verifiable evidence
- Income or ROI promises framed as universally applicable ("you will earn", "anyone can make")

ALLOWED MONEY FRAMING (compliant — do NOT flag these):
- Personal progress journaling with no guarantee: "Day 1 of my journey toward $5k/month — here are the tools I'm using"
- Third-party user results with explicit evidence: "My friend made $X — here's their proof and the process they used"
- Process/skill teaching with no income promise: "How to land your first clients using Manus"
- Scenario-specific time savings with proof: "This report took 4 hours manually, Manus did it in 12 minutes (with evidence)"
- Payment capability demos: "I built a site that can accept payments via Stripe" (showing the flow, not promising profit)

═══════════════════════════════════════
WHAT TO CHECK (Internal — use these to inform your paragraphs)
═══════════════════════════════════════

LEGAL COMPLIANCE CHECKS (v1.2 Checklist):
1. Income & Money Claims — See LAYER 0.5 above. AUTO-REJECT for banned phrases; [HIGH] for implied guarantees; compliant framing is fine. (HIGH RISK / AUTO-REJECT)
2. Absolute Claims — Phrases like "100%", "zero errors", "fully replaces humans", "best AI" without proof. (HIGH RISK)
3. Efficiency Numbers Without Proof — Time-saved or speed claims (e.g., "build a site in 10 minutes", "save 5 hours") require real supporting data or evidence. Flag if none is visible. (MEDIUM RISK)
4. Copyrighted Characters or Content — Disney, Marvel, anime, Netflix, movie scenes, third-party characters. (HIGH RISK)
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
  "content_paragraph": "Write a SHORT conversational paragraph (3-5 sentences max) summarizing the UGC fundamentals. Cover safe zones, lighting/audio, the hook (mention which of the 12 categories it fits and whether it's strong or could be improved — suggest a specific alternative if weak), and pacing. Be constructive and specific. Example tone: 'Lighting and audio are solid — your face is well-lit and the sound is crisp. Safe zones look good for IG and TikTok. Your hook falls into the Demo/How-To category and it's decent, but it could be punchier — try opening with something like \"I built an entire website in 30 seconds\" to create more instant curiosity. Pacing is smooth throughout with no dead air.'",
  "quick_verdict": "LOOKS GOOD / NEEDS REVIEW / COACH ATTENTION NEEDED / AUTO-REJECT / NOT MANUS CONTENT",
  "overall_summary": "One final sentence. Always include: 'A human coach will review this shortly for final approval.' If AUTO-REJECT, start with: 'This video contains auto-reject language and must be reviewed by a coach before any use.'"
}

CRITICAL RULES FOR THE PARAGRAPHS:
- Keep each paragraph SHORT — 3-5 sentences max. Do NOT write essays.
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
    """Download a public social media video via yt-dlp. Returns (file_path, error)."""
    import yt_dlp

    tmp_dir = tempfile.mkdtemp(prefix="vexi_study_")
    output_template = os.path.join(tmp_dir, "video.%(ext)s")

    ydl_opts = {
        "outtmpl": output_template,
        "format": "best[ext=mp4][filesize<100M]/best[filesize<100M]/best",
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 100 * 1024 * 1024,
    }

    def _download():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        files = list(Path(tmp_dir).glob("video.*"))
        if files:
            return str(files[0]), None
        return None, "Download completed but output file not found."

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _download)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, str(e)


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


async def _gemini_generate(contents: list, retries: int = 3) -> object:
    delays = [5, 15, 30]
    last_exc = None
    for attempt in range(retries):
        try:
            return gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
            )
        except Exception as e:
            last_exc = e
            if _is_retryable(e) and attempt < retries - 1:
                wait = delays[attempt]
                log.warning(f"Gemini transient error (attempt {attempt + 1}/{retries}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
    raise last_exc


async def analyze_video_with_gemini(video_url: str, mime_type: str = "video/mp4", prompt: str = None) -> dict:
    """Send video URL directly to Gemini for review (no local download needed)."""
    if prompt is None:
        prompt = REVIEW_PROMPT
    raw_text = ""
    try:
        log.info(f"Sending video URL to Gemini: {video_url[:120]}...")
        log.info(f"MIME type: {mime_type}")

        video_part = genai_types.Part.from_uri(
            file_uri=video_url,
            mime_type=mime_type,
        )

        log.info("Calling Gemini 2.5 Flash for review...")
        response = await _gemini_generate([video_part, prompt])

        raw_text = response.text.strip()
        log.info(f"Gemini response length: {len(raw_text)} chars")
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        result = json.loads(raw_text)
        log.info(f"Review complete. Verdict: {result.get('quick_verdict', 'N/A')}")
        return result

    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}\nRaw: {raw_text[:500]}")
        return {"error": f"AI returned invalid JSON. Raw excerpt: {raw_text[:300]}"}
    except Exception as e:
        log.error(f"Gemini error: {type(e).__name__}: {e}")
        import traceback
        log.error(traceback.format_exc())
        return {"error": str(e)}


async def _analyze_local_file_with_gemini(file_path: str, prompt: str = None) -> dict:
    """Upload a local file to Gemini File API and analyze it."""
    if prompt is None:
        prompt = REVIEW_PROMPT
    raw_text = ""
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

        response = await _gemini_generate([uploaded_file, prompt])
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        result = json.loads(raw_text)

        try:
            gemini_client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        return result

    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}\nRaw: {raw_text[:500]}")
        return {"error": "AI returned invalid JSON."}
    except Exception as e:
        log.error(f"Local file Gemini upload error: {type(e).__name__}: {e}")
        return {"error": str(e)}


async def analyze_video_with_gemini_upload(video_url: str, session: aiohttp.ClientSession, prompt: str = None) -> dict:
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

        return await _analyze_local_file_with_gemini(video_path, prompt)

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
        err_embed = discord.Embed(
            title="Vexi Study — Error",
            description=f"Something went wrong:\n```{study['error'][:500]}```\nMake sure the video is public and in a supported format.",
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
            if dl_error or not tmp_path:
                study = {"error": f"Could not download video: {dl_error or 'Unknown error'}. Make sure the video is public."}
            else:
                tmp_dir = os.path.dirname(tmp_path)
                log.info(f"yt-dlp downloaded to: {tmp_path}")
                downloaded_video_path_for_discord = tmp_path
                study = await _analyze_local_file_with_gemini(tmp_path, prompt=active_prompt)
        elif is_discord_cdn:
            log.info("Discord CDN URL — using upload fallback.")
            async with aiohttp.ClientSession() as sess:
                study = await analyze_video_with_gemini_upload(source_url, sess, prompt=active_prompt)
        elif is_gdrive:
            direct_url = convert_gdrive_to_direct(source_url)
            study = await analyze_video_with_gemini(direct_url, prompt=active_prompt)
            if "error" in study:
                log.info(f"Direct GDrive failed, trying upload fallback...")
                async with aiohttp.ClientSession() as sess:
                    study = await analyze_video_with_gemini_upload(direct_url, sess, prompt=active_prompt)
        else:
            mime = guess_mime_type(source_url, video.filename if video else "")
            study = await analyze_video_with_gemini(source_url, mime_type=mime, prompt=active_prompt)
            if "error" in study:
                log.info(f"Direct URL failed, trying upload fallback...")
                async with aiohttp.ClientSession() as sess:
                    study = await analyze_video_with_gemini_upload(source_url, sess, prompt=active_prompt)

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
# Auto-Detect in Configured Channels
# ---------------------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
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
