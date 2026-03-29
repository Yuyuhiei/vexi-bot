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
WHAT TO CHECK (Internal — use these to inform your paragraphs)
═══════════════════════════════════════

LEGAL COMPLIANCE CHECKS:
1. Copyrighted Characters or Content — Disney, Marvel, anime, Netflix, movie scenes, third-party characters. (HIGH RISK)
2. Fake Reviews or Testimonials — Actors scripting fake customer stories, fake "first-time" reactions. (HIGH RISK)
3. Exaggerated or Unproven Claims — Specific unprovable numerical claims like "10x your revenue guaranteed." Personal honest experiences are fine. (MEDIUM RISK)
4. People Without Permission — Identifiable bystanders, friends, or children without release. (MEDIUM RISK)
5. Competitor Logos or Products — Visible competitor logos, mocking competitors. (MEDIUM RISK)
6. AI-Generated Faces or Voices — AI-generated people as testimonials without disclosure. (MEDIUM RISK)
7. Font Licensing — Premium fonts without commercial license. Google Fonts are fine. (LOW RISK)
8. Filming Locations — Inside recognizable branded private spaces. (LOW RISK)
9. Platform Rules — Missing branded content toggles. (LOW RISK)
10. Cultural Sensitivity — Stereotypes, accents as jokes, religious/political imagery. (LOW RISK)
11. Music — If you hear music, give a soft reminder to confirm it's from TikTok/IG library or approved royalty-free source. NEVER flag music as a risk.
12. Ad Disclosure — Remind creators to include at least one ad-disclosure hashtag (#ManusAd, #ManusPartner, #Ad, #Sponsored) in their caption when posting. Generic tags like #Manus alone are NOT enough. NEVER suggest putting hashtags on the video itself. NEVER flag as a risk.

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
  "legal_paragraph": "Write a SHORT conversational paragraph (3-5 sentences max) summarizing the legal compliance findings. Naturally weave in any flags you found — mention the specific issue, the risk level in brackets like [HIGH] or [MEDIUM], and the timestamp if applicable. If there are no flags, say so briefly. Always end with the music soft reminder (if music was detected) and the ad-disclosure hashtag reminder as natural sentences, not as separate sections. Example tone: 'No major legal flags here! I didn't spot any copyrighted content or fake testimonials. One thing to note — at 0:15 there's a claim about saving 20 hours a week [MEDIUM], your coach might want to verify that. I hear some background music, so just confirm it's from a licensed source. And remember to pop an ad-disclosure hashtag like #ManusAd in your caption when posting!'",
  "content_paragraph": "Write a SHORT conversational paragraph (3-5 sentences max) summarizing the UGC fundamentals. Cover safe zones, lighting/audio, the hook (mention which of the 12 categories it fits and whether it's strong or could be improved — suggest a specific alternative if weak), and pacing. Be constructive and specific. Example tone: 'Lighting and audio are solid — your face is well-lit and the sound is crisp. Safe zones look good for IG and TikTok. Your hook falls into the Demo/How-To category and it's decent, but it could be punchier — try opening with something like \"I built an entire website in 30 seconds\" to create more instant curiosity. Pacing is smooth throughout with no dead air.'",
  "quick_verdict": "LOOKS GOOD / NEEDS REVIEW / COACH ATTENTION NEEDED / NOT MANUS CONTENT",
  "overall_summary": "One final sentence. Always include: 'A human coach will review this shortly for final approval.'"
}

CRITICAL RULES FOR THE PARAGRAPHS:
- Keep each paragraph SHORT — 3-5 sentences max. Do NOT write essays.
- Be conversational and friendly, like a peer creator giving feedback in a chat.
- Naturally mention ALL relevant checks within the paragraph flow — don't use headers, bullet points, or field labels.
- If something is fine, you can group multiple "all good" items in one sentence (e.g., "No copyrighted content, fake testimonials, or competitor logos spotted.").
- If something needs attention, be specific but brief (mention what, where/when, and risk level).
- Music and ad disclosure reminders should feel like natural sentences at the end of the legal paragraph, not separate callouts.
- For hooks, always mention which of the 12 categories it falls into.
- If the video is in a foreign language, still review it fully. Use "script_summary" for the English translation.
- NEVER use markdown formatting (no bold, no headers, no bullets) inside the paragraph strings — just plain conversational text.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
GDRIVE_PATTERN = re.compile(r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)")
GDRIVE_OPEN_PATTERN = re.compile(r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)")
YOUTUBE_PATTERN = re.compile(r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)")


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


async def analyze_video_with_gemini(video_url: str, mime_type: str = "video/mp4") -> dict:
    """Send video URL directly to Gemini for review (no local download needed)."""
    raw_text = ""
    try:
        log.info(f"Sending video URL to Gemini: {video_url[:120]}...")
        log.info(f"MIME type: {mime_type}")

        video_part = genai_types.Part.from_uri(
            file_uri=video_url,
            mime_type=mime_type,
        )

        log.info("Calling Gemini 2.5 Flash for review...")
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[video_part, REVIEW_PROMPT],
        )

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


async def analyze_video_with_gemini_upload(video_url: str, session: aiohttp.ClientSession) -> dict:
    """Fallback: Download video and upload to Gemini File API."""
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

        log.info(f"Uploading {video_path} to Gemini File API...")
        uploaded_file = gemini_client.files.upload(file=video_path)
        log.info(f"Upload complete: {uploaded_file.name}, state={uploaded_file.state}")

        max_wait = 120
        waited = 0
        while uploaded_file.state.name == "PROCESSING" and waited < max_wait:
            await asyncio.sleep(5)
            waited += 5
            uploaded_file = gemini_client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            return {"error": f"File processing failed. State: {uploaded_file.state.name}"}

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[uploaded_file, REVIEW_PROMPT],
        )

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
        return {"error": f"AI returned invalid JSON."}
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
        gate_embed.set_footer(text="Vexi • Derived from Latin 'vexillum' (flag) • v1.0")
        return (None, [gate_embed])

    # --- Build single compact embed ---
    verdict = review.get("quick_verdict", "NEEDS REVIEW")
    verdict_color = {
        "LOOKS GOOD": discord.Color.green(),
        "NEEDS REVIEW": discord.Color.gold(),
        "COACH ATTENTION NEEDED": discord.Color.orange(),
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
    embed.set_footer(text="Vexi • Derived from Latin 'vexillum' (flag) • v1.0")

    return (None, [embed])


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
        log.error(f"Slash command error: {type(e).__name__}: {e}")
        try:
            await progress_msg.edit(
                content=f"❌ Something went wrong during the review. Please try again.\nError: {str(e)[:200]}"
            )
        except Exception:
            pass


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
        log.error(f"Auto-detect error: {type(e).__name__}: {e}")
        try:
            await thinking_msg.edit(
                content=f"❌ Something went wrong during the review. Please try again.\nError: {str(e)[:200]}"
            )
        except Exception:
            pass

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
