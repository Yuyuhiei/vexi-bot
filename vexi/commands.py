"""Discord layer: bot instance, slash commands, mention handler, auto-detect,
per-user review queue, and the revise context menu. Logic moved from bot.py;
the four duplicated progress-bar closures are consolidated into
vexi.progress.ProgressAnimation and the three duplicated Gemini routing blocks
into vexi.pipeline.run_legacy_review — behavior unchanged."""

import asyncio
import io
import os
import re
import shutil
from pathlib import Path

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from vexi.config import (
    APIFY_API_TOKEN,
    COACH_ROLE_ID,
    VEXI_CHANNEL_IDS,
    VEXI_PIPELINE,
    log,
)
from vexi.downloaders import (
    GDRIVE_OPEN_PATTERN,
    GDRIVE_PATTERN,
    INSTAGRAM_PATTERN,
    TIKTOK_PATTERN,
    VIDEO_EXTENSIONS,
    YOUTUBE_PATTERN,
    _is_social_media_url,
    download_with_apify,
    download_with_ytdlp,
    extract_video_source,
    guess_mime_type,
)
from vexi.gemini import _analyze_local_file_with_gemini, _gemini_call_and_parse
from vexi.pipeline import run_legacy_review, run_review
from vexi.progress import REVIEW_STAGES, STUDY_STAGES, ProgressAnimation
from vexi.prompts import REVISE_PROMPT, STUDY_PROMPT
from vexi.render import (
    DetailsButton,
    _cache_script,
    _extract_script_from_attachment,
    _extract_script_from_embed,
    _get_cached_script,
    build_review_layout,
    build_review_message,
    build_study_message,
)

# ---------------------------------------------------------------------------
# Discord Bot Setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Track whether we've already synced commands (avoid re-syncing on reconnect)
_commands_synced = False


@bot.event
async def setup_hook():
    # Persistent "View full analysis" buttons keep working across restarts:
    # DynamicItem dispatches on custom_id pattern, state lives on the message.
    bot.add_dynamic_items(DetailsButton)


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
# Per-user FIFO review queue
# ---------------------------------------------------------------------------
# Videos submitted via /vexi are processed one at a time PER USER. Different
# users run in parallel; a single user's videos run serially so no one can
# monopolize Gemini quota. Queue and worker state live in-memory only.
_user_queues: dict[int, asyncio.Queue] = {}
_user_workers: dict[int, asyncio.Task] = {}
_queue_lock = asyncio.Lock()


async def enqueue_review(user_id: int, job: dict) -> None:
    async with _queue_lock:
        q = _user_queues.setdefault(user_id, asyncio.Queue())
        await q.put(job)
        w = _user_workers.get(user_id)
        if w is None or w.done():
            _user_workers[user_id] = asyncio.create_task(_user_worker(user_id))


async def _user_worker(user_id: int) -> None:
    q = _user_queues.get(user_id)
    if q is None:
        return
    try:
        while True:
            try:
                job = await asyncio.wait_for(q.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # Recheck under lock — a job may have been enqueued mid-drain.
                async with _queue_lock:
                    if q.empty():
                        _user_queues.pop(user_id, None)
                        _user_workers.pop(user_id, None)
                        return
                continue
            try:
                await process_single_review(**job)
            except Exception:
                log.exception(f"review job failed for user {user_id}")
            finally:
                q.task_done()
    except Exception:
        log.exception(f"user worker crashed for user {user_id}")
        async with _queue_lock:
            _user_workers.pop(user_id, None)


async def process_single_review(
    submitter: discord.User | discord.Member,
    coach: discord.Member | None,
    channel: discord.abc.Messageable,
    master_msg: discord.Message,
    attachment: discord.Attachment | None,
    url: str | None,
    display_url: str | None,
    index: int,
    total: int,
) -> None:
    """Runs one video through the full review pipeline. Posts the video (or its
    URL), a live progress bar, and the final review as replies to master_msg.
    Called by the per-user worker — this is where the actual Gemini call happens."""
    label = f"Video {index} of {total}" if total > 1 else "Your video"
    header = f"🎬 **{label}** — from {submitter.mention}"

    # Post the video header (with re-attached file if we can)
    video_msg: discord.Message
    if attachment is not None:
        try:
            video_file = await attachment.to_file()
            video_msg = await channel.send(content=header, file=video_file, reference=master_msg)
        except Exception as e:
            log.warning(f"Could not re-attach video for video {index}: {e}")
            video_msg = await channel.send(content=f"{header}\n(couldn't re-attach the file)", reference=master_msg)
    else:
        video_msg = await channel.send(content=f"{header}\n🎬 Video: {display_url or url}", reference=master_msg)

    progress_header = f"⏳ **Vexi is analyzing {label.lower()}...**"
    progress_msg = await video_msg.reply(content=f"{progress_header}\n▁▁▁▁▁▁▁▁▁▁ Sending to AI...")

    # Multiagent pipeline reports REAL stages; legacy keeps the cosmetic bar.
    progress: ProgressAnimation | None = None
    progress_cb = None
    if VEXI_PIPELINE == "multiagent":
        async def progress_cb(stage: str) -> None:
            try:
                await progress_msg.edit(content=f"{progress_header}\n{stage}")
            except Exception:
                pass
    else:
        progress = ProgressAnimation(progress_msg, progress_header, REVIEW_STAGES)
        progress.start()

    source_url = attachment.url if attachment is not None else (url or "")
    filename = attachment.filename if attachment is not None else ""
    mime = guess_mime_type(source_url, filename)

    try:
        review, pipeline_used = await run_review(
            source_url,
            mime_type=mime,
            creator_name=getattr(submitter, "display_name", str(submitter)),
            progress=progress_cb,
        )

        if progress:
            await progress.finish()

        # Ping coach only on the FIRST video's review (so a batch of 5 doesn't
        # spam @coach five times). Index 1 is the first video.
        coach_ping = coach and index == 1

        if pipeline_used == "multiagent" and "error" not in review:
            coach_line = f"🏷️ {coach.mention} — new review ready!" if coach_ping else None
            view, state_file = build_review_layout(review, submitter.mention, coach_line)
            # Send FIRST, delete the progress message only after success — if
            # the send fails, the except handler can still edit progress_msg.
            await channel.send(
                view=view,
                files=[state_file],
                reference=video_msg,
                allowed_mentions=discord.AllowedMentions(users=True, roles=True, everyone=False),
            )
            try:
                await progress_msg.delete()
            except Exception:
                pass
        else:
            content_text, embeds = build_review_message(review, creator=f"{submitter.mention}")
            if coach_ping:
                prefix = f"🏷️ {coach.mention} — "
                content_text = (prefix + (content_text or "")).strip()
            await progress_msg.edit(content=content_text or None, embeds=embeds)
    except Exception as e:
        if progress:
            await progress.finish()
        log.error(f"process_single_review error (video {index}/{total}): {type(e).__name__}: {e}")
        try:
            await progress_msg.edit(
                content=f"❌ Video {index}/{total} review failed. Error: {str(e)[:200]}"
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Slash Command: /vexi
# ---------------------------------------------------------------------------
@bot.tree.command(name="vexi", description="Submit up to 5 videos for Vexi AI review")
@app_commands.describe(
    video1="Attach a video file (required unless you use a URL)",
    video2="Optional 2nd video file",
    video3="Optional 3rd video file",
    video4="Optional 4th video file",
    video5="Optional 5th video file",
    video_url1="Or paste a video URL (Google Drive, YouTube, or direct link)",
    video_url2="Optional 2nd URL",
    video_url3="Optional 3rd URL",
    video_url4="Optional 4th URL",
    video_url5="Optional 5th URL",
    coach="Tag a coach to notify them about this batch",
)
async def vexi_command(
    interaction: discord.Interaction,
    video1: discord.Attachment = None,
    video2: discord.Attachment = None,
    video3: discord.Attachment = None,
    video4: discord.Attachment = None,
    video5: discord.Attachment = None,
    video_url1: str = None,
    video_url2: str = None,
    video_url3: str = None,
    video_url4: str = None,
    video_url5: str = None,
    coach: discord.Member = None,
):
    """Handle /vexi slash command with 1-5 videos, queued per-user."""
    try:
        await interaction.response.defer(thinking=True)
    except (discord.errors.NotFound, discord.errors.HTTPException) as e:
        log.warning(f"Interaction defer failed: {e}")
        return

    videos = [v for v in (video1, video2, video3, video4, video5) if v is not None]
    urls = [u for u in (video_url1, video_url2, video_url3, video_url4, video_url5) if u]

    # Validate every attachment's extension before enqueuing anything.
    for v in videos:
        suffix = Path(v.filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            await interaction.followup.send(
                f"❌ `{v.filename}` doesn't look like a video file.\n"
                f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}"
            )
            return

    # Build source list — attachments in slot order first, then URLs in slot order.
    sources: list[dict] = []
    for v in videos:
        sources.append({"attachment": v, "url": None, "display_url": None})
    for u in urls:
        sources.append({"attachment": None, "url": u, "display_url": u})

    if not sources:
        await interaction.followup.send(
            "👋 Hey! Please attach at least one video or provide a URL.\n\n"
            "**Attach up to 5 files:** `/vexi video1: video2: ...`\n"
            "**Or paste up to 5 URLs:** `/vexi video_url1: ...`\n"
            "**Mix both** — Vexi will review them one at a time in the order you gave them.\n"
            "**Tag a coach** with `coach:@CoachName` to notify them."
        )
        return

    total = len(sources)
    coach_ping = f"🏷️ {coach.mention} — " if coach else ""
    if total == 1:
        header = (
            f"{coach_ping}🔍 **Vexi is reviewing a video from {interaction.user.mention}...** Hang tight!"
        )
    else:
        header = (
            f"{coach_ping}📥 **Vexi received {total} videos from {interaction.user.mention}.**\n"
            f"Processing one at a time — I'll reply with each review below as it's ready."
        )
    master_msg = await interaction.followup.send(content=header)

    for i, src in enumerate(sources, start=1):
        await enqueue_review(
            interaction.user.id,
            {
                "submitter": interaction.user,
                "coach": coach,
                "channel": interaction.channel,
                "master_msg": master_msg,
                "attachment": src["attachment"],
                "url": src["url"],
                "display_url": src["display_url"],
                "index": i,
                "total": total,
            },
        )


# ---------------------------------------------------------------------------
# Study flow (shared by /study and @Vexi study mentions)
# ---------------------------------------------------------------------------
def _study_source_label(source_url: str) -> str:
    if INSTAGRAM_PATTERN.search(source_url):
        return "Instagram"
    if TIKTOK_PATTERN.search(source_url):
        return "TikTok"
    if YOUTUBE_PATTERN.search(source_url):
        return "YouTube"
    if GDRIVE_PATTERN.search(source_url) or GDRIVE_OPEN_PATTERN.search(source_url):
        return "Google Drive"
    return "Direct Link"


async def _execute_study(
    visible_msg: discord.Message,
    progress_msg: discord.Message,
    source_url: str,
    source_label: str,
    active_prompt: str,
    is_social: bool,
    attachment_filename: str = "",
) -> None:
    """Download (if needed), analyze, and post a Study Mode result. Shared by
    /study and the @Vexi study mention handler — the callers only differ in how
    they build the visible/progress messages and validate input."""
    progress = ProgressAnimation(progress_msg, "⏳ **Vexi is studying the format...**", STUDY_STAGES)
    progress.start()

    tmp_path = None
    tmp_dir = None
    downloaded_video_path_for_discord = None

    try:
        if is_social:
            log.info(f"Social media URL — downloading with yt-dlp: {source_url[:80]}")
            try:
                await progress_msg.edit(
                    content="⏳ **Vexi is studying the format...**\n▓▁▁▁▁▁▁▁▁▁ 📥 Downloading from social media..."
                )
            except Exception:
                pass
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
        else:
            mime = guess_mime_type(source_url, attachment_filename)
            study = await run_legacy_review(source_url, mime_type=mime, prompt=active_prompt, response_json=True)

        await progress.finish()

        _, embeds, script_file, script_text = build_study_message(study, source_label=source_label)

        # Delete the progress bar so only two messages remain: intro + results
        try:
            await progress_msg.delete()
        except Exception:
            pass

        # Build the single reply: video (if downloaded) + embeds + optional script.txt
        files: list[discord.File] = []
        if downloaded_video_path_for_discord and os.path.exists(downloaded_video_path_for_discord):
            ext = Path(downloaded_video_path_for_discord).suffix.lower()
            friendly_name = f"{source_label.lower().replace(' ', '_')}{ext}"
            files.append(discord.File(downloaded_video_path_for_discord, filename=friendly_name))
        if script_file is not None:
            files.append(script_file)

        send_kwargs: dict = {"embeds": embeds}
        if files:
            send_kwargs["files"] = files

        try:
            posted = await visible_msg.reply(**send_kwargs)
        except discord.HTTPException as e:
            if e.status == 413 and "files" in send_kwargs:
                # Drop the (large) video file but keep the script.txt if present.
                keep = [f for f in files if f.filename == "manus_script.txt"]
                if keep:
                    send_kwargs["files"] = keep
                else:
                    send_kwargs.pop("files", None)
                posted = await visible_msg.reply(**send_kwargs)
                await visible_msg.reply(content=f"📹 Video too large to attach — original: {source_url}")
            else:
                raise

        if script_text and posted is not None:
            _cache_script(posted.id, script_text)

    except Exception as e:
        await progress.finish()
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


def _study_prompt_with_niche(niche: str | None) -> str:
    if not niche:
        return STUDY_PROMPT
    return STUDY_PROMPT + (
        f"\n\nNICHE CONTEXT PROVIDED BY SUBMITTER: {niche}\n"
        "Tailor your Manus adaptation and suggested outline to this niche specifically.\n"
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
        source_label = _study_source_label(video_url)
        visible_msg = await interaction.followup.send(
            content=f"{intro_text}\n🎬 Video: {source_url}"
        )

    progress_msg = await visible_msg.reply(
        content="⏳ **Vexi is studying the format...**\n▁▁▁▁▁▁▁▁▁▁ Loading video..."
    )

    await _execute_study(
        visible_msg=visible_msg,
        progress_msg=progress_msg,
        source_url=source_url,
        source_label=source_label,
        active_prompt=_study_prompt_with_niche(niche),
        is_social=(video is None) and _is_social_media_url(source_url),
        attachment_filename=video.filename if video else "",
    )


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

    source_label = _study_source_label(source_url)

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

    await _execute_study(
        visible_msg=visible_msg,
        progress_msg=progress_msg,
        source_url=source_url,
        source_label=source_label,
        active_prompt=_study_prompt_with_niche(niche),
        is_social=is_social,
    )

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

    thinking_header = "⏳ **Vexi is analyzing your video...**"
    thinking_msg = await message.reply(f"{thinking_header}\n▁▁▁▁▁▁▁▁▁▁ Sending to AI...")

    progress: ProgressAnimation | None = None
    progress_cb = None
    if VEXI_PIPELINE == "multiagent":
        async def progress_cb(stage: str) -> None:
            try:
                await thinking_msg.edit(content=f"{thinking_header}\n{stage}")
            except Exception:
                pass
    else:
        progress = ProgressAnimation(thinking_msg, thinking_header, REVIEW_STAGES)
        progress.start()

    att_filename = ""
    for att in message.attachments:
        if Path(att.filename).suffix.lower() in VIDEO_EXTENSIONS:
            att_filename = att.filename
            break
    mime = guess_mime_type(video_url, att_filename)

    try:
        review, pipeline_used = await run_review(
            video_url,
            mime_type=mime,
            creator_name=message.author.display_name,
            progress=progress_cb,
        )

        if progress:
            await progress.finish()

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

        coach_line = None
        if COACH_ROLE_ID:
            coach_line = f"🏷️ <@&{COACH_ROLE_ID}> — new video submitted by {message.author.mention} for review!"

        if pipeline_used == "multiagent":
            view, state_file = build_review_layout(review, message.author.mention, coach_line)
            # Send first, delete the progress message only after success.
            await message.reply(
                view=view,
                files=[state_file],
                allowed_mentions=discord.AllowedMentions(users=True, roles=True, everyone=False),
            )
            try:
                await thinking_msg.delete()
            except Exception:
                pass
        else:
            content_text, embeds = build_review_message(review, creator=f"{message.author.mention}")
            ping_text = coach_line or content_text or ""
            await thinking_msg.edit(content=ping_text if ping_text else None, embeds=embeds)
    except Exception as e:
        if progress:
            await progress.finish()
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
# Revise Mode — context menu on Vexi's /study output
# ---------------------------------------------------------------------------
async def revise_script_via_gemini(original: str, instruction: str) -> dict:
    """Text-only Gemini call: revise a script per the user's instruction."""
    prompt = (
        REVISE_PROMPT
        + f"\n\nORIGINAL SCRIPT:\n{original}\n\nUSER INSTRUCTION:\n{instruction}\n"
    )
    return await _gemini_call_and_parse(
        contents=[prompt],
        response_json=True,
        label="revise",
    )


class ReviseModal(discord.ui.Modal, title="Revise this script"):
    instruction = discord.ui.TextInput(
        label="How should Vexi revise the script?",
        style=discord.TextStyle.paragraph,
        placeholder="e.g. make it more TikTok, shorten to 30s, make it casual, alternative angle",
        required=True,
        max_length=500,
    )

    def __init__(self, original_script: str, target_message: discord.Message):
        super().__init__()
        self.original_script = original_script
        self.target_message = target_message

    async def on_submit(self, interaction: discord.Interaction):
        try:
            await interaction.response.defer(thinking=True, ephemeral=True)
        except Exception as e:
            log.warning(f"Revise defer failed: {e}")
            return

        instruction_text = str(self.instruction).strip()
        log.info(f"Revise request from {interaction.user}: {instruction_text[:100]}")

        result = await revise_script_via_gemini(self.original_script, instruction_text)
        if "error" in result:
            await interaction.followup.send(
                f"❌ Couldn't revise the script: {result['error'][:200]}", ephemeral=True
            )
            return

        new_script = (result.get("full_script") or "").strip()
        if not new_script:
            await interaction.followup.send(
                "❌ Gemini didn't return a revised script. Try again with a clearer instruction.",
                ephemeral=True,
            )
            return

        wrapped = f"```\n{new_script}\n```"
        files: list[discord.File] = []
        if len(wrapped) <= 3900:
            desc = (
                f"**Instruction:** *{instruction_text}*\n"
                f"{wrapped}\n"
                "*Right-click this message → Apps → 'Revise this script' to iterate again.*"
            )
        else:
            files.append(discord.File(io.BytesIO(new_script.encode("utf-8")), filename="manus_script_revised.txt"))
            desc = (
                f"**Instruction:** *{instruction_text}*\n"
                "Revised script was long — attached as `manus_script_revised.txt`.\n"
                "*Right-click this message → Apps → 'Revise this script' to iterate again.*"
            )

        embed = discord.Embed(
            title="🔁 Vexi Revised Script",
            description=desc,
            color=discord.Color.blurple(),
        )
        embed.set_footer(text="Vexi Study Mode • Revised • v1.3")

        posted: discord.Message | None = None
        try:
            posted = await self.target_message.reply(embed=embed, files=files)
        except (discord.HTTPException, discord.NotFound) as e:
            log.warning(f"Revise reply failed ({e}), posting via followup instead.")
            try:
                posted = await interaction.followup.send(embed=embed, files=files)
            except Exception as e2:
                log.error(f"Revise followup send also failed: {e2}")

        if posted is not None:
            _cache_script(posted.id, new_script)
            try:
                await interaction.followup.send("✅ Revised script posted below.", ephemeral=True)
            except Exception:
                pass


@bot.tree.context_menu(name="Revise this script")
async def revise_context_menu(interaction: discord.Interaction, message: discord.Message):
    if message.author.id != bot.user.id:
        await interaction.response.send_message(
            "This only works on Vexi's own /study messages.", ephemeral=True
        )
        return

    is_study = any(
        (e.footer and e.footer.text and "Vexi Study Mode" in e.footer.text)
        for e in message.embeds
    )
    if not is_study:
        await interaction.response.send_message(
            "This doesn't look like a Vexi /study result. Right-click a Vexi Study Mode message and try again.",
            ephemeral=True,
        )
        return

    script = _get_cached_script(message.id) or _extract_script_from_embed(message)
    if not script:
        script = await _extract_script_from_attachment(message)
    if not script:
        await interaction.response.send_message(
            "I couldn't find the script in this message (cache may have expired). Run /study again and try revise on the fresh result.",
            ephemeral=True,
        )
        return

    await interaction.response.send_modal(ReviseModal(script, message))
