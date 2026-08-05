"""Message rendering: the legacy embed builders (moved verbatim from bot.py,
plus a previously-missing 4096-char truncation guard) and the study-script
cache used by /revise."""

import io
import re
import time

import discord

from vexi.config import LUMISCRIPT_URL


def _fit_embed_description(parts: list[str], tail: str | None = None, limit: int = 4096) -> str:
    """Join parts into an embed description, truncating the middle if needed so
    the total stays under Discord's 4096-char embed-description limit. `tail`
    (e.g. the risk grid) is always preserved at the bottom."""
    tail_text = f"\n{tail}" if tail else ""
    body = "\n".join(parts)
    if len(body) + len(tail_text) <= limit:
        return body + tail_text
    marker = "\n…*(review truncated — ask a coach if anything looks cut off)*"
    budget = limit - len(tail_text) - len(marker)
    return body[:budget].rstrip() + marker + tail_text


def build_review_message(review: dict, creator: str = None) -> tuple[str | None, list[discord.Embed]]:
    """Convert the Gemini JSON review into a single compact Discord embed.

    Returns (content_text, embeds_list).
    """
    # --- Error ---
    if "error" in review:
        err_msg = review["error"]
        # Friendly wording for the common "response got cut off" case.
        is_truncation = (
            "finish_reason" in err_msg
            or "cut off" in err_msg.lower()
            or "invalid json" in err_msg.lower()
        )
        if is_truncation:
            description = (
                "😅 My reply got cut off before I could finish reviewing your video. "
                "This usually clears up on its own — please try `/vexi` again on the "
                "same video in a moment. If it keeps happening, tag a coach."
            )
        else:
            description = (
                f"Something went wrong during the review:\n```{err_msg[:500]}```\n"
                "Please try again or ask a coach for help."
            )
        err_embed = discord.Embed(
            title="Vexi — Review Error",
            description=description,
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

    # Layer 1 — Compliance paragraph
    legal = review.get("legal_paragraph", "")
    if legal:
        parts.append(f"\n🛡️ **Compliance Check:**\n{legal}")

    # Layer 2 — Content paragraph
    content = review.get("content_paragraph", "")
    if content:
        parts.append(f"\n🎬 **Content Review:**\n{content}")

    # Layer 3 — Manus plug paragraph
    plug = review.get("manus_plug_paragraph", "")
    if plug:
        parts.append(f"\n🔌 **Manus Plug:**\n{plug}")

    # Layer 4 — Viral checklist paragraph
    viral = review.get("viral_checklist_paragraph", "")
    if viral:
        parts.append(f"\n🚀 **Viral Video Checklist:**\n{viral}")

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
    risk_tail = f"\n**{grid_label}**\n{grid_block}"

    embed = discord.Embed(
        title=f"Vexi Review — {verdict}",
        description=_fit_embed_description(parts, tail=risk_tail),
        color=verdict_color,
    )
    embed.set_footer(text="Vexi • Derived from Latin 'vexillum' (flag) • v1.2")

    return (None, [embed])


# In-memory cache of study scripts keyed by the Discord message ID Vexi posted
# them in. Used by /revise (context-menu command) to fetch the original script
# without re-parsing the embed. Entries are evicted lazily on read.
_recent_scripts: dict[int, tuple[str, float]] = {}
SCRIPT_CACHE_TTL_SEC = 24 * 3600


def _cache_script(message_id: int, script: str) -> None:
    _recent_scripts[message_id] = (script, time.time())
    # Evict anything older than TTL to keep the dict bounded.
    now = time.time()
    stale = [mid for mid, (_, ts) in _recent_scripts.items() if now - ts > SCRIPT_CACHE_TTL_SEC]
    for mid in stale:
        _recent_scripts.pop(mid, None)


def _get_cached_script(message_id: int) -> str | None:
    entry = _recent_scripts.get(message_id)
    if not entry:
        return None
    script, ts = entry
    if time.time() - ts > SCRIPT_CACHE_TTL_SEC:
        _recent_scripts.pop(message_id, None)
        return None
    return script


def _extract_script_from_embed(message: discord.Message) -> str | None:
    """Pull the script out of Vexi's Study embed3 code block (fallback path)."""
    for e in message.embeds:
        desc = e.description or ""
        m = re.search(r"```(?:\w+)?\n(.*?)\n```", desc, re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            # Guard: must look like a beat-marked script, not some random code block
            if "[" in candidate and "]" in candidate:
                return candidate
    return None


async def _extract_script_from_attachment(message: discord.Message) -> str | None:
    for att in message.attachments:
        if att.filename.lower().endswith(".txt") and "script" in att.filename.lower():
            try:
                data = await att.read()
                return data.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
    return None


def build_study_message(study: dict, source_label: str = "Video") -> tuple[str | None, list[discord.Embed], discord.File | None, str | None]:
    """Convert a Gemini Study Mode JSON result into (content, embeds, attachment, script).

    The optional discord.File is a `script.txt` attachment used when the script
    is too long for the third embed. The final str is the raw script text
    (returned so the caller can cache it against the posted message ID for /revise).
    """
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
        return (None, [err_embed], None, None)

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
    embed1.set_footer(text="Vexi Study Mode • 1 of 3 — Format Analysis")

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
    embed2.set_footer(text="Vexi Study Mode • 2 of 3 — Manus Adaptation Brief")

    # --- Embed 3: Copyable Script (or .txt attachment if too long) ---
    embeds: list[discord.Embed] = [embed1, embed2]
    script_file: discord.File | None = None
    full_script = (study.get("full_script") or "").strip()

    if full_script:
        # Discord embed description max is 4096 chars. Add 8 for the ``` fences
        # and a safety margin — if the wrapped block exceeds 3900, attach as file.
        wrapped = f"```\n{full_script}\n```"
        if len(wrapped) <= 3900:
            embed3 = discord.Embed(
                title="📝 Copyable Script — Manus UGC",
                description=(
                    f"Ready to record. Copy the block below.\n{wrapped}\n"
                    "*Right-click this message → Apps → 'Revise this script' to iterate.*"
                ),
                color=diff_color,
            )
            embed3.set_footer(text="Vexi Study Mode • 3 of 3 — Copyable Script • v1.3")
            embeds.append(embed3)
        else:
            # Too long for a single embed — attach as file, keep a short embed pointer.
            script_bytes = full_script.encode("utf-8")
            script_file = discord.File(io.BytesIO(script_bytes), filename="manus_script.txt")
            embed3 = discord.Embed(
                title="📝 Copyable Script — Manus UGC",
                description=(
                    "Script was long — attached as `manus_script.txt` (open it, copy the text).\n"
                    "*Right-click this message → Apps → 'Revise this script' to iterate.*"
                ),
                color=diff_color,
            )
            embed3.set_footer(text="Vexi Study Mode • 3 of 3 — Copyable Script • v1.3")
            embeds.append(embed3)

    return (None, embeds, script_file, full_script or None)
