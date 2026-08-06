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


# ═══════════════════════════════════════════════════════════════════════════
# v3 renderer — conversational Components V2 message + details button
# ═══════════════════════════════════════════════════════════════════════════
import io as _io
import json as _json

STATE_FILENAME = "vexi_review.json"
DETAILS_CUSTOM_ID = "vexi:details:v3"

VERDICT_ACCENT = {
    "LOOKS GOOD": 0x2ECC71,
    "NEEDS REVIEW": 0xF1C40F,
    "COACH ATTENTION NEEDED": 0xE67E22,
    "AUTO-REJECT": 0xE74C3C,
    "NOT MANUS CONTENT": 0x95A5A6,
}

RISK_LINE = {
    "LOOKS GOOD": "🟢 **Low risk** — looking clean!",
    "NEEDS REVIEW": "🟡 **Medium risk** — a coach should double-check the notes above.",
    "COACH ATTENTION NEEDED": "🟠 **High risk** — a few edits needed before this can go out.",
    "AUTO-REJECT": "🔴 **Critical — auto-reject language detected.** A coach must clear this before any use.",
    "NOT MANUS CONTENT": "⚪ **Not Manus content** — nothing to review here.",
}

# Order in which finding categories appear in the main message.
_CATEGORY_ORDER = [
    "money_claims", "copyright", "legal", "spelling", "manus_plug",
    "features", "viral", "content", "hook", "cta",
]


class DetailsButton(discord.ui.DynamicItem[discord.ui.Button], template=r"vexi:details:v(?P<schema>[0-9]+)"):
    """Persistent '📋 View full analysis' button. State lives in an invisible
    attachment on the message itself, so this works across bot restarts with
    no database — register with bot.add_dynamic_items(DetailsButton)."""

    def __init__(self) -> None:
        super().__init__(
            discord.ui.Button(
                label="📋 View full analysis",
                style=discord.ButtonStyle.secondary,
                custom_id=DETAILS_CUSTOM_ID,
            )
        )

    @classmethod
    async def from_custom_id(cls, interaction: discord.Interaction, item: discord.ui.Button, match):
        return cls()

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        state = None
        for att in interaction.message.attachments:
            if att.filename == STATE_FILENAME:
                try:
                    state = _json.loads(await att.read())
                except Exception:
                    state = None
                break
        if state is None:
            await interaction.followup.send(
                "😕 I couldn't find the analysis data on this message anymore — "
                "it may have been edited. Re-run `/vexi` for a fresh review.",
                ephemeral=True,
            )
            return
        for embed in build_details_pages(state):
            await interaction.followup.send(embed=embed, ephemeral=True)


def _ts_str(timestamps: list) -> str:
    clean = [str(t) for t in (timestamps or []) if str(t).strip()]
    return f" ({', '.join(clean[:3])})" if clean else ""


def _finding_line(f: dict) -> str:
    icon = "⚠️" if f.get("severity") == "flag" else "💡"
    if f.get("risk") == "AUTO-REJECT":
        icon = "⛔"
    return f"{icon} {f.get('message', '').strip()}{_ts_str(f.get('evidence_timestamps'))}"


def _grouped_finding_blocks(findings: list[dict]) -> list[str]:
    """Findings grouped by category — flags first, then recommends — each group
    one block; blank lines between blocks come from the join."""
    def sort_key(f):
        sev = 0 if f.get("severity") == "flag" else 1
        risk_rank = {"AUTO-REJECT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(f.get("risk"), 2)
        cat_rank = _CATEGORY_ORDER.index(f["category"]) if f.get("category") in _CATEGORY_ORDER else 99
        return (sev, risk_rank, cat_rank)

    ordered = sorted(findings, key=sort_key)
    blocks: list[str] = []
    current_cat, current_sev, lines = None, None, []
    for f in ordered:
        key = (f.get("category"), f.get("severity"))
        if key != (current_cat, current_sev) and lines:
            blocks.append("\n".join(lines))
            lines = []
        current_cat, current_sev = key
        lines.append(_finding_line(f))
    if lines:
        blocks.append("\n".join(lines))
    return blocks


def build_main_sections(review: dict, creator_mention: str, coach_line: str | None) -> tuple[list[tuple[int, str]], str]:
    """Returns ([(priority, section_text)], risk_line_text). Lower priority
    number = kept longer under length pressure."""
    verdict = review.get("quick_verdict", "NEEDS REVIEW")
    sections: list[tuple[int, str]] = []

    if coach_line:
        sections.append((0, coach_line))

    kudos = (review.get("kudos_line") or "").strip()
    if not kudos:
        kudos = f"Thanks for the submission, {creator_mention} — here's my first pass!"
    sections.append((0, kudos))

    if verdict == "NOT MANUS CONTENT":
        sections.append((0,
            "I took a look, but this video doesn't appear to be related to Manus — "
            "I couldn't find a Manus mention, logo, UI, or feature anywhere. If you think "
            "I got this wrong, ask a coach to review it manually."))
        risk = RISK_LINE[verdict]
        return sections, risk

    if verdict == "AUTO-REJECT":
        sections.append((0,
            "⛔ **Auto-reject language detected** — do not approve or publish until a coach "
            "reviews and clears this video."))

    findings = review.get("findings", [])
    flags = [f for f in findings if f.get("severity") == "flag"]
    recommends = [f for f in findings if f.get("severity") != "flag"]

    for block in _grouped_finding_blocks(flags):
        sections.append((1, block))
    for block in _grouped_finding_blocks(recommends):
        sections.append((3, block))

    if not findings:
        sections.append((1, "No issues from me — this one's clean on every check I ran. 🎉"))

    reminders = review.get("reminders") or []
    if reminders:
        sections.append((4, "-# 📌 Before posting: " + " • ".join(r.strip().rstrip(".") for r in reminders[:3])))

    if review.get("needs_human_review"):
        sections.append((2, "-# 🧑‍🏫 I wasn't 100% sure about parts of this one — flagged for a closer coach look."))

    sections.append((2, "-# I'm an AI flagger — a real coach makes the final call. Tap **📋 View full analysis** for the full breakdown (script summary, transcript & every check)."))

    summary = (review.get("overall_summary") or "").strip()
    risk = RISK_LINE.get(verdict, RISK_LINE["NEEDS REVIEW"])
    if summary:
        risk = f"{risk}\n-# {summary}"
    return sections, risk


def fit_sections(sections: list[tuple[int, str]], budget: int) -> list[str]:
    """Keep sections in order, dropping the lowest-priority ones from the end
    until the total fits the budget. Priority-0/1 sections are never dropped —
    they get truncated as a last resort."""
    def total(secs):
        return sum(len(s) for _, s in secs) + 2 * max(0, len(secs) - 1)

    kept = list(sections)
    dropped = 0
    while total(kept) > budget:
        drop_candidates = [i for i, (p, _) in enumerate(kept) if p >= 3]
        if drop_candidates:
            kept.pop(drop_candidates[-1])
            dropped += 1
            continue
        drop_candidates = [i for i, (p, _) in enumerate(kept) if p == 2]
        if drop_candidates:
            kept.pop(drop_candidates[-1])
            continue
        # Only priority 0/1 left — truncate the longest section, repeatedly,
        # until we fit (or everything is at the floor, then drop from the end).
        idx = max(range(len(kept)), key=lambda i: len(kept[i][1]))
        p, text = kept[idx]
        if len(text) <= 80:
            kept.pop()
            dropped += 1
            continue
        overshoot = total(kept) - budget
        kept[idx] = (p, text[: max(80, len(text) - overshoot - 2)].rstrip() + "…")

    # Discord caps a LayoutView at 40 components total; leave room for the
    # risk container + button row + state-file chip by merging overflow
    # sections together.
    MAX_SECTIONS = 24
    if len(kept) > MAX_SECTIONS:
        head = kept[:MAX_SECTIONS - 1]
        merged = "\n\n".join(s for _, s in kept[MAX_SECTIONS - 1:])
        kept = head + [(1, merged)]

    out = [s for _, s in kept]
    if dropped:
        out.append(f"-# +{dropped} more note{'s' if dropped > 1 else ''} — tap **📋 View full analysis**.")
    return out


class ReviewLayout(discord.ui.LayoutView):
    """The v3 conversational review message: bare TextDisplays (full-width chat
    look), a small accent-colored container for the risk line, and the
    persistent details button."""

    def __init__(self, review: dict, creator_mention: str, coach_line: str | None = None):
        super().__init__(timeout=None)
        verdict = review.get("quick_verdict", "NEEDS REVIEW")
        sections, risk_text = build_main_sections(review, creator_mention, coach_line)

        # CV2 budget: 4,000 chars across all TextDisplays. Reserve room for the
        # risk container.
        body_budget = 3800 - len(risk_text)
        for text in fit_sections(sections, body_budget):
            self.add_item(discord.ui.TextDisplay(text))

        container = discord.ui.Container(
            discord.ui.TextDisplay(risk_text),
            accent_colour=discord.Colour(VERDICT_ACCENT.get(verdict, 0xF1C40F)),
        )
        self.add_item(container)

        row = discord.ui.ActionRow()
        row.add_item(DetailsButton())
        self.add_item(row)

        # Discord REMOVES attachments that no component references on a
        # Components V2 message, so the state file must be referenced here or
        # the details button finds nothing. Spoilered = collapsed grey chip.
        self.add_item(discord.ui.File(f"attachment://{STATE_FILENAME}", spoiler=True))


def build_state_file(review: dict, creator_mention: str) -> discord.File:
    """The invisible state attachment: everything the details button needs,
    stored on the message itself (restart-proof, no DB). Never put secrets or
    coach-only notes in here — anyone can download message attachments."""
    state = {
        "v": 3,
        "review": {k: v for k, v in review.items() if k != "_details"},
        "details": review.get("_details", {}),
        "creator": creator_mention,
    }
    payload = _json.dumps(state, ensure_ascii=False).encode("utf-8")
    return discord.File(_io.BytesIO(payload), filename=STATE_FILENAME)


def build_review_layout(review: dict, creator_mention: str, coach_line: str | None = None) -> tuple["ReviewLayout", discord.File]:
    return ReviewLayout(review, creator_mention, coach_line), build_state_file(review, creator_mention)


# ---------------------------------------------------------------------------
# Ephemeral details pages
# ---------------------------------------------------------------------------
def _chunk_lines(lines: list[str], limit: int = 3800) -> list[str]:
    chunks, cur = [], ""
    for line in lines:
        # Hard-wrap any single line longer than the limit (e.g. one giant
        # transcript segment) so no chunk can ever exceed the embed cap.
        while len(line) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit - 1] + "…")
            line = "…" + line[limit - 1:]
        if cur and len(cur) + len(line) + 1 > limit:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def _fmt_t(seconds: float | int | None) -> str:
    try:
        m, s = divmod(int(round(float(seconds))), 60)
        return f"{m}:{s:02d}"
    except (TypeError, ValueError):
        return "?"


def build_details_pages(state: dict) -> list[discord.Embed]:
    review = state.get("review", {})
    details = state.get("details", {})
    verdict = review.get("quick_verdict", "NEEDS REVIEW")
    colour = discord.Colour(VERDICT_ACCENT.get(verdict, 0xF1C40F))
    pages: list[discord.Embed] = []

    # --- Page 1: overview + all checks -------------------------------------
    p1: list[str] = []
    if state.get("creator"):
        p1.append(f"**Creator:** {state['creator']}")
    p1.append(f"🌐 **Language:** {review.get('language_detected', 'Unknown')}")
    p1.append(f"🏁 **Verdict:** {verdict}")
    model_used = review.get("adjudicator_model_used")
    if model_used:
        p1.append(f"-# Reviewed by the multi-agent pipeline • head model: {model_used}")
    script = review.get("script_summary")
    if script:
        p1.append(f"\n📜 **Script summary:** {script}")

    findings = review.get("findings", [])
    if findings:
        p1.append("\n**All findings:**")
        for f in findings:
            sev = "FLAG" if f.get("severity") == "flag" else "suggestion"
            p1.append(f"{_finding_line(f)}  `[{f.get('risk', '?')} • {sev}]`")
    green = review.get("green_checks", [])
    if green:
        p1.append("\n**Passed checks:**")
        p1.extend(f"✅ {g}" for g in green)
    reminders = review.get("reminders", [])
    if reminders:
        p1.append("\n**Before posting:**")
        p1.extend(f"📌 {r}" for r in reminders)

    for i, chunk in enumerate(_chunk_lines(p1)):
        e = discord.Embed(title="📋 Full analysis" + (" (cont.)" if i else ""), description=chunk, colour=colour)
        e.set_footer(text="Vexi • multi-agent review • only you can see this")
        pages.append(e)

    # --- Page 2: witness + evidence facts ----------------------------------
    p2: list[str] = []
    witness_summary = review.get("witness_summary")
    if witness_summary:
        p2.append(f"🎬 **Watch-through impressions:** {witness_summary}")
    witness = details.get("witness", {})
    if isinstance(witness, dict) and witness.get("status") == "ok":
        hook = witness.get("hook", {})
        if hook:
            p2.append(f"🪝 **Hook:** {hook.get('category', '?')} — {hook.get('strength', '?')}. {hook.get('comment', '')}")
        for key, label in (
            ("pacing_dead_air", "⏱️ **Pacing**"),
            ("visual_quality_safe_zones", "💡 **Visuals/safe zones**"),
            ("product_formula", "🧪 **Product formula**"),
            ("overall_impression", "✨ **Overall**"),
        ):
            if witness.get(key):
                p2.append(f"{label}: {witness[key]}")

    det = details.get("deterministic", {})
    if det:
        plug = det.get("plug", {})
        feats = det.get("features", {})
        sites = det.get("websites", {})
        p2.append("\n**Measured facts (computed, not guessed):**")
        p2.append(f"• Manus logo on screen: {plug.get('logo_total_s', 0)}s (≥2s: {'yes' if plug.get('logo_ge_2s') else 'no'})")
        p2.append(f"• Manus UI on screen: {plug.get('manus_ui_total_s', 0)}s")
        p2.append(f"• Mentioned: spoken {'yes' if plug.get('mentioned_spoken') else 'no'} / on-screen {'yes' if plug.get('mentioned_ocr') else 'no'}")
        p2.append(f"• CTA keyword: {plug.get('cta_keyword') or 'none detected'}")
        if feats:
            names = ", ".join(h.get("feature", "?") for h in feats.get("list", [])) or "none detected"
            p2.append(f"• Manus features shown ({feats.get('count', 0)}): {names}")
        if sites:
            p2.append(f"• Distinct sites showcased: {sites.get('count', 0)} ({', '.join(sites.get('names', [])) or '—'})")

    disagreements = review.get("disagreements", [])
    if disagreements:
        p2.append("\n**Cross-check notes:**")
        for d in disagreements:
            p2.append(f"⚖️ {d.get('topic', '?')}: witness said “{d.get('witness_said', '?')}”, logs say “{d.get('logs_say', '?')}” → {d.get('resolution', '?')}")
    degraded = review.get("degraded_inputs", [])
    if degraded:
        p2.append(f"\n⚠️ Some analysis inputs were unavailable this run: {', '.join(degraded)} — a coach should take a closer look.")

    if p2:
        for i, chunk in enumerate(_chunk_lines(p2)):
            e = discord.Embed(title="🎬 Watch-through & measured evidence" + (" (cont.)" if i else ""), description=chunk, colour=colour)
            e.set_footer(text="Vexi • multi-agent review • only you can see this")
            pages.append(e)

    # --- Page 3+: transcript -------------------------------------------------
    transcript = details.get("transcript", {})
    t_lines: list[str] = []
    status = transcript.get("status") if isinstance(transcript, dict) else None
    if status in ("no_speech", "no_audio"):
        t_lines.append("🎵 No spoken dialogue in this video (music/visual-only) — speech checks were skipped, which is totally fine.")
    elif status == "ok":
        translated = transcript.get("translation_en")
        segs = translated if translated else transcript.get("segments", [])
        if translated:
            t_lines.append("_Translated to English (original language: "
                           f"{transcript.get('language_name') or transcript.get('language') or 'unknown'})_")
        for s in segs:
            text = (s.get("text") or "").strip()
            if text:
                t_lines.append(f"`[{_fmt_t(s.get('start'))}]` {text}")
        corrections = transcript.get("corrections", [])
        if corrections:
            t_lines.append("\n**Auto-caption/speech corrections applied:**")
            for c in corrections:
                t_lines.append(f"🔤 heard “{c.get('read')}” at {_fmt_t(c.get('t'))} → read as “{c.get('corrected_to')}”")
    if t_lines:
        for i, chunk in enumerate(_chunk_lines(t_lines)):
            e = discord.Embed(title="🗣️ Transcript" + (" (cont.)" if i else ""), description=chunk, colour=colour)
            e.set_footer(text="Vexi • multi-agent review • only you can see this")
            pages.append(e)

    return pages
