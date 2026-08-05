"""Review pipeline dispatch.

run_legacy_review() is the consolidated v2 routing (previously duplicated in
/vexi, /study, and the auto-detect handler): Discord CDN / Drive URLs go
straight to the download-and-upload path, everything else tries the direct-URI
path with an upload fallback.

run_review() dispatches on VEXI_PIPELINE ('legacy' | 'multiagent'). The
multiagent orchestrator lands in a later increment; until then it falls back
to legacy transparently.
"""

import aiohttp

from vexi.config import VEXI_PIPELINE, log
from vexi.downloaders import DISCORD_CDN_PATTERN, is_gdrive_url
from vexi.gemini import (
    analyze_video_with_gemini,
    analyze_video_with_gemini_upload,
)


async def run_legacy_review(
    source_url: str,
    mime_type: str = "video/mp4",
    prompt: str | None = None,
    response_json: bool = False,
) -> dict:
    """The v2 single-call review routing, shared by every command."""
    is_discord_cdn = bool(DISCORD_CDN_PATTERN.match(source_url))
    is_gdrive = is_gdrive_url(source_url)

    if is_discord_cdn or is_gdrive:
        log.info(f"{'Discord CDN' if is_discord_cdn else 'Drive'} URL — using upload path directly.")
        async with aiohttp.ClientSession() as session:
            return await analyze_video_with_gemini_upload(
                source_url, session, prompt=prompt, response_json=response_json
            )

    result = await analyze_video_with_gemini(
        source_url, mime_type=mime_type, prompt=prompt, response_json=response_json
    )
    if "error" in result:
        log.info(f"Direct URL failed ({result['error']}), trying upload fallback...")
        async with aiohttp.ClientSession() as session:
            result = await analyze_video_with_gemini_upload(
                source_url, session, prompt=prompt, response_json=response_json
            )
    return result


async def run_review(
    source_url: str,
    mime_type: str = "video/mp4",
    creator_name: str | None = None,
    progress=None,
) -> tuple[dict, str]:
    """Dispatch a /vexi-style review. Returns (review_dict, pipeline_used).

    pipeline_used ∈ {"legacy", "multiagent", "legacy-fallback"} — the renderer
    picks the legacy embed for anything that isn't "multiagent". `progress` is
    an optional async callable(stage_label) for real progress updates."""
    if VEXI_PIPELINE == "multiagent":
        from vexi.multiagent import run_multiagent_review  # deferred: keeps legacy path import-light

        try:
            review = await run_multiagent_review(
                source_url, mime_type=mime_type, creator_name=creator_name, progress=progress
            )
            return review, "multiagent"
        except Exception:
            log.exception("Multiagent pipeline hard-failed — falling back to legacy for this review.")
            return await run_legacy_review(source_url, mime_type=mime_type), "legacy-fallback"
    return await run_legacy_review(source_url, mime_type=mime_type), "legacy"
