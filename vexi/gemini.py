"""Gemini client layer: retry ladder, JSON parse-and-retry, File API helpers,
and the legacy analyze entry points. Moved from bot.py; the only change is that
the model name is now a parameter (default GEMINI_MODEL_LEGACY) instead of a
hardcoded string."""

import asyncio
import json
import os
import re
import tempfile

import aiohttp
import google.genai as genai
from google.genai import types as genai_types

from vexi.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_LEGACY,
    MAX_VIDEO_BYTES,
    log,
)
from vexi.downloaders import download_gdrive, guess_mime_type, is_gdrive_url
from vexi.prompts import REVIEW_PROMPT, STUDY_PROMPT

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


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


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(code in msg for code in ("503", "429", "unavailable", "resource_exhausted", "resourceexhausted", "too many requests"))


def _is_hard_model_error(exc: Exception) -> bool:
    """Errors that won't clear on retry with the SAME model — quota exhausted,
    model not found (e.g. after the Gemini 2.5 retirement), or permission
    denied (e.g. Pro on a free-tier key). Callers with a fallback model should
    switch instead of retrying."""
    msg = str(exc).lower()
    return any(s in msg for s in ("quota", "permission", "not found", "not_found", "404", "403"))


async def _gemini_generate(contents: list, retries: int = 3, config=None, model: str | None = None) -> object:
    """Wrap the SYNCHRONOUS Gemini SDK call in asyncio.to_thread so it doesn't
    block the event loop (which would starve Discord's 3s slash-command ACK and
    cause 'The application did not respond' errors on concurrent commands)."""
    delays = [5, 15, 30]
    last_exc = None
    for attempt in range(retries):
        try:
            kwargs = {"model": model or GEMINI_MODEL_LEGACY, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            return await asyncio.to_thread(gemini_client.models.generate_content, **kwargs)
        except Exception as e:
            last_exc = e
            if _is_retryable(e) and attempt < retries - 1:
                wait = delays[attempt]
                log.warning(f"Gemini transient error (attempt {attempt + 1}/{retries}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
    raise last_exc


# Explicit output-token ceiling for JSON review calls. Truncated responses
# (finish_reason=STOP mid-string) are the main cause of "AI returned invalid
# JSON" errors — the model quietly capped itself. 16k gives every paragraph
# room to breathe including the new viral_checklist_paragraph.
GEMINI_JSON_MAX_OUTPUT_TOKENS = 16384


def _build_generate_config(
    response_json: bool, temperature: float = 0.2, media_resolution=None
) -> "genai_types.GenerateContentConfig":
    kwargs: dict = {"temperature": temperature}
    if response_json:
        kwargs["response_mime_type"] = "application/json"
        kwargs["max_output_tokens"] = GEMINI_JSON_MAX_OUTPUT_TOKENS
    if media_resolution is not None:
        kwargs["media_resolution"] = media_resolution
    return genai_types.GenerateContentConfig(**kwargs)


async def _gemini_call_and_parse(
    contents: list,
    response_json: bool,
    label: str = "gemini",
    model: str | None = None,
    retries: int = 3,
    short_circuit_hard_errors: bool = False,
    media_resolution=None,
) -> dict:
    """Call Gemini, parse the JSON reply, and retry once on parse/truncation
    failures. Returns a dict — either the parsed result or {"error": ...} with
    a creator-friendly message."""
    last_raw = ""
    last_fr = "unknown"
    for attempt in (1, 2):
        temp = 0.2 if attempt == 1 else 0.35  # nudge temp up on retry to avoid deterministic re-truncation
        config = _build_generate_config(
            response_json=response_json, temperature=temp, media_resolution=media_resolution
        )
        try:
            response = await _gemini_generate(contents, retries=retries, config=config, model=model)
        except Exception as e:
            log.error(f"{label} Gemini call failed on attempt {attempt}: {type(e).__name__}: {e}")
            if short_circuit_hard_errors and _is_hard_model_error(e):
                # Caller has a fallback model — no point re-calling this one on
                # quota/permission/not-found errors. (Legacy callers keep the
                # full two-attempt behavior: opt-in only.)
                return {"error": f"Gemini call failed: {e}", "hard_model_error": True}
            if attempt == 2:
                return {"error": f"Gemini call failed: {e}"}
            continue

        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        last_raw = raw
        try:
            last_fr = str(response.candidates[0].finish_reason)
        except Exception:
            last_fr = "unknown"

        log.info(f"{label} response length: {len(raw)} chars, finish_reason={last_fr} (attempt {attempt})")
        result = _parse_json_with_repair(raw)
        if result is not None:
            return result

        log.warning(f"{label} JSON parse failed on attempt {attempt} (finish_reason={last_fr}). Raw excerpt: {raw[:300]}")
        # fall through to retry

    log.error(f"{label} JSON parse failed after retry. finish_reason={last_fr}. Raw: {last_raw[:500]}")
    return {
        "error": (
            "The AI's reply got cut off before it could finish (finish_reason="
            f"{last_fr}). This usually clears up on a retry — please try running "
            "the command again in a moment."
        )
    }


async def upload_file_and_wait(file_path: str, max_wait: int = 120):
    """Upload a local file to the Gemini File API and poll until ACTIVE.
    Returns the uploaded file object, or raises RuntimeError on failure."""
    mime = guess_mime_type(file_path)
    log.info(f"Uploading local file to Gemini File API: {file_path} ({mime})")
    uploaded_file = await asyncio.to_thread(gemini_client.files.upload, file=file_path)
    log.info(f"Upload complete: {uploaded_file.name}, state={uploaded_file.state}")

    waited = 0
    while uploaded_file.state.name == "PROCESSING" and waited < max_wait:
        await asyncio.sleep(5)
        waited += 5
        uploaded_file = await asyncio.to_thread(gemini_client.files.get, name=uploaded_file.name)

    if uploaded_file.state.name != "ACTIVE":
        await delete_uploaded(uploaded_file.name)
        raise RuntimeError(f"File processing failed. State: {uploaded_file.state.name}")
    return uploaded_file


async def delete_uploaded(name: str) -> None:
    try:
        await asyncio.to_thread(gemini_client.files.delete, name=name)
    except Exception:
        pass


async def analyze_video_with_gemini(video_url: str, mime_type: str = "video/mp4", prompt: str = None, response_json: bool = False) -> dict:
    """Send video URL directly to Gemini for review (no local download needed).

    If response_json=True, asks Gemini to enforce JSON output via response_mime_type
    and retries once on parse failure.
    """
    if prompt is None:
        prompt = REVIEW_PROMPT
    # For the URL-fetch path we always want JSON structured output from the
    # review/study prompts. Callers pass response_json=True for /study; the
    # legacy /vexi callers relied on the model returning JSON by convention.
    # Always force JSON when the prompt is one of ours to eliminate ambiguity.
    force_json = response_json or (prompt in (REVIEW_PROMPT, STUDY_PROMPT))
    try:
        log.info(f"Sending video URL to Gemini: {video_url[:120]}... (mime={mime_type})")
        video_part = genai_types.Part.from_uri(file_uri=video_url, mime_type=mime_type)
        log.info("Calling Gemini for review...")
        result = await _gemini_call_and_parse(
            contents=[video_part, prompt],
            response_json=force_json,
            label="review(url)",
        )
        if "error" not in result:
            log.info(f"Review complete. Verdict: {result.get('quick_verdict', 'N/A')}")
        return result
    except Exception as e:
        log.error(f"Gemini error: {type(e).__name__}: {e}")
        import traceback
        log.error(traceback.format_exc())
        return {"error": str(e)}


async def _analyze_local_file_with_gemini(file_path: str, prompt: str = None, response_json: bool = False) -> dict:
    """Upload a local file to Gemini File API and analyze it.

    Uses the shared JSON parse + retry helper. Deletes the uploaded file when done.
    """
    if prompt is None:
        prompt = REVIEW_PROMPT
    force_json = response_json or (prompt in (REVIEW_PROMPT, STUDY_PROMPT))
    uploaded_file = None
    try:
        uploaded_file = await upload_file_and_wait(file_path)

        result = await _gemini_call_and_parse(
            contents=[uploaded_file, prompt],
            response_json=force_json,
            label="review(upload)",
        )

        await delete_uploaded(uploaded_file.name)

        return result
    except Exception as e:
        log.error(f"Local file Gemini upload error: {type(e).__name__}: {e}")
        if uploaded_file is not None:
            await delete_uploaded(uploaded_file.name)
        return {"error": str(e)}


async def analyze_video_with_gemini_upload(video_url: str, session: aiohttp.ClientSession, prompt: str = None, response_json: bool = False) -> dict:
    """Fallback: Download video and upload to Gemini File API."""
    if prompt is None:
        prompt = REVIEW_PROMPT
    video_path = None
    try:
        # Drive links: use the dedicated helper that handles the >25MB interstitial.
        if is_gdrive_url(video_url):
            gdrive_path, gdrive_err = await download_gdrive(video_url, session)
            if not gdrive_path:
                return {"error": gdrive_err or "Drive download failed."}
            video_path = gdrive_path
            log.info(f"Drive download complete → {video_path}")
            return await _analyze_local_file_with_gemini(video_path, prompt, response_json=response_json)

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
                if total > MAX_VIDEO_BYTES:
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
