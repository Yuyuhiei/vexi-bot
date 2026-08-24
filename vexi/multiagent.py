"""The v3 multiagent orchestrator: download → ffmpeg → parallel extractors
(vision+OCR, ASR, full-video witness) → deterministic computations →
adjudicator (Gemini Pro with automatic Flash fallback).

Failure philosophy:
- One extractor down → the review proceeds; the adjudicator is told which
  input is UNAVAILABLE and the review is marked needs_human_review.
- Vision AND transcript both down, ffmpeg failure, or adjudicator failure →
  PipelineHardFailure, which the dispatcher catches to fall back to the
  legacy single-call pipeline (creators always get a review).
"""

import asyncio
import contextlib
import copy
import hashlib
import json
import shutil
import tempfile
import time

import aiohttp

from vexi.config import (
    GEMINI_MODEL_ADJUDICATOR,
    GEMINI_MODEL_ADJUDICATOR_FALLBACK,
    VEXI_MAX_CONCURRENT_PIPELINES,
    log,
)
from vexi.deterministic import (
    _BRAND_WORD_RE,
    compute_deterministic,
    correct_brand_homophones,
    derive_verdict,
)
from vexi.downloaders import acquire_video
from vexi.extractors import (
    run_vision_extractor,
    run_witness,
    transcribe_audio,
    translate_segments,
)
from vexi.gemini import (
    _gemini_call_and_parse,
    delete_uploaded,
    start_usage_tally,
    tally_cost,
    upload_file_and_wait,
)
from vexi.media import MediaError, extract_media
from vexi.prompts import ADJUDICATOR_PROMPT

_pipeline_sema = asyncio.Semaphore(VEXI_MAX_CONCURRENT_PIPELINES)


class PipelineHardFailure(Exception):
    """The multiagent pipeline can't produce a review — fall back to legacy."""


async def _notify(progress, stage: str) -> None:
    if progress is None:
        return
    try:
        await progress(stage)
    except Exception:
        pass


def build_adjudicator_envelope(
    creator_name: str | None,
    bundle_meta: dict,
    vision_log: dict,
    transcript: dict,
    det: dict,
    witness: dict,
) -> dict:
    def _slot(d: dict, ok_statuses: tuple) -> object:
        return d if isinstance(d, dict) and d.get("status") in ok_statuses else "UNAVAILABLE"

    return {
        "video_meta": {
            "duration_s": bundle_meta.get("duration_s"),
            "creator": creator_name or "the creator",
            "raw_frames": bundle_meta.get("raw_frame_count"),
            "kept_frames": bundle_meta.get("kept_frames"),
        },
        "vision_log": _slot(vision_log, ("ok",)),
        "transcript": _slot(transcript, ("ok", "no_speech", "no_audio")),
        "deterministic": det,
        "witness_report": _slot(witness, ("ok",)),
    }


_REVIEW_DEFAULTS: dict = {
    "schema_version": 3,
    "manus_relevant": True,
    "language_detected": "Unknown",
    "script_summary": "",
    "kudos_line": "",
    "findings": [],
    "green_checks": [],
    "reminders": [],
    "disagreements": [],
    "needs_human_review": False,
    "witness_summary": "",
    "quick_verdict": "NEEDS REVIEW",
    "overall_summary": "A human coach will review this shortly for final approval.",
}

_FINDING_DEFAULTS = {
    "category": "content",
    "severity": "recommend",
    "risk": "LOW",
    "message": "",
    "evidence_timestamps": [],
    "source": "adjudicator",
}


def validate_review(result: dict) -> dict:
    """Coerce the adjudicator output so the renderer can never KeyError, and
    let code (not the model) have the final word on the verdict."""
    # Fresh list objects every time — sharing _REVIEW_DEFAULTS' lists would
    # leak appended entries across reviews.
    review = dict(_REVIEW_DEFAULTS)
    for key in ("findings", "green_checks", "reminders", "disagreements"):
        review[key] = []
    review.update({k: v for k, v in result.items() if v is not None})

    findings = []
    for f in review.get("findings") or []:
        if not isinstance(f, dict) or not str(f.get("message", "")).strip():
            continue
        entry = dict(_FINDING_DEFAULTS)
        entry.update(f)
        entry["severity"] = str(entry["severity"]).lower()
        if entry["severity"] not in ("flag", "recommend"):
            entry["severity"] = "recommend"
        entry["risk"] = str(entry["risk"]).upper().replace("_", "-")
        if entry["risk"] not in ("AUTO-REJECT", "HIGH", "MEDIUM", "LOW"):
            entry["risk"] = "MEDIUM"
        if not isinstance(entry.get("evidence_timestamps"), list):
            entry["evidence_timestamps"] = []
        findings.append(entry)
    review["findings"] = findings

    # Coerce list contents so a sloppy model can't crash the renderer.
    for key in ("green_checks", "reminders"):
        if not isinstance(review.get(key), list):
            review[key] = []
        review[key] = [str(x) for x in review[key] if isinstance(x, (str, int, float)) and str(x).strip()]
    if not isinstance(review.get("disagreements"), list):
        review["disagreements"] = []
    review["disagreements"] = [d for d in review["disagreements"] if isinstance(d, dict)]

    model_verdict = str(review.get("quick_verdict", "")).upper()
    derived = derive_verdict(bool(review.get("manus_relevant", True)), findings)
    if model_verdict != derived:
        review["disagreements"].append({
            "topic": "verdict routing",
            "witness_said": f"adjudicator wrote {model_verdict or '(none)'}",
            "logs_say": f"findings imply {derived}",
            "resolution": "code-derived verdict wins",
        })
    review["quick_verdict"] = derived
    return review


async def run_adjudicator(envelope: dict) -> dict:
    """Head-master call with the Pro→Flash fallback ladder. The primary gets a
    short retry budget so a rate-limited Pro doesn't stall the review; hard
    model errors (quota/permission/not-found) short-circuit to the fallback."""
    contents = [
        ADJUDICATOR_PROMPT
        + "\n\nEVIDENCE ENVELOPE:\n"
        + json.dumps(envelope, ensure_ascii=False)
    ]
    ladder = [
        (GEMINI_MODEL_ADJUDICATOR, 2),
        (GEMINI_MODEL_ADJUDICATOR_FALLBACK, 3),
    ]
    # Dedupe in case both env vars point at the same model.
    seen = set()
    result: dict = {"error": "adjudicator not attempted"}
    for model, retries in ladder:
        if model in seen:
            continue
        seen.add(model)
        result = await _gemini_call_and_parse(
            contents, response_json=True, label=f"adjudicator({model})", model=model,
            retries=retries, short_circuit_hard_errors=True,
            # Rule application benefits from reasoning, but bounded: 4k thinking
            # still leaves 12k of the 16k output ceiling for the review JSON.
            thinking_budget=4096,
        )
        if "error" not in result:
            review = validate_review(result)
            review["adjudicator_model_used"] = model
            return review
        log.warning(f"adjudicator on {model} failed: {result['error'][:200]} — "
                    f"{'switching to fallback' if model != ladder[-1][0] else 'no fallback left'}")
    return result


async def run_multiagent_review(
    source_url: str,
    mime_type: str = "video/mp4",
    creator_name: str | None = None,
    progress=None,
) -> dict:
    """Full extract-then-adjudicate review of one video. Returns the v3 review
    dict (with `_details` attached for the renderer/state file), or raises
    PipelineHardFailure."""
    async with _pipeline_sema:
        return await _run(source_url, creator_name, progress)


# Exact-duplicate review cache: same video bytes + same creator → same review,
# reused for free instead of re-running the whole pipeline. Keyed on content
# hash so URL differences don't matter; creator included so kudos lines stay
# personal. In-memory only — restarts (every deploy) naturally invalidate it,
# so prompt changes never serve stale reviews.
_review_cache: dict[tuple[str, str], tuple[float, dict]] = {}
_REVIEW_CACHE_TTL_S = 24 * 3600
_REVIEW_CACHE_MAX = 40


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _review_cache_get(key: tuple[str, str]) -> dict | None:
    entry = _review_cache.get(key)
    if entry is None:
        return None
    ts, review = entry
    if time.time() - ts > _REVIEW_CACHE_TTL_S:
        _review_cache.pop(key, None)
        return None
    return copy.deepcopy(review)


def _review_cache_put(key: tuple[str, str], review: dict) -> None:
    if len(_review_cache) >= _REVIEW_CACHE_MAX:
        oldest = min(_review_cache, key=lambda k: _review_cache[k][0])
        _review_cache.pop(oldest, None)
    _review_cache[key] = (time.time(), copy.deepcopy(review))


async def _run(source_url: str, creator_name: str | None, progress) -> dict:
    usage = start_usage_tally()
    workdir = tempfile.mkdtemp(prefix="vexi_pipe_")
    local_path: str | None = None
    extra_tmp: str | None = None
    upload_holder: dict = {}
    upload_task: asyncio.Task | None = None

    async def _witness_after_upload(up_task: asyncio.Task) -> dict:
        try:
            uploaded = await asyncio.shield(up_task)
        except Exception as e:
            return {"status": "unavailable", "error": f"File API upload failed: {e}"}
        upload_holder["file"] = uploaded
        return await run_witness(uploaded)

    try:
        async with aiohttp.ClientSession() as session:
            await _notify(progress, "📥 Downloading video...")
            local_path, extra_tmp, err = await acquire_video(source_url, session)
            if not local_path:
                raise PipelineHardFailure(f"download failed: {err}")

            cache_key = (await asyncio.to_thread(_file_sha256, local_path), creator_name or "")
            cached = _review_cache_get(cache_key)
            if cached is not None:
                log.info(f"review cache hit ({cache_key[0][:12]}…) — reusing previous review, $0")
                return cached

            # File-API upload (for the witness) runs concurrently with ffmpeg.
            upload_task = asyncio.create_task(upload_file_and_wait(local_path))
            witness_task = asyncio.create_task(_witness_after_upload(upload_task))

            await _notify(progress, "🎞️ Extracting frames & audio...")
            try:
                bundle = await extract_media(local_path, workdir)
            except MediaError as e:
                witness_task.cancel()
                with contextlib.suppress(BaseException):
                    await witness_task
                raise PipelineHardFailure(f"media extraction failed: {e}")

            await _notify(progress, "🔍 Reading on-screen text & transcribing audio...")
            vision_res, transcript_res = await asyncio.gather(
                run_vision_extractor(bundle.frames),
                transcribe_audio(bundle.audio_path, session),
                return_exceptions=True,
            )
            vision_log = vision_res if isinstance(vision_res, dict) else {
                "status": "unavailable", "error": str(vision_res)[:300], "frames": [],
            }
            transcript = transcript_res if isinstance(transcript_res, dict) else {
                "status": "unavailable", "segments": [], "error": str(transcript_res)[:300],
            }

            # Without the vision log AND without any speech evidence there is
            # nothing trustworthy to adjudicate on (a music-only video with a
            # dead vision extractor would otherwise read as NOT MANUS CONTENT).
            if vision_log.get("status") != "ok" and not transcript.get("segments"):
                witness_task.cancel()
                with contextlib.suppress(BaseException):
                    await witness_task
                raise PipelineHardFailure(
                    f"vision extractor failed with no transcript evidence "
                    f"(vision: {vision_log.get('error')}, asr status: {transcript.get('status')})"
                )

            transcript = correct_brand_homophones(transcript)
            transcript = await translate_segments(transcript)

            await _notify(progress, "🎬 Full watch-through (witness pass)...")
            witness = await witness_task

            det = compute_deterministic(vision_log, transcript, bundle.duration_s)

            degraded = []
            if vision_log.get("status") != "ok":
                degraded.append("vision")
            if transcript.get("status") == "unavailable":
                degraded.append("transcript")
            if witness.get("status") != "ok":
                degraded.append("witness")

            envelope = build_adjudicator_envelope(
                creator_name,
                {
                    "duration_s": round(bundle.duration_s, 1),
                    "raw_frame_count": bundle.raw_frame_count,
                    "kept_frames": len(bundle.frames),
                },
                vision_log, transcript, det, witness,
            )

            await _notify(progress, "🧑‍⚖️ Head reviewer writing the verdict...")
            review = await run_adjudicator(envelope)
            if "error" in review:
                raise PipelineHardFailure(f"adjudicator failed on both models: {review['error'][:200]}")

            # Dead-sensor backstop (code, not prompt): a NOT MANUS verdict is
            # untrustworthy when the vision extractor was down and the witness
            # — who actually watched the video — described Manus content.
            # Absence of evidence from a failed extractor is not evidence.
            if (
                not review.get("manus_relevant", True)
                and vision_log.get("status") != "ok"
                and witness.get("status") == "ok"
                and _BRAND_WORD_RE.search(json.dumps(witness, ensure_ascii=False))
            ):
                log.warning("relevance override: vision was down but the witness describes "
                            "Manus content — NOT MANUS CONTENT verdict rejected")
                review["manus_relevant"] = True
                review["quick_verdict"] = "NEEDS REVIEW"
                review["needs_human_review"] = True
                review["kudos_line"] = ""
                review["overall_summary"] = (
                    "The frame-by-frame extractor failed this run, but the full watch-through "
                    "clearly saw Manus content — visual checks were skipped, so a coach should "
                    "give this one a proper look."
                )
                review.setdefault("disagreements", []).append({
                    "topic": "Manus relevance",
                    "witness_said": "video contains Manus content",
                    "logs_say": "vision log unavailable (extractor failure — zeros are void)",
                    "resolution": "code override: witness trusted, flagged for human review",
                })

            review["degraded_inputs"] = degraded
            if degraded:
                review["needs_human_review"] = True

            review["_details"] = {
                "transcript": transcript,
                "deterministic": det,
                "witness": witness if witness.get("status") == "ok" else {"status": witness.get("status")},
                "vision_frames": [
                    {
                        "t_start": f["t_start"], "t_end": f["t_end"],
                        "scene_description": f["scene_description"], "ocr_text": f["ocr_text"],
                    }
                    for f in vision_log.get("frames", [])
                ],
                "pipeline": "multiagent",
            }
            _review_cache_put(cache_key, review)
            return review
    finally:
        if usage["calls"]:
            log.info(f"review total: {usage['calls']} Gemini calls, in={usage['in']} "
                     f"out={usage['out']} think={usage['think']} ≈ ${tally_cost(usage):.4f}")
        await asyncio.to_thread(shutil.rmtree, workdir, ignore_errors=True)
        if extra_tmp:
            await asyncio.to_thread(shutil.rmtree, extra_tmp, ignore_errors=True)
        if local_path and not extra_tmp:
            with contextlib.suppress(Exception):
                import os
                await asyncio.to_thread(os.unlink, local_path)
        # The upload task may still be in flight (e.g. ffmpeg failed while the
        # File API upload was running) — settle it so the file is never leaked.
        uploaded = upload_holder.get("file")
        if uploaded is None and upload_task is not None:
            with contextlib.suppress(BaseException):
                uploaded = await asyncio.wait_for(asyncio.shield(upload_task), timeout=60)
        if uploaded is not None:
            await delete_uploaded(uploaded.name)
