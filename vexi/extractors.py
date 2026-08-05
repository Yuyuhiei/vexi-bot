"""v3 extractors: batched vision+OCR over deduplicated frames, ASR (Groq
Whisper with word timestamps, Gemini audio fallback), transcript translation,
and the full-video witness pass.

Every extractor returns a dict with a "status" key ("ok" | "no_audio" |
"no_speech" | "unavailable") — the orchestrator degrades gracefully on
anything that isn't "ok" instead of failing the review.
"""

import asyncio
import json

import aiohttp
from google.genai import types as genai_types

from vexi.config import (
    GEMINI_MODEL_TRANSLATE,
    GEMINI_MODEL_VISION,
    GEMINI_MODEL_WITNESS,
    GROQ_API_KEY,
    GROQ_ASR_MODEL,
    VEXI_ASR,
    log,
)
from vexi.gemini import _gemini_call_and_parse
from vexi.media import FrameSpan
from vexi.prompts import (
    GEMINI_ASR_PROMPT,
    MANUS_FEATURE_KEYS,
    TRANSLATE_PROMPT,
    VISION_EXTRACT_PROMPT,
    WITNESS_PROMPT,
)

VISION_FRAMES_PER_CALL = 30
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
NO_SPEECH_PROB_THRESHOLD = 0.8


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Extractor V — vision + OCR (one batched call per ≤30 frames)
# ---------------------------------------------------------------------------
def _read_frame_bytes(spans: list[FrameSpan]) -> list[bytes]:
    return [open(s.path, "rb").read() for s in spans]


def _coerce_vision_entry(raw: dict, span: FrameSpan) -> dict:
    """Validate/normalize one model-produced frame entry; timestamps always
    come from OUR span data, never from the model."""
    feat = raw.get("manus_feature")
    if feat not in MANUS_FEATURE_KEYS:
        feat = None
    return {
        "t_start": round(span.t_start, 1),
        "t_end": round(span.t_end, 1),
        "ocr_text": str(raw.get("ocr_text") or ""),
        "manus_logo_visible": bool(raw.get("manus_logo_visible")),
        "manus_ui_visible": bool(raw.get("manus_ui_visible")),
        "website_or_app_shown": raw.get("website_or_app_shown") or None,
        "inside_manus_panel": bool(raw.get("inside_manus_panel")),
        "manus_feature": feat,
        "scene_description": str(raw.get("scene_description") or ""),
        "people_present": bool(raw.get("people_present")),
    }


async def _vision_call(chunk: list[FrameSpan], first_index: int) -> list[dict]:
    frame_bytes = await asyncio.to_thread(_read_frame_bytes, chunk)
    contents: list = [VISION_EXTRACT_PROMPT]
    for i, (span, data) in enumerate(zip(chunk, frame_bytes)):
        n = first_index + i
        contents.append(
            f"FRAME {n} — represents {span.t_start:.0f}s to {span.t_end:.0f}s "
            f"({_fmt_ts(span.t_start)}–{_fmt_ts(span.t_end)}):"
        )
        contents.append(genai_types.Part.from_bytes(data=data, mime_type="image/jpeg"))

    result = await _gemini_call_and_parse(
        contents, response_json=True, label=f"vision[{first_index}..]", model=GEMINI_MODEL_VISION
    )
    if "error" in result:
        raise RuntimeError(f"vision extractor failed: {result['error'][:200]}")

    raw_frames = result.get("frames")
    if not isinstance(raw_frames, list):
        raise RuntimeError("vision extractor returned no frames array")

    # Align by "frame" number when present, else by position.
    by_number = {}
    for entry in raw_frames:
        if isinstance(entry, dict) and isinstance(entry.get("frame"), int):
            by_number[entry["frame"]] = entry
    out = []
    for i, span in enumerate(chunk):
        n = first_index + i
        raw = by_number.get(n)
        if raw is None and i < len(raw_frames) and isinstance(raw_frames[i], dict):
            raw = raw_frames[i]
        out.append(_coerce_vision_entry(raw or {}, span))
    return out


async def run_vision_extractor(frames: list[FrameSpan]) -> dict:
    """One (or a few, for long videos) batched Gemini calls over labeled,
    deduplicated frames → the per-frame vision log."""
    if not frames:
        return {"status": "unavailable", "error": "no frames extracted", "frames": []}
    try:
        chunks: list[tuple[int, list[FrameSpan]]] = []
        for start in range(0, len(frames), VISION_FRAMES_PER_CALL):
            chunks.append((start + 1, frames[start:start + VISION_FRAMES_PER_CALL]))
        # return_exceptions so a failed chunk doesn't leave siblings running
        # unobserved; partial coverage is treated as unavailable (the
        # deterministic layer must never see a silently truncated timeline).
        results = await asyncio.gather(
            *(_vision_call(chunk, first) for first, chunk in chunks),
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
        merged = [entry for part in results for entry in part]
        return {
            "status": "ok",
            "frames": merged,
            "frames_analyzed": len(merged),
        }
    except Exception as e:
        log.error(f"vision extractor unavailable: {type(e).__name__}: {e}")
        return {"status": "unavailable", "error": str(e)[:300], "frames": []}


# ---------------------------------------------------------------------------
# Extractor A — ASR
# ---------------------------------------------------------------------------
def _is_english(language: str | None) -> bool:
    if not language:
        return True  # unknown → don't attempt translation
    lang = language.strip().lower()
    return lang in ("en", "eng", "english") or lang.startswith("en-")


async def _transcribe_groq(audio_path: str, session: aiohttp.ClientSession) -> dict:
    """Groq-hosted Whisper via their OpenAI-compatible endpoint. One multipart
    POST — no SDK needed. Word + segment timestamps."""
    form = aiohttp.FormData()
    form.add_field(
        "file",
        await asyncio.to_thread(lambda: open(audio_path, "rb").read()),
        filename="audio.wav",
        content_type="audio/wav",
    )
    form.add_field("model", GROQ_ASR_MODEL)
    form.add_field("response_format", "verbose_json")
    form.add_field("timestamp_granularities[]", "word")
    form.add_field("timestamp_granularities[]", "segment")
    # Decode-biasing prompt: nudges Whisper toward the brand spelling.
    form.add_field("prompt", "Manus, manus.im, the Manus AI agent")

    async with session.post(
        GROQ_TRANSCRIPTION_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        data=form,
        timeout=aiohttp.ClientTimeout(total=120, connect=20),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Groq HTTP {resp.status}: {body[:200]}")
        data = await resp.json()

    segments = [
        {
            "start": round(float(s.get("start", 0.0)), 2),
            "end": round(float(s.get("end", 0.0)), 2),
            "text": str(s.get("text", "")).strip(),
            "no_speech_prob": float(s.get("no_speech_prob", 0.0)),
        }
        for s in (data.get("segments") or [])
    ]
    words = [
        {"w": str(w.get("word", "")).strip(), "start": round(float(w.get("start", 0.0)), 2), "end": round(float(w.get("end", 0.0)), 2)}
        for w in (data.get("words") or [])
    ]

    spoken = [s for s in segments if s["text"] and s["no_speech_prob"] < NO_SPEECH_PROB_THRESHOLD]
    if not spoken:
        return {"status": "no_speech", "backend": f"groq-{GROQ_ASR_MODEL}", "language": None, "segments": []}

    return {
        "status": "ok",
        "backend": f"groq-{GROQ_ASR_MODEL}",
        "language": str(data.get("language") or "en"),
        "granularity": "word",
        "segments": spoken,
        "words": words,
    }


async def _transcribe_gemini(audio_path: str) -> dict:
    """Fallback: Gemini audio-only transcription (~1s segment granularity)."""
    audio_bytes = await asyncio.to_thread(lambda: open(audio_path, "rb").read())
    audio_part = genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
    result = await _gemini_call_and_parse(
        [audio_part, GEMINI_ASR_PROMPT], response_json=True, label="asr(gemini)", model=GEMINI_MODEL_WITNESS
    )
    if "error" in result:
        raise RuntimeError(f"gemini ASR failed: {result['error'][:200]}")
    if result.get("no_speech"):
        return {"status": "no_speech", "backend": "gemini", "language": None, "segments": []}
    segments = [
        {
            "start": round(float(s.get("start", 0.0)), 2),
            "end": round(float(s.get("end", 0.0)), 2),
            "text": str(s.get("text", "")).strip(),
        }
        for s in (result.get("segments") or [])
        if str(s.get("text", "")).strip()
    ]
    if not segments:
        return {"status": "no_speech", "backend": "gemini", "language": None, "segments": []}
    return {
        "status": "ok",
        "backend": "gemini",
        "language": str(result.get("language") or "en"),
        "language_name": result.get("language_name"),
        "granularity": "segment",
        "segments": segments,
        "words": [],
    }


# 16kHz mono 16-bit WAV ≈ 1.9MB/min. 40MB ≈ 20+ minutes of audio — far beyond
# any UGC clip, and past both Groq's and Gemini's inline comfort zone.
MAX_AUDIO_BYTES = 40 * 1024 * 1024


async def transcribe_audio(audio_path: str | None, session: aiohttp.ClientSession) -> dict:
    """ASR facade with the full degradation chain:
    no audio stream → Groq (if configured) → Gemini → unavailable."""
    if audio_path is None:
        return {"status": "no_audio", "backend": None, "language": None, "segments": []}

    try:
        import os
        audio_size = os.path.getsize(audio_path)
    except OSError:
        audio_size = 0
    if audio_size > MAX_AUDIO_BYTES:
        log.warning(f"audio track too large for ASR ({audio_size / 1e6:.0f}MB) — skipping transcription")
        return {"status": "unavailable", "backend": None, "language": None, "segments": [],
                "error": "audio track too long to transcribe"}

    backend = VEXI_ASR
    if backend == "auto":
        backend = "groq" if GROQ_API_KEY else "gemini"

    if backend == "groq":
        if not GROQ_API_KEY:
            log.warning("VEXI_ASR=groq but GROQ_API_KEY unset — using Gemini ASR.")
        else:
            try:
                return await _transcribe_groq(audio_path, session)
            except Exception as e:
                log.warning(f"Groq ASR failed ({e}) — falling back to Gemini ASR.")

    try:
        return await _transcribe_gemini(audio_path)
    except Exception as e:
        log.error(f"All ASR backends failed: {e}")
        return {"status": "unavailable", "backend": None, "language": None, "segments": [], "error": str(e)[:300]}


async def translate_segments(transcript: dict) -> dict:
    """Translate non-English segments to English for the details view. Adds
    `translation_en`; non-fatal on failure."""
    if transcript.get("status") != "ok" or _is_english(transcript.get("language")):
        return transcript
    try:
        payload = json.dumps({"segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in transcript.get("segments", [])
        ]}, ensure_ascii=False)
        result = await _gemini_call_and_parse(
            [TRANSLATE_PROMPT + "\n\n" + payload],
            response_json=True,
            label="translate",
            model=GEMINI_MODEL_TRANSLATE,
        )
        if "error" not in result and isinstance(result.get("segments"), list):
            transcript["translation_en"] = result["segments"]
    except Exception as e:
        log.warning(f"transcript translation failed (non-fatal): {e}")
    return transcript


# ---------------------------------------------------------------------------
# Witness — full-video holistic pass
# ---------------------------------------------------------------------------
async def run_witness(uploaded_file) -> dict:
    """Full-video Gemini call with the slimmed holistic prompt. Takes an
    already-ACTIVE File API object (the orchestrator uploads once)."""
    try:
        result = await _gemini_call_and_parse(
            [uploaded_file, WITNESS_PROMPT], response_json=True, label="witness", model=GEMINI_MODEL_WITNESS
        )
        if "error" in result:
            return {"status": "unavailable", "error": result["error"][:300]}
        result["status"] = "ok"
        return result
    except Exception as e:
        log.error(f"witness unavailable: {type(e).__name__}: {e}")
        return {"status": "unavailable", "error": str(e)[:300]}
