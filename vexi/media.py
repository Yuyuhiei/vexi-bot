"""Local media extraction for the multiagent pipeline.

One ffprobe pass (duration + audio presence), then ONE ffmpeg invocation that
produces both the 1fps frame set and a 16kHz mono WAV (decode once — matters on
a single shared vCPU). Frames are then deduplicated with a perceptual hash so
the vision extractor only sees visually distinct moments, with each kept frame
carrying the time span it represents.

All subprocess work runs via asyncio.create_subprocess_exec with a hard
timeout; pHash runs in a thread. Nothing here blocks the event loop.
"""

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from vexi.config import (
    VEXI_FFMPEG_TIMEOUT_S,
    VEXI_FRAME_FPS,
    VEXI_FRAME_MAX_W,
    VEXI_MAX_FRAMES,
    VEXI_PHASH_THRESHOLD,
    log,
)

# Frames in the first HOOK_KEEP_SECONDS are never dropped by dedupe — the hook
# check needs full temporal resolution at the start of the video.
HOOK_KEEP_SECONDS = 4.0


class MediaError(Exception):
    """ffprobe/ffmpeg failed — the multiagent pipeline cannot run on this file."""


@dataclass
class FrameSpan:
    path: str
    t_start: float
    t_end: float
    phash: str = ""


@dataclass
class MediaBundle:
    workdir: str
    video_path: str
    duration_s: float
    has_audio: bool
    frames: list[FrameSpan] = field(default_factory=list)
    audio_path: str | None = None
    raw_frame_count: int = 0

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


async def _run_subprocess(args: list[str], timeout: int) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await proc.communicate()
        except Exception:
            pass
        raise MediaError(f"{args[0]} timed out after {timeout}s")
    return proc.returncode, stdout, stderr


async def probe_video(video_path: str) -> tuple[float, bool]:
    """Return (duration_seconds, has_audio_stream) via ffprobe."""
    rc, stdout, stderr = await _run_subprocess(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            video_path,
        ],
        timeout=30,
    )
    if rc != 0:
        raise MediaError(f"ffprobe failed (rc={rc}): {stderr.decode(errors='ignore')[:300]}")
    try:
        info = json.loads(stdout.decode(errors="ignore"))
        duration = float(info.get("format", {}).get("duration") or 0.0)
        has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    except (ValueError, KeyError) as e:
        raise MediaError(f"ffprobe output unparsable: {e}")
    if duration <= 0:
        raise MediaError("ffprobe reported zero/unknown duration")
    return duration, has_audio


def _phash_dedupe(
    frame_paths: list[str],
    fps: float,
    duration_s: float,
    threshold: int,
) -> tuple[list[FrameSpan], int]:
    """Blocking: hash every frame, keep only frames that differ from the last
    KEPT frame by more than `threshold` Hamming distance. Frames inside the
    hook window are always kept. Returns (kept_spans, raw_count).

    Runs inside asyncio.to_thread — Pillow opens one image at a time so peak
    RAM stays tiny."""
    import imagehash
    from PIL import Image

    kept: list[FrameSpan] = []
    last_hash = None
    for i, path in enumerate(frame_paths):
        t_start = i / fps
        try:
            with Image.open(path) as img:
                h = imagehash.phash(img)
        except Exception:
            continue
        in_hook = t_start <= HOOK_KEEP_SECONDS
        is_distinct = last_hash is None or (h - last_hash) > threshold
        if in_hook or is_distinct:
            kept.append(FrameSpan(path=path, t_start=t_start, t_end=t_start, phash=str(h)))
            last_hash = h
        # dropped frames simply extend the previous kept frame's span (fixed below)

    # Fix up spans: each kept frame represents [its t_start, next kept t_start)
    for j in range(len(kept) - 1):
        kept[j].t_end = kept[j + 1].t_start
    if kept:
        kept[-1].t_end = max(duration_s, kept[-1].t_start + 1.0 / fps)
    return kept, len(frame_paths)


def _subsample(frames: list[FrameSpan], cap: int) -> list[FrameSpan]:
    """Uniformly subsample kept frames down to `cap`, merging spans so timing
    information survives. First and last frames are always kept."""
    if len(frames) <= cap:
        return frames
    step = (len(frames) - 1) / (cap - 1)
    picked_idx = sorted({round(i * step) for i in range(cap)})
    picked = [frames[i] for i in picked_idx]
    for j in range(len(picked) - 1):
        picked[j].t_end = picked[j + 1].t_start
    picked[-1].t_end = frames[-1].t_end
    return picked


async def extract_media(video_path: str, workdir: str) -> MediaBundle:
    """Probe, extract frames + audio in one ffmpeg run, dedupe frames."""
    duration_s, has_audio = await probe_video(video_path)

    frames_dir = Path(workdir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(Path(workdir) / "audio.wav")

    fps = float(VEXI_FRAME_FPS)
    vf = f"fps={fps},scale='min({VEXI_FRAME_MAX_W},iw)':-2"
    args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", vf, "-q:v", "3", str(frames_dir / "f_%04d.jpg"),
    ]
    if has_audio:
        args += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", audio_path]

    rc, _stdout, stderr = await _run_subprocess(args, timeout=VEXI_FFMPEG_TIMEOUT_S)
    if rc != 0:
        raise MediaError(f"ffmpeg failed (rc={rc}): {stderr.decode(errors='ignore')[:300]}")

    frame_paths = sorted(str(p) for p in frames_dir.glob("f_*.jpg"))
    if not frame_paths:
        raise MediaError("ffmpeg produced no frames")

    kept, raw_count = await asyncio.to_thread(
        _phash_dedupe, frame_paths, fps, duration_s, VEXI_PHASH_THRESHOLD
    )
    if len(kept) > VEXI_MAX_FRAMES:
        # Long / visually busy video — dedupe harder, then uniform-subsample.
        harder, _ = await asyncio.to_thread(
            _phash_dedupe, frame_paths, fps, duration_s, VEXI_PHASH_THRESHOLD + 4
        )
        kept = _subsample(harder, VEXI_MAX_FRAMES)

    log.info(
        f"media: {duration_s:.1f}s video → {raw_count} raw frames → {len(kept)} kept "
        f"(audio={'yes' if has_audio else 'no'})"
    )
    return MediaBundle(
        workdir=workdir,
        video_path=video_path,
        duration_s=duration_s,
        has_audio=has_audio,
        frames=kept,
        audio_path=audio_path if has_audio else None,
        raw_frame_count=raw_count,
    )
