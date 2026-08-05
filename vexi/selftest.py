"""Selftest CLI — run any stage of the multiagent pipeline on a local video
file or URL without Discord. The primary dev loop for prompt/pipeline work.

    python -m vexi.selftest video.mp4 --stage media
    python -m vexi.selftest video.mp4 --stage vision --json-out /tmp/logs
    python -m vexi.selftest "https://drive.google.com/file/d/..." --stage all

Stages: probe, media, vision, asr, deterministic, witness, adjudicate, all
(each stage runs its prerequisites fresh). Requires GEMINI_API_KEY for
vision/asr/witness/adjudicate; GROQ_API_KEY optional (better ASR).
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile

import aiohttp


def _dump(json_out: str | None, name: str, data) -> None:
    print(f"\n===== {name} =====")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str)[:8000])
    if json_out:
        os.makedirs(json_out, exist_ok=True)
        with open(os.path.join(json_out, f"{name}.json"), "w") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        print(f"(saved to {json_out}/{name}.json)")


async def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m vexi.selftest")
    parser.add_argument("source", help="local video file path or URL")
    parser.add_argument("--stage", default="all", choices=[
        "probe", "media", "vision", "asr", "deterministic", "witness", "adjudicate", "all",
    ])
    parser.add_argument("--json-out", default=None, help="directory to save stage logs")
    parser.add_argument("--creator", default="selftest", help="creator name for the envelope")
    args = parser.parse_args()

    from vexi.deterministic import compute_deterministic, correct_brand_homophones
    from vexi.downloaders import acquire_video
    from vexi.extractors import run_vision_extractor, run_witness, transcribe_audio, translate_segments
    from vexi.gemini import delete_uploaded, upload_file_and_wait
    from vexi.media import extract_media, probe_video
    from vexi.multiagent import build_adjudicator_envelope, run_adjudicator

    stage = args.stage
    local_path = args.source
    extra_tmp = None
    workdir = tempfile.mkdtemp(prefix="vexi_selftest_")
    uploaded = None

    try:
        async with aiohttp.ClientSession() as session:
            if not os.path.exists(local_path):
                print(f"Downloading {args.source} ...")
                local_path, extra_tmp, err = await acquire_video(args.source, session)
                if not local_path:
                    print(f"DOWNLOAD FAILED: {err}")
                    return 1
                print(f"→ {local_path}")

            duration, has_audio = await probe_video(local_path)
            _dump(args.json_out, "probe", {"duration_s": duration, "has_audio": has_audio})
            if stage == "probe":
                return 0

            bundle = await extract_media(local_path, workdir)
            _dump(args.json_out, "media", {
                "duration_s": bundle.duration_s,
                "has_audio": bundle.has_audio,
                "raw_frame_count": bundle.raw_frame_count,
                "kept_frames": [
                    {"path": f.path, "t_start": f.t_start, "t_end": f.t_end} for f in bundle.frames
                ],
            })
            if stage == "media":
                return 0

            if stage == "witness":
                uploaded = await upload_file_and_wait(local_path)
                witness = await run_witness(uploaded)
                _dump(args.json_out, "witness", witness)
                return 0

            vision_log: dict = {"status": "skipped", "frames": []}
            transcript: dict = {"status": "skipped", "segments": []}

            if stage in ("vision", "deterministic", "adjudicate", "all"):
                vision_log = await run_vision_extractor(bundle.frames)
                _dump(args.json_out, "vision", vision_log)
                if stage == "vision":
                    return 0

            if stage in ("asr", "deterministic", "adjudicate", "all"):
                transcript = await transcribe_audio(bundle.audio_path, session)
                transcript = correct_brand_homophones(transcript)
                transcript = await translate_segments(transcript)
                _dump(args.json_out, "transcript", transcript)
                if stage == "asr":
                    return 0

            det = compute_deterministic(vision_log, transcript, bundle.duration_s)
            _dump(args.json_out, "deterministic", det)
            if stage == "deterministic":
                return 0

            uploaded = await upload_file_and_wait(local_path)
            witness = await run_witness(uploaded)
            _dump(args.json_out, "witness", witness)

            envelope = build_adjudicator_envelope(
                args.creator,
                {
                    "duration_s": round(bundle.duration_s, 1),
                    "raw_frame_count": bundle.raw_frame_count,
                    "kept_frames": len(bundle.frames),
                },
                vision_log, transcript, det, witness,
            )
            review = await run_adjudicator(envelope)
            _dump(args.json_out, "review", review)
            if "error" in review:
                print("\nADJUDICATOR FAILED")
                return 1
            print(f"\nVERDICT: {review.get('quick_verdict')}  "
                  f"(model: {review.get('adjudicator_model_used')}, "
                  f"findings: {len(review.get('findings', []))})")
            return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if extra_tmp:
            shutil.rmtree(extra_tmp, ignore_errors=True)
        elif local_path != args.source and local_path and os.path.exists(local_path):
            # Downloaded (GDrive/direct) temp file outside a tmp dir.
            try:
                os.unlink(local_path)
            except OSError:
                pass
        if uploaded is not None:
            await delete_uploaded(uploaded.name)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
