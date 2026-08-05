"""Video source detection and downloaders (Discord CDN, Google Drive, yt-dlp,
Apify). Moved verbatim from bot.py — behavior unchanged."""

import asyncio
import os
import shutil
import re
import tempfile
from pathlib import Path

import aiohttp
import discord

from vexi.config import (
    APIFY_API_TOKEN,
    APIFY_INSTAGRAM_ACTOR,
    APIFY_TIKTOK_ACTOR,
    APIFY_USE_PROXY,
    INSTAGRAM_COOKIES_FILE,
    MAX_VIDEO_BYTES,
    log,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
GDRIVE_PATTERN = re.compile(r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)")
GDRIVE_OPEN_PATTERN = re.compile(r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)")
GDRIVE_UC_PATTERN = re.compile(r"https?://drive\.google\.com/uc\?[^ ]*id=([a-zA-Z0-9_-]+)")
GDRIVE_USERCONTENT_PATTERN = re.compile(r"https?://drive\.usercontent\.google\.com/download\?[^ ]*id=([a-zA-Z0-9_-]+)")
# On the >25MB interstitial page, Drive returns a form with hidden confirm/uuid
# inputs. Parse those and re-request to get the actual bytes.
GDRIVE_HIDDEN_INPUT_PATTERN = re.compile(r'name="([^"]+)"\s+value="([^"]+)"')
YOUTUBE_PATTERN = re.compile(r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)")
DISCORD_CDN_PATTERN = re.compile(r"https?://cdn\.discordapp\.com/attachments/")
INSTAGRAM_PATTERN = re.compile(r"https?://(www\.)?instagram\.com/(reel|p|tv)/([a-zA-Z0-9_-]+)")
TIKTOK_PATTERN = re.compile(r"https?://(www\.|vm\.|vt\.)?tiktok\.com/")


def _is_social_media_url(url: str) -> bool:
    return bool(
        INSTAGRAM_PATTERN.search(url)
        or TIKTOK_PATTERN.search(url)
        or YOUTUBE_PATTERN.search(url)
    )


async def download_with_ytdlp(url: str) -> tuple[str | None, str | None]:
    """Download a public social media video via yt-dlp. Returns (file_path, error).

    Hardened: captures yt-dlp logs, retries once with backoff, sends desktop UA,
    optionally loads cookies from INSTAGRAM_COOKIES_FILE.
    """
    import yt_dlp

    tmp_dir = tempfile.mkdtemp(prefix="vexi_study_")
    output_template = os.path.join(tmp_dir, "video.%(ext)s")

    log_buffer: list[str] = []

    class _YtdlpLogger:
        def debug(self, msg): pass
        def info(self, msg): log_buffer.append(f"[info] {msg}")
        def warning(self, msg): log_buffer.append(f"[warn] {msg}")
        def error(self, msg): log_buffer.append(f"[err] {msg}")

    ydl_opts = {
        "outtmpl": output_template,
        "format": "best[ext=mp4][filesize<100M]/best[filesize<100M]/best",
        "max_filesize": MAX_VIDEO_BYTES,
        "logger": _YtdlpLogger(),
        "retries": 3,
        "extractor_retries": 2,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "sleep_interval": 1,
        "max_sleep_interval": 5,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        },
    }
    if INSTAGRAM_COOKIES_FILE and os.path.exists(INSTAGRAM_COOKIES_FILE):
        ydl_opts["cookiefile"] = INSTAGRAM_COOKIES_FILE
        log.info(f"yt-dlp using cookie file: {INSTAGRAM_COOKIES_FILE}")

    def _download():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        files = list(Path(tmp_dir).glob("video.*"))
        if files:
            return str(files[0]), None
        return None, "Download completed but output file not found."

    loop = asyncio.get_event_loop()
    last_err: str | None = None
    for attempt in range(2):
        try:
            path, err = await loop.run_in_executor(None, _download)
            if path:
                return path, None
            last_err = err
        except Exception as e:
            last_err = str(e)
            log.warning(f"yt-dlp attempt {attempt + 1}/2 failed: {e}")
            if log_buffer:
                log.warning("yt-dlp log tail: " + " | ".join(log_buffer[-5:]))
        for f in Path(tmp_dir).glob("video.*"):
            try:
                f.unlink()
            except Exception:
                pass
        if attempt == 0:
            await asyncio.sleep(8)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return None, last_err or "yt-dlp failed for unknown reason."


async def download_with_apify(url: str, session: aiohttp.ClientSession) -> tuple[str | None, str | None]:
    """Apify fallback for Instagram/TikTok. Returns (file_path, error).

    Uses run-sync-get-dataset-items so we don't have to poll. Picks an actor
    based on URL host. Downloads the resolved video URL to a tmp file.
    """
    if not APIFY_API_TOKEN:
        return None, "APIFY_API_TOKEN not configured."

    if INSTAGRAM_PATTERN.search(url):
        actor = APIFY_INSTAGRAM_ACTOR
        actor_input = {
            "directUrls": [url],
            "resultsType": "details",
            "resultsLimit": 1,
            "addParentData": False,
        }
    elif TIKTOK_PATTERN.search(url):
        actor = APIFY_TIKTOK_ACTOR
        actor_input = {
            "postURLs": [url],
            "resultsPerPage": 1,
            "shouldDownloadVideos": False,
        }
    else:
        return None, "No Apify actor configured for this URL type."

    actor_path = actor.replace("/", "~")
    api_url = f"https://api.apify.com/v2/acts/{actor_path}/run-sync-get-dataset-items?token={APIFY_API_TOKEN}"

    log.info(f"Apify fallback: actor={actor}, url={url[:80]}")
    try:
        async with session.post(
            api_url,
            json=actor_input,
            timeout=aiohttp.ClientTimeout(total=180, connect=30),
        ) as resp:
            if resp.status not in (200, 201):
                body = await resp.text()
                return None, f"Apify HTTP {resp.status}: {body[:200]}"
            items = await resp.json()
    except Exception as e:
        return None, f"Apify request failed: {e}"

    if not isinstance(items, list) or not items:
        return None, "Apify returned no items (video may be private/deleted)."

    item = items[0]
    video_url = (
        (item.get("mediaUrls") or [None])[0]
        or item.get("videoUrl")
        or item.get("video_url")
        or (item.get("videoMeta") or {}).get("downloadAddr")
    )
    if not video_url:
        return None, f"Apify item missing video URL. Available keys: {list(item.keys())[:10]}"

    log.info(f"Apify scrape OK, downloading video: {video_url[:80]}")

    # Instagram CDN returns 403 to bare datacenter requests. Mimic a real Safari
    # request initiated from instagram.com — User-Agent, Referer, Range, and the
    # Sec-Fetch-* trio together unblock most 403s when the signed URL is fresh.
    ig_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity;q=1, *;q=0",
        "Range": "bytes=0-",
        "Referer": "https://www.instagram.com/",
        "Origin": "https://www.instagram.com",
        "Sec-Fetch-Dest": "video",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
        "Connection": "keep-alive",
    }

    proxy = None
    proxy_auth = None
    if APIFY_USE_PROXY and APIFY_API_TOKEN:
        proxy = "http://proxy.apify.com:8000"
        proxy_auth = aiohttp.BasicAuth("groups-RESIDENTIAL", APIFY_API_TOKEN)
        log.info("Routing video download through Apify residential proxy")

    tmp_dir = tempfile.mkdtemp(prefix="vexi_apify_")
    tmp_path = os.path.join(tmp_dir, "video.mp4")
    try:
        async with session.get(
            video_url,
            timeout=aiohttp.ClientTimeout(total=120, connect=30),
            headers=ig_headers,
            proxy=proxy,
            proxy_auth=proxy_auth,
            allow_redirects=True,
        ) as r:
            # Range header makes IG return 206 Partial Content — accept both.
            if r.status not in (200, 206):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                proxy_note = " (via Apify proxy)" if proxy else " (direct)"
                return None, f"Apify scrape OK but CDN download blocked: HTTP {r.status}{proxy_note}"
            total = 0
            with open(tmp_path, "wb") as fh:
                async for chunk in r.content.iter_chunked(256 * 1024):
                    fh.write(chunk)
                    total += len(chunk)
                    if total > MAX_VIDEO_BYTES:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        return None, "Apify video exceeds 100MB."
        log.info(f"Apify download complete: {total / 1024 / 1024:.1f}MB")
        return tmp_path, None
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, f"Apify video download error: {e}"


def extract_gdrive_id(url: str) -> str | None:
    """Return the Drive file ID from any recognized Drive URL shape, or None."""
    for pat in (GDRIVE_PATTERN, GDRIVE_OPEN_PATTERN, GDRIVE_UC_PATTERN, GDRIVE_USERCONTENT_PATTERN):
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


def is_gdrive_url(url: str) -> bool:
    return extract_gdrive_id(url) is not None


async def download_gdrive(url_or_id: str, session: aiohttp.ClientSession) -> tuple[str | None, str | None]:
    """Fetch a public Google Drive video to a local temp file.

    Handles the >25MB virus-scan interstitial by parsing the returned HTML for
    the confirm/uuid tokens and re-requesting with those params. Returns
    (file_path, error). File is capped at 100MB to match the rest of the pipeline.
    """
    file_id = extract_gdrive_id(url_or_id) or url_or_id
    base = "https://drive.usercontent.google.com/download"
    params: dict = {"id": file_id, "export": "download", "confirm": "t"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    tmp_path: str | None = None

    try:
        for attempt in range(2):
            async with session.get(
                base,
                params=params,
                timeout=aiohttp.ClientTimeout(total=300, connect=30),
                headers=headers,
                allow_redirects=True,
            ) as resp:
                if resp.status != 200:
                    return None, f"Drive HTTP {resp.status}"
                content_type = resp.headers.get("Content-Type", "")

                if "text/html" in content_type.lower():
                    # Either the interstitial (parse confirm token & retry) or a
                    # sign-in page (private file — surface a clean error).
                    body = await resp.text(errors="ignore")
                    lowered = body.lower()
                    if "sign in" in lowered and "google" in lowered and attempt == 0:
                        return None, "This Drive file isn't public. Set sharing to 'Anyone with the link' and try again."
                    if attempt == 1:
                        return None, "Google Drive returned an HTML page instead of the file — check that the link is public."
                    # Extract hidden form fields to resubmit
                    extracted = dict(GDRIVE_HIDDEN_INPUT_PATTERN.findall(body))
                    if not extracted:
                        return None, "Couldn't parse Drive confirmation page. Make sure the file is public."
                    for key in ("id", "export", "confirm", "uuid", "at"):
                        if key in extracted:
                            params[key] = extracted[key]
                    # Loop back and retry with new params
                    continue

                # Real bytes — pick extension from content-type
                ext = ".mp4"
                if "quicktime" in content_type:
                    ext = ".mov"
                elif "webm" in content_type:
                    ext = ".webm"
                elif "matroska" in content_type or "x-matroska" in content_type:
                    ext = ".mkv"

                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False, prefix="vexi_gdrive_")
                total = 0
                async for chunk in resp.content.iter_chunked(1024 * 256):
                    tmp.write(chunk)
                    total += len(chunk)
                    if total > MAX_VIDEO_BYTES:
                        tmp.close()
                        os.unlink(tmp.name)
                        return None, "Drive file exceeds 100MB limit."
                tmp.close()
                tmp_path = tmp.name
                log.info(f"Downloaded {total / 1024 / 1024:.1f}MB from Drive to {tmp_path}")
                return tmp_path, None

        return None, "Drive download failed after retry."
    except Exception as e:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return None, f"Drive download error: {e}"


def extract_video_source(message: discord.Message) -> str | None:
    """Return a video URL from the message (attachment or link)."""
    for att in message.attachments:
        suffix = Path(att.filename).suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return att.url

    text = message.content or ""
    file_id = None
    m = GDRIVE_PATTERN.search(text)
    if m:
        file_id = m.group(1)
    else:
        m = GDRIVE_OPEN_PATTERN.search(text)
        if m:
            file_id = m.group(1)
    if file_id:
        # Return the original-shape URL so downstream helpers can detect + route.
        return f"https://drive.google.com/file/d/{file_id}/view"

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
