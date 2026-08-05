"""Shared progress-bar animation for review/study messages.

Replaces the four identical update_progress closures that used to live in
process_single_review, /study, the mention handler, and the auto-detect
handler. Behavior is unchanged: edit through the stage list, 15s per stage,
stop as soon as .finish() is called or an edit fails."""

import asyncio

import discord

REVIEW_STAGES = [
    ("▓▓▁▁▁▁▁▁▁▁", "🧠 AI is watching your video..."),
    ("▓▓▓▓▁▁▁▁▁▁", "🧠 AI is watching your video..."),
    ("▓▓▓▓▓▁▁▁▁▁", "🔍 Checking legal compliance..."),
    ("▓▓▓▓▓▓▁▁▁▁", "🔍 Checking legal compliance..."),
    ("▓▓▓▓▓▓▓▁▁▁", "🎬 Reviewing hooks & storytelling..."),
    ("▓▓▓▓▓▓▓▓▁▁", "🎬 Reviewing hooks & storytelling..."),
    ("▓▓▓▓▓▓▓▓▓▁", "📝 Writing up the review..."),
    ("▓▓▓▓▓▓▓▓▓▓", "✅ Almost done..."),
]

STUDY_STAGES = [
    ("▓▓▁▁▁▁▁▁▁▁", "🎬 Watching the video..."),
    ("▓▓▓▓▁▁▁▁▁▁", "🎬 Watching the video..."),
    ("▓▓▓▓▓▁▁▁▁▁", "🔍 Breaking down the format..."),
    ("▓▓▓▓▓▓▁▁▁▁", "🔍 Analyzing hook & structure..."),
    ("▓▓▓▓▓▓▓▁▁▁", "🔄 Building Manus adaptation brief..."),
    ("▓▓▓▓▓▓▓▓▁▁", "🔄 Building Manus adaptation brief..."),
    ("▓▓▓▓▓▓▓▓▓▁", "📋 Writing the outline..."),
    ("▓▓▓▓▓▓▓▓▓▓", "✅ Almost done..."),
]


class ProgressAnimation:
    """Animates a progress message through cosmetic stages until finished."""

    def __init__(self, message: discord.Message, header: str, stages: list[tuple[str, str]], stage_seconds: int = 15):
        self._message = message
        self._header = header
        self._stages = stages
        self._stage_seconds = stage_seconds
        self._done = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        for bar, status in self._stages:
            if self._done.is_set():
                return
            try:
                await self._message.edit(content=f"{self._header}\n{bar} {status}")
            except Exception:
                return
            try:
                await asyncio.wait_for(self._done.wait(), timeout=self._stage_seconds)
                return
            except asyncio.TimeoutError:
                continue

    async def finish(self) -> None:
        """Signal completion and wait for the animation task to exit."""
        self._done.set()
        if self._task is not None:
            try:
                await self._task
            except Exception:
                pass
