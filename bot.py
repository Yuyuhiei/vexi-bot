"""
Vexi — AI-Powered UGC Video Review Discord Bot
Derived from Latin "vexillum" (flag). Vexi flags potential issues in UGC videos
so human coaches can make the final call.

Entrypoint only — the implementation lives in the vexi/ package:
  vexi/config.py        env vars, model names, VEXI_PIPELINE flag
  vexi/prompts.py       all Gemini prompts
  vexi/gemini.py        Gemini client layer (retry, JSON parse, File API)
  vexi/downloaders.py   Discord CDN / GDrive / yt-dlp / Apify downloaders
  vexi/pipeline.py      review pipeline dispatch (legacy | multiagent)
  vexi/render.py        Discord message builders
  vexi/commands.py      slash commands, mention handler, auto-detect, queue

Triggers:
  1. /vexi slash command (works in any channel)
  2. Auto-detect video posts in configured channels (VEXI_CHANNELS env var)
"""

from vexi.config import DISCORD_BOT_TOKEN, GEMINI_API_KEY, log
from vexi.commands import bot

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        log.error("DISCORD_BOT_TOKEN not set!")
        exit(1)
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set!")
        exit(1)
    bot.run(DISCORD_BOT_TOKEN)
