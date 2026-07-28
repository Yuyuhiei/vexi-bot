"""
Vexi — AI-Powered UGC Video Review Discord Bot
Derived from Latin "vexillum" (flag). Vexi flags potential issues in UGC videos
so human coaches can make the final call.

Triggers:
  1. /vexi slash command (works in any channel)
  2. Auto-detect video posts in configured channels (VEXI_CHANNELS env var)
"""

import os
import io
import json
import re
import shutil
import tempfile
import time
import asyncio
import logging
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands
import aiohttp
import google.genai as genai
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
VEXI_CHANNELS_RAW = os.environ.get("VEXI_CHANNELS", "")

VEXI_CHANNEL_IDS: list[int] = []
if VEXI_CHANNELS_RAW.strip():
    VEXI_CHANNEL_IDS = [int(ch.strip()) for ch in VEXI_CHANNELS_RAW.split(",") if ch.strip()]

# Guild ID for instant slash command sync (set via env var or auto-detected)
GUILD_ID = os.environ.get("GUILD_ID", "")

# Coach role ID for auto-detect channel pings (optional)
COACH_ROLE_ID = os.environ.get("COACH_ROLE_ID", "")

# Apify fallback (used by /study only when yt-dlp fails on Instagram/TikTok)
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
APIFY_INSTAGRAM_ACTOR = os.environ.get("APIFY_INSTAGRAM_ACTOR", "apify/instagram-scraper")
APIFY_TIKTOK_ACTOR = os.environ.get("APIFY_TIKTOK_ACTOR", "clockworks/free-tiktok-scraper")
# Set APIFY_USE_PROXY=true to route the video-bytes download through Apify's
# residential proxy. Costs Apify proxy bandwidth (~$8/GB) but bypasses Instagram
# CDN's datacenter-IP block that returns 403 on raw downloads.
APIFY_USE_PROXY = os.environ.get("APIFY_USE_PROXY", "").strip().lower() in ("1", "true", "yes", "on")
INSTAGRAM_COOKIES_FILE = os.environ.get("INSTAGRAM_COOKIES_FILE", "")

# Last-resort fallback when both yt-dlp and Apify can't fetch the video
LUMISCRIPT_URL = "https://lumiscript.manus.space/"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vexi")

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Manus knowledge base — injected into review + study + revise prompts so the
# model has ground truth about the product it's evaluating.
# ---------------------------------------------------------------------------
MANUS_KNOWLEDGE = """
MANUS GROUND TRUTH (use this to verify what you actually see on screen):

- Product: Manus (manus.im) — a general-purpose AI agent that autonomously
  executes multi-step tasks in a cloud workspace. It plans, uses tools, and
  produces artifacts end-to-end without step-by-step user guidance.

- Core capabilities creators typically demo:
  • Autonomous web research (opens sites, reads pages, compiles reports)
  • Browser automation (fills forms, navigates flows, scrapes data)
  • Document, spreadsheet, and slide generation (PDF, DOCX, XLSX, PPTX)
  • Website and landing-page builds (HTML/CSS/JS, sometimes deployed live)
  • App / prototype building (small full-stack apps, dashboards)
  • Image generation and editing
  • Data analysis, cleaning, and visualization
  • Long-running background tasks the user can check on later

- UI signals (any of these strongly indicate a Manus video):
  • Left sidebar listing task threads / previous runs
  • A live task-execution panel showing a virtual browser, terminal, or
    file tree while the agent works
  • A chat input where the user gives Manus a high-level instruction
  • The word "Manus" as wordmark, tab title, or in the URL bar (manus.im)
  • Dark blue / near-black palette; clean minimal chrome

- Spoken / text signals: "manus.im", "Manus", "the Manus agent",
  "I asked Manus to...", "Manus built this for me", "let Manus do it"

- Correct spellings ONLY: "Manus" and "manus.im".
  Wrong (flag as [HIGH] misspelling): Manis, Mannus, Maus, Mauns, manis.im,
  mannus.im, and any other variant. The domain is manus.im — never .com/.io
  unless the creator is explicitly referencing a different product.
  Auto-caption homophones to catch (CapCut/TikTok/Reels often mishear "Manus"):
  Manners, Menace, Man is, Manis, Manas, Manuscript, Manor, Menus, Minus.
  These are real English words, so judge by SENTENCE CONTEXT — if the sentence
  only makes sense with "Manus" swapped back in, it's a caption error, flag it.
"""

# ---------------------------------------------------------------------------
# The Core AI Prompt — Compact Conversational Output
# ---------------------------------------------------------------------------
REVIEW_PROMPT = MANUS_KNOWLEDGE + r"""
You are Vexi, a friendly, non-authoritative AI assistant helping human coaches review UGC videos for Manus. Your job is to flag potential issues for the human coach to make the final decision. You are NOT the approver — you are a helpful first-pass flagger.

═══════════════════════════════════════
LAYER 0: MANUS RELEVANCE — CHAIN OF THOUGHT (RUN BEFORE DECIDING)
═══════════════════════════════════════
Do NOT decide manus_relevant until you have walked through these three steps. Bias STRONGLY toward relevant — if ANY signal exists at Step 1, this is a Manus video.

STEP 1 — First Manus signal. Scan the entire video for the FIRST moment any of the following appears, and note the timestamp and signal type:
  (a) The Manus wordmark or logo (top-left, footer, watermark, or anywhere on screen — even tiny)
  (b) Spoken mention of "Manus", "manus.im", "Manus agent", or a common misspelling ("Manis", "Mannus", etc.)
  (c) On-screen text or captions referencing Manus or manus.im
  (d) The Manus web app UI (sidebar of task threads, task-execution panel with live browser/terminal, chat prompt input, dark blue/black palette matching MANUS GROUND TRUTH)
  (e) A URL bar or browser tab showing manus.im
If NONE of (a)–(e) appears anywhere, note "no Manus signal found".

STEP 2 — Manus scene description. If Step 1 found a signal, describe in one sentence WHAT is happening in the Manus portion: what task is the creator giving the agent? What is the agent producing? Is the UI visible?

STEP 3 — Manus features shown. Enumerate every Manus capability visible in the video (research report, browser automation, doc/slide creation, website build, data work, image generation, etc.). Refer to MANUS GROUND TRUTH.

NOW decide "manus_relevant":
- If Step 1 found ANY signal (even a 1-second logo flash, or a single spoken mention), set "manus_relevant": true and proceed with the full review.
- Only set "manus_relevant": false if Steps 1, 2, and 3 are all empty — i.e. the video contains zero Manus visuals, zero mentions, and zero features. When in doubt, choose TRUE. A missed Manus video wastes a coach's time; a false-flag "NOT MANUS" wastes the creator's.

If "manus_relevant" is false:
- Set "quick_verdict" to "NOT MANUS CONTENT"
- Set "legal_paragraph" and "content_paragraph" and "manus_plug_paragraph" to empty strings
- In "overall_summary" write: "I couldn't find any Manus mention, logo, UI, or feature in this video. If you think I'm wrong, tag your coach for a manual review."

Include your Step 1/2/3 findings implicitly when writing manus_plug_paragraph — the coach benefits from seeing the exact timestamp of the first Manus signal.

═══════════════════════════════════════
LAYER 0.5: MONEY & INCOME CLAIM SCAN (v1.3 — RUN BEFORE ALL OTHER CHECKS)
═══════════════════════════════════════
Scan ALL spoken audio, on-screen text, captions, and visuals for money and income language. Use the tiers below — not everything money-adjacent is an auto-reject.

IMPORTANT — IGNORE COMPLETELY (do not flag):
- Song lyrics or audio slang referencing money ("getting my paper wet", "money on my mind", etc.) — these are background music/culture, not claims
- Productivity or client-work framing with no income claim: "I use Manus to deliver client projects faster", "helps me take on more clients", "saves me time"
- Tool capability demos with no income angle: building a Shopify store, e-commerce site, or app to show Manus features — fine even if the tool could theoretically be used to earn money

─────────────────────────────────────
AUTO-REJECT — set "quick_verdict" to "AUTO-REJECT" if ANY of these are present:
─────────────────────────────────────
1. Explicit personal income claims (with or without numbers):
   - "I made $5k with Manus" / "I earned ₱10,000" / "I make $2,000/month"
   - "this replaced my 9-5" / "I quit my job because of Manus" / "this is my full-time income"
   - "I make a full-time living with this" / "this pays my bills"

2. Third-party earnings claims:
   - "My friend made $3,000 using Manus" / "creators are earning $X with this"

3. Cost-savings with a dollar amount:
   - "I saved $500 by using Manus instead of hiring a freelancer"
   - Any specific currency amount framed as money saved or replaced

4. Banned income phrases regardless of context:
   - "passive income" / "financial freedom" / "get rich quick" / "easy money" / "guaranteed income" / "make you rich" / "zero risk"

5. On-screen revenue proof as the focus:
   - Showing a Shopify dashboard, YouTube analytics, or any earnings screenshot where real revenue numbers are the point being made

─────────────────────────────────────
MEDIUM RISK [MEDIUM] — flag for coach review, not auto-reject:
─────────────────────────────────────
- Aspirational/goal language with numbers but no claim: "I'm working toward my $10k month" / "my goal is $5k"
- Revenue dashboards or analytics accidentally visible in the background but clearly not the focus of the video

─────────────────────────────────────
If AUTO-REJECT is triggered:
- In "legal_paragraph", name the exact phrase(s) with an [AUTO-REJECT] tag and timestamp if possible
- Still complete all remaining checks below (the coach needs the full picture)
- End "overall_summary" with: "If you think this is a mistake, please tag your coach for a manual review."

═══════════════════════════════════════
WHAT TO CHECK (Internal — use these to inform your paragraphs)
═══════════════════════════════════════

LEGAL COMPLIANCE CHECKS (v1.2 Checklist):
1. Income & Money Claims — See LAYER 0.5 above. Follow the three-tier system: AUTO-REJECT for explicit claims, third-party earnings, cost-saving amounts, banned phrases, and on-screen revenue proof; [MEDIUM] for aspirational goal language or background dashboards; ignore song lyrics, slang, and productivity framing entirely. (HIGH RISK / AUTO-REJECT)
2. Absolute Claims — Phrases like "100%", "zero errors", "fully replaces humans", "best AI" without proof. (HIGH RISK)
3. Efficiency Numbers Without Proof — Time-saved or speed claims (e.g., "build a site in 10 minutes", "save 5 hours") require real supporting data or evidence. Flag if none is visible. (MEDIUM RISK)
4. Copyrighted / Trademarked Material — UGC is a paid advertisement, so any famous brand, logo, celebrity, or copyrighted character that the creator does not own CANNOT be published as-is. This includes: brand logos (Nike, Adidas, Apple, etc.), copyrighted characters (Disney, Marvel, anime, Iron Man, etc.), celebrity names/images/likenesses (e.g. David Goggins, Robert Downey Jr.), and protected event branding (FIFA World Cup, Olympics, named teams/players/jersey numbers). When you do flag, name the exact element and timestamp and tell the creator it must be REMOVED — they can regenerate any AI image with a prompt that omits the logo/character/likeness.
   ANTI-HALLUCINATION RULES (CRITICAL — read before flagging anything as copyrighted):
   - Only flag a specific brand/character/celebrity when it is CLEARLY and UNAMBIGUOUSLY identifiable. Cite the concrete visual evidence you actually see (readable logo text, an exact name on screen, a truly distinctive and unmistakable design).
   - Do NOT name-match. A generic yellow square character is NOT automatically SpongeBob; generic round candy is NOT automatically M&M's; a muscular bald man is NOT automatically David Goggins. Original, generic, or merely similar-looking characters are FINE and must NOT be flagged.
   - If something only RESEMBLES a known IP but you are not confident, do NOT assert the IP. Either say nothing, or note it softly as [MEDIUM] "this generic character may read as similar to <X> — worth a human glance" without claiming it IS that IP.
   - When uncertain, default to NOT flagging. A false copyright flag is worse than a missed borderline one, because a coach reviews everything anyway.
   EXCEPTIONS (do NOT flag): a copyrighted character appearing incidentally in the background or on a desktop screen for under 2 seconds; a creator simply wearing a branded jersey or shirt (e.g. an Adidas tee) as everyday clothing.
   NEVER FLAG these categories — they are not creator-published IP use:
   - Brand names inside a browser tab, URL bar, address bar, or app UI chrome (e.g. a Chrome tab reading "nike.com" is not a Nike endorsement — it's a URL)
   - Brand names or logos appearing INSIDE the Manus agent's browser or task-execution panel — Manus is USING the site to complete a task, that's not brand placement by the creator
   - Logos smaller than roughly 5% of the frame that sit incidentally in the background (a laptop sticker, a distant sign)
   - Generic English words that happen to match brand names (e.g. "Apple" as fruit, "Nike" as the Greek goddess, "Amazon" as the river)
   - Product names visible only on a laptop or phone screen that is displaying a normal website in the ordinary course of the video
   - Search-engine result pages listing many brand names as text — that's search UI, not endorsement
   When in doubt, do not flag. (HIGH RISK)
5. Fake Reviews or Testimonials — Actors scripting fake customer stories, fake "first-time" reactions. (HIGH RISK)
6. Exaggerated or Unproven Claims — Unprovable numerical claims beyond income (e.g., "10x your revenue"). Personal honest experiences without guarantees are fine. (MEDIUM RISK)
7. People Without Permission — Identifiable bystanders, friends, or children without release. (MEDIUM RISK)
8. Competitor Logos or Products — Visible competitor logos, mocking competitors. (MEDIUM RISK)
9. AI-Generated Faces or Voices — AI-generated people used as testimonials without disclosure. (MEDIUM RISK)
10. Privacy Claims — Any "data security/privacy" statements must match Manus's privacy policy; flag vague or absolute privacy promises. (MEDIUM RISK)
11. Product Demo Accuracy — Only show Manus UI/features that exist and work in the current version. Flag if demo shows non-existent or unshippable features. (MEDIUM RISK)
12. Real-Person Likeness — AI-generated faces or voices resembling real people without authorization. (MEDIUM RISK)
13. Font Licensing — Premium fonts without commercial license. Google Fonts are fine. (LOW RISK)
14. Filming Locations — Inside recognizable branded private spaces. (LOW RISK)
15. Platform Rules — Missing branded content toggles. (LOW RISK)
16. Cultural Sensitivity — Stereotypes, accents as jokes, religious/political imagery. (LOW RISK)
17. Music — If you hear music, give a soft reminder to confirm it's from TikTok/IG library or approved royalty-free source. NEVER flag music as a risk.
18. Ad Disclosure — Remind creators to include at least one ad-disclosure hashtag (#ManusAd, #ManusPartner, #Ad, #Sponsored) in their caption when posting. Generic tags like #Manus alone are NOT enough. NEVER suggest putting hashtags on the video itself. NEVER flag as a risk.
19. Website / Tool Showcase Cap — Listicle-style videos ("5 websites you need", "top AI tools", "sites that will change your workflow", etc.) that feature multiple websites or apps alongside Manus are FINE, but total distinct sites/tools shown (INCLUDING Manus itself) must be 5 or fewer. This keeps Manus from getting lost in the list. Count each named site/tool once. Examples:
    - refero.design + motion.dev + animejs + skiper-ui.com + manus.im = 5 total → OK.
    - 4 external sites + Manus = 5 → OK. 3 + Manus = 4 → OK.
    - 6 or more distinct sites/tools featured (whether or not Manus is one of them) → EDIT REQUIRED [HIGH]: tell the creator to trim the list to 5 max and keep Manus as one of them. Name every site you counted.
    Do NOT count: sites that only appear inside the Manus agent's browser as part of a task Manus is performing (Manus is using them, not showcasing them); a browser tab bar showing unrelated tabs; search-engine results pages. Only count sites the creator explicitly presents, points to, or narrates as an item in the list. (HIGH RISK when 6+)

MANUS PLUG & BRAND PRESENCE CHECKS (a weak or missing plug is a potential rejection — verdict COACH ATTENTION NEEDED). Report these in the "manus_plug_paragraph" field. The four EDIT-REQUIRED cases below must each be flagged [HIGH] when present:
1. Clear Manus Mention — Manus must be clearly mentioned in the video, spoken or as on-screen text. EDIT REQUIRED if there is NO in-video mention of Manus. (HIGH RISK)
2. Manus Logo Present — A Manus logo should appear (a small/tiny logo at the bottom is acceptable). EDIT REQUIRED if there is NO Manus logo anywhere. (HIGH RISK)
3. Logo on Interface — If the Manus interface is shown but there is NO Manus logo on screen, EDIT REQUIRED — tell them to add a logo. (HIGH RISK)
4. CTA Present — The video should end with a clear CTA that ties back to Manus, e.g. "comment 'Manus' for the tool" or "comment 'PROMPT' and I'll send you the exact one". EDIT REQUIRED if no CTA is present. (HIGH RISK)
5. Correct Spelling — Check the brand name is spelled correctly everywhere it appears, in captions, on-screen text, and the website/domain. The brand is "Manus" and the site is "manus.im". EDIT REQUIRED if it's misspelled, e.g. "Mauns", "Manis", "Manus.im" typo'd as "manis.im", "Mannus", "Maus", or any wrong domain. Name the exact misspelling and where it appears. (HIGH RISK)
   AUTO-CAPTION HOMOPHONES — Read burned-in captions carefully (CapCut / TikTok / Reels auto-captions often mishear "Manus"). Common wrong words to watch for: "Manners", "Menace", "Man is", "Manis", "Manas", "Manuscript", "Manor", "Menus", "Minus". These are REAL words, so the test is CONTEXT: does the sentence make sense with the real word, or does it only make sense if you swap "Manus" back in? Example: "I asked Manners to build me a website" — Manners doesn't fit; this is an auto-caption error, flag as [HIGH] "captions transcribed 'Manus' as 'Manners' at 0:07 — fix the captions". Do NOT flag when the real word fits ("mind your manners"). Cite the exact sentence and timestamp.
Also: the Manus interface/website/logo should ideally be on screen for at least ~4 seconds total — note it if it only flashes briefly. A pure low-effort plug (e.g. just a 2-second "made with Manus" text card with no real demo) is [HIGH] and routes to COACH ATTENTION NEEDED.

UGC FUNDAMENTALS CHECKS:
1. Safe Zones — Critical text/face in bottom 350px (caption area) or top 250px (UI overlay)?
2. Lighting & Audio — Face visible? Audio clear? No echo/background noise?
3. Hook & Storytelling — Evaluate the first 3 seconds against these 12 hook categories:
   (1) Curiosity / "Feels Illegal to Know"  (2) Challenge / Speed Run  (3) Before & After / Transformation
   (4) Hot Take / Controversial / Pattern Interrupt  (5) Demo / How-To (Punchy Openers)
   (6) Social Proof / Flex / Authority  (7) Skits  (8) News & Presentation  (9) FOMO / Urgency
   (10) Anti-Hook / Reverse Psychology  (11) Comparison / "This vs. That"  (12) Emotional / Relatable
4. Pacing & Dead Air — Awkward silences, long pauses, or dead space?

═══════════════════════════════════════
MANUS VIRAL VIDEO CHECKLIST (v1.0 — from the official Viral Video Checklist)
═══════════════════════════════════════
This is the official checklist the coach applies before approving a viral-format Manus video. Any missing item that is testable from the video itself must be flagged as [VIRAL] and routed to COACH ATTENTION NEEDED. Report all findings in the "viral_checklist_paragraph" field. Skip items that can only be checked off-video (like ManyChat setup) — just remind the creator to confirm those before posting.

SECTION 1 — TEXT HOOK (first 3 seconds).
At least ONE on-screen text hook must exist. Acceptable hook types:
  a) Curiosity or Shock — text that makes the viewer stop scrolling
  b) FOMO / Loss Aversion — e.g. "The colleague dumber than you is getting promoted — because he uses AI and you don't"
  c) Identity / ICP Match — e.g. "Are you an interior designer still burning midnight oil?"
  d) Value List — e.g. "5 AI tips every business owner should know"
Flag [VIRAL] if there is NO on-screen text hook in the first 3 seconds, or if the text hook fits none of the four categories.

SECTION 2 — VISUAL HOOK (first 3 seconds).
  a) High-quality signal: camera is clear, bright, not blurry
  b) Real person presence: a real face, side profile, or authentic reaction on screen
  c) Strong emotion: stressed, proud, happy, shocked (shocked expressions perform best)
Flag [VIRAL] if the first 3 seconds are blurry/low-quality, have no real-person presence, OR the person's expression is flat/neutral with no clear emotion.

SECTION 3 — PRODUCT FORMULA (only if the video showcases a product).
The proven 4-step structure:
  Step 1: Shocking hook (text + visual)
  Step 2: Visually stunning product — cut directly to the amazing end result to create the "I want that too" feeling
  Step 3: The how — go to Manus, prompt it, use /plan mode to get started
  Step 4: The build process — show Manus generating the result via a replay link ("watch again" → open "manus computer")
Flag [VIRAL] any missing step. If the video skips the end-result reveal (Step 2) or skips showing the Manus build process (Step 4), call it out specifically.

SECTION 4 — PRODUCT CHECKLIST (if demonstrating Manus or a Manus-built product).
  a) Manus logo appears for at least 2 seconds — OR Manus is mentioned — OR is used as the CTA comment keyword
  b) Functional demo: if the product is functional, the creator clicks buttons to show it working live
  c) Publish/share features: for web-app builds, showcase publish-and-share, analytics, and/or SEO
  d) AT LEAST 2 Manus features/showcases must appear in the video (websites, ads, IG carousels, lead gen, workflow setup, competitor research, etc.)
Flag [VIRAL] if fewer than 2 distinct Manus features are shown, if a functional product is shown but never actually clicked, or if a web-app build never mentions publish/analytics/SEO.

SECTION 5 — PACING & SUBTITLES.
  a) Subtitles are localized to the audience's language (if non-English audience implied by spoken language, subtitles should match)
  b) No dead space longer than 1 second (silence, still screen, no motion)
  c) No information overload — screen isn't wall-to-wall wordy; avoid huge chunks of text
Flag [VIRAL] on dead space >1s, subtitles missing or in the wrong language, or on visibly overloaded text-heavy screens.

SECTION 6 — CTA & AUTOMATION.
  a) Clear SPOKEN CTA with a comment trigger keyword (e.g. "Comment MANUS and I'll send you the link")
  b) [OFF-VIDEO — cannot verify] ManyChat / Super Profile DM automation, follow requirement, DM message body, and link. In the paragraph, remind the creator to confirm the DM automation is set up and tested BEFORE posting.
Flag [VIRAL] if there is no clear spoken CTA with a comment keyword.

SECTION 7 — COMPLIANCE (Zero Tolerance).
  a) NO "AI replaces humans" framing — never frame AI as firing people or replacing jobs. Use empowerment framing.
  b) NO forbidden IP: zero Disney, Marvel, or anime IP. (This is stricter than the general copyright rule — for these three franchises there is no gray area.)
  c) Hashtag reminder: creator's caption should include #PR / #ad / #sponsored AND #manus (remind them; do not flag as [VIRAL] since captions are set at post time)
  d) Brand accuracy: uses the correct Manus logo (already covered by the Manus Plug checks)
  e) NO false promises: no money guarantees, no absolute claims like "100% success" or "guaranteed results" (already covered by Legal Layer 0.5 and check #2 — cross-reference, don't double-count)
Flag [VIRAL] on any "AI replaces humans" framing or any visible Disney/Marvel/anime IP.

CRITICAL: The Product Formula (Section 3) and Product Checklist (Section 4) only apply if the video is showcasing Manus or a Manus-built product. If the video is a pure talking-head testimonial with no product demo, mark those two sections N/A and skip them.

═══════════════════════════════════════
OUTPUT FORMAT — COMPACT & CONVERSATIONAL
═══════════════════════════════════════
Return ONLY a valid JSON object (no markdown, no code fences) with this exact structure:

{
  "manus_relevant": true,
  "language_detected": "English",
  "script_summary": "2-3 sentence English summary of what the creator says and shows. If non-English, this serves as the translation for coaches.",
  "legal_paragraph": "Write a SHORT conversational paragraph (3-5 sentences max) summarizing the legal compliance findings. If AUTO-REJECT keywords were found, lead with them clearly using [AUTO-REJECT] and the exact phrase. Then naturally weave in any other flags — mention the specific issue, the risk level in brackets like [HIGH] or [MEDIUM], and the timestamp if applicable. If there are no flags, say so briefly. Always end with the music soft reminder (if music was detected) and the ad-disclosure hashtag reminder as natural sentences. Example tone for clean video: 'No major legal flags here! No income guarantees, absolute claims, or copyrighted content spotted. One soft note — at 0:15 there's a time-saved claim without visible proof [MEDIUM], so your coach might want to verify that. I hear some background music, so just confirm it's from a licensed source. And remember to pop an ad-disclosure hashtag like #ManusAd in your caption when posting!'",
  "content_paragraph": "Write a SHORT conversational paragraph (3-5 sentences max) summarizing the UGC fundamentals ONLY. Cover safe zones, lighting/audio, the hook (mention which of the 12 categories it fits and whether it's strong or could be improved — suggest a specific alternative if weak), and pacing. Do NOT cover the Manus plug here — that goes in its own field. Be constructive and specific. Example tone: 'Lighting and audio are solid — your face is well-lit and the sound is crisp. Safe zones look good for IG and TikTok. Your hook falls into the Demo/How-To category and it's decent, but it could be punchier — try opening with something like \"I built an entire website in 30 seconds\" to create more instant curiosity. Pacing is smooth throughout with no dead air.'",
  "manus_plug_paragraph": "Write a SHORT conversational paragraph (2-4 sentences max) evaluating ONLY the Manus plug. Check these four things explicitly and call out any that need an edit as [HIGH]: (1) Is Manus clearly mentioned in the video (spoken or on-screen text)? Flag if there is no in-video mention. (2) Is the Manus logo present? A tiny logo at the bottom is fine, but flag if there is NO Manus logo at all. (3) If the Manus interface is shown but there is no Manus logo on screen, flag that the logo should be added. (4) Is there a clear CTA, e.g. 'comment Manus for the tool/prompt'? Flag if no CTA is present. (5) Is 'Manus' (and the domain 'manus.im') spelled correctly everywhere it appears? Flag any misspelling like 'Mauns', 'Manis', 'manis.im' and name where it appears. If everything is good, say so briefly. Example tone: 'Your Manus plug is clear — you mention Manus at 0:03 and the interface is on screen for a good 6 seconds. One edit though [HIGH]: I don't see a Manus logo anywhere, so add at least a small one (bottom corner is fine). Also, there's no closing CTA — try ending with \"comment MANUS and I'll send you the tool.\"'",
  "viral_checklist_paragraph": "Write a SHORT conversational paragraph (4-6 sentences max) walking through the Manus Viral Video Checklist. Cover the applicable sections in order: (1) Text hook — was there an on-screen text hook in the first 3s and did it fit Curiosity/Shock, FOMO, Identity/ICP, or Value List? (2) Visual hook — clear camera, real person, strong emotion? (3) Product formula — if a product is shown, did they follow shock hook → stunning result → the how (going to Manus, /plan) → build process (replay link)? (4) Product checklist — Manus logo ≥2s or mentioned or as CTA keyword, functional demo actually clicked, publish/analytics/SEO shown for web apps, at least 2 Manus features/showcases? (5) Pacing/subtitles — dead space, subtitle language, info overload? (6) CTA & automation — clear spoken CTA with a comment keyword, and remind them to have ManyChat/Super Profile DM automation live before posting. (7) Compliance — no 'AI replaces humans' framing, no Disney/Marvel/anime IP. Flag anything missing as [VIRAL]. Skip sections that don't apply (e.g. Product Formula for a pure testimonial) — say so briefly. Example tone: 'Viral checklist walk-through — your text hook at 0:00 (\"5 AI tips every founder needs\") lands as Value List, nice. Visual hook is solid: clear camera, your face on screen, mild curious expression. Product Formula [VIRAL]: you jump straight from the hook to Manus without showing the finished result first — cut to the end product between 0:03 and 0:06 so viewers get the wow moment. Only 1 Manus feature shown (website build) [VIRAL] — the checklist wants at least 2; add a quick clip of publishing or analytics. Pacing looks tight, no dead air. Spoken CTA is present at 0:32. Make sure ManyChat is set up and tested before you post. No 'AI replaces humans' framing and no forbidden IP — compliance is clean.'",
  "quick_verdict": "LOOKS GOOD / NEEDS REVIEW / COACH ATTENTION NEEDED / AUTO-REJECT / NOT MANUS CONTENT",
  "overall_summary": "One final sentence. Always include: 'A human coach will review this shortly for final approval.' If AUTO-REJECT, start with: 'This video contains auto-reject language and must be reviewed by a coach before any use.' then end with: 'If you think this is a mistake, please tag your coach for a manual review.'"
}

VERDICT ROUTING:
- Any AUTO-REJECT trigger (see LAYER 0.5) → "quick_verdict" = "AUTO-REJECT".
- Any [HIGH] flag — including copyrighted/trademarked material or a weak/missing/low-effort Manus plug → "quick_verdict" = "COACH ATTENTION NEEDED".
- Any [VIRAL] flag from the Manus Viral Video Checklist → "quick_verdict" = "COACH ATTENTION NEEDED".
- Only [MEDIUM] flags and no higher → "quick_verdict" = "NEEDS REVIEW".
- No flags at all → "quick_verdict" = "LOOKS GOOD".

CRITICAL RULES FOR THE PARAGRAPHS:
- Keep paragraphs SHORT: legal 3-5 sentences, content 3-5 sentences, Manus plug 2-4 sentences, viral checklist 4-6 sentences. Do NOT write essays.
- Be conversational and friendly, like a peer creator giving feedback in a chat.
- Naturally mention ALL relevant checks within the paragraph flow — don't use headers, bullet points, or field labels.
- If something is fine, you can group multiple "all good" items in one sentence (e.g., "No income guarantees, copyrighted content, or fake testimonials spotted.").
- If something needs attention, be specific but brief (mention what, where/when, and risk level).
- AUTO-REJECT findings must always be called out first and clearly, with the exact phrase detected.
- Music and ad disclosure reminders should feel like natural sentences at the end of the legal paragraph, not separate callouts.
- For hooks, always mention which of the 12 categories it falls into.
- If the video is in a foreign language, still review it fully. Use "script_summary" for the English translation.
- NEVER use markdown formatting (no bold, no headers, no bullets) inside the paragraph strings — just plain conversational text.
"""

# ---------------------------------------------------------------------------
# Study Mode Prompt — Format Analysis + Manus Adaptation Brief
# ---------------------------------------------------------------------------
STUDY_PROMPT = MANUS_KNOWLEDGE + r"""
You are Vexi in "Study Mode" — a creative strategist helping the Manus UGC team learn from high-performing content in any niche or brand.

Your job: watch this video, break down WHY it works, write a concise Manus adaptation brief, AND deliver a fully copyable script the creator can record as a Manus UGC video.

═══════════════════════════════════════
WHAT TO ANALYZE
═══════════════════════════════════════

PART 1 — FORMAT ANALYSIS:
- Source context: niche, creator style, platform
- Hook: which of these 12 categories does it use?
  (1) Curiosity / "Feels Illegal to Know"  (2) Challenge / Speed Run  (3) Before & After / Transformation
  (4) Hot Take / Controversial / Pattern Interrupt  (5) Demo / How-To (Punchy Openers)
  (6) Social Proof / Flex / Authority  (7) Skits  (8) News & Presentation  (9) FOMO / Urgency
  (10) Anti-Hook / Reverse Psychology  (11) Comparison / "This vs. That"  (12) Emotional / Relatable
- Narrative structure (hook → problem → solution → CTA, etc.)
- Pacing, editing style, key visual/audio techniques
- The single core reason this format works

PART 2 — MANUS ADAPTATION:
- How to port this exact format to a Manus UGC video, beat by beat
- Which Manus features fill each narrative role (refer to MANUS GROUND TRUTH — agentic tasks, browser automation, research, doc/slide generation, website builds, image generation, etc.)
- A numbered shot/beat outline a creator can follow (5-7 beats max)
- What NOT to copy — flag anything that would fail a Vexi compliance check (income claims, absolute promises, competitor mentions, fake testimonials)

PART 3 — FULL COPYABLE SCRIPT:
Write a ready-to-record script the creator can paste and shoot. It must be:
- 25-60 seconds of runtime total
- Structured with beat markers on their own lines: [HOOK 0-3s], [BEAT 1 3-8s], [BEAT 2 ...], ..., [CTA]
- Every beat has: spoken line(s) in plain prose, then a "(visual: ...)" cue in parentheses on the SAME line or the next line
- Tailored to a Manus feature that fits the source video's format (name the specific Manus capability). Aim to showcase AT LEAST 2 distinct Manus features across the beats (per the Viral Video Checklist).
- Compliance-safe: NO income claims, NO absolute claims like "100%" or "replaces humans", NO "AI replaces humans" framing, NO Disney/Marvel/anime IP, NO competitor brand mentions or logos, NO fake testimonials. Include a soft ad-disclosure reminder in the CTA area (e.g. "and tag #ManusAd #ad").
- End with a clear CTA that ties back to Manus with a comment keyword (e.g. "comment MANUS and I'll send the exact prompt") — this is what triggers ManyChat/Super Profile DM automation on the creator's end.

MUST follow the MANUS VIRAL VIDEO CHECKLIST:
- HOOK 0-3s must have BOTH:
  (a) an on-screen text hook that fits ONE of: Curiosity/Shock, FOMO/Loss Aversion, Identity/ICP Match ("Are you a [role] struggling with X?"), or Value List ("5 things every [role] should know") — write this text hook explicitly in the (visual: ...) cue
  (b) a spoken hook and a visible face/emotion (creator on camera, ideally shocked or curious)
- If the source video is a product showcase, the script MUST follow the Product Formula: Step 1 Shocking Hook → Step 2 cut to visually stunning end result ("I want that too" moment) → Step 3 "the how" (opening Manus, prompting it, using /plan) → Step 4 build process (Manus computer / replay link footage)
- Keep pacing tight — no beat should imply more than ~1 second of silence or a static screen

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════
Return ONLY a valid JSON object (no markdown, no code fences):

{
  "source_context": "One sentence: niche, creator style, platform.",
  "format_breakdown": "2-3 sentences max. Name the hook category explicitly. Cover structure and pacing briefly.",
  "what_makes_it_work": "One sentence only.",
  "manus_adaptation": "2-3 sentences max. What to keep, what to swap, which Manus features fill each beat.",
  "suggested_outline": "Numbered plain-text outline, 5-7 beats, each on its own line. No markdown symbols.",
  "copy_guardrails": "1-2 sentences. Flag compliance risks or confirm it's clean.",
  "adaptation_difficulty": "EASY / MODERATE / COMPLEX",
  "full_script": "Full ready-to-record script with beat markers on their own lines. Example shape:\n[HOOK 0-3s] Spoken line here. (visual: creator on camera, quick zoom in)\n[BEAT 1 3-8s] Spoken line. (visual: screen recording of Manus opening a task)\n[BEAT 2 8-15s] ...\n[CTA] Comment MANUS and I'll send the prompt. #ManusAd"
}

CRITICAL RULES:
- Be concise for the analysis fields; every field except full_script has a strict length cap.
- full_script may be up to ~1500 characters — enough for 25-60 seconds of dialogue plus visual cues.
- NEVER use markdown formatting inside string values — plain text only. No bold, no headers, no bullets.
- If the video has no audio or is very short, work with what is visible.
- The full_script must be about Manus (use MANUS GROUND TRUTH), not the source video's brand.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
        "max_filesize": 100 * 1024 * 1024,
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
                    if total > 100 * 1024 * 1024:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        return None, "Apify video exceeds 100MB."
        log.info(f"Apify download complete: {total / 1024 / 1024:.1f}MB")
        return tmp_path, None
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, f"Apify video download error: {e}"


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


def extract_gdrive_id(url: str) -> str | None:
    """Return the Drive file ID from any recognized Drive URL shape, or None."""
    for pat in (GDRIVE_PATTERN, GDRIVE_OPEN_PATTERN, GDRIVE_UC_PATTERN, GDRIVE_USERCONTENT_PATTERN):
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


def is_gdrive_url(url: str) -> bool:
    return extract_gdrive_id(url) is not None


def convert_gdrive_to_direct(url: str) -> str:
    """Convert a Google Drive sharing URL to a direct download URL.

    Uses the drive.usercontent.google.com endpoint, which bypasses the classic
    /uc virus-scan interstitial for most public files. Files >~100MB still
    return an HTML confirmation page — download_gdrive() handles that case.
    """
    file_id = extract_gdrive_id(url)
    if file_id:
        return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    return url


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
                    if total > 100 * 1024 * 1024:
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


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(code in msg for code in ("503", "429", "unavailable", "resource_exhausted", "resourceexhausted", "too many requests"))


async def _gemini_generate(contents: list, retries: int = 3, config=None) -> object:
    delays = [5, 15, 30]
    last_exc = None
    for attempt in range(retries):
        try:
            kwargs = {"model": "gemini-2.5-flash", "contents": contents}
            if config is not None:
                kwargs["config"] = config
            return gemini_client.models.generate_content(**kwargs)
        except Exception as e:
            last_exc = e
            if _is_retryable(e) and attempt < retries - 1:
                wait = delays[attempt]
                log.warning(f"Gemini transient error (attempt {attempt + 1}/{retries}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
    raise last_exc


async def analyze_video_with_gemini(video_url: str, mime_type: str = "video/mp4", prompt: str = None, response_json: bool = False) -> dict:
    """Send video URL directly to Gemini for review (no local download needed).

    If response_json=True, asks Gemini to enforce JSON output via response_mime_type.
    """
    if prompt is None:
        prompt = REVIEW_PROMPT
    raw_text = ""
    response = None
    try:
        log.info(f"Sending video URL to Gemini: {video_url[:120]}...")
        log.info(f"MIME type: {mime_type}")

        video_part = genai_types.Part.from_uri(
            file_uri=video_url,
            mime_type=mime_type,
        )

        # Low temperature reduces hallucinated copyright/IP matches (e.g. calling a
        # generic character "SpongeBob"). 0.2 keeps it factual but not robotic.
        config_kwargs = {"temperature": 0.2}
        if response_json:
            config_kwargs["response_mime_type"] = "application/json"
        config = genai_types.GenerateContentConfig(**config_kwargs)

        log.info("Calling Gemini 2.5 Flash for review...")
        response = await _gemini_generate([video_part, prompt], config=config)

        raw_text = response.text.strip()
        log.info(f"Gemini response length: {len(raw_text)} chars")
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        result = _parse_json_with_repair(raw_text)
        if result is None:
            try:
                fr = response.candidates[0].finish_reason
            except Exception:
                fr = "unknown"
            log.error(f"JSON parse failed (finish_reason={fr}). Raw: {raw_text[:500]}")
            return {"error": f"AI returned invalid JSON (finish_reason={fr}). Raw excerpt: {raw_text[:300]}"}
        log.info(f"Review complete. Verdict: {result.get('quick_verdict', 'N/A')}")
        return result

    except Exception as e:
        log.error(f"Gemini error: {type(e).__name__}: {e}")
        import traceback
        log.error(traceback.format_exc())
        return {"error": str(e)}


async def _analyze_local_file_with_gemini(file_path: str, prompt: str = None, response_json: bool = False) -> dict:
    """Upload a local file to Gemini File API and analyze it.

    If response_json=True, asks Gemini to enforce JSON output via response_mime_type.
    On JSON parse failure, includes a raw excerpt and finish_reason in the error.
    """
    if prompt is None:
        prompt = REVIEW_PROMPT
    raw_text = ""
    response = None
    uploaded_file = None
    try:
        mime = guess_mime_type(file_path)
        log.info(f"Uploading local file to Gemini File API: {file_path} ({mime})")
        uploaded_file = gemini_client.files.upload(file=file_path)
        log.info(f"Upload complete: {uploaded_file.name}, state={uploaded_file.state}")

        max_wait = 120
        waited = 0
        while uploaded_file.state.name == "PROCESSING" and waited < max_wait:
            await asyncio.sleep(5)
            waited += 5
            uploaded_file = gemini_client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            return {"error": f"File processing failed. State: {uploaded_file.state.name}"}

        # Low temperature reduces hallucinated copyright/IP matches (e.g. calling a
        # generic character "SpongeBob"). 0.2 keeps it factual but not robotic.
        config_kwargs = {"temperature": 0.2}
        if response_json:
            config_kwargs["response_mime_type"] = "application/json"
        config = genai_types.GenerateContentConfig(**config_kwargs)

        response = await _gemini_generate([uploaded_file, prompt], config=config)
        raw_text = response.text.strip()
        log.info(f"Gemini response length: {len(raw_text)} chars")
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        result = _parse_json_with_repair(raw_text)

        try:
            gemini_client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        if result is None:
            try:
                fr = response.candidates[0].finish_reason
            except Exception:
                fr = "unknown"
            log.error(f"JSON parse failed (finish_reason={fr}). Raw: {raw_text[:500]}")
            return {"error": f"AI returned invalid JSON (finish_reason={fr}). Raw excerpt: {raw_text[:300]}"}
        return result

    except Exception as e:
        log.error(f"Local file Gemini upload error: {type(e).__name__}: {e}")
        return {"error": str(e)}


async def analyze_video_with_gemini_upload(video_url: str, session: aiohttp.ClientSession, prompt: str = None, response_json: bool = False) -> dict:
    """Fallback: Download video and upload to Gemini File API."""
    if prompt is None:
        prompt = REVIEW_PROMPT
    raw_text = ""
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
                if total > 100 * 1024 * 1024:
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


def build_review_message(review: dict, creator: str = None) -> tuple[str | None, list[discord.Embed]]:
    """Convert the Gemini JSON review into a single compact Discord embed.
    
    Returns (content_text, embeds_list).
    """
    # --- Error ---
    if "error" in review:
        err_embed = discord.Embed(
            title="Vexi — Review Error",
            description=f"Something went wrong during the review:\n```{review['error'][:500]}```\nPlease try again or ask a coach for help.",
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
    parts.append(f"\n**{grid_label}**\n{grid_block}")

    embed = discord.Embed(
        title=f"Vexi Review — {verdict}",
        description="\n".join(parts),
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


# ---------------------------------------------------------------------------
# Discord Bot Setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Track whether we've already synced commands (avoid re-syncing on reconnect)
_commands_synced = False


@bot.event
async def on_ready():
    global _commands_synced
    log.info(f"Vexi is online as {bot.user} (ID: {bot.user.id})")
    log.info(f"Auto-detect channels: {VEXI_CHANNEL_IDS}")

    if _commands_synced:
        log.info("Reconnected — skipping command sync (already synced).")
        return

    try:
        for g in bot.guilds:
            bot.tree.copy_global_to(guild=g)
            synced = await bot.tree.sync(guild=g)
            log.info(f"Synced {len(synced)} slash command(s) to guild {g.name} ({g.id})")

        bot.tree.clear_commands(guild=None)
        await bot.tree.sync()
        log.info("Cleared global commands to prevent duplicates")
        _commands_synced = True
    except Exception as e:
        log.error(f"Failed to sync commands: {e}")


# ---------------------------------------------------------------------------
# Per-user FIFO review queue
# ---------------------------------------------------------------------------
# Videos submitted via /vexi are processed one at a time PER USER. Different
# users run in parallel; a single user's videos run serially so no one can
# monopolize Gemini quota. Queue and worker state live in-memory only.
_user_queues: dict[int, asyncio.Queue] = {}
_user_workers: dict[int, asyncio.Task] = {}
_queue_lock = asyncio.Lock()


async def enqueue_review(user_id: int, job: dict) -> None:
    async with _queue_lock:
        q = _user_queues.setdefault(user_id, asyncio.Queue())
        await q.put(job)
        w = _user_workers.get(user_id)
        if w is None or w.done():
            _user_workers[user_id] = asyncio.create_task(_user_worker(user_id))


async def _user_worker(user_id: int) -> None:
    q = _user_queues.get(user_id)
    if q is None:
        return
    try:
        while True:
            try:
                job = await asyncio.wait_for(q.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # Recheck under lock — a job may have been enqueued mid-drain.
                async with _queue_lock:
                    if q.empty():
                        _user_queues.pop(user_id, None)
                        _user_workers.pop(user_id, None)
                        return
                continue
            try:
                await process_single_review(**job)
            except Exception:
                log.exception(f"review job failed for user {user_id}")
            finally:
                q.task_done()
    except Exception:
        log.exception(f"user worker crashed for user {user_id}")
        async with _queue_lock:
            _user_workers.pop(user_id, None)


async def process_single_review(
    submitter: discord.User | discord.Member,
    coach: discord.Member | None,
    channel: discord.abc.Messageable,
    master_msg: discord.Message,
    attachment: discord.Attachment | None,
    url: str | None,
    display_url: str | None,
    index: int,
    total: int,
) -> None:
    """Runs one video through the full review pipeline. Posts the video (or its
    URL), a live progress bar, and the final review embed as replies to master_msg.
    Called by the per-user worker — this is where the actual Gemini call happens."""
    label = f"Video {index} of {total}" if total > 1 else "Your video"
    header = f"🎬 **{label}** — from {submitter.mention}"

    # Post the video header (with re-attached file if we can)
    video_msg: discord.Message
    if attachment is not None:
        try:
            video_file = await attachment.to_file()
            video_msg = await channel.send(content=header, file=video_file, reference=master_msg)
        except Exception as e:
            log.warning(f"Could not re-attach video for video {index}: {e}")
            video_msg = await channel.send(content=f"{header}\n(couldn't re-attach the file)", reference=master_msg)
    else:
        video_msg = await channel.send(content=f"{header}\n🎬 Video: {display_url or url}", reference=master_msg)

    progress_msg = await video_msg.reply(
        content=f"⏳ **Vexi is analyzing {label.lower()}...**\n▁▁▁▁▁▁▁▁▁▁ Sending to AI..."
    )

    progress_done = asyncio.Event()

    async def update_progress():
        stages = [
            ("▓▓▁▁▁▁▁▁▁▁", "🧠 AI is watching your video..."),
            ("▓▓▓▓▁▁▁▁▁▁", "🧠 AI is watching your video..."),
            ("▓▓▓▓▓▁▁▁▁▁", "🔍 Checking legal compliance..."),
            ("▓▓▓▓▓▓▁▁▁▁", "🔍 Checking legal compliance..."),
            ("▓▓▓▓▓▓▓▁▁▁", "🎬 Reviewing hooks & storytelling..."),
            ("▓▓▓▓▓▓▓▓▁▁", "🎬 Reviewing hooks & storytelling..."),
            ("▓▓▓▓▓▓▓▓▓▁", "📝 Writing up the review..."),
            ("▓▓▓▓▓▓▓▓▓▓", "✅ Almost done..."),
        ]
        for bar, status in stages:
            if progress_done.is_set():
                return
            try:
                await progress_msg.edit(
                    content=f"⏳ **Vexi is analyzing {label.lower()}...**\n{bar} {status}"
                )
            except Exception:
                return
            try:
                await asyncio.wait_for(progress_done.wait(), timeout=15)
                return
            except asyncio.TimeoutError:
                continue

    progress_task = asyncio.create_task(update_progress())

    source_url = attachment.url if attachment is not None else (url or "")
    filename = attachment.filename if attachment is not None else ""
    mime = guess_mime_type(source_url, filename)

    try:
        is_discord_cdn = bool(DISCORD_CDN_PATTERN.match(source_url))
        is_gdrive = is_gdrive_url(source_url)

        if is_discord_cdn or is_gdrive:
            log.info(f"{'Discord CDN' if is_discord_cdn else 'Drive'} URL (video {index}/{total}) — using upload path directly.")
            async with aiohttp.ClientSession() as session:
                review = await analyze_video_with_gemini_upload(source_url, session)
        else:
            review = await analyze_video_with_gemini(source_url, mime_type=mime)
            if "error" in review:
                log.info(f"Direct URL failed ({review['error']}), trying upload fallback...")
                async with aiohttp.ClientSession() as session:
                    review = await analyze_video_with_gemini_upload(source_url, session)

        progress_done.set()
        await progress_task

        content_text, embeds = build_review_message(review, creator=f"{submitter.mention}")

        # Ping coach only on the FIRST video's review (so a batch of 5 doesn't
        # spam @coach five times). Index 1 is the first video.
        if coach and index == 1:
            prefix = f"🏷️ {coach.mention} — "
            content_text = (prefix + (content_text or "")).strip()

        await progress_msg.edit(content=content_text or None, embeds=embeds)
    except Exception as e:
        progress_done.set()
        try:
            await progress_task
        except Exception:
            pass
        log.error(f"process_single_review error (video {index}/{total}): {type(e).__name__}: {e}")
        try:
            await progress_msg.edit(
                content=f"❌ Video {index}/{total} review failed. Error: {str(e)[:200]}"
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Slash Command: /vexi
# ---------------------------------------------------------------------------
@bot.tree.command(name="vexi", description="Submit up to 5 videos for Vexi AI review")
@app_commands.describe(
    video1="Attach a video file (required unless you use a URL)",
    video2="Optional 2nd video file",
    video3="Optional 3rd video file",
    video4="Optional 4th video file",
    video5="Optional 5th video file",
    video_url1="Or paste a video URL (Google Drive, YouTube, or direct link)",
    video_url2="Optional 2nd URL",
    video_url3="Optional 3rd URL",
    video_url4="Optional 4th URL",
    video_url5="Optional 5th URL",
    coach="Tag a coach to notify them about this batch",
)
async def vexi_command(
    interaction: discord.Interaction,
    video1: discord.Attachment = None,
    video2: discord.Attachment = None,
    video3: discord.Attachment = None,
    video4: discord.Attachment = None,
    video5: discord.Attachment = None,
    video_url1: str = None,
    video_url2: str = None,
    video_url3: str = None,
    video_url4: str = None,
    video_url5: str = None,
    coach: discord.Member = None,
):
    """Handle /vexi slash command with 1-5 videos, queued per-user."""
    try:
        await interaction.response.defer(thinking=True)
    except (discord.errors.NotFound, discord.errors.HTTPException) as e:
        log.warning(f"Interaction defer failed: {e}")
        return

    videos = [v for v in (video1, video2, video3, video4, video5) if v is not None]
    urls = [u for u in (video_url1, video_url2, video_url3, video_url4, video_url5) if u]

    # Validate every attachment's extension before enqueuing anything.
    for v in videos:
        suffix = Path(v.filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            await interaction.followup.send(
                f"❌ `{v.filename}` doesn't look like a video file.\n"
                f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}"
            )
            return

    # Build source list — attachments in slot order first, then URLs in slot order.
    sources: list[dict] = []
    for v in videos:
        sources.append({"attachment": v, "url": None, "display_url": None})
    for u in urls:
        sources.append({"attachment": None, "url": u, "display_url": u})

    if not sources:
        await interaction.followup.send(
            "👋 Hey! Please attach at least one video or provide a URL.\n\n"
            "**Attach up to 5 files:** `/vexi video1: video2: ...`\n"
            "**Or paste up to 5 URLs:** `/vexi video_url1: ...`\n"
            "**Mix both** — Vexi will review them one at a time in the order you gave them.\n"
            "**Tag a coach** with `coach:@CoachName` to notify them."
        )
        return

    total = len(sources)
    coach_ping = f"🏷️ {coach.mention} — " if coach else ""
    if total == 1:
        header = (
            f"{coach_ping}🔍 **Vexi is reviewing a video from {interaction.user.mention}...** Hang tight!"
        )
    else:
        header = (
            f"{coach_ping}📥 **Vexi received {total} videos from {interaction.user.mention}.**\n"
            f"Processing one at a time — I'll reply with each review below as it's ready."
        )
    master_msg = await interaction.followup.send(content=header)

    for i, src in enumerate(sources, start=1):
        await enqueue_review(
            interaction.user.id,
            {
                "submitter": interaction.user,
                "coach": coach,
                "channel": interaction.channel,
                "master_msg": master_msg,
                "attachment": src["attachment"],
                "url": src["url"],
                "display_url": src["display_url"],
                "index": i,
                "total": total,
            },
        )


# ---------------------------------------------------------------------------
# Slash Command: /study
# ---------------------------------------------------------------------------
@bot.tree.command(name="study", description="Study any creator's video and get a Manus UGC adaptation brief")
@app_commands.describe(
    video="Attach a downloaded video file",
    video_url="Or paste an Instagram, TikTok, YouTube, or direct video URL",
    niche="Optional: describe the niche or brand for extra context (e.g. 'productivity SaaS', 'fitness app')",
)
async def study_command(
    interaction: discord.Interaction,
    video: discord.Attachment = None,
    video_url: str = None,
    niche: str = None,
):
    """Handle /study slash command — format analysis + Manus adaptation brief."""
    try:
        await interaction.response.defer(thinking=True)
    except (discord.errors.NotFound, discord.errors.HTTPException) as e:
        log.warning(f"Study interaction defer failed: {e}")
        return

    if video is None and not video_url:
        await interaction.followup.send(
            "👋 Give me a video to study!\n\n"
            "**Option 1:** `/study video:` → attach a downloaded video\n"
            "**Option 2:** `/study video_url:https://www.tiktok.com/@creator/video/123`\n"
            "**Option 3:** Add `niche:` for context — e.g. `niche:productivity SaaS`\n\n"
            "Supports Instagram Reels, TikTok, YouTube, Google Drive, and direct links."
        )
        return

    source_label = "Video"
    source_url = None
    tmp_path = None
    tmp_dir = None
    downloaded_video_path_for_discord = None

    intro_text = f"🔍 **Vexi is studying a video for {interaction.user.mention}...** Hang tight!"
    if niche:
        intro_text += f"\n📌 Niche context: *{niche}*"

    if video is not None:
        suffix = Path(video.filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            await interaction.followup.send(
                f"❌ That doesn't look like a video file (`{video.filename}`).\n"
                f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}"
            )
            return
        source_url = video.url
        source_label = f"Uploaded: {video.filename}"
        try:
            video_file = await video.to_file()
            visible_msg = await interaction.followup.send(content=intro_text, file=video_file)
        except Exception as e:
            log.warning(f"Could not re-attach video: {e}")
            visible_msg = await interaction.followup.send(content=intro_text)
    else:
        source_url = video_url
        if INSTAGRAM_PATTERN.search(video_url):
            source_label = "Instagram"
        elif TIKTOK_PATTERN.search(video_url):
            source_label = "TikTok"
        elif YOUTUBE_PATTERN.search(video_url):
            source_label = "YouTube"
        elif GDRIVE_PATTERN.search(video_url) or GDRIVE_OPEN_PATTERN.search(video_url):
            source_label = "Google Drive"
        else:
            source_label = "Direct Link"
        visible_msg = await interaction.followup.send(
            content=f"{intro_text}\n🎬 Video: {source_url}"
        )

    progress_msg = await visible_msg.reply(
        content="⏳ **Vexi is studying the format...**\n▁▁▁▁▁▁▁▁▁▁ Loading video..."
    )

    progress_done = asyncio.Event()

    async def update_progress():
        stages = [
            ("▓▓▁▁▁▁▁▁▁▁", "🎬 Watching the video..."),
            ("▓▓▓▓▁▁▁▁▁▁", "🎬 Watching the video..."),
            ("▓▓▓▓▓▁▁▁▁▁", "🔍 Breaking down the format..."),
            ("▓▓▓▓▓▓▁▁▁▁", "🔍 Analyzing hook & structure..."),
            ("▓▓▓▓▓▓▓▁▁▁", "🔄 Building Manus adaptation brief..."),
            ("▓▓▓▓▓▓▓▓▁▁", "🔄 Building Manus adaptation brief..."),
            ("▓▓▓▓▓▓▓▓▓▁", "📋 Writing the outline..."),
            ("▓▓▓▓▓▓▓▓▓▓", "✅ Almost done..."),
        ]
        for bar, status in stages:
            if progress_done.is_set():
                return
            try:
                await progress_msg.edit(
                    content=f"⏳ **Vexi is studying the format...**\n{bar} {status}"
                )
            except Exception:
                return
            try:
                await asyncio.wait_for(progress_done.wait(), timeout=15)
                return
            except asyncio.TimeoutError:
                continue

    progress_task = asyncio.create_task(update_progress())

    active_prompt = STUDY_PROMPT
    if niche:
        active_prompt = STUDY_PROMPT + f"\n\nNICHE CONTEXT PROVIDED BY SUBMITTER: {niche}\nTailor your Manus adaptation and suggested outline to this niche specifically.\n"

    try:
        is_discord_cdn = bool(DISCORD_CDN_PATTERN.match(source_url))
        is_social = (video is None) and _is_social_media_url(source_url)
        is_gdrive = bool(GDRIVE_PATTERN.search(source_url) or GDRIVE_OPEN_PATTERN.search(source_url))

        if is_social:
            log.info(f"Social media URL — downloading with yt-dlp: {source_url[:80]}")
            await progress_msg.edit(
                content="⏳ **Vexi is studying the format...**\n▓▁▁▁▁▁▁▁▁▁ 📥 Downloading from social media..."
            )
            tmp_path, dl_error = await download_with_ytdlp(source_url)

            # Fallback: Apify (Instagram/TikTok only) when yt-dlp fails
            if (not tmp_path) and (INSTAGRAM_PATTERN.search(source_url) or TIKTOK_PATTERN.search(source_url)):
                if APIFY_API_TOKEN:
                    log.warning(f"yt-dlp failed ({dl_error}). Trying Apify fallback...")
                    try:
                        await progress_msg.edit(
                            content="⏳ **Vexi is studying the format...**\n▓▓▁▁▁▁▁▁▁▁ 🔄 yt-dlp blocked, trying Apify..."
                        )
                    except Exception:
                        pass
                    async with aiohttp.ClientSession() as sess:
                        tmp_path, apify_err = await download_with_apify(source_url, sess)
                    if not tmp_path:
                        dl_error = f"yt-dlp: {dl_error} | apify: {apify_err}"
                else:
                    log.warning("yt-dlp failed and APIFY_API_TOKEN not set — skipping Apify fallback.")

            if not tmp_path:
                study = {"error": f"DOWNLOAD_FAILED: {dl_error or 'Unknown error'}"}
            else:
                tmp_dir = os.path.dirname(tmp_path)
                log.info(f"Download succeeded: {tmp_path}")
                downloaded_video_path_for_discord = tmp_path
                study = await _analyze_local_file_with_gemini(tmp_path, prompt=active_prompt, response_json=True)
        elif is_discord_cdn:
            log.info("Discord CDN URL — using upload fallback.")
            async with aiohttp.ClientSession() as sess:
                study = await analyze_video_with_gemini_upload(source_url, sess, prompt=active_prompt, response_json=True)
        elif is_gdrive:
            async with aiohttp.ClientSession() as sess:
                study = await analyze_video_with_gemini_upload(source_url, sess, prompt=active_prompt, response_json=True)
        else:
            mime = guess_mime_type(source_url, video.filename if video else "")
            study = await analyze_video_with_gemini(source_url, mime_type=mime, prompt=active_prompt, response_json=True)
            if "error" in study:
                log.info(f"Direct URL failed, trying upload fallback...")
                async with aiohttp.ClientSession() as sess:
                    study = await analyze_video_with_gemini_upload(source_url, sess, prompt=active_prompt, response_json=True)

        progress_done.set()
        await progress_task

        _, embeds, script_file, script_text = build_study_message(study, source_label=source_label)

        # Delete the progress bar so only two messages remain: intro + results
        try:
            await progress_msg.delete()
        except Exception:
            pass

        # Build the single reply: video (if downloaded) + embeds + optional script.txt
        files: list[discord.File] = []
        if downloaded_video_path_for_discord and os.path.exists(downloaded_video_path_for_discord):
            ext = Path(downloaded_video_path_for_discord).suffix.lower()
            friendly_name = f"{source_label.lower().replace(' ', '_')}{ext}"
            files.append(discord.File(downloaded_video_path_for_discord, filename=friendly_name))
        if script_file is not None:
            files.append(script_file)

        send_kwargs: dict = {"embeds": embeds}
        if files:
            send_kwargs["files"] = files

        try:
            posted = await visible_msg.reply(**send_kwargs)
        except discord.HTTPException as e:
            if e.status == 413 and "files" in send_kwargs:
                # Drop the (large) video file but keep the script.txt if present.
                keep = [f for f in files if f.filename == "manus_script.txt"]
                if keep:
                    send_kwargs["files"] = keep
                else:
                    send_kwargs.pop("files", None)
                posted = await visible_msg.reply(**send_kwargs)
                await visible_msg.reply(content=f"📹 Video too large to attach — original: {source_url}")
            else:
                raise

        if script_text and posted is not None:
            _cache_script(posted.id, script_text)

    except Exception as e:
        progress_done.set()
        await progress_task
        log.error(f"Study command error: {type(e).__name__}: {e}")
        await progress_msg.edit(
            content=f"❌ Something went wrong. Please try again.\nError: {str(e)[:200]}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Mention Handler: @Vexi study <url>
# ---------------------------------------------------------------------------
async def handle_mention_study(message: discord.Message, content_after_mention: str):
    """Handle @Vexi study <url> [niche:...] mentions from any channel."""
    text = content_after_mention.strip()

    # No keyword at all — generic help
    if not text:
        await message.reply(
            "👋 Hey! I'm **Vexi**, your UGC review buddy.\n\n"
            "To study a video format, use:\n"
            "`@Vexi study <Instagram/TikTok/YouTube link>`\n\n"
            "**Example:** `@Vexi study https://www.instagram.com/reel/abc123`\n\n"
            "You can also add niche context:\n"
            "`@Vexi study https://www.tiktok.com/... niche:productivity SaaS`\n\n"
            "Not sure what to do? Ask your coach for help! 🙌"
        )
        return

    # Wrong keyword (mentioned bot but typed something other than "study")
    if not text.lower().startswith("study"):
        await message.reply(
            "🤔 Hmm, I didn't catch that!\n\n"
            "To study a video, type:\n"
            "`@Vexi study <Instagram/TikTok/YouTube link>`\n\n"
            "**Example:** `@Vexi study https://www.instagram.com/reel/abc123`\n\n"
            "If you need help, reach out to your coach! 🙌"
        )
        return

    # Strip "study" and parse the rest
    rest = text[len("study"):].strip()

    # Parse optional niche: param (e.g. "niche:productivity SaaS")
    niche = None
    niche_match = re.search(r'\bniche:(.+?)(?=\s+https?://|\s*$)', rest, re.IGNORECASE)
    if niche_match:
        niche = niche_match.group(1).strip()

    # Extract URL
    url_match = re.search(r'https?://\S+', rest)
    if not url_match:
        await message.reply(
            "❌ I need a link to study!\n\n"
            "**Format:** `@Vexi study <link>`\n"
            "**Example:** `@Vexi study https://www.instagram.com/reel/abc123`\n\n"
            "Supports Instagram Reels, TikTok, YouTube, Google Drive, and direct video links.\n"
            "If you're stuck, ask your coach for help! 🙌"
        )
        return

    source_url = url_match.group(0).rstrip(".,;)")  # strip trailing punctuation

    # Validate it's a URL type we can handle
    is_social = _is_social_media_url(source_url)
    is_gdrive = bool(GDRIVE_PATTERN.search(source_url) or GDRIVE_OPEN_PATTERN.search(source_url))
    has_video_ext = any(source_url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

    if not (is_social or is_gdrive or has_video_ext):
        await message.reply(
            "❌ That doesn't look like a supported video link.\n\n"
            "I can study links from:\n"
            "• **Instagram** — `instagram.com/reel/...`\n"
            "• **TikTok** — `tiktok.com/...`\n"
            "• **YouTube** — `youtube.com/watch?...` or `youtu.be/...`\n"
            "• **Google Drive** — `drive.google.com/file/d/...`\n"
            "• **Direct video links** (.mp4, .mov, etc.)\n\n"
            "If you're unsure, ask your coach for help! 🙌"
        )
        return

    # Determine source label
    if INSTAGRAM_PATTERN.search(source_url):
        source_label = "Instagram"
    elif TIKTOK_PATTERN.search(source_url):
        source_label = "TikTok"
    elif YOUTUBE_PATTERN.search(source_url):
        source_label = "YouTube"
    elif is_gdrive:
        source_label = "Google Drive"
    else:
        source_label = "Direct Link"

    # Kick off — react and show intro
    try:
        await message.add_reaction("🔍")
    except Exception:
        pass

    intro_text = f"🔍 **Vexi is studying a video for {message.author.mention}...** Hang tight!"
    if niche:
        intro_text += f"\n📌 Niche context: *{niche}*"
    intro_text += f"\n🎬 Video: {source_url}"
    visible_msg = await message.reply(content=intro_text)

    progress_msg = await visible_msg.reply(
        content="⏳ **Vexi is studying the format...**\n▁▁▁▁▁▁▁▁▁▁ Loading video..."
    )

    progress_done = asyncio.Event()

    async def update_progress():
        stages = [
            ("▓▓▁▁▁▁▁▁▁▁", "🎬 Watching the video..."),
            ("▓▓▓▓▁▁▁▁▁▁", "🎬 Watching the video..."),
            ("▓▓▓▓▓▁▁▁▁▁", "🔍 Breaking down the format..."),
            ("▓▓▓▓▓▓▁▁▁▁", "🔍 Analyzing hook & structure..."),
            ("▓▓▓▓▓▓▓▁▁▁", "🔄 Building Manus adaptation brief..."),
            ("▓▓▓▓▓▓▓▓▁▁", "🔄 Building Manus adaptation brief..."),
            ("▓▓▓▓▓▓▓▓▓▁", "📋 Writing the outline..."),
            ("▓▓▓▓▓▓▓▓▓▓", "✅ Almost done..."),
        ]
        for bar, status in stages:
            if progress_done.is_set():
                return
            try:
                await progress_msg.edit(content=f"⏳ **Vexi is studying the format...**\n{bar} {status}")
            except Exception:
                return
            try:
                await asyncio.wait_for(progress_done.wait(), timeout=15)
                return
            except asyncio.TimeoutError:
                continue

    progress_task = asyncio.create_task(update_progress())

    active_prompt = STUDY_PROMPT
    if niche:
        active_prompt = STUDY_PROMPT + f"\n\nNICHE CONTEXT PROVIDED BY SUBMITTER: {niche}\nTailor your Manus adaptation and suggested outline to this niche specifically.\n"

    tmp_path = None
    tmp_dir = None
    downloaded_video_path_for_discord = None

    try:
        if is_social:
            log.info(f"[mention] Social URL — yt-dlp: {source_url[:80]}")
            await progress_msg.edit(
                content="⏳ **Vexi is studying the format...**\n▓▁▁▁▁▁▁▁▁▁ 📥 Downloading from social media..."
            )
            tmp_path, dl_error = await download_with_ytdlp(source_url)

            if (not tmp_path) and (INSTAGRAM_PATTERN.search(source_url) or TIKTOK_PATTERN.search(source_url)):
                if APIFY_API_TOKEN:
                    log.warning(f"[mention] yt-dlp failed ({dl_error}). Trying Apify...")
                    try:
                        await progress_msg.edit(
                            content="⏳ **Vexi is studying the format...**\n▓▓▁▁▁▁▁▁▁▁ 🔄 yt-dlp blocked, trying Apify..."
                        )
                    except Exception:
                        pass
                    async with aiohttp.ClientSession() as sess:
                        tmp_path, apify_err = await download_with_apify(source_url, sess)
                    if not tmp_path:
                        dl_error = f"yt-dlp: {dl_error} | apify: {apify_err}"

            if not tmp_path:
                study = {"error": f"DOWNLOAD_FAILED: {dl_error or 'Unknown error'}"}
            else:
                tmp_dir = os.path.dirname(tmp_path)
                downloaded_video_path_for_discord = tmp_path
                study = await _analyze_local_file_with_gemini(tmp_path, prompt=active_prompt, response_json=True)
        elif is_gdrive:
            async with aiohttp.ClientSession() as sess:
                study = await analyze_video_with_gemini_upload(source_url, sess, prompt=active_prompt, response_json=True)
        else:
            mime = guess_mime_type(source_url)
            study = await analyze_video_with_gemini(source_url, mime_type=mime, prompt=active_prompt, response_json=True)
            if "error" in study:
                async with aiohttp.ClientSession() as sess:
                    study = await analyze_video_with_gemini_upload(source_url, sess, prompt=active_prompt, response_json=True)

        progress_done.set()
        await progress_task

        _, embeds, script_file, script_text = build_study_message(study, source_label=source_label)

        try:
            await progress_msg.delete()
        except Exception:
            pass

        files: list[discord.File] = []
        if downloaded_video_path_for_discord and os.path.exists(downloaded_video_path_for_discord):
            ext = Path(downloaded_video_path_for_discord).suffix.lower()
            friendly_name = f"{source_label.lower().replace(' ', '_')}{ext}"
            files.append(discord.File(downloaded_video_path_for_discord, filename=friendly_name))
        if script_file is not None:
            files.append(script_file)

        send_kwargs: dict = {"embeds": embeds}
        if files:
            send_kwargs["files"] = files

        try:
            posted = await visible_msg.reply(**send_kwargs)
        except discord.HTTPException as e:
            if e.status == 413 and "files" in send_kwargs:
                keep = [f for f in files if f.filename == "manus_script.txt"]
                if keep:
                    send_kwargs["files"] = keep
                else:
                    del send_kwargs["files"]
                posted = await visible_msg.reply(**send_kwargs)
                await visible_msg.reply(content=f"📹 Video too large to attach — original: {source_url}")
            else:
                raise

        if script_text and posted is not None:
            _cache_script(posted.id, script_text)

    except Exception as e:
        progress_done.set()
        await progress_task
        log.error(f"[mention] Study error: {type(e).__name__}: {e}")
        await progress_msg.edit(
            content=f"❌ Something went wrong. Please try again.\nError: {str(e)[:200]}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        await message.remove_reaction("🔍", bot.user)
        await message.add_reaction("✅")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Auto-Detect in Configured Channels
# ---------------------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # @Vexi study <url> — works in any channel
    if bot.user in message.mentions:
        # Ignore replies to Vexi's own messages (Discord auto-includes the mention on reply)
        if message.reference is not None:
            ref = message.reference.resolved
            if isinstance(ref, discord.Message) and ref.author.id == bot.user.id:
                await bot.process_commands(message)
                return
        content = message.content
        for mention_str in (f"<@{bot.user.id}>", f"<@!{bot.user.id}>"):
            content = content.replace(mention_str, "").strip()
        await handle_mention_study(message, content)
        return

    if message.channel.id not in VEXI_CHANNEL_IDS:
        await bot.process_commands(message)
        return

    video_url = extract_video_source(message)
    if not video_url:
        await bot.process_commands(message)
        return

    try:
        await message.add_reaction("🔍")
    except Exception:
        pass

    thinking_msg = await message.reply(
        "⏳ **Vexi is analyzing your video...**\n▁▁▁▁▁▁▁▁▁▁ Sending to AI..."
    )

    # Progress animation
    progress_done = asyncio.Event()

    async def update_progress():
        stages = [
            ("▓▓▁▁▁▁▁▁▁▁", "🧠 AI is watching your video..."),
            ("▓▓▓▓▁▁▁▁▁▁", "🧠 AI is watching your video..."),
            ("▓▓▓▓▓▁▁▁▁▁", "🔍 Checking legal compliance..."),
            ("▓▓▓▓▓▓▁▁▁▁", "🔍 Checking legal compliance..."),
            ("▓▓▓▓▓▓▓▁▁▁", "🎬 Reviewing hooks & storytelling..."),
            ("▓▓▓▓▓▓▓▓▁▁", "🎬 Reviewing hooks & storytelling..."),
            ("▓▓▓▓▓▓▓▓▓▁", "📝 Writing up the review..."),
            ("▓▓▓▓▓▓▓▓▓▓", "✅ Almost done..."),
        ]
        for bar, status in stages:
            if progress_done.is_set():
                return
            try:
                await thinking_msg.edit(
                    content=f"⏳ **Vexi is analyzing your video...**\n{bar} {status}"
                )
            except Exception:
                return
            try:
                await asyncio.wait_for(progress_done.wait(), timeout=15)
                return
            except asyncio.TimeoutError:
                continue

    progress_task = asyncio.create_task(update_progress())

    att_filename = ""
    for att in message.attachments:
        if Path(att.filename).suffix.lower() in VIDEO_EXTENSIONS:
            att_filename = att.filename
            break
    mime = guess_mime_type(video_url, att_filename)

    try:
        # Discord CDN URLs are signed/temporary and Drive URLs need interstitial
        # handling — both go straight to the upload path.
        is_discord_cdn = bool(DISCORD_CDN_PATTERN.match(video_url))
        is_gdrive = is_gdrive_url(video_url)

        if is_discord_cdn or is_gdrive:
            log.info(f"{'Discord CDN' if is_discord_cdn else 'Drive'} URL — using upload path directly.")
            async with aiohttp.ClientSession() as session:
                review = await analyze_video_with_gemini_upload(video_url, session)
        else:
            review = await analyze_video_with_gemini(video_url, mime_type=mime)
            if "error" in review:
                log.info(f"Direct URL failed ({review['error']}), trying upload fallback...")
                async with aiohttp.ClientSession() as session:
                    review = await analyze_video_with_gemini_upload(video_url, session)

        # Stop progress
        progress_done.set()
        await progress_task

        if "error" in review:
            await thinking_msg.edit(
                content=(
                    f"❌ I couldn't process that video. Error: {review['error']}\n\n"
                    "Please make sure:\n"
                    "• The file is a supported format (.mp4, .mov, .webm)\n"
                    "• Google Drive links have sharing set to 'Anyone with the link'\n"
                    "• The file is under 100MB"
                )
            )
            return

        content_text, embeds = build_review_message(review, creator=f"{message.author.mention}")

        # Coach ping
        ping_text = content_text or ""
        if COACH_ROLE_ID:
            ping_text = f"🏷️ <@&{COACH_ROLE_ID}> — new video submitted by {message.author.mention} for review!"

        await thinking_msg.edit(content=ping_text if ping_text else None, embeds=embeds)
    except Exception as e:
        progress_done.set()
        await progress_task
        log.error(f"Auto-detect error: {type(e).__name__}: {e}")
        await thinking_msg.edit(
            content=f"❌ Something went wrong during the review. Please try again.\nError: {str(e)[:200]}"
        )

    try:
        await message.remove_reaction("🔍", bot.user)
        await message.add_reaction("✅")
    except Exception:
        pass

    await bot.process_commands(message)


# ---------------------------------------------------------------------------
# Revise Mode — context menu on Vexi's /study output
# ---------------------------------------------------------------------------
REVISE_PROMPT = MANUS_KNOWLEDGE + """
You are Vexi in Revise Mode. Rewrite the ORIGINAL SCRIPT below to follow the USER INSTRUCTION.

Rules:
- Keep it about Manus. Use MANUS GROUND TRUTH.
- Preserve the beat-marker structure: [HOOK 0-3s], [BEAT 1 3-8s], [BEAT 2 ...], ..., [CTA]. Adjust timings if the user asked for a shorter/longer video.
- Every beat has spoken line(s) plus a "(visual: ...)" cue.
- Compliance-safe: no income claims, no absolute claims ("100%", "replaces humans"), no competitor brand mentions, no fake testimonials.
- End with a Manus CTA and a soft ad-disclosure hashtag (#ManusAd or similar).
- Return ONLY a valid JSON object (no markdown, no code fences):

{"full_script": "..."}
"""


async def revise_script_via_gemini(original: str, instruction: str) -> dict:
    """Text-only Gemini call: revise a script per the user's instruction."""
    prompt = (
        REVISE_PROMPT
        + f"\n\nORIGINAL SCRIPT:\n{original}\n\nUSER INSTRUCTION:\n{instruction}\n"
    )
    try:
        config = genai_types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json",
        )
        response = await _gemini_generate([prompt], config=config)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        result = _parse_json_with_repair(raw)
        if result is None:
            return {"error": f"AI returned invalid JSON. Raw excerpt: {raw[:300]}"}
        return result
    except Exception as e:
        log.error(f"revise_script_via_gemini error: {type(e).__name__}: {e}")
        return {"error": str(e)}


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


class ReviseModal(discord.ui.Modal, title="Revise this script"):
    instruction = discord.ui.TextInput(
        label="How should Vexi revise the script?",
        style=discord.TextStyle.paragraph,
        placeholder="e.g. make it more TikTok, shorten to 30s, make it casual, alternative angle",
        required=True,
        max_length=500,
    )

    def __init__(self, original_script: str, target_message: discord.Message):
        super().__init__()
        self.original_script = original_script
        self.target_message = target_message

    async def on_submit(self, interaction: discord.Interaction):
        try:
            await interaction.response.defer(thinking=True, ephemeral=True)
        except Exception as e:
            log.warning(f"Revise defer failed: {e}")
            return

        instruction_text = str(self.instruction).strip()
        log.info(f"Revise request from {interaction.user}: {instruction_text[:100]}")

        result = await revise_script_via_gemini(self.original_script, instruction_text)
        if "error" in result:
            await interaction.followup.send(
                f"❌ Couldn't revise the script: {result['error'][:200]}", ephemeral=True
            )
            return

        new_script = (result.get("full_script") or "").strip()
        if not new_script:
            await interaction.followup.send(
                "❌ Gemini didn't return a revised script. Try again with a clearer instruction.",
                ephemeral=True,
            )
            return

        wrapped = f"```\n{new_script}\n```"
        files: list[discord.File] = []
        if len(wrapped) <= 3900:
            desc = (
                f"**Instruction:** *{instruction_text}*\n"
                f"{wrapped}\n"
                "*Right-click this message → Apps → 'Revise this script' to iterate again.*"
            )
        else:
            files.append(discord.File(io.BytesIO(new_script.encode("utf-8")), filename="manus_script_revised.txt"))
            desc = (
                f"**Instruction:** *{instruction_text}*\n"
                "Revised script was long — attached as `manus_script_revised.txt`.\n"
                "*Right-click this message → Apps → 'Revise this script' to iterate again.*"
            )

        embed = discord.Embed(
            title="🔁 Vexi Revised Script",
            description=desc,
            color=discord.Color.blurple(),
        )
        embed.set_footer(text="Vexi Study Mode • Revised • v1.3")

        posted: discord.Message | None = None
        try:
            posted = await self.target_message.reply(embed=embed, files=files)
        except (discord.HTTPException, discord.NotFound) as e:
            log.warning(f"Revise reply failed ({e}), posting via followup instead.")
            try:
                posted = await interaction.followup.send(embed=embed, files=files)
            except Exception as e2:
                log.error(f"Revise followup send also failed: {e2}")

        if posted is not None:
            _cache_script(posted.id, new_script)
            try:
                await interaction.followup.send("✅ Revised script posted below.", ephemeral=True)
            except Exception:
                pass


@bot.tree.context_menu(name="Revise this script")
async def revise_context_menu(interaction: discord.Interaction, message: discord.Message):
    if message.author.id != bot.user.id:
        await interaction.response.send_message(
            "This only works on Vexi's own /study messages.", ephemeral=True
        )
        return

    is_study = any(
        (e.footer and e.footer.text and "Vexi Study Mode" in e.footer.text)
        for e in message.embeds
    )
    if not is_study:
        await interaction.response.send_message(
            "This doesn't look like a Vexi /study result. Right-click a Vexi Study Mode message and try again.",
            ephemeral=True,
        )
        return

    script = _get_cached_script(message.id) or _extract_script_from_embed(message)
    if not script:
        script = await _extract_script_from_attachment(message)
    if not script:
        await interaction.response.send_message(
            "I couldn't find the script in this message (cache may have expired). Run /study again and try revise on the fresh result.",
            ephemeral=True,
        )
        return

    await interaction.response.send_modal(ReviseModal(script, message))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        log.error("DISCORD_BOT_TOKEN not set!")
        exit(1)
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set!")
        exit(1)
    bot.run(DISCORD_BOT_TOKEN)
