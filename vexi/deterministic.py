"""Deterministic evidence computations for the multiagent pipeline.

Pure functions, zero I/O, zero LLM calls — everything here is arithmetic and
pattern matching over the extractor logs, so it can never hallucinate and is
trivially unit-testable. The adjudicator receives these results as ground
truth that outranks every model opinion.
"""

import re

# ---------------------------------------------------------------------------
# Brand lexicons
# ---------------------------------------------------------------------------
BRAND = "manus"
BRAND_DOMAIN = "manus.im"

# Real English words auto-captions/ASR substitute for "Manus". Judged by
# sentence context (agent-verb cues nearby), never blindly.
HOMOPHONES = {
    "manners", "menace", "manis", "manas", "mannus", "manuscript",
    "manor", "menus", "minus", "magnus", "maus", "mauns", "manos", "mantis",
    "maneus", "manius",
}
# Two-word mishearing: "man is"
HOMOPHONE_BIGRAM = re.compile(r"\bman\s+is\b", re.IGNORECASE)

# Non-word misspellings — these are never legitimate English, so no context
# gate is needed when they appear on screen.
HARD_MISSPELLINGS = {"manis", "mannus", "maus", "mauns", "manuss", "mansu", "maneus", "manius"}

# Note: manus.ai is NOT in this list — open.manus.ai (API) and mail.manus.ai
# (Mail Manus) are official Manus domains.
WRONG_DOMAIN_RE = re.compile(
    r"\b(?:manus\.(?:com|io|co|net)|man[ia]s\.im|mannus\.im|maus\.im|mauns\.im)\b",
    re.IGNORECASE,
)

# Context cues that suggest the sentence is talking about the Manus product —
# the gate that separates "I asked Manners to build a site" from "mind your
# manners".
CONTEXT_CUES = re.compile(
    r"\b(?:ask(?:ed|ing|s)?|tell|told|open(?:ed|ing|s)?|use(?:d|s)?|using|"
    r"built|build(?:s|ing)?|go(?:es|ing)?\s+to|went\s+to|tr(?:y|ied|ying)|"
    r"prompt(?:ed|ing|s)?|agent|ai|tool|app|website|site|create[ds]?|"
    r"generat(?:e|ed|es|ing)|comment|link|automation|task)\b"
    r"|\.im\b",
    re.IGNORECASE,
)

CTA_KEYWORD_RE = re.compile(r"\bcomment\s+[\"'“‘]?([A-Za-z]{2,20})", re.IGNORECASE)
# Filler words after "comment" that are NOT trigger keywords ("comment below…").
CTA_STOPWORDS = {
    "below", "down", "here", "now", "this", "that", "your", "what", "and",
    "the", "for", "with", "on", "in", "if", "me", "us", "it", "them", "to",
    "something", "anything", "done", "away",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
# Word-boundary brand match — "manuscript" must NOT count as a Manus mention.
_BRAND_WORD_RE = re.compile(r"\bmanus\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Homophone / spelling scanning
# ---------------------------------------------------------------------------
def _context_supports_brand(text: str) -> bool:
    return bool(CONTEXT_CUES.search(text))


def scan_text_for_homophones(text: str, t: float | None, source: str) -> list[dict]:
    """Return correction records for brand homophones in `text`, gated on the
    sentence containing agent/product context cues. `source` is
    'speech' or 'caption'."""
    if not text:
        return []
    found: list[dict] = []
    lowered = text.lower()

    has_context = _context_supports_brand(text)

    for m in _WORD_RE.finditer(text):
        word = m.group(0).lower()
        if word == BRAND:
            continue
        if word in HARD_MISSPELLINGS and source == "caption":
            # Non-words on screen are always errors, context or not.
            found.append({
                "read": m.group(0), "corrected_to": "Manus", "t": t,
                "source": source, "context": text.strip()[:160], "reason": "misspelling",
            })
            continue
        if word in HOMOPHONES and has_context:
            found.append({
                "read": m.group(0), "corrected_to": "Manus", "t": t,
                "source": source, "context": text.strip()[:160], "reason": "homophone",
            })

    if HOMOPHONE_BIGRAM.search(text) and has_context and BRAND not in lowered:
        bigram = HOMOPHONE_BIGRAM.search(text).group(0)
        found.append({
            "read": bigram, "corrected_to": "Manus", "t": t,
            "source": source, "context": text.strip()[:160], "reason": "homophone",
        })
    return found


def correct_brand_homophones(transcript: dict) -> dict:
    """Apply homophone corrections to transcript segments in place (adding a
    `corrections` list). Original wording is preserved inside each correction
    record; segment text is rewritten so downstream consumers see 'Manus'."""
    if not isinstance(transcript, dict) or transcript.get("status") != "ok":
        return transcript

    corrections: list[dict] = []
    for seg in transcript.get("segments", []):
        text = seg.get("text", "")
        hits = scan_text_for_homophones(text, seg.get("start"), "speech")
        if not hits:
            continue
        corrections.extend(hits)
        new_text = text
        for h in hits:
            new_text = re.sub(rf"\b{re.escape(h['read'])}\b", "Manus", new_text)
        seg["text"] = new_text

    transcript["corrections"] = corrections
    return transcript


def spelling_findings_from_vision(frames: list[dict]) -> list[dict]:
    """Scan verbatim OCR text for misspellings, wrong domains, and contextual
    homophones burned into captions."""
    findings: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for f in frames:
        text = f.get("ocr_text") or ""
        if not text:
            continue
        t = f.get("t_start")
        for m in WRONG_DOMAIN_RE.finditer(text):
            key = ("domain", m.group(0).lower())
            if key not in seen:
                seen.add(key)
                findings.append({
                    "found": m.group(0), "expected": BRAND_DOMAIN, "t": t,
                    "source": "ocr", "context": text.strip()[:160], "reason": "wrong_domain",
                })
        for h in scan_text_for_homophones(text, t, "caption"):
            key = (h["reason"], h["read"].lower())
            if key not in seen:
                seen.add(key)
                findings.append({
                    "found": h["read"], "expected": "Manus", "t": t,
                    "source": "ocr", "context": h["context"], "reason": h["reason"],
                })
    return findings


# ---------------------------------------------------------------------------
# Span arithmetic over the vision log
# ---------------------------------------------------------------------------
def _visible_spans(frames: list[dict], key: str) -> list[list[float]]:
    """Merge consecutive frames where `key` is true into [start, end] spans.
    Frames tile the timeline (t_end of one == t_start of the next), so
    adjacency in the list is adjacency in time."""
    spans: list[list[float]] = []
    for f in frames:
        if not f.get(key):
            continue
        start, end = float(f["t_start"]), float(f["t_end"])
        if spans and abs(spans[-1][1] - start) < 1e-6:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return spans


def _spans_total(spans: list[list[float]]) -> float:
    return round(sum(end - start for start, end in spans), 2)


def _norm_site(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.split("/")[0].strip()
    # Alias every Manus surface to one canonical entry so "Manus" + "manus.im"
    # + a published manus.space site never count as separate showcased tools.
    if s in ("manus", "manus.im", "manus.space", "manus app", "manus.ai") or s.endswith(".manus.space"):
        return "manus.im"
    return s


def distinct_websites(frames: list[dict]) -> list[str]:
    """Distinct sites/tools the CREATOR showcased (sites the Manus agent was
    merely using — inside_manus_panel — are excluded, matching the ≤5 rule)."""
    seen: list[str] = []
    for f in frames:
        site = f.get("website_or_app_shown")
        if not site or f.get("inside_manus_panel"):
            continue
        norm = _norm_site(str(site))
        if norm and norm not in seen:
            seen.append(norm)
    return seen


def manus_feature_hits(frames: list[dict]) -> list[dict]:
    """Distinct Manus features tagged in the vision log, with timestamps."""
    hits: dict[str, dict] = {}
    for f in frames:
        feat = f.get("manus_feature")
        if not feat:
            continue
        entry = hits.setdefault(str(feat), {"feature": str(feat), "timestamps": []})
        entry["timestamps"].append(float(f["t_start"]))
    return list(hits.values())


def _text_mentions_brand(text: str) -> bool:
    return bool(_BRAND_WORD_RE.search(text or ""))


def first_manus_signal(frames: list[dict], transcript: dict) -> dict | None:
    """Earliest Manus evidence across every log. Returns {'t':, 'type':} or None."""
    candidates: list[tuple[float, str]] = []
    for f in frames:
        t = float(f.get("t_start", 0.0))
        if f.get("manus_logo_visible"):
            candidates.append((t, "logo"))
        if f.get("manus_ui_visible"):
            candidates.append((t, "ui"))
        if _text_mentions_brand(f.get("ocr_text", "")):
            candidates.append((t, "ocr"))
        if f.get("manus_feature"):
            candidates.append((t, "feature"))
        site = _norm_site(str(f.get("website_or_app_shown") or ""))
        if site in (BRAND_DOMAIN, "manus.space") or site.endswith(".manus.space"):
            candidates.append((t, "url"))
    if isinstance(transcript, dict) and transcript.get("status") == "ok":
        for seg in transcript.get("segments", []):
            if _text_mentions_brand(seg.get("text", "")):
                candidates.append((float(seg.get("start", 0.0)), "spoken"))
                break
    if not candidates:
        return None
    t, sig_type = min(candidates, key=lambda c: c[0])
    return {"t": round(t, 1), "type": sig_type}


# ---------------------------------------------------------------------------
# Evidence windows (hook = first 5s, CTA = last 8s)
# ---------------------------------------------------------------------------
def evidence_window(frames: list[dict], transcript: dict, t0: float, t1: float) -> dict:
    ocr, scenes, speech = [], [], []
    for f in frames:
        if float(f.get("t_end", 0)) <= t0 or float(f.get("t_start", 0)) >= t1:
            continue
        if f.get("ocr_text"):
            ocr.append(f["ocr_text"])
        if f.get("scene_description"):
            scenes.append(f["scene_description"])
    if isinstance(transcript, dict) and transcript.get("status") == "ok":
        for seg in transcript.get("segments", []):
            if float(seg.get("end", 0)) <= t0 or float(seg.get("start", 0)) >= t1:
                continue
            if seg.get("text"):
                speech.append(seg["text"])
    return {
        "t0": round(t0, 1), "t1": round(t1, 1),
        "ocr_text": ocr, "scene_descriptions": scenes, "speech": speech,
    }


def detect_cta_keyword(window: dict) -> str | None:
    for text in list(window.get("speech", [])) + list(window.get("ocr_text", [])):
        for m in CTA_KEYWORD_RE.finditer(text or ""):
            word = m.group(1)
            if word.lower() not in CTA_STOPWORDS:
                return word.upper()
    return None


# ---------------------------------------------------------------------------
# The full deterministic block
# ---------------------------------------------------------------------------
def compute_deterministic(vision_log: dict, transcript: dict, duration_s: float) -> dict:
    vision_ok = isinstance(vision_log, dict) and vision_log.get("status") == "ok"
    frames = vision_log.get("frames", []) if isinstance(vision_log, dict) else []

    logo_spans = _visible_spans(frames, "manus_logo_visible")
    ui_spans = _visible_spans(frames, "manus_ui_visible")
    logo_total = _spans_total(logo_spans)

    speech_ok = isinstance(transcript, dict) and transcript.get("status") == "ok"
    mentioned_spoken = speech_ok and any(
        _text_mentions_brand(s.get("text", "")) for s in transcript.get("segments", [])
    )
    mentioned_ocr = any(_text_mentions_brand(f.get("ocr_text", "")) for f in frames)

    hook_window = evidence_window(frames, transcript, 0.0, 5.0)
    cta_window = evidence_window(frames, transcript, max(0.0, duration_s - 8.0), duration_s)
    cta_keyword = detect_cta_keyword(cta_window)

    websites = distinct_websites(frames)
    features = manus_feature_hits(frames)

    # Brand-presence bar: logo ≥2s OR any Manus mention OR "comment MANUS"
    # specifically as the CTA trigger word (a generic keyword like PROMPT
    # counts as a CTA, but not as brand presence).
    cta_is_brand = cta_keyword is not None and cta_keyword.lower() == BRAND
    plug_satisfied = bool(logo_total >= 2.0 or mentioned_spoken or mentioned_ocr or cta_is_brand)

    det = {
        "video_duration_s": round(duration_s, 1),
        # When the vision extractor failed, every vision-derived number below is
        # VOID, not zero — the adjudicator is told to skip those checks.
        "vision_available": vision_ok,
        "speech_present": speech_ok and bool(transcript.get("segments")),
        "language": transcript.get("language") if speech_ok else None,
        "homophone_corrections": transcript.get("corrections", []) if speech_ok else [],
        "spelling_findings": spelling_findings_from_vision(frames),
        "plug": {
            "logo_total_s": logo_total,
            "logo_spans": logo_spans,
            "logo_ge_2s": logo_total >= 2.0,
            "manus_ui_total_s": _spans_total(ui_spans),
            "mentioned_spoken": bool(mentioned_spoken),
            "mentioned_ocr": bool(mentioned_ocr),
            "cta_keyword": cta_keyword,
            "cta_keyword_is_brand": cta_is_brand,
            "plug_satisfied": plug_satisfied,
        },
        "first_manus_signal": first_manus_signal(frames, transcript),
        "websites": {
            "count": len(websites),
            "names": websites,
            "over_cap": len(websites) > 5,
        },
        "features": {
            "count": len(features),
            "list": features,
        },
        "hook_window": hook_window,
        "cta_window": cta_window,
    }

    if not vision_ok:
        det["plug"] = {"unavailable": True, "cta_keyword": cta_keyword, "cta_keyword_is_brand": cta_is_brand,
                       "mentioned_spoken": bool(mentioned_spoken)}
        det["websites"] = {"unavailable": True}
        det["features"] = {"unavailable": True}
        det["spelling_findings"] = []
    return det


# ---------------------------------------------------------------------------
# Verdict routing — code has the final word, not the model
# ---------------------------------------------------------------------------
VERDICTS = ("LOOKS GOOD", "NEEDS REVIEW", "COACH ATTENTION NEEDED", "AUTO-REJECT", "NOT MANUS CONTENT")


def derive_verdict(manus_relevant: bool, findings: list[dict]) -> str:
    if not manus_relevant:
        return "NOT MANUS CONTENT"
    risks = set()
    for f in findings or []:
        severity = str(f.get("severity", "")).lower()
        risk = str(f.get("risk", "")).upper()
        if risk == "AUTO-REJECT":
            return "AUTO-REJECT"
        if severity == "flag":
            risks.add(risk)
    if "HIGH" in risks:
        return "COACH ATTENTION NEEDED"
    if "MEDIUM" in risks:
        return "NEEDS REVIEW"
    return "LOOKS GOOD"
