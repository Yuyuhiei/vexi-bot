"""Unit tests for the pure deterministic evidence layer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vexi.deterministic import (  # noqa: E402
    compute_deterministic,
    correct_brand_homophones,
    derive_verdict,
    detect_cta_keyword,
    distinct_websites,
    evidence_window,
    first_manus_signal,
    manus_feature_hits,
    scan_text_for_homophones,
    spelling_findings_from_vision,
    _visible_spans,
)


def frame(t0, t1, **kw):
    base = {
        "t_start": t0, "t_end": t1, "ocr_text": "",
        "manus_logo_visible": False, "manus_ui_visible": False,
        "website_or_app_shown": None, "inside_manus_panel": False,
        "manus_feature": None, "scene_description": "", "people_present": False,
    }
    base.update(kw)
    return base


def transcript(*segs, status="ok", language="en"):
    return {
        "status": status, "language": language,
        "segments": [{"start": s, "end": e, "text": t} for (s, e, t) in segs],
    }


# --- homophones -----------------------------------------------------------
def test_homophone_flagged_with_agent_context():
    hits = scan_text_for_homophones("I asked Manners to build me a website", 7.0, "speech")
    assert len(hits) == 1
    assert hits[0]["read"] == "Manners"
    assert hits[0]["corrected_to"] == "Manus"


def test_homophone_not_flagged_without_context():
    assert scan_text_for_homophones("Please mind your manners at dinner", 3.0, "speech") == []


def test_man_is_bigram_flagged_in_context():
    hits = scan_text_for_homophones("Man is building my app with one prompt", 2.0, "caption")
    assert any(h["reason"] == "homophone" for h in hits)


def test_hard_misspelling_on_captions_needs_no_context():
    hits = scan_text_for_homophones("Mannus changed everything for me", 1.0, "caption")
    assert len(hits) == 1
    assert hits[0]["reason"] in ("misspelling", "homophone")


def test_correct_brand_homophones_rewrites_segments():
    t = transcript((0.0, 3.0, "I asked Manners to build my site"))
    out = correct_brand_homophones(t)
    assert out["segments"][0]["text"] == "I asked Manus to build my site"
    assert out["corrections"][0]["read"] == "Manners"
    assert out["corrections"][0]["t"] == 0.0


def test_correct_brand_homophones_leaves_clean_speech():
    t = transcript((0.0, 3.0, "Mind your manners, kids"))
    out = correct_brand_homophones(t)
    assert out["segments"][0]["text"] == "Mind your manners, kids"
    assert out["corrections"] == []


# --- OCR spelling ----------------------------------------------------------
def test_wrong_domain_detected():
    frames = [frame(0, 5, ocr_text="Go to manus.com right now to try the AI agent")]
    findings = spelling_findings_from_vision(frames)
    assert any(f["reason"] == "wrong_domain" and f["found"].lower() == "manus.com" for f in findings)


def test_caption_homophone_detected_with_timestamp():
    frames = [frame(7, 9, ocr_text="I asked Manners to build me a website")]
    findings = spelling_findings_from_vision(frames)
    assert len(findings) == 1
    assert findings[0]["t"] == 7


# --- span arithmetic -------------------------------------------------------
def test_logo_spans_merge_consecutive_frames():
    frames = [
        frame(11, 12), frame(12, 13, manus_logo_visible=True),
        frame(13, 14, manus_logo_visible=True), frame(14, 15, manus_logo_visible=True),
        frame(15, 16), frame(20, 22, manus_logo_visible=True),
    ]
    spans = _visible_spans(frames, "manus_logo_visible")
    assert spans == [[12.0, 15.0], [20.0, 22.0]]


# --- websites --------------------------------------------------------------
def test_website_count_excludes_inside_manus_panel():
    frames = [
        frame(0, 5, website_or_app_shown="refero.design"),
        frame(5, 10, website_or_app_shown="https://www.motion.dev/docs"),
        frame(10, 15, website_or_app_shown="nike.com", inside_manus_panel=True),
        frame(15, 20, website_or_app_shown="manus.im"),
        frame(20, 25, website_or_app_shown="refero.design"),  # duplicate
    ]
    sites = distinct_websites(frames)
    assert sites == ["refero.design", "motion.dev", "manus.im"]


# --- features --------------------------------------------------------------
def test_feature_late_in_video_counts():
    frames = [frame(10, 12, manus_feature="website_builder"), frame(40, 45, manus_feature="publishing")]
    hits = manus_feature_hits(frames)
    assert {h["feature"] for h in hits} == {"website_builder", "publishing"}


# --- windows + CTA ---------------------------------------------------------
def test_windows_at_clip_edges():
    frames = [frame(0, 3, ocr_text="5 AI tips"), frame(41, 45, ocr_text="Comment MANUS")]
    t = transcript((0.5, 2.5, "you need to see this"), (42.0, 44.5, "comment Manus and I'll send it"))
    hook = evidence_window(frames, t, 0.0, 5.0)
    cta = evidence_window(frames, t, 45.0 - 8.0, 45.0)
    assert hook["ocr_text"] == ["5 AI tips"] and hook["speech"] == ["you need to see this"]
    assert cta["ocr_text"] == ["Comment MANUS"]
    assert detect_cta_keyword(cta) == "MANUS"


def test_no_cta_keyword():
    assert detect_cta_keyword({"speech": ["thanks for watching"], "ocr_text": []}) is None


# --- first signal ----------------------------------------------------------
def test_first_manus_signal_earliest_wins():
    frames = [frame(3, 4, manus_ui_visible=True)]
    t = transcript((1.0, 2.0, "I let Manus do it"))
    sig = first_manus_signal(frames, t)
    assert sig == {"t": 1.0, "type": "spoken"}


def test_first_manus_signal_none():
    assert first_manus_signal([frame(0, 5)], transcript(status="no_speech")) is None


# --- full block ------------------------------------------------------------
def test_compute_deterministic_plug_any_one_suffices():
    # No logo, no mention — but CTA keyword present → plug satisfied
    frames = [frame(50, 58, ocr_text="Comment MANUS for the link")]
    det = compute_deterministic({"frames": frames}, transcript(status="no_speech"), 58.0)
    assert det["plug"]["cta_keyword"] == "MANUS"
    assert det["plug"]["plug_satisfied"] is True
    assert det["plug"]["logo_ge_2s"] is False


def test_compute_deterministic_six_websites_over_cap():
    frames = [
        frame(i * 5, i * 5 + 5, website_or_app_shown=site)
        for i, site in enumerate(["a.com", "b.com", "c.com", "d.com", "e.com", "manus.im"])
    ]
    det = compute_deterministic({"frames": frames}, transcript(status="no_speech"), 30.0)
    assert det["websites"]["count"] == 6
    assert det["websites"]["over_cap"] is True


# --- verdict routing -------------------------------------------------------
def test_derive_verdict_truth_table():
    assert derive_verdict(False, []) == "NOT MANUS CONTENT"
    assert derive_verdict(True, []) == "LOOKS GOOD"
    assert derive_verdict(True, [{"severity": "recommend", "risk": "LOW"}]) == "LOOKS GOOD"
    assert derive_verdict(True, [{"severity": "flag", "risk": "MEDIUM"}]) == "NEEDS REVIEW"
    assert derive_verdict(True, [{"severity": "flag", "risk": "HIGH"}]) == "COACH ATTENTION NEEDED"
    assert derive_verdict(True, [
        {"severity": "flag", "risk": "MEDIUM"},
        {"severity": "recommend", "risk": "LOW"},
        {"severity": "flag", "risk": "AUTO-REJECT"},
    ]) == "AUTO-REJECT"
    # Recommends never escalate, even at HIGH risk labels
    assert derive_verdict(True, [{"severity": "recommend", "risk": "HIGH"}]) == "LOOKS GOOD"
