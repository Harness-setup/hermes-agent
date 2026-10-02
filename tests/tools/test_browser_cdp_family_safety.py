"""Unit tests for browser_cdp_tool's Windows Family Safety block detection.

Pure-function tests -- no CDP server needed, unlike test_browser_cdp_tool.py's
integration suite. Covers _contains_family_safety_redirect (recursive scan) and
_annotate_family_safety_block (payload annotation), used by both browser_cdp's
own result and browser_navigate's response (tools/browser_tool.py)."""
from tools import browser_cdp_tool as cdp_tool


def test_no_match_on_an_ordinary_result():
    payload = {"success": True, "method": "Target.getTargets",
               "result": {"targetInfos": [{"url": "https://example.com/"}]}}
    assert cdp_tool._contains_family_safety_redirect(payload) is False


def test_matches_restricted_web_url_nested_in_target_infos():
    payload = {"success": True, "method": "Target.getTargets", "result": {
        "targetInfos": [{"targetId": "T1",
                          "url": "https://sdx.microsoft.com/family/restricted-web?url=example.com"}]
    }}
    assert cdp_tool._contains_family_safety_redirect(payload) is True


def test_matches_familysafety_hostname_variant():
    payload = {"result": {"url": "https://familysafety.microsoft.com/something"}}
    assert cdp_tool._contains_family_safety_redirect(payload) is True


def test_matches_inside_a_list():
    payload = {"result": [{"url": "https://sdx.microsoft.com/family/restricted-web"}]}
    assert cdp_tool._contains_family_safety_redirect(payload) is True


def test_case_insensitive_match():
    payload = {"result": {"url": "HTTPS://SDX.MICROSOFT.COM/FAMILY/RESTRICTED-WEB"}}
    assert cdp_tool._contains_family_safety_redirect(payload) is True


def test_non_string_non_container_values_are_ignored():
    assert cdp_tool._contains_family_safety_redirect(None) is False
    assert cdp_tool._contains_family_safety_redirect(42) is False
    assert cdp_tool._contains_family_safety_redirect(True) is False


def test_annotate_adds_notice_and_flag_when_detected():
    payload = {"success": True, "method": "Page.navigate",
               "result": {"url": "https://sdx.microsoft.com/family/restricted-web?url=example.com"}}
    annotated = cdp_tool._annotate_family_safety_block(payload)
    assert annotated is payload
    assert annotated["family_safety_block_detected"] is True
    assert "Family Safety" in annotated["notice"]
    assert "never attempt to bypass" in annotated["notice"]


def test_annotate_is_a_noop_when_not_detected():
    payload = {"success": True, "method": "Target.getTargets", "result": {"targetInfos": []}}
    annotated = cdp_tool._annotate_family_safety_block(payload)
    assert "family_safety_block_detected" not in annotated
    assert "notice" not in annotated


def test_annotate_never_raises_on_a_scan_failure(monkeypatch):
    def boom(_value):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(cdp_tool, "_contains_family_safety_redirect", boom)
    payload = {"success": True, "result": {"url": "https://example.com"}}
    annotated = cdp_tool._annotate_family_safety_block(payload)  # must not raise
    assert "family_safety_block_detected" not in annotated


def test_annotate_works_on_a_flat_top_level_payload_like_browser_navigate():
    # browser_navigate's response has "url"/"title" at the top level, not nested
    # under "result" -- confirm the scan isn't hardcoded to the CDP shape.
    payload = {"success": True, "url": "https://sdx.microsoft.com/family/restricted-web?url=example.com",
               "title": "Restricted"}
    annotated = cdp_tool._annotate_family_safety_block(payload)
    assert annotated["family_safety_block_detected"] is True
