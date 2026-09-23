"""computer_use's capture path surfaces a Windows Family Safety redirect the same way the CDP
browser tools already do (tools/browser_cdp_tool.py's _contains_family_safety_redirect).

computer_use drives the OS window via screenshots/AX-tree, not CDP, so it never saw that
signature at all before -- if the AX walk captures the address bar (or any visible element)
showing the redirect URL, `_capture_response` must surface the same notice regardless of which
of its three return shapes ends up used (plain text, multimodal image envelope, or aux-vision
routed text).
"""
from __future__ import annotations

import base64
import json
from unittest.mock import patch

from tools.computer_use import tool as cu_tool
from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use.tool import _capture_response

_BLOCKED_URL = "https://sdx.microsoft.com/family/restricted-web?url=https%3A%2F%2Fexample.com"
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)


def _make_capture(*, mode: str, label: str, png_b64: str = _PNG_B64):
    elements = [UIElement(index=0, role="AXTextField", label=label, bounds=(0, 0, 200, 24))]
    raw = base64.b64decode(png_b64, validate=False)
    return CaptureResult(
        mode=mode, width=1280, height=800, png_b64=png_b64, elements=elements,
        app="Edge", window_title="This content is blocked", png_bytes_len=len(raw),
    )


def test_plain_text_capture_surfaces_the_family_safety_notice():
    """mode='ax' forces the has_image=False / plain-text return path."""
    cap = _make_capture(mode="ax", label=_BLOCKED_URL)

    result = _capture_response(cap)

    assert isinstance(result, str)
    payload = json.loads(result)
    assert "FAMILY SAFETY BLOCK DETECTED" in payload["summary"]
    assert "same-day access" in payload["summary"]


def test_multimodal_capture_surfaces_the_family_safety_notice():
    """mode='som' with a real image keeps the multimodal envelope, which must ALSO carry the
    notice -- this is the return path the plain-text path's fix does NOT automatically cover,
    since the two paths build their payload from different points in the function."""
    cap = _make_capture(mode="som", label=_BLOCKED_URL)

    with patch.object(cu_tool, "_should_route_through_aux_vision", return_value=False):
        result = _capture_response(cap)

    assert isinstance(result, dict) and result.get("_multimodal")
    assert "FAMILY SAFETY BLOCK DETECTED" in result["text_summary"]
    assert "FAMILY SAFETY BLOCK DETECTED" in result["content"][0]["text"]


def test_normal_capture_has_no_false_positive():
    cap = _make_capture(mode="ax", label="https://example.com/dashboard")

    result = _capture_response(cap)

    payload = json.loads(result)
    assert "FAMILY SAFETY" not in payload["summary"]
