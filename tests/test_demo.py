"""Tests for the demo page: the saved session is inlined safely into the template."""

import json

from hcs.demo import render


def test_the_session_is_inlined_and_cannot_close_the_script_tag(tmp_path):
    session = tmp_path / "session.json"
    page = tmp_path / "index.html"
    session.write_text(json.dumps({"case": {"id": "present-1"}, "note": "</script><b>x"}))

    render(session, page)
    html = page.read_text(encoding="utf-8")

    assert "/*SESSION*/null" not in html
    assert '"id": "present-1"' in html
    assert "</script><b>" not in html
    assert html.count("</script>") == 1
