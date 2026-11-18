"""Tests for the Team Room's web tools (teamroom/browser_tools.py).

No API key, no public internet, no real Chromium needed. web_fetch is
tested against a local HTTP server; the Playwright browser is tested
with a fake playwright module injected into sys.modules, which proves
registration, the navigate/read/click/type/screenshot flow, and the
graceful-degradation path without launching a real browser.

Run:  python tests/test_browser_tools.py
"""
import json
import os
import sys
import tempfile
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["TEAMROOM_MOCK_MODEL"] = "1"  # noqa: E402 — must precede teamroom import
os.environ["TEAMROOM_HOME"] = tempfile.mkdtemp()  # noqa: E402

from looper.registry import registry  # noqa: E402
from teamroom import agents, browser_tools  # noqa: E402
from teamroom.store import RoomStore  # noqa: E402

_MINIMAL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082")

BROWSER_TOOLS = ("browser_navigate", "browser_read", "browser_click",
                 "browser_type", "browser_screenshot")


def _store():
    return RoomStore(Path(tempfile.mkdtemp()) / "t.db")


class _ScriptedClient:
    """Fake provider that pops scripted responses in order."""
    def __init__(self, script):
        self.script = list(script)

    def call(self, messages, tools, deadline=None):
        if not self.script:
            return {"content": "done", "tool_calls": []}
        return self.script.pop(0)


# --- fake playwright ------------------------------------------------------

class _FakeLocator:
    def __init__(self, page, text):
        self.page = page
        self.text = text

    @property
    def first(self):
        return self

    def click(self, timeout=None):
        self.page.clicked.append(("text", self.text))


class _FakePage:
    def __init__(self):
        self.url = "about:blank"
        self._title = ""
        self._text = ""
        self.clicked = []
        self.filled = {}
        self.shots = []
        self.closed = False

    def goto(self, url, timeout=None, wait_until=None):
        self.url = url
        self._title = "Title of " + url
        self._text = "Visible text of " + url

    def title(self):
        return self._title

    def inner_text(self, selector):
        return self._text

    def evaluate(self, expr):
        return self._text

    def click(self, selector, timeout=None):
        if selector.startswith(("#", ".", "input", "button", "a", "[")):
            self.clicked.append(("selector", selector))
            return
        raise Exception(f"no element for selector {selector!r}")

    def get_by_text(self, text, exact=False):
        return _FakeLocator(self, text)

    def fill(self, selector, text, timeout=None):
        self.filled[selector] = text

    def wait_for_load_state(self, state, timeout=None):
        return None

    def is_closed(self):
        return self.closed

    def screenshot(self, path=None, timeout=None):
        Path(path).write_bytes(_MINIMAL_PNG)
        self.shots.append(path)

    def close(self):
        self.closed = True


class _FakeBrowser:
    def __init__(self):
        self.page = _FakePage()

    def new_page(self):
        return self.page

    def close(self):
        pass


class _FakeChromium:
    def __init__(self, browser):
        self.browser = browser

    def launch(self, headless=True, timeout=None):
        return self.browser


class _FakePlaywright:
    def __init__(self):
        self.browser = _FakeBrowser()
        self.chromium = _FakeChromium(self.browser)

    def start(self):
        return self

    def stop(self):
        pass


class _FakeSyncContext:
    """sync_playwright() return value: .start() gives the Playwright object."""
    def __init__(self, pw):
        self._pw = pw

    def start(self):
        return self._pw


def _reset_browser_state():
    browser_tools._state["playwright"] = None
    browser_tools._state["browser"] = None
    browser_tools._state["page"] = None


def _install_fake_playwright():
    pw = _FakePlaywright()
    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.sync_playwright = lambda: _FakeSyncContext(pw)
    pkg = types.ModuleType("playwright")
    pkg.sync_api = sync_api
    sys.modules["playwright"] = pkg
    sys.modules["playwright.sync_api"] = sync_api
    browser_tools._AVAILABLE = None
    _reset_browser_state()
    return pw


def _remove_fake_playwright():
    sys.modules.pop("playwright", None)
    sys.modules.pop("playwright.sync_api", None)
    browser_tools._AVAILABLE = None
    _reset_browser_state()


# --- web_fetch ------------------------------------------------------------

def test_web_fetch_local_server():
    """web_fetch reads a real local page: title + visible text, script
    content stripped; a 404 and an unreachable host are clear errors,
    not crashes."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/page":
                body = (b"<html><head><title>Test Page</title></head><body>"
                        b"<h1>Hello</h1><p>Some visible text here.</p>"
                        b"<script>var hidden = 1;</script></body></html>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        out = json.loads(browser_tools.web_fetch(
            f"http://127.0.0.1:{port}/page"))
        assert out["status"] == 200, out
        assert out["title"] == "Test Page", out
        assert "Hello" in out["text"] and "Some visible text" in out["text"], out
        assert "hidden" not in out["text"], out  # script content stripped
        out = json.loads(browser_tools.web_fetch(
            f"http://127.0.0.1:{port}/missing"))
        assert "error" in out and "404" in out["error"], out
    finally:
        srv.shutdown()
        srv.server_close()
    # unreachable host: a clear error, not a crash
    out = json.loads(browser_tools.web_fetch("http://127.0.0.1:1/nope"))
    assert "error" in out, out


def test_web_fetch_plain_text_and_bad_url():
    """A non-HTML body comes back as text; a bad URL is an error."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"hello": "world"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        out = json.loads(browser_tools.web_fetch(
            f"http://127.0.0.1:{port}/data"))
        assert out["status"] == 200 and '"hello"' in out["text"], out
    finally:
        srv.shutdown()
        srv.server_close()
    out = json.loads(browser_tools.web_fetch(""))
    assert "error" in out, out


# --- graceful degradation -------------------------------------------------

def test_browser_tools_hidden_without_playwright():
    """In this sandbox Playwright is not installed: the browser tools
    are registered but hidden from the model, while web_fetch is
    offered — and a direct call still returns a clear error."""
    _remove_fake_playwright()
    assert browser_tools.browser_available() is False
    assert browser_tools.browser_status()["available"] is False
    names = {d["function"]["name"] for d in registry.get_definitions()}
    assert "web_fetch" in names, names
    for name in BROWSER_TOOLS:
        assert name not in names, names
    out = json.loads(browser_tools.browser_navigate("https://example.com"))
    assert "error" in out and "not available" in out["error"], out


def test_browser_graceful_error_when_launch_fails():
    """Playwright imports fine but Chromium cannot launch: the call
    returns a clear error, the turn does not crash."""
    pw = _install_fake_playwright()
    orig_launch = pw.chromium.launch

    def _boom(headless=True, timeout=None):
        raise Exception("Executable doesn't exist")

    pw.chromium.launch = _boom
    try:
        out = json.loads(browser_tools.browser_navigate("https://example.com"))
        assert "error" in out, out
        assert ("not available" in out["error"]
                or "could not launch" in out["error"]), out
    finally:
        pw.chromium.launch = orig_launch
        _remove_fake_playwright()


# --- browser ops with a fake playwright -----------------------------------

def test_browser_tools_with_mock_playwright():
    """With Playwright importable the tools are offered, and the
    navigate/read/click/type/screenshot flow works against a fake
    browser — one shared session, real PNG bytes on disk."""
    pw = _install_fake_playwright()
    try:
        assert browser_tools.browser_available() is True
        names = {d["function"]["name"] for d in registry.get_definitions()}
        for name in BROWSER_TOOLS:
            assert name in names, names
        # navigate + read share one session
        out = json.loads(browser_tools.browser_navigate("https://example.com"))
        assert out["url"] == "https://example.com", out
        assert out["title"] == "Title of https://example.com", out
        assert "Visible text" in out["text"], out
        out = json.loads(browser_tools.browser_read())
        assert out["url"] == "https://example.com", out
        # click by selector, then by visible text
        out = json.loads(browser_tools.browser_click("button.submit"))
        assert out["url"] == "https://example.com", out
        assert pw.browser.page.clicked == [("selector", "button.submit")], \
            pw.browser.page.clicked
        out = json.loads(browser_tools.browser_click("Sign in"))
        assert pw.browser.page.clicked[-1] == ("text", "Sign in"), \
            pw.browser.page.clicked
        # type into a field
        out = json.loads(browser_tools.browser_type("input[name=q]", "nvidia"))
        assert out["filled"] == "input[name=q]", out
        assert pw.browser.page.filled == {"input[name=q]": "nvidia"}
        # screenshot saves a PNG and returns its URL
        out = json.loads(browser_tools.browser_screenshot())
        assert out["screenshot_url"].startswith("/api/screenshots/shot-"), out
        assert out["screenshot_url"].endswith(".png"), out
        path = browser_tools.screenshot_dir() / out["screenshot"]
        assert path.is_file() and path.stat().st_size > 0, path
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", path
    finally:
        _remove_fake_playwright()


def test_browser_read_before_navigate_is_a_clear_error():
    pw = _install_fake_playwright()
    try:
        out = json.loads(browser_tools.browser_read())
        assert "error" in out and "navigate" in out["error"], out
    finally:
        _remove_fake_playwright()


# --- screenshot serving ---------------------------------------------------

def test_screenshot_endpoint_serves_and_rejects():
    """GET /api/screenshots/{name} serves a saved PNG, rejects
    traversal/wrong-extension names with 400, and unknown names
    with 404."""
    from fastapi import HTTPException
    import teamroom.server as server

    path = browser_tools._next_screenshot_path()
    path.write_bytes(_MINIMAL_PNG)
    resp = server.get_screenshot(path.name)
    assert resp.media_type == "image/png", resp
    assert str(path) in str(resp.path), resp
    for bad in ("../secret.png", "foo.txt", "shot-x.png/../../etc/passwd"):
        try:
            server.get_screenshot(bad)
        except HTTPException as exc:
            assert exc.status_code == 400, (bad, exc.status_code)
        else:
            raise AssertionError(f"expected 400 for {bad!r}")
    try:
        server.get_screenshot("shot-does-not-exist-0001.png")
    except HTTPException as exc:
        assert exc.status_code == 404, exc.status_code
    else:
        raise AssertionError("expected 404 for a missing screenshot")


def test_status_reports_browser():
    """ /api/status reports whether the browser is available on this
    host — False here, where Playwright is not installed."""
    import teamroom.server as server
    _remove_fake_playwright()
    assert server.status()["browser"]["available"] is False


# --- screenshot safety net in run_member ----------------------------------

def test_run_member_appends_screenshot_url():
    """A browser_screenshot during a member run lands in the posted
    reply even when the model's final text never mentions the URL —
    the on_tool safety net in run_member."""
    from looper import Looper
    from looper.state import SessionStore as LooperSessionStore

    shot_url = "/api/screenshots/shot-20261118-0001.png"
    orig_run = browser_tools._run_on_browser_thread

    def _fake_run(fn, *args):
        return {"url": "https://example.com", "title": "Example",
                "screenshot": "shot-20261118-0001.png",
                "screenshot_url": shot_url}

    browser_tools._run_on_browser_thread = _fake_run
    script = [
        {"content": "", "tool_calls": [
            {"id": "c1", "name": "browser_screenshot", "arguments": {}}]},
        {"content": "I took a screenshot of the page.", "tool_calls": []},
    ]
    orig_make = agents.make_looper

    def _fake_make(member, max_turns=12, provider=None, deadline=None,
                   on_tool=None):
        return Looper(client=_ScriptedClient(script),
                      store=LooperSessionStore(":memory:"),
                      home=Path(tempfile.mkdtemp()), auto_approve=True,
                      on_tool=on_tool, deadline=deadline)

    agents.make_looper = _fake_make
    try:
        store = _store()
        researcher = agents.member_by_handle(store, "researcher")
        out = agents.run_member(researcher, "screenshot the homepage", "")
        assert "I took a screenshot" in out, out
        assert shot_url in out, out  # appended by the safety net
        assert out.count(shot_url) == 1, out
    finally:
        browser_tools._run_on_browser_thread = orig_run
        agents.make_looper = orig_make


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all browser-tools tests passed")
