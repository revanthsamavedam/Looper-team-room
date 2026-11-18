"""Web tools for the team room: a lightweight HTTP fetch and a
Playwright browser.

web_fetch works everywhere (stdlib urllib only), including Vercel —
it is how the Researcher reads live pages on the hosted room.

The browser tools need a host with Chromium (a persistent VM, not
Vercel serverless). They register only when Playwright is importable;
on a host without it they are hidden from the model entirely, so a
turn never crashes because the browser is unavailable. If Playwright
is present but Chromium cannot launch, each call returns a clear
error instead of raising.

Threading: the looper dispatches tool calls in parallel, and
Playwright's sync API is bound to the thread that created the
browser. All browser work therefore runs on one dedicated
single-thread executor, so the browser is created once per process
and reused across calls — navigate, then read, then click, then
screenshot is one coherent session.
"""
from __future__ import annotations

import html
import itertools
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

from looper.registry import tool
from looper.tools import _props, _s

FETCH_TIMEOUT_S = 15
FETCH_TEXT_LIMIT = 6000
FETCH_MAX_BYTES = 200_000

BROWSER_TIMEOUT_MS = 15000
BROWSER_TEXT_LIMIT = 4000


# --- web_fetch (works everywhere) ---------------------------------------

class _TextExtractor(HTMLParser):
    """Pull visible text out of HTML, skipping script/style/head."""

    _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5",
              "tr", "section", "article", "header", "footer"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._in_head = False

    def handle_starttag(self, tag, attrs):
        if tag == "head":
            self._in_head = True
        elif tag in ("script", "style", "noscript"):
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "head":
            self._in_head = False
        elif tag in ("script", "style", "noscript"):
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if self._skip or self._in_head:
            return
        self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


def _html_to_text(raw_html: str) -> tuple[str, str]:
    """Return (title, visible text) for an HTML document."""
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.S | re.I)
    if m:
        title = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
    ext = _TextExtractor()
    try:
        ext.feed(raw_html)
    except Exception:
        pass
    return title, ext.text()


def _fetch(url: str) -> dict:
    url = (url or "").strip()
    if not url:
        return {"error": "no URL given"}
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Looper Team Room; web_fetch)"})
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            raw = resp.read(FETCH_MAX_BYTES + 1)
            content_type = resp.headers.get("Content-Type", "")
            status = resp.status
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code} fetching {url}", "url": url}
    except Exception as exc:  # noqa: BLE001 — the model gets JSON, not a crash
        return {"error": f"could not fetch {url}: {type(exc).__name__}: {exc}",
                "url": url}
    truncated = len(raw) > FETCH_MAX_BYTES
    raw = raw[:FETCH_MAX_BYTES]
    charset = "utf-8"
    m = re.search(r"charset=([^\s;]+)", content_type, re.I)
    if m:
        charset = m.group(1)
    try:
        text = raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")
    if "html" in content_type.lower() or "<html" in text[:2000].lower():
        title, visible = _html_to_text(text)
    else:
        title, visible = "", text
    visible = visible.strip()
    if len(visible) > FETCH_TEXT_LIMIT:
        visible = visible[:FETCH_TEXT_LIMIT]
        truncated = True
    out: dict = {"url": url, "status": status, "text": visible,
                 "truncated": truncated}
    if title:
        out["title"] = title
    return out


@tool("web_fetch", "web",
      "Fetch a web page (HTTP GET) and return its visible text. Use for plain readable pages.",
      _props(url=_s("The URL to fetch, e.g. https://example.com", required=True)),
      emoji="🌐")
def web_fetch(url: str) -> str:
    return json.dumps(_fetch(url))


# --- Playwright browser (needs Chromium) ---------------------------------

_AVAILABLE: bool | None = None


def browser_available() -> bool:
    """True when the Playwright library is importable. Cached — the
    registry calls this once per provider call, and importing is cheap
    after the first time. This is only the library: if Chromium itself
    is missing, calls fail with a clear error telling the user to run
    `python -m playwright install chromium`."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import playwright.sync_api  # noqa: F401
            _AVAILABLE = True
        except ImportError:
            _AVAILABLE = False
    return _AVAILABLE


def browser_status() -> dict:
    if browser_available():
        return {"available": True,
                "note": "Playwright on this host — browser tools work when Chromium is installed"}
    return {"available": False,
            "note": "Playwright is not installed on this host — browser tools are hidden; web_fetch still works"}


def screenshot_dir() -> Path:
    """Where browser screenshots live. Reads TEAMROOM_HOME fresh so
    the server endpoint and the tool agree even when the env changes
    between import and request (e.g. in tests)."""
    home = Path(os.environ.get("TEAMROOM_HOME", Path.home() / ".looper-teamroom"))
    return home / "screenshots"


_counter = itertools.count(1)


def _next_screenshot_path() -> Path:
    d = screenshot_dir()
    d.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    return d / f"shot-{ts}-{next(_counter):04d}.png"


def _truncate(text: str, limit: int = BROWSER_TEXT_LIMIT) -> str:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(text) > limit:
        return text[:limit] + f"\n…[truncated, {len(text)} chars total]"
    return text


# The browser lives on one dedicated thread (see module docstring).
_executor: ThreadPoolExecutor | None = None
_state: dict = {"playwright": None, "browser": None, "page": None}


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1,
                                       thread_name_prefix="teamroom-browser")
    return _executor


def _reset_browser() -> None:
    """Tear the browser down so the next call starts fresh. Runs on
    the browser thread."""
    for key, closer in (("page", "close"), ("browser", "close"),
                        ("playwright", "stop")):
        obj = _state.get(key)
        _state[key] = None
        if obj is None:
            continue
        try:
            getattr(obj, closer)()
        except Exception:
            pass


def _ensure_page():
    """Lazily start Chromium and return the shared page. Runs on the
    browser thread."""
    page = _state.get("page")
    if page is not None:
        try:
            if not page.is_closed():
                return page
        except Exception:
            pass
        _reset_browser()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "browser not available on this host: Playwright is not installed")
    try:
        pw = sync_playwright().start()
        _state["playwright"] = pw
        _state["browser"] = pw.chromium.launch(headless=True,
                                               timeout=BROWSER_TIMEOUT_MS)
        _state["page"] = _state["browser"].new_page()
    except Exception as exc:
        _reset_browser()
        raise RuntimeError(
            "browser not available on this host: Chromium could not launch "
            f"({exc}). On a VM, run: python -m playwright install chromium")
    return _state["page"]


def _visible_text(page) -> str:
    try:
        text = page.inner_text("body")
    except Exception:
        try:
            text = page.evaluate(
                "() => document.body ? document.body.innerText : ''")
        except Exception:
            text = ""
    return _truncate(text)


def _page_summary(page) -> dict:
    return {"url": page.url, "title": page.title(), "text": _visible_text(page)}


def _op_navigate(url: str) -> dict:
    page = _ensure_page()
    url = (url or "").strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    page.goto(url, timeout=BROWSER_TIMEOUT_MS, wait_until="domcontentloaded")
    return _page_summary(page)


def _op_read() -> dict:
    page = _ensure_page()
    if (page.url or "") in ("", "about:blank"):
        return {"error": "no page loaded yet — use browser_navigate with a URL first"}
    return _page_summary(page)


def _op_click(target: str) -> dict:
    page = _ensure_page()
    target = (target or "").strip()
    if not target:
        return {"error": "nothing to click — give a selector or visible text"}
    last_exc = None
    for attempt in ("selector", "text"):
        try:
            if attempt == "selector":
                page.click(target, timeout=5000)
            else:
                page.get_by_text(target, exact=False).first.click(timeout=5000)
            break
        except Exception as exc:  # noqa: BLE001 — fall through to the next attempt
            last_exc = exc
    else:
        return {"error": f"could not click {target!r}: {last_exc}"}
    try:
        page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass  # no navigation happened; the click still landed
    return _page_summary(page)


def _op_type(selector: str, text: str) -> dict:
    page = _ensure_page()
    selector = (selector or "").strip()
    if not selector:
        return {"error": "nothing to type into — give a field selector"}
    page.fill(selector, text or "", timeout=BROWSER_TIMEOUT_MS)
    out = _page_summary(page)
    out["filled"] = selector
    return out


def _op_screenshot() -> dict:
    page = _ensure_page()
    path = _next_screenshot_path()
    page.screenshot(path=str(path), timeout=BROWSER_TIMEOUT_MS)
    out = _page_summary(page)
    out["screenshot"] = path.name
    out["screenshot_url"] = f"/api/screenshots/{path.name}"
    return out


def _run_on_browser_thread(fn, *args) -> dict:
    """Run a browser op on the dedicated browser thread. Returns the
    op's dict, or an error dict when the browser is unavailable or the
    call times out. Never raises — a tool call must return JSON, not a
    traceback."""
    if not browser_available():
        return {"error": "browser not available on this host: Playwright is not installed"}
    try:
        future = _get_executor().submit(fn, *args)
        return future.result(timeout=BROWSER_TIMEOUT_MS / 1000 + 5)
    except Exception as exc:  # noqa: BLE001 — the model gets JSON, not a crash
        return {"error": f"browser call failed: {type(exc).__name__}: {exc}"}


@tool("browser_navigate", "browser",
      "Open a URL in the browser (JavaScript runs) and return the page title + visible text.",
      _props(url=_s("The URL to open, e.g. https://example.com", required=True)),
      check_fn=browser_available, emoji="🧭")
def browser_navigate(url: str) -> str:
    return json.dumps(_run_on_browser_thread(_op_navigate, url))


@tool("browser_read", "browser",
      "Read the visible text of the page currently open in the browser.",
      _props(), check_fn=browser_available, emoji="📖")
def browser_read() -> str:
    return json.dumps(_run_on_browser_thread(_op_read))


@tool("browser_click", "browser",
      "Click an element in the browser, by CSS selector or by its visible text. Returns the new page.",
      _props(target=_s("CSS selector or visible text of the element to click, e.g. 'Sign in' or 'button.submit'", required=True)),
      check_fn=browser_available, emoji="🖱️")
def browser_click(target: str) -> str:
    return json.dumps(_run_on_browser_thread(_op_click, target))


@tool("browser_type", "browser",
      "Type text into a field in the browser, by CSS selector.",
      _props(selector=_s("CSS selector of the field, e.g. 'input[name=q]'", required=True),
             text=_s("Text to type into the field", required=True)),
      check_fn=browser_available, emoji="⌨️")
def browser_type(selector: str, text: str) -> str:
    return json.dumps(_run_on_browser_thread(_op_type, selector, text))


@tool("browser_screenshot", "browser",
      "Take a screenshot of the current browser page. Returns a screenshot URL — include it in your reply so the room can see the page.",
      _props(), check_fn=browser_available, emoji="📸")
def browser_screenshot() -> str:
    return json.dumps(_run_on_browser_thread(_op_screenshot))
