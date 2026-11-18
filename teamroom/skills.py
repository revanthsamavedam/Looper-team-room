"""Skill-file parsing — a SKILL.md-style playbook defines a member.

Accepted shape (tolerant, order-free):

  ---
  name: Dana
  specialty: Data analysis — turns raw numbers into findings
  ---
  # Dana — Data Analyst
  specialty: Data analysis ...
  ## System prompt
  You are Dana, the Data Analyst in a team chat room. ...

  ## Instructions          <- accepted as an alias for System prompt
  ## Prompt                <- also accepted

Validation is minimal and honest: a member must yield a name and a
system prompt. If either is missing the file is rejected with a plain
error — no defaults are invented. Specialty is optional for parsing
(it falls back to "Custom specialist" for display only); the name and
the instructions always come from the file itself.
"""
from __future__ import annotations

import re

_PROMPT_HEADINGS = ("system prompt", "instructions", "prompt")
_MISSING = object()


def slugify(name: str) -> str:
    """'Dana the Data Analyst' -> 'dana-the-data-analyst'."""
    slug = re.sub(r"[\s_]+", "-", name.strip().lower())
    slug = re.sub(r"[^a-z0-9-]", "", slug).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug


def _kv_lines(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(name|specialty)\s*:\s*(.+?)\s*$", line, re.IGNORECASE)
        if m:
            out[m.group(1).lower()] = m.group(2).strip().strip('"').strip("'")
    return out


def _sections(body: str) -> dict[str, str]:
    """Map lower-cased '## heading' -> section text."""
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in body.splitlines():
        m = re.match(r"##\s+(.+?)\s*$", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1).strip().lower()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def parse_skill_text(text: str) -> dict:
    """Parse a skill file into {name, handle, specialty, system_prompt}.

    Raises ValueError with a plain message when the file cannot yield
    a name or a system prompt.
    """
    if not text or not text.strip():
        raise ValueError("skill file is empty — it needs a name and a system prompt")

    body = text.strip()
    kv: dict[str, str] = {}

    # Optional YAML frontmatter: ---\nname: ...\n---\n
    if body.startswith("---"):
        end = body.find("\n---", 3)
        if end != -1:
            kv.update(_kv_lines(body[3:end]))
            body = body[end + 4:].strip()

    kv.update(_kv_lines(body))
    sections = _sections(body)

    # -- name: `name:` line, else the first `# ` heading -----------------
    name = kv.get("name", "")
    if not name:
        m = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
        name = m.group(1).strip() if m else ""
    if not name:
        raise ValueError(
            "could not find a name — put `name: Dana` in the file or a `# Dana` heading")

    # -- specialty: `specialty:` line, else a ## Specialty section ------
    specialty = kv.get("specialty", "") or sections.get("specialty", "")
    specialty = " ".join(specialty.split())
    if not specialty:
        specialty = "Custom specialist"

    # -- system prompt: a ## System prompt / Instructions / Prompt ------
    prompt = ""
    for heading in _PROMPT_HEADINGS:
        if sections.get(heading):
            prompt = sections[heading]
            break
    if not prompt:
        # Fallback: the whole body minus name/specialty lines and headings.
        lines = []
        for line in body.splitlines():
            if re.match(r"\s*(name|specialty)\s*:", line, re.IGNORECASE):
                continue
            if line.startswith("#"):
                continue
            lines.append(line)
        prompt = "\n".join(lines).strip()
    if not prompt or len(prompt) < 10:
        raise ValueError(
            "could not find a system prompt — add a `## System prompt` section "
            "with the member's instructions")

    handle = slugify(name)
    if not handle:
        raise ValueError(f"could not make a handle from the name {name!r}")
    return {"name": name, "handle": handle, "specialty": specialty,
            "system_prompt": prompt}
