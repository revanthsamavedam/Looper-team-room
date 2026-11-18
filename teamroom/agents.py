"""Specialists — each a looper AIAgent loop with its own system prompt.

Built-ins: Dev, QA, Researcher, Writer, Reviewer. Custom members are
loaded from the room store and get the same treatment: a name, a
handle (@-mentionable), a colour, a specialty line, and the system
prompt from their skill file.

Model handling: constructing a real Looper calls resolve_provider(),
which raises ValueError when no key is configured. probe_model() does
that check once at startup so the server can say plainly that no
model is configured instead of crashing. TEAMROOM_MOCK_MODEL=1 swaps
in a scripted client that returns clearly-labelled canned replies —
verification only, never presented as a real model.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from looper.loop import Looper
from looper.provider import (OpenAICompatibleClient, ResolvedProvider,
                             resolve_provider)
from looper.state import SessionStore

from . import projects  # noqa: F401 — import installs the project-aware file tools
from .store import RoomStore

DEFAULT_MODEL = "gpt-4o-mini"

RESERVED_HANDLES = {"dev", "qa", "researcher", "writer", "reviewer",
                    "leader", "room", "orchestrator", "system", "you", "all"}

LEADER_HANDLE = "leader"

CUSTOM_COLOURS = ["#0ea5e9", "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#84cc16"]

ROOM_HOME = Path(os.environ.get("TEAMROOM_HOME", Path.home() / ".looper-teamroom"))
MOCK_MODEL = os.environ.get("TEAMROOM_MOCK_MODEL") == "1"


@dataclass
class Member:
    name: str
    handle: str
    colour: str
    emoji: str
    specialty: str
    system_prompt: str
    builtin: bool = True
    source: str = ""


def _room_rule(name: str, specialty: str) -> str:
    return (
        f"You are {name}, the {specialty} in a team chat room with other "
        "specialist agents. Answer only your part of the task. Be concrete, "
        "use your tools to verify instead of guessing, and end with a short "
        "summary of what you did. Never claim work you did not do, and never "
        "answer for another specialist.\n"
        "If you make a decision, state it as a line starting 'Decision: '. "
        "If you learn something about this team's work, state it as a line "
        "starting 'Learned: '. If you notice a lesson that would help ANY "
        "team, state it as a line starting 'Best practice: '. These lines "
        "are saved as the team's memory — only state ones that are true."
    )


BUILTINS: list[Member] = [
    Member(name="Dev", handle="dev", colour="#3b82f6", emoji="💻",
           specialty="Writes and fixes code",
           system_prompt=_room_rule("Dev", "developer")
           + " Write working code, show the key snippets, and verify by "
             "reading the file back or running it. If a fix needs a decision, "
             "say which option you picked and why."),
    Member(name="QA", handle="qa", colour="#22c55e", emoji="🧪",
           specialty="Tests, finds bugs, verifies fixes",
           system_prompt=_room_rule("QA", "QA engineer")
           + " Test the work: list the cases you checked, what passed and "
             "what failed, and the exact steps to reproduce any bug. "
             "If you find a bug, state it as a line starting 'Bug: '. "
             "Do not approve work you did not actually test."),
    Member(name="Researcher", handle="researcher", colour="#a855f7", emoji="🔍",
           specialty="Finds facts, compares options",
           system_prompt=_room_rule("Researcher", "researcher")
           + " Gather the facts first: what exists, how options compare, "
             "and what the trade-offs are. Cite what you found and say what "
             "you could not verify."),
    Member(name="Writer", handle="writer", colour="#f59e0b", emoji="✍️",
           specialty="Writes docs, drafts and summaries",
           system_prompt=_room_rule("Writer", "technical writer")
           + " Write clearly and briefly: short sentences, plain words, "
             "structure with headings or bullets when it helps. Drafts go "
             "in full so they can be used as-is."),
    Member(name="Reviewer", handle="reviewer", colour="#ef4444", emoji="👀",
           specialty="Reviews work for quality, risk and gaps",
           system_prompt=_room_rule("Reviewer", "reviewer")
           + " Review the work in the room, not your own. Call out real "
             "risks, gaps and inconsistencies with specifics, and say "
             "plainly whether the work is good to go or what must change."),
    Member(name="Leader", handle=LEADER_HANDLE, colour="#0d9488", emoji="🧭",
           specialty="Plans and coordinates the team",
           system_prompt=(
               "You are the Leader of a team chat room of specialist agents. "
               "You plan work, delegate it to the right specialists, and "
               "integrate their results into one coherent final answer. You "
               "do not do the specialists' work yourself — you coordinate it.\n"
               "When planning, break the instruction into a short ordered "
               "plan of at most 3 steps, each assigned to one specialist by "
               "@handle. Order steps so dependencies come first: research "
               "before building, building before testing, testing before "
               "reviewing, reviewing before writing it up. Reply with ONLY "
               "the plan, one step per line, in this exact format:\n"
               "STEP @handle: <one-sentence subtask for that specialist>\n"
               "Use only the handles listed as available. Do not add "
               "commentary, explanation, or any other text.\n"
               "When integrating, combine the specialists' results into one "
               "coherent final answer: what was done, the key findings, and "
               "what is still open. Be concise. Do not repeat their full text.")),
]


def list_members(store: RoomStore) -> list[Member]:
    """Built-ins first, then custom members from SQLite (persisted)."""
    members = list(BUILTINS)
    for i, row in enumerate(store.list_members()):
        members.append(Member(
            name=row["name"], handle=row["handle"],
            colour=CUSTOM_COLOURS[i % len(CUSTOM_COLOURS)], emoji="🤖",
            specialty=row["specialty"], system_prompt=row["system_prompt"],
            builtin=False, source=row["source"]))
    return members


def list_specialists(store: RoomStore) -> list[Member]:
    """Members the Leader can delegate to — everyone except the Leader.

    The Leader is the orchestrator, not a delegate: it plans, assigns
    and integrates, so it never appears in its own delegation pool.
    The roster (list_members) still shows it, so it is visible and
    @-mentionable as the explicit trigger for a leader turn.
    """
    return [m for m in list_members(store) if m.handle != LEADER_HANDLE]


def member_by_handle(store: RoomStore, handle: str) -> Member | None:
    for m in list_members(store):
        if m.handle == handle:
            return m
    return None


def build_provider(api_key: str | None, model: str | None) -> ResolvedProvider | None:
    """Build a per-request provider from a browser-supplied (BYOK) key.

    Returns None when no key is given. The key lives only in the
    returned object, in memory, for the one turn that uses it. It is
    never written to the DB, logs, or disk anywhere in this package.
    Raises ValueError for an unusable model name (e.g. anthropic/).
    """
    key = (api_key or "").strip()
    if not key:
        return None
    name = (model or "").strip() or DEFAULT_MODEL
    return resolve_provider(model=name, api_key=key)


def probe_model() -> dict:
    """Check once whether a live provider is configured. Never raises."""
    if MOCK_MODEL:
        return {"configured": True, "model": "mock (TEAMROOM_MOCK_MODEL=1)",
                "error": None, "mock": True}
    try:
        provider = resolve_provider()
        return {"configured": True, "model": provider.model,
                "error": None, "mock": False}
    except ValueError as exc:
        return {"configured": False, "model": None, "error": str(exc), "mock": False}


class MockClient:
    """Scripted stand-in for a real provider — verification only.

    Enabled by TEAMROOM_MOCK_MODEL=1. Every reply is labelled
    '[mock model]' so it is never mistaken for a real model result.
    It proves routing, identity and posting work end to end.
    """

    def __init__(self, member: Member):
        self.member = member

    def call(self, messages: list[dict], tools: list[dict],
             deadline: float | None = None) -> dict:
        # The mock never calls a real provider, so the deadline is
        # accepted and ignored — the Looper still checks it before
        # each call, which is what bounds a mock turn.
        full = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                full = m.get("content") or ""
                break
        if self.member.handle == LEADER_HANDLE:
            return {"content": self._leader_reply(full), "tool_calls": []}
        task = ""
        task_lines = [ln for ln in full.replace("Your task:", "").strip().splitlines()
                      if ln.strip()]
        task = task_lines[-1][:160] if task_lines else ""
        return {"content": (
            f"[mock model] {self.member.name} ({self.member.specialty}) here. "
            f"Working on: {task} — this is a scripted reply proving the "
            f"@{self.member.handle} route works; set a real model for live answers.\n"
            f"Decision: use the simplest approach that passes the checks.\n"
            f"Learned: mock replies are scripted, so verify routing, not content.\n"
            f"Best practice: keep each agent's reply focused on its own specialty."),
            "tool_calls": []}

    def _leader_reply(self, prompt: str) -> str:
        """Scripted Leader replies — verification only.

        A plan prompt gets STEP lines for the first three handles on
        its "Available specialists" line, in that order; an integrate
        prompt gets a short integrated summary; a consolidate prompt
        gets one consolidated status naming every member in its
        "## Name (@handle)" sections. A real model plans for real;
        this proves the plan → delegate → integrate mechanism.
        """
        if "consolidate" in prompt.lower():
            handles = re.findall(r"## [^(@]*\(@([A-Za-z0-9][A-Za-z0-9_-]*)\)",
                                 prompt)
            if not handles:
                handles = re.findall(r"@([A-Za-z0-9][A-Za-z0-9_-]*)", prompt)
            lines = [f"- @{h}: working on the mock task, nothing blocked."
                     for h in list(dict.fromkeys(handles))]
            return ("[mock model] 📋 Team status:\n" + "\n".join(lines)
                    if lines else
                    "[mock model] 📋 Team status: nobody to report on.")
        if "integrate" in prompt.lower():
            return ("[mock model] Leader integrated the specialists' results: "
                    "the work is done, the key findings are in the posts "
                    "above, and nothing is left open in this mock turn.\n"
                    "Decision: keep the integrated result short and concrete.")
        handles: list[str] = []
        for line in prompt.splitlines():
            if "available specialists" in line.lower():
                handles = re.findall(r"@([A-Za-z0-9][A-Za-z0-9_-]*)", line)
                break
        instr = ""
        for line in prompt.splitlines():
            if line.strip().lower().startswith("instruction:"):
                instr = line.strip()[len("instruction:"):].strip()[:80]
                break
        steps = [f"STEP @{h}: work on '{instr}' as the {h}."
                 for h in list(dict.fromkeys(handles))[:3]]
        if not steps:
            return ("[mock model] Leader could not make a plan: no "
                    "specialists were listed as available.")
        return "\n".join(steps)


def make_looper(member: Member, max_turns: int = 12,
                provider: ResolvedProvider | None = None,
                deadline: float | None = None) -> Looper:
    """One Looper per member per run: own home dir, own session store.

    Provider precedence: mock client (verification only) wins, then an
    explicit per-request provider (BYOK key), then the environment.
    `deadline` is a monotonic timestamp: once it is past, the Looper
    starts no new provider call, so a member's run ends with the
    partial work it has instead of blowing the hosting time limit.
    """
    home = ROOM_HOME / "homes" / member.handle
    home.mkdir(parents=True, exist_ok=True)
    if MOCK_MODEL:
        client = MockClient(member)
    elif provider is not None:
        client = OpenAICompatibleClient(provider)
    else:
        client = None
    return Looper(client=client, store=SessionStore(":memory:"), home=home,
                  max_turns=max_turns, auto_approve=False, deadline=deadline)


def run_member(member: Member, task: str, room_context: str,
               provider: ResolvedProvider | None = None,
               max_turns: int = 12, deadline: float | None = None,
               project: projects.ProjectWorkspace | None = None) -> str:
    """Run one specialist on one task; returns its final response text.

    max_turns bounds the loop (the leader path uses a smaller cap so a
    delegated turn stays inside the hosting time limit). `deadline` is
    a monotonic timestamp past which no new provider call is started.
    `project` is the active project workspace (or None): while it is
    set, the member's file tools (list_files / read_file / write_file)
    operate on the project's files in the store instead of the local
    filesystem. The context is set here, in this thread, so it reaches
    the tool dispatch even when this run is a parallel-path pool
    worker.
    """
    looper = make_looper(member, max_turns=max_turns, provider=provider,
                         deadline=deadline)
    message = ""
    if room_context.strip():
        message += f"{room_context}\n\n"
    message += f"Your task: {task}"
    try:
        with projects.active_project(project):
            result = looper.run_conversation(message, system_message=member.system_prompt)
        if result.get("stopped") == "deadline":
            text = (result.get("final_response") or "").strip()
            return (text + "\n⏱️ (ran out of time — partial work above)"
                    if text else "⏱️ (ran out of time before finishing)")
        return (result.get("final_response") or "").strip() or "(no answer)"
    except Exception as exc:  # noqa: BLE001 — the room reports, never crashes
        return f"⚠️ {member.name} could not finish: {type(exc).__name__}: {exc}"
