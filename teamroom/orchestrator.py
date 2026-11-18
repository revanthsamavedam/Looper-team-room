"""Orchestrator — the Leader plans, delegates, and integrates.

Leader/subagent model (the primary path when a model is configured):
  1. The Leader (a real looper loop, @leader) plans: it breaks the
     instruction into at most 3 ordered steps, each assigned to one
     specialist by @handle, dependencies first (research before
     building, building before testing, testing before reviewing).
  2. It delegates: the specialists run in plan order, each seeing the
     earlier specialists' results, so later steps build on earlier ones.
  3. It integrates: the Leader combines the specialists' outputs into
     one coherent final answer. The plan and the integrated result are
     posted as coordination messages, so they never count toward a
     goal's task progress — only specialist work does.

The Leader respects the existing rules: with a goal active it
delegates only to that goal's members; @-mentions of specific members
restrict the delegation pool to them; otherwise it picks the right
specialists itself. It works with built-ins and custom members alike.

Fallback: when the Leader cannot run — no model configured, or it
produces no usable plan — the room falls back to the original path:
route by @-mention/keyword, run the selected specialists in parallel,
and post a Room synthesis. If no model is configured at all the room
gets one plain message saying so and the turn stops.

Status: a message that is essentially a status ask ("Status", "where
are we", "how are we doing") is not work — it triggers a roundup in
which the Leader loops through every member in the pool, collects one
short status each, and posts exactly ONE consolidated status message.
The single message is author_type='orchestrator', so it never counts
toward goal progress and earns no XP or memory.

Goals and teams:
  - When a goal is active, the member pool is that goal's members
    only — routing never picks someone outside the goal.
  - The goal's title/description, the member's team-scoped memory
    (did/decided/learned, this team only) and the shared best
    practices are fed back as context for every agent run.
  - After a run, what each member did is recorded as team-scoped
    memory; `Decision:` / `Learned:` lines in an agent's reply are
    recorded as decided/learned memory, and `Best practice:` lines go
    to the shared best-practices layer (visible to all teams).

Every message in the turn is posted to the room as it lands, so the
SSE stream shows the plan, then each specialist's work, then the
integrated result — and on a hosting timeout the partial results stay
visible. If no model is configured nothing is faked.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import agents, projects
from .agents import LEADER_HANDLE, Member
from .projects import ProjectWorkspace
from .store import (RoomStore, XP_BEST_PRACTICE, XP_BUG, XP_DECISION,
                    XP_LEARNED)

# A turn's work is bounded by one deadline (a monotonic timestamp,
# set by the server from TEAMROOM_TURN_DEADLINE_S). Every loop in the
# turn — the Leader's plan, each delegated specialist, the integrate
# step, the parallel path — checks it before starting new work, so a
# turn aborts gracefully with the partial results above instead of
# letting the hosting time limit kill the whole function mid-stream.
# The default here is only used when the caller passes none.
DEFAULT_TURN_DEADLINE_S = 45.0


def _past(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline

MENTION_RE = re.compile(r"@([A-Za-z0-9][A-Za-z0-9_-]*)")

# One STEP line per plan step: `STEP @handle: subtask`.
STEP_RE = re.compile(r"^\s*STEP\s+@([A-Za-z0-9][A-Za-z0-9_-]*)\s*:\s*(.+?)\s*$",
                     re.IGNORECASE | re.MULTILINE)
MAX_LEADER_STEPS = 3          # a leader plan is at most 3 delegated steps
LEADER_PLAN_TURNS = 1         # the plan call is one short answer, no tools
LEADER_INTEGRATE_TURNS = 2    # the integrate call is short too
LEADER_DELEGATE_TURNS = 4     # delegated specialists run a bounded loop so a
                              # leader turn (plan + steps + integrate) stays
                              # inside the hosting time limit (~45s on Vercel)
PARALLEL_TURNS = 6            # specialists on the parallel path run
                              # concurrently, so 6 turns each still fits

KEYWORDS: dict[str, list[str]] = {
    "dev": ["build", "implement", "code", "fix", "bug", "refactor", "script",
            "function", "api", "deploy", "write code", "program"],
    "qa": ["test", "qa", "verify", "check", "edge case", "regression",
           "passes", "fails", "broken"],
    "researcher": ["research", "find", "compare", "investigate", "what is",
                   "how does", "options", "best", "look up", "search"],
    "writer": ["write", "doc", "document", "draft", "readme", "explain",
               "summarise", "summarize", "summary"],
    "reviewer": ["review", "audit", "risk", "quality", "look over", "approve",
                 "is this good"],
}

_STOPWORDS = {"the", "and", "for", "with", "from", "into", "that", "this",
              "your", "their", "about", "turns", "raw", "clear", "findings"}

# Lines an agent may end its reply with; the orchestrator saves them.
_STRUCTURED_LINES = {
    "decision:": "decided",
    "learned:": "learned",
}


def parse_mentions(text: str) -> list[str]:
    seen: list[str] = []
    for handle in MENTION_RE.findall(text):
        handle = handle.lower()
        if handle not in seen:
            seen.append(handle)
    return seen


def keywords_for(member: Member) -> list[str]:
    if member.handle in KEYWORDS:
        return KEYWORDS[member.handle]
    words = re.findall(r"[a-z]{4,}", f"{member.name} {member.specialty}".lower())
    return [w for w in words if w not in _STOPWORDS] + [member.handle]


def _score(text: str, words: list[str]) -> int:
    low = f" {text.lower()} "
    score = 0
    for w in words:
        if f" {w} " in low or low.startswith(f"{w} ") or low.endswith(f" {w}"):
            score += 1
        elif " " not in w and re.search(rf"\b{re.escape(w)}\w*\b", low):
            score += 1
    return score


def select_members(text: str, members: list[Member]) -> tuple[list[Member], list[str]]:
    """Returns (selected members, unknown @-mentions). Members are drawn
    from the given pool only — callers pass the goal's members when a
    goal is active, so routing can never pick someone off-goal."""
    by_handle = {m.handle: m for m in members}
    mentions = parse_mentions(text)
    unknown = [h for h in mentions if h not in by_handle]
    picked = [by_handle[h] for h in mentions if h in by_handle]
    if picked:
        return picked, unknown
    scored = [(m, _score(text, keywords_for(m))) for m in members]
    picked = [m for m, s in sorted(scored, key=lambda p: -p[1]) if s > 0]
    if picked:
        return picked, unknown
    default = by_handle.get("researcher") or (members[0] if members else None)
    return ([default] if default else []), unknown


def parse_plan(text: str, pool: list[Member]) -> list[tuple[Member, str]]:
    """Parse a Leader's plan into ordered (member, subtask) steps.

    The plan is one STEP line per step: `STEP @handle: subtask`.
    Only handles in the given pool are kept (goal/@-mention scoping —
    a step for someone outside the pool is dropped, never run),
    duplicates are dropped, and the plan is capped at MAX_LEADER_STEPS.
    Returns [] when nothing usable is in the plan.
    """
    by_handle = {m.handle: m for m in pool}
    steps: list[tuple[Member, str]] = []
    seen: set[str] = set()
    for m in STEP_RE.finditer(text or ""):
        handle = m.group(1).lower()
        subtask = m.group(2).strip()
        if handle not in by_handle or handle in seen or not subtask:
            continue
        seen.add(handle)
        steps.append((by_handle[handle], subtask))
        if len(steps) >= MAX_LEADER_STEPS:
            break
    return steps


def plan_skipped_handles(text: str, pool: list[Member]) -> list[str]:
    """Handles the Leader named in its plan that are not in the pool."""
    pool_handles = {m.handle for m in pool}
    skipped: list[str] = []
    for m in STEP_RE.finditer(text or ""):
        handle = m.group(1).lower()
        if handle not in pool_handles and handle not in skipped:
            skipped.append(handle)
    return skipped


# --- status roundup ---------------------------------------------------------
# "Status" is a question, not work: when the user asks for status every
# member reports in, instead of the Leader treating it as an instruction
# and routing it to one member. The matcher is deliberately conservative
# — when in doubt, the message is work, not a status ask.

_STATUS_VERBS = ("build", "create", "add", "fix", "show", "implement",
                 "write", "code", "design", "make", "develop", "deploy",
                 "test", "change", "set", "put", "include", "remove",
                 "delete", "improve", "refactor", "update")
_STATUS_OBJECTS = ("page", "field", "ui", "button", "screen", "component",
                   "endpoint", "api", "dashboard", "bar", "indicator",
                   "column", "feature", "app", "site", "website", "form",
                   "table", "modal", "badge", "widget")
_STATUS_PREFIXES = ("whats the status", "what is the status",
                    "whats the progress", "what is the progress",
                    "where are we", "how are we doing",
                    "how is it going", "hows it going",
                    "how is it going", "whats going on",
                    "what is going on", "give me a status",
                    "give me the status", "give me an update",
                    "team status", "status update", "any update",
                    "any progress", "how is the project",
                    "hows the project", "how is the work",
                    "hows the work")
_STATUS_EXACT = {"status", "progress", "status please", "please status",
                 "status update", "team status", "progress please",
                 "please progress", "status check", "any status",
                 "status report", "check the status", "check status"}


def _is_status_request(text: str) -> bool:
    """True when the message is essentially a status ask.

    Normalised (lowercase, punctuation stripped, @-mentions removed —
    "@dev status" is a status ask for Dev). Unambiguous status-ask
    phrasings win first ("what is the status of the build" is a status
    ask even though "build" is in it). Otherwise a work verb before
    "status"/"progress" ("build a status page") or a work object
    ("status field") makes it work. When in doubt, it is work.
    """
    raw = MENTION_RE.sub("", (text or "").lower())
    # drop apostrophes so "what's" reads as "whats", not "what s"
    raw = raw.replace("'", "").replace("’", "")
    norm = re.sub(r"[^a-z0-9\s]", " ", raw)
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return False
    if any(norm.startswith(p) for p in _STATUS_PREFIXES):
        return True
    if norm.startswith(("status of ", "status for ", "progress on ",
                        "progress for ")):
        return True
    words = norm.split()
    for key in ("status", "progress"):
        if key in words:
            ki = words.index(key)
            if any(v in words[:ki] for v in _STATUS_VERBS):
                return False
            if any(o in words for o in _STATUS_OBJECTS):
                return False
    return norm in _STATUS_EXACT


def _room_context(store: RoomStore, room_id: str) -> str:
    lines = []
    for m in store.recent(room_id, limit=10):
        lines.append(f"- {m['author']}: {(m['content'] or '')[:300]}")
    return "\n".join(lines)


def build_agent_context(store: RoomStore, member: Member,
                        goal: dict | None,
                        project: ProjectWorkspace | None = None) -> str:
    """Context fed to one member's run: goal, its team-scoped memory,
    and the shared best practices. Memory is fetched by (team, handle)
    only — it cannot leak across teams. When a project is active the
    member is told its file tools operate on the project's files, so
    deliverables are written there instead of only pasted in chat."""
    parts: list[str] = []
    if goal:
        parts.append(f"[Goal: {goal['title']}"
                     + (f" — {goal['description']}" if goal.get("description") else "")
                     + "]")
    if project is not None:
        parts.append(
            f"[Project: {project.name} — the deliverable files for this "
            "work live in the project. Use the write_file tool to write "
            "them (paths like main.py, requirements.txt, README.md) and "
            "list_files / read_file to check what is already there. Files "
            "in the project persist; code only pasted in chat does not.]")
    if goal:
        mem = store.list_memory(goal["team_id"], handle=member.handle, limit=6)
        if mem:
            lines = [f"- [{e['kind']}] {e['content'][:300]}"
                     for e in reversed(mem)]
            parts.append("[Your memory in this team — what you did, decided "
                         "and learned here]\n" + "\n".join(lines))
    practices = store.list_best_practices()
    if practices:
        lines = [f"- {p['title']}: {p['content'][:200]}"
                 for p in practices[-8:]]
        parts.append("[Shared best practices — lessons from every team]\n"
                     + "\n".join(lines))
    return "\n\n".join(parts)


def _post_best_effort(store: RoomStore, room_id: str, goal_id: str | None,
                      **kwargs) -> None:
    """Append a room message, best-effort. A transient store failure
    (e.g. a Turso hiccup) must not kill the turn — the message is
    already in the turn's results, so the integrate step can still use
    it and the room stays usable. The store retries transient errors
    internally; this is the last resort."""
    try:
        store.append_message(room_id, goal_id=goal_id, **kwargs)
    except Exception:  # noqa: BLE001 — best-effort post, never fatal
        pass


def _record_outcome(store: RoomStore, team_id: str | None, member: Member,
                    instruction: str, response: str) -> None:
    """Save what a member did/decided/learned (team-scoped) and any
    generally-useful lesson to the shared best-practices layer.
    Also awards XP for real completed work (room-wide, even with no
    goal): a finished turn, and bonuses for Decision:/Learned:/
    Best practice:/Bug: lines. Badges are awarded once each.

    Best-effort by design: memory and XP are non-critical writes. If
    the store is having a bad moment, a failed write is swallowed here
    so the turn itself still completes — the agent's reply is already
    posted in the room."""
    try:
        _record_outcome_inner(store, team_id, member, instruction, response)
    except Exception:  # noqa: BLE001 — non-critical writes never kill a turn
        pass


def _record_outcome_inner(store: RoomStore, team_id: str | None, member: Member,
                          instruction: str, response: str) -> None:
    totals = store.record_turn(member.handle)
    if totals["turns"] == 1:
        store.award_badge(member.handle, "first-steps")
    if totals["turns"] == 10:
        store.award_badge(member.handle, "workhorse")
    if totals["xp"] >= 100:
        store.award_badge(member.handle, "centurion")
    for line in response.splitlines():
        low = line.strip().lower()
        if low.startswith("decision:"):
            store.add_xp(member.handle, XP_DECISION)
            store.award_badge(member.handle, "decider")
        elif low.startswith("learned:"):
            store.add_xp(member.handle, XP_LEARNED)
        elif low.startswith("best practice:"):
            store.add_xp(member.handle, XP_BEST_PRACTICE)
            store.award_badge(member.handle, "sharer")
        elif low.startswith("bug:"):
            store.add_xp(member.handle, XP_BUG)
            store.award_badge(member.handle, "bug-hunter")
    if not team_id:
        return
    first = (response.splitlines() or [""])[0][:200]
    store.record_memory(team_id, member.handle, "did",
                        f"Worked on '{instruction[:120]}': {first}")
    for line in response.splitlines():
        low = line.strip().lower()
        for prefix, kind in _STRUCTURED_LINES.items():
            if low.startswith(prefix):
                store.record_memory(team_id, member.handle, kind,
                                    line.strip()[len(prefix):].strip())
        if low.startswith("best practice:"):
            store.add_best_practice(
                title=f"From @{member.handle}: {line.strip()[14:].strip()[:80]}",
                content=line.strip()[14:].strip(),
                source_team_id=team_id, source_handle=member.handle)
        if low.startswith("bug:"):
            store.record_memory(team_id, member.handle, "did",
                                "Found a bug: " + line.strip()[4:].strip())
    if store.teams_worked(member.handle) >= 2:
        store.award_badge(member.handle, "team-player")


def _synthesise(results: list[tuple[Member, str]], mock: bool,
                provider=None, deadline: float | None = None) -> str:
    """Short synthesis of the specialists' results, via a coordinator
    Looper when a model is available, else a plain template summary.
    Past the deadline the coordinator call is skipped and the template
    is used — the turn is already over budget."""
    if not _past(deadline):
        try:
            coordinator = Member(
                name="Room", handle="room", colour="#64748b", emoji="🧑‍🤝‍🧑",
                specialty="coordination",
                system_prompt=(
                    "You are the room coordinator. Summarise the specialists' "
                    "results below into 3–6 short bullets: what was done, the "
                    "key findings, and what is still open. Do not repeat their "
                    "full text."),
                builtin=True)
            joined = "\n\n".join(f"## {m.name}\n{r}" for m, r in results)
            looper = agents.make_looper(coordinator, max_turns=4, provider=provider,
                                        deadline=deadline)
            out = looper.run_conversation(
                f"Synthesise these results:\n\n{joined}",
                system_message=coordinator.system_prompt).get("final_response", "")
            if out.strip():
                return out.strip()
        except Exception:  # noqa: BLE001 — fall back to the template
            pass
    names = ", ".join(m.name for m, _ in results)
    first = (results[0][1].splitlines() or [""])[0][:200]
    return (f"Done — {names} posted above.\nKey point: {first}"
            if first else f"Done — {names} posted above.")


def _consolidate_status(pool: list[Member],
                        results: list[tuple[Member, str]],
                        provider, deadline: float | None = None) -> str:
    """One consolidated status update from the collected per-member
    statuses. The Leader synthesises them in one short call; if that
    call fails or the deadline is past, a template lists each member's
    status instead. Either way every pool member is named, and a
    member with nothing to report says so plainly."""
    by_handle = {m.handle: r for m, r in results}
    if not _past(deadline):
        try:
            leader = Member(
                name="Leader", handle=LEADER_HANDLE, colour="#0ea5e9",
                emoji="🧭", specialty="coordination",
                system_prompt=(
                    "You are the Leader. Combine the team members' status "
                    "reports below into ONE short consolidated status "
                    "update. Name every member, one short line each: what "
                    "they are working on, what is done, what is blocked. "
                    "Say plainly when a member has nothing to report."),
                builtin=True)
            joined = "\n\n".join(
                f"## {m.name} (@{m.handle})\n"
                f"{by_handle.get(m.handle, '(no report)')}"
                for m in pool)
            looper = agents.make_looper(leader, max_turns=1, provider=provider,
                                        deadline=deadline)
            out = (looper.run_conversation(
                "Consolidate these status reports into one team status "
                f"update:\n\n{joined}",
                system_message=leader.system_prompt).get("final_response", "")
                or "").strip()
            if out:
                return out
        except Exception:  # noqa: BLE001 — fall back to the template
            pass
    lines = []
    for m in pool:
        report = by_handle.get(m.handle)
        first = (report.splitlines() or [""])[0][:200] if report else ""
        lines.append(f"- @{m.handle}: {first}" if first
                     else f"- @{m.handle}: nothing to report.")
    return "📋 Team status:\n" + "\n".join(lines)


def _run_status_roundup(store: RoomStore, room_id: str, goal: dict | None,
                        delegation_pool: list[Member], provider,
                        goal_id: str | None,
                        deadline: float | None = None) -> None:
    """A status roundup: the Leader loops through every member in the
    pool, collects one short status each, and posts exactly ONE
    consolidated status message.

    Each member gets one short call grounded in its team-scoped memory
    and the recent room history, so the report reflects what it
    actually did here. The statuses are never posted individually.
    The Leader then wraps them up into one update (template fallback
    when the wrap-up call fails or the deadline is past).

    Statuses are coordination, not work: the single message is posted
    with author_type='orchestrator' (goal_task_count only counts
    author_type='agent', so goal progress never moves) and no XP or
    memory is recorded for it. The deadline is checked before each
    member; past it, the roundup wraps up with whatever was collected
    — or, if nothing was collected, posts the timeout note and returns.
    """
    context = _room_context(store, room_id)
    results: list[tuple[Member, str]] = []
    for member in delegation_pool:
        if _past(deadline):
            break
        agent_context = build_agent_context(store, member, goal)
        combined = ((agent_context + "\n\n") if agent_context else "") + context
        task = ("Give a short status report: what you are working on, "
                "what is done, and what is blocked. Keep it to a few "
                "short lines, grounded in what you actually did here — "
                "do not invent work.")
        response = agents.run_member(member, task, combined, provider,
                                     max_turns=1, deadline=deadline)
        results.append((member, response))
    if not results:
        _post_best_effort(
            store, room_id, goal_id, author="Room",
            author_type="system",
            content="⏱️ The run hit the time limit — partial results are "
                    "above. Send the instruction again to continue.")
        return
    _post_best_effort(store, room_id, goal_id, author="Leader",
                      author_type="orchestrator", handle=LEADER_HANDLE,
                      content=_consolidate_status(delegation_pool, results,
                                                  provider, deadline))


def _run_leader(store: RoomStore, room_id: str, instruction: str,
                goal: dict | None, team_id: str | None,
                delegation_pool: list[Member], model_status: dict, provider,
                goal_id: str | None, deadline: float | None = None,
                project: ProjectWorkspace | None = None) -> bool:
    """One leader turn: plan, delegate in order, integrate.

    Returns True when the Leader completed the turn, False when it
    could not (no usable plan) and the caller should fall back to the
    parallel path. The plan and the integrated result are posted as
    coordination messages, so they never count toward a goal's task
    progress — only the specialists' work does.

    The deadline is checked before every step. When it is past, the
    turn ends right there with the partial results above (True — the
    turn is done, just short) and the caller must NOT fall back to the
    parallel path, which would only start more work.

    When a project is active, the plan is told to include a step that
    writes the deliverable files into it, and every delegated
    specialist runs with the project workspace — its file tools then
    write to the project's files in the store, not the local disk.
    """
    leader = agents.member_by_handle(store, LEADER_HANDLE)
    if leader is None or _past(deadline):
        return False
    # Status: the plan call can take a while, and nothing is visible
    # until it returns — say so up front so the room is not silent.
    _post_best_effort(store, room_id, goal_id, author="Leader",
                      author_type="orchestrator", handle=LEADER_HANDLE,
                      content="🧭 Leader is planning how to handle this...")
    handles = ", ".join(f"@{m.handle}" for m in delegation_pool)
    plan_prompt = (
        "Plan how to handle this instruction.\n"
        f"Instruction: {instruction}\n"
        + (f"Goal: {goal['title']}"
           + (f" — {goal['description']}" if goal.get("description") else "")
           + "\n" if goal else "")
        + (f"A project '{project.name}' is active for this work: include "
           "a step in which the specialist writes the deliverable files "
           "into the project with the write_file tool (e.g. main.py, "
           "requirements.txt, README.md). Files in the project persist; "
           "code only pasted in chat does not.\n" if project else "")
        + f"Available specialists (pick only from these handles): {handles}\n"
        "Reply with the plan only, one STEP line per step, at most "
        f"{MAX_LEADER_STEPS} steps.")
    try:
        looper = agents.make_looper(leader, max_turns=LEADER_PLAN_TURNS,
                                    provider=provider, deadline=deadline)
        with projects.active_project(project):
            plan_text = (looper.run_conversation(
                plan_prompt, system_message=leader.system_prompt)
                .get("final_response", "") or "").strip()
    except Exception:  # noqa: BLE001 — a failed plan falls back to parallel
        return False
    steps = parse_plan(plan_text, delegation_pool)
    if not steps:
        return False

    plan_lines = [f"{i + 1}. @{m.handle}: {sub}" for i, (m, sub) in enumerate(steps)]
    plan_content = "🧭 Leader's plan:\n" + "\n".join(plan_lines)
    if model_status.get("mock"):
        plan_content = "[mock model] " + plan_content
    skipped = plan_skipped_handles(plan_text, delegation_pool)
    if skipped:
        plan_content += ("\n(Skipped: " + ", ".join(f"@{h}" for h in skipped)
                         + " — not available for this turn.)")
    _post_best_effort(store, room_id, goal_id, author="Leader",
                      author_type="orchestrator", handle=LEADER_HANDLE,
                      content=plan_content)

    context = _room_context(store, room_id)
    results: list[tuple[Member, str]] = []
    prior: list[tuple[Member, str]] = []
    for i, (member, subtask) in enumerate(steps, start=1):
        if _past(deadline):
            _post_best_effort(
                store, room_id, goal_id, author="Leader",
                author_type="orchestrator", handle=LEADER_HANDLE,
                content="⏱️ The run hit the time limit — the plan and the "
                        "finished steps are above. Send the instruction again "
                        "to continue from here.")
            return True
        # Status: specialists run one at a time and each reply only
        # posts when it finishes, so narrate each handoff.
        _post_best_effort(store, room_id, goal_id, author="Leader",
                          author_type="orchestrator", handle=LEADER_HANDLE,
                          content=f"➡️ Step {i} of {len(steps)}: delegating "
                                  f"to @{member.handle}.")
        agent_context = build_agent_context(store, member, goal, project)
        combined = ((agent_context + "\n\n") if agent_context else "") + context
        if prior:
            combined += "\n\n[Results so far — build on these]\n" + "\n".join(
                f"- {pm.name}: {pr[:400]}" for pm, pr in prior)
        task = f"{subtask}\n(Original instruction: {instruction})"
        response = agents.run_member(member, task, combined, provider,
                                     max_turns=LEADER_DELEGATE_TURNS,
                                     deadline=deadline, project=project)
        results.append((member, response))
        prior.append((member, response))
        _post_best_effort(store, room_id, goal_id, author=member.name,
                          author_type="agent", handle=member.handle,
                          content=response)
        _record_outcome(store, team_id, member, instruction, response)

    if _past(deadline):
        _post_best_effort(
            store, room_id, goal_id, author="Leader",
            author_type="orchestrator", handle=LEADER_HANDLE,
            content="⏱️ The run hit the time limit before the final answer — "
                    "the specialists' work is above. Send the instruction "
                    "again to continue from here.")
        return True

    # Status: the integrate call is the last silent stretch.
    _post_best_effort(store, room_id, goal_id, author="Leader",
                      author_type="orchestrator", handle=LEADER_HANDLE,
                      content=f"✅ All {len(steps)} steps done — integrating "
                              "the results.")
    joined = "\n\n".join(f"## {m.name}\n{r}" for m, r in results)
    integrate_prompt = (
        "Integrate these results into one final answer.\n"
        f"Instruction: {instruction}\n\n{joined}")
    try:
        looper = agents.make_looper(leader, max_turns=LEADER_INTEGRATE_TURNS,
                                    provider=provider, deadline=deadline)
        with projects.active_project(project):
            integrated = (looper.run_conversation(
                integrate_prompt, system_message=leader.system_prompt)
                .get("final_response", "") or "").strip()
    except Exception:  # noqa: BLE001 — still close the turn honestly
        integrated = ""
    if not integrated:
        names = ", ".join(m.name for m, _ in results)
        integrated = f"Done — {names} posted above."
    _post_best_effort(store, room_id, goal_id, author="Leader",
                      author_type="orchestrator", handle=LEADER_HANDLE,
                      content=integrated)
    _record_outcome(store, team_id, leader, instruction, integrated)
    return True


def _run_parallel(store: RoomStore, room_id: str, instruction: str,
                  goal: dict | None, team_id: str | None,
                  pool: list[Member], model_status: dict, provider,
                  goal_id: str | None, deadline: float | None = None,
                  project: ProjectWorkspace | None = None) -> None:
    """The original path: route by @-mention/keyword, run the selected
    specialists in parallel, post a Room synthesis. Used when the
    Leader cannot run. The deadline stops new work from starting —
    specialists already running finish on their own deadline, and the
    synthesis is built from whatever finished. When a project is
    active each specialist runs with its workspace, so its file tools
    write to the project's files."""
    selected, _ = select_members(instruction, pool)
    _post_best_effort(
        store, room_id, goal_id, author="Room", author_type="orchestrator",
        content="Routing to " + ", ".join(f"@{m.handle}" for m in selected) + "."
                + (f" Goal: {goal['title']}." if goal else ""))

    context = _room_context(store, room_id)
    results: list[tuple[Member, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool_exec:
        futures = {}
        for m in selected:
            if _past(deadline):
                break
            agent_context = build_agent_context(store, m, goal, project)
            combined = ((agent_context + "\n\n") if agent_context else "") + context
            futures[pool_exec.submit(agents.run_member, m, instruction, combined,
                                     provider, PARALLEL_TURNS, deadline,
                                     project=project)] = m
        for fut in as_completed(futures):
            member = futures[fut]
            try:
                response = fut.result()
            except Exception as exc:  # noqa: BLE001 — one agent failing must not kill the room
                response = f"⚠️ {member.name} failed: {type(exc).__name__}: {exc}"
            results.append((member, response))
            _post_best_effort(store, room_id, goal_id, author=member.name,
                              author_type="agent", handle=member.handle,
                              content=response)
            _record_outcome(store, team_id, member, instruction, response)
    # keep results in the order the room was told about, not finish order
    order = {m.handle: i for i, m in enumerate(selected)}
    results.sort(key=lambda p: order.get(p[0].handle, 99))

    _post_best_effort(store, room_id, goal_id, author="Room",
                      author_type="orchestrator",
                      content=_synthesise(results, model_status.get("mock", False),
                                          provider=provider, deadline=deadline))


def run_turn(store: RoomStore, room_id: str, instruction: str,
             model_status: dict, goal_id: str | None = None,
             provider=None, deadline: float | None = None,
             project_id: str | None = None) -> None:
    """Run one user instruction end to end, appending room messages.

    All state lands in the store, so the SSE stream (which polls the
    store) picks it up and the room survives restarts. Every message
    in the turn is tagged with the goal it belongs to (if any).

    The Leader runs first when a model is configured: it plans, the
    specialists work in plan order, and it integrates. @-mentions of
    specific members restrict the delegation pool to them (an @leader
    mention just invokes the Leader explicitly); with a goal active
    the pool is the goal's members. A status ask ("Status", "where
    are we") never reaches the Leader's work path: it triggers a
    roundup in which the Leader collects one short status from every
    pool member and posts exactly one consolidated update. When the
    Leader cannot make a plan, the room falls back to the parallel
    path.

    `project_id` is the active project for this turn: the explicitly
    named project wins, otherwise the goal's linked project (if the
    turn has a goal). With a project active, the Leader's plan
    includes a step that writes the deliverable files into it, and
    the specialists' file tools operate on the project's files in the
    store instead of the local disk.

    `provider` is a per-request BYOK provider (built from the browser's
    key). It counts as "configured" for this turn only and is never
    stored. The mock client still wins when TEAMROOM_MOCK_MODEL=1.

    `deadline` is a monotonic timestamp: once it is past, no new work
    is started anywhere in the turn (no new plan, specialist, or
    integrate call, and no parallel fallback — that would only start
    more work). The turn ends with the partial results above, and the
    stream reports turn_timeout. Without a deadline the turn is
    bounded only by the per-agent turn caps.
    """
    if deadline is None:
        deadline = time.monotonic() + DEFAULT_TURN_DEADLINE_S
    all_members = agents.list_members(store)
    specialists = agents.list_specialists(store)
    by_handle = {m.handle: m for m in all_members}
    goal = store.get_goal(goal_id) if goal_id else None
    team_id = goal["team_id"] if goal else None

    if goal_id and not goal:
        _post_best_effort(
            store, room_id, goal_id, author="Room", author_type="system",
            content=f"Unknown goal {goal_id} — running without a goal.")
        goal_id = None

    if goal:
        pool = [m for m in specialists if m.handle in goal["members"]]
        if not pool:
            _post_best_effort(
                store, room_id, goal_id, author="Room", author_type="system",
                content=f"Goal '{goal['title']}' has no members — add some first.")
            return
    else:
        pool = specialists

    # The active project for this turn: the explicitly named project
    # wins, otherwise the goal's linked project. Without one, the
    # file tools fall back to the local filesystem as before.
    project = projects.resolve_project(store, project_id, goal_id)

    # @-mentions: the Leader is the orchestrator, not a delegate, so an
    # @leader mention invokes it explicitly and never lands in the pool.
    # Other mentions restrict the delegation pool to the named members.
    mentions = parse_mentions(instruction)
    pool_handles = {m.handle for m in pool}
    mentioned = [by_handle[h] for h in mentions
                 if h in pool_handles and h != LEADER_HANDLE]
    delegation_pool = mentioned if mentioned else pool

    # Report mentions the room cannot honour — unknown handles, and
    # members that exist but are outside this turn's pool (off-goal).
    nowhere = [h for h in mentions if h not in by_handle]
    off_pool = [h for h in mentions
                if h in by_handle and h != LEADER_HANDLE and h not in pool_handles]
    if nowhere or off_pool:
        notes = []
        if nowhere:
            notes.append(f"@{', @'.join(nowhere)} — no member with that handle")
        if off_pool:
            if goal:
                notes.append(f"@{', @'.join(off_pool)} — not on goal "
                             f"'{goal['title']}' (its members: "
                             f"{', '.join('@' + h for h in goal['members'])})")
            else:
                notes.append(f"@{', @'.join(off_pool)} — not available here")
        _post_best_effort(store, room_id, goal_id, author="Room",
                          author_type="system",
                          content="I don't know " + "; ".join(notes) + ".")

    if not (model_status.get("configured") or provider is not None):
        _post_best_effort(
            store, room_id, goal_id, author="Room", author_type="system",
            content="No model is configured, so nobody in the room can work yet. "
                    f"({model_status.get('error') or 'no provider'}) "
                    "Paste your OpenAI key in the sidebar — it stays in your "
                    "browser and is only used for that turn — or set "
                    "LOOPER_API_KEY (or LOOPER_MODEL, e.g. ollama/llama3.1) "
                    "and restart the server. The room, roster and history "
                    "all work in the meantime.")
        return

    if _is_status_request(instruction):
        # "Status" is a question for the whole room, not work for the
        # Leader — every member in the pool reports in.
        _run_status_roundup(store, room_id, goal, delegation_pool, provider,
                            goal_id, deadline)
        return

    if delegation_pool and _run_leader(store, room_id, instruction, goal,
                                       team_id, delegation_pool, model_status,
                                       provider, goal_id, deadline,
                                       project=project):
        return
    if _past(deadline):
        # The Leader stopped on the deadline (or the deadline passed
        # before it could plan). Do NOT fall back to parallel — that
        # would start more work after the budget is spent. The partial
        # results are already above; the stream reports turn_timeout.
        _post_best_effort(
            store, room_id, goal_id, author="Room", author_type="system",
            content="⏱️ The run hit the time limit — partial results are "
                    "above. Send the instruction again to continue.")
        return
    if delegation_pool:
        _post_best_effort(
            store, room_id, goal_id, author="Room", author_type="system",
            content="The Leader couldn't make a plan, so the room is "
                    "running the parallel path instead.")
    _run_parallel(store, room_id, instruction, goal, team_id, pool,
                  model_status, provider, goal_id, deadline, project=project)
