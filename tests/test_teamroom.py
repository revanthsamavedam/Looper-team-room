"""Tests for the Team Room — no API key, no network, no server needed.

Run:  python tests/test_teamroom.py
Routing and the no-model path are tested for real; the live-model path
is exercised with the scripted mock client (TEAMROOM_MOCK_MODEL=1 in
the server, MockClient here), which proves routing, identity and
posting work end to end without a key.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["TEAMROOM_MOCK_MODEL"] = "1"  # noqa: E402 — must precede teamroom import

from teamroom import agents, orchestrator  # noqa: E402
from teamroom.skills import parse_skill_text  # noqa: E402
from teamroom.store import RoomStore, level_for_xp  # noqa: E402

SAMPLE_SKILL = """---
name: Dana
specialty: Data analysis — turns raw numbers into clear findings
---
# Dana — Data Analyst

## System prompt
You are Dana, the Data Analyst in a team chat room. When asked to
analyse numbers, compute the key figures and list the top 3 findings.
"""


def _store():
    tmp = tempfile.mkdtemp()
    return RoomStore(Path(tmp) / "t.db")


def _store_with_dana():
    store = _store()
    parsed = parse_skill_text(SAMPLE_SKILL)
    store.add_member(source="test", **parsed)
    return store


def test_skill_parse_frontmatter():
    p = parse_skill_text(SAMPLE_SKILL)
    assert p["name"] == "Dana" and p["handle"] == "dana", p
    assert "Data analysis" in p["specialty"], p
    assert "top 3 findings" in p["system_prompt"], p


def test_skill_parse_plain_no_frontmatter():
    p = parse_skill_text("# Rex\nspecialty: Release herding\n\n## Instructions\n"
                         "You are Rex. Keep releases on rails.")
    assert p["name"] == "Rex" and p["handle"] == "rex", p
    assert p["specialty"] == "Release herding", p
    assert "on rails" in p["system_prompt"], p


def test_skill_parse_missing_name():
    try:
        parse_skill_text("## System prompt\nDo the thing.")
    except ValueError as exc:
        assert "name" in str(exc).lower(), exc
    else:
        raise AssertionError("expected ValueError for missing name")


def test_skill_parse_missing_prompt():
    try:
        parse_skill_text("# Ghost\nspecialty: Haunting\n")
    except ValueError as exc:
        assert "system prompt" in str(exc).lower(), exc
    else:
        raise AssertionError("expected ValueError for missing prompt")


def test_skill_handle_slug():
    p = parse_skill_text("# Dana the Data Analyst\n\n## System prompt\n"
                         "You analyse numbers for the team.")
    assert p["handle"] == "dana-the-data-analyst", p


def test_routing_mention_custom_member():
    store = _store_with_dana()
    selected, unknown = orchestrator.select_members(
        "@dana analyse the numbers 10, 20, 30", agents.list_members(store))
    assert [m.handle for m in selected] == ["dana"], selected
    assert unknown == []


def test_routing_mention_builtin():
    store = _store()
    selected, _ = orchestrator.select_members(
        "@qa check the login flow", agents.list_members(store))
    assert [m.handle for m in selected] == ["qa"], selected


def test_routing_unknown_mention_reported():
    store = _store()
    selected, unknown = orchestrator.select_members(
        "@bob do the thing", agents.list_members(store))
    assert unknown == ["bob"], unknown
    assert selected, "falls back to keyword/default routing"


def test_routing_keywords():
    store = _store()
    members = agents.list_members(store)
    selected, _ = orchestrator.select_members("research the best vector database", members)
    assert "researcher" in [m.handle for m in selected], selected
    selected, _ = orchestrator.select_members("review this design for risks", members)
    assert "reviewer" in [m.handle for m in selected], selected


def test_routing_default_is_one_member():
    store = _store()
    selected, _ = orchestrator.select_members("hello", agents.list_members(store))
    assert len(selected) == 1, selected


def test_store_persists_members_and_messages():
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "t.db"
    s1 = RoomStore(path)
    s1.create_room("Team Room", room_id="main")
    s1.add_member(source="test", **parse_skill_text(SAMPLE_SKILL))
    s1.append_message("main", "you", "user", "hello room")
    s2 = RoomStore(path)  # reopen: nothing lost
    assert s2.get_member("dana")["name"] == "Dana"
    assert len(s2.messages("main")) == 1
    assert s2.messages("main")[0]["content"] == "hello room"


def test_build_provider_defaults_to_gpt4o_mini():
    p = agents.build_provider("sk-fake-test-key", None)
    assert p is not None
    assert p.model == "gpt-4o-mini"
    assert p.base_url == "https://api.openai.com/v1"
    assert agents.build_provider("", None) is None
    assert agents.build_provider(None, "gpt-4o") is None


def test_run_turn_byok_provider_counts_as_configured():
    """A browser-supplied key counts as configured for that turn only.

    Mock is on, so no network is touched and no real call is made; the
    turn runs (mock) instead of posting the no-model warning.
    """
    store = _store()
    store.create_room("Team Room", room_id="main")
    provider = agents.build_provider("sk-fake-test-key", "gpt-4o-mini")
    orchestrator.run_turn(store, "main", "@qa check the login flow",
                          {"configured": False, "model": None,
                           "error": "no API key: set LOOPER_API_KEY"},
                          provider=provider)
    msgs = store.messages("main")
    assert not any("No model is configured" in (m["content"] or "")
                   for m in msgs), msgs
    assert any(m["author_type"] == "agent" and m["handle"] == "qa"
               for m in msgs), msgs
    # the key itself never lands in the room log
    blob = " ".join((m["content"] or "") for m in msgs)
    assert "sk-fake-test-key" not in blob


def test_run_turn_no_model_posts_plain_warning():
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "@qa check the login flow",
                          {"configured": False, "model": None,
                           "error": "no API key: set LOOPER_API_KEY"})
    msgs = store.messages("main")
    assert any(m["author_type"] == "system" and "No model is configured" in m["content"]
               for m in msgs), msgs
    assert not any(m["author_type"] == "agent" for m in msgs), msgs  # nothing faked


def test_run_turn_mock_custom_member_responds_as_itself():
    store = _store_with_dana()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "@dana analyse the numbers 10, 20, 30",
                          {"configured": True, "model": "mock", "mock": True})
    msgs = store.messages("main")
    dana = [m for m in msgs if m["handle"] == "dana"]
    assert len(dana) == 1, msgs
    assert "[mock model]" in dana[0]["content"] and "Dana" in dana[0]["content"], dana
    assert any(m["author_type"] == "orchestrator" for m in msgs), msgs  # synthesis posted


def test_run_turn_mock_multi_agent():
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "build a fizzbuzz script and test it",
                          {"configured": True, "model": "mock", "mock": True})
    handles = {m["handle"] for m in store.messages("main") if m["author_type"] == "agent"}
    assert "dev" in handles and "qa" in handles, handles


# --- goals & teams --------------------------------------------------------

def _two_teams():
    """Team Alpha (dev, qa, dana) and Team Beta (qa, writer, dana)."""
    store = _store_with_dana()
    alpha = store.create_team("Alpha", ["dev", "qa", "dana"])
    beta = store.create_team("Beta", ["qa", "writer", "dana"])
    goal_a = store.create_goal(alpha["id"], "Ship v1", "First release", ["dev", "dana"])
    goal_b = store.create_goal(beta["id"], "Docs push", "Write the guides", ["writer", "dana"])
    return store, alpha, beta, goal_a, goal_b


def test_goal_members_must_come_from_the_team():
    store = _store()
    team = store.create_team("Alpha", ["dev", "qa"])
    try:
        store.create_goal(team["id"], "Bad goal", "", ["dev", "writer"])
    except ValueError as exc:
        assert "not on this team" in str(exc), exc
    else:
        raise AssertionError("expected ValueError for a non-team goal member")
    goal = store.create_goal(team["id"], "Good goal", "", ["dev"])
    assert goal["members"] == ["dev"] and goal["team_id"] == team["id"]


def test_member_memory_stays_scoped_per_team():
    store, alpha, beta, _, _ = _two_teams()
    store.record_memory(alpha["id"], "dana", "did", "Analysed Alpha sales")
    store.record_memory(beta["id"], "dana", "did", "Analysed Beta docs")
    a = store.list_memory(alpha["id"], handle="dana")
    b = store.list_memory(beta["id"], handle="dana")
    assert len(a) == 1 and "Alpha sales" in a[0]["content"], a
    assert len(b) == 1 and "Beta docs" in b[0]["content"], b
    assert all(e["team_id"] == alpha["id"] for e in a)
    assert all(e["team_id"] == beta["id"] for e in b)


def test_best_practice_is_shared_across_teams():
    store, alpha, beta, goal_a, goal_b = _two_teams()
    store.add_best_practice("Keep replies focused",
                            "Each agent answers only its own specialty.",
                            source_team_id=alpha["id"], source_handle="dev")
    assert len(store.list_best_practices()) == 1
    dana = agents.member_by_handle(store, "dana")
    ctx_b = orchestrator.build_agent_context(store, dana, goal_b)
    assert "Keep replies focused" in ctx_b, ctx_b  # visible in the other team


def test_agent_context_feeds_team_memory_not_other_teams():
    store, alpha, beta, goal_a, goal_b = _two_teams()
    store.record_memory(alpha["id"], "dana", "learned", "Alpha uses euros")
    store.record_memory(beta["id"], "dana", "learned", "Beta uses dollars")
    dana = agents.member_by_handle(store, "dana")
    ctx_a = orchestrator.build_agent_context(store, dana, goal_a)
    ctx_b = orchestrator.build_agent_context(store, dana, goal_b)
    assert "euros" in ctx_a and "dollars" not in ctx_a, ctx_a
    assert "dollars" in ctx_b and "euros" not in ctx_b, ctx_b


def test_goal_routing_never_leaves_the_goal_pool():
    store, _, _, goal_a, _ = _two_teams()
    members = agents.list_members(store)
    pool = [m for m in members if m.handle in goal_a["members"]]
    selected, _ = orchestrator.select_members("build a thing", pool)
    assert selected and all(m.handle in goal_a["members"] for m in selected), selected


def test_run_turn_mock_goal_scoped_memory_and_shared_practice():
    store, alpha, beta, goal_a, goal_b = _two_teams()
    store.create_room("Team Room", room_id="main")
    mock = {"configured": True, "model": "mock", "mock": True}

    orchestrator.run_turn(store, "main", "@dana analyse the numbers 10, 20, 30",
                          mock, goal_id=goal_a["id"])
    mem_a = store.list_memory(alpha["id"], handle="dana")
    mem_b = store.list_memory(beta["id"], handle="dana")
    assert mem_a, "Dana's Alpha work should be recorded in Alpha"
    assert mem_b == [], f"Dana's Alpha work leaked into Beta: {mem_b}"
    assert any(e["kind"] == "did" for e in mem_a), mem_a
    assert any(e["kind"] == "decided" for e in mem_a), mem_a
    practices = store.list_best_practices()
    assert practices, "the mock Best practice line should land in the shared layer"
    assert practices[0]["source_team_id"] == alpha["id"]

    orchestrator.run_turn(store, "main", "@dana analyse the numbers 1, 2, 3",
                          mock, goal_id=goal_b["id"])
    mem_b = store.list_memory(beta["id"], handle="dana")
    assert mem_b, "Dana's Beta work should be recorded in Beta"
    assert len(store.list_memory(alpha["id"], handle="dana")) == len(mem_a), \
        "Beta work must not change Alpha's memory"
    # every message in the goal-B turn is tagged with that goal
    msgs_b = [m for m in store.messages("main") if m.get("goal_id") == goal_b["id"]]
    assert any(m["handle"] == "dana" for m in msgs_b), msgs_b


# --- gamification ------------------------------------------------------

def test_xp_awarded_for_mock_turn():
    """A mock turn pays real XP: turn + Decision + Learned + Best practice.

    The mock reply carries all three structured lines, so QA earns
    10 + 5 + 5 + 15 = 35 XP, 1 turn, and the first-steps / decider /
    sharer badges. Nothing is invented — the XP follows the reply.
    """
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "@qa check the login flow",
                          {"configured": True, "model": "mock", "mock": True})
    xp = store.get_xp("qa")
    assert xp["xp"] == 35, xp
    assert xp["turns"] == 1, xp
    assert xp["level"] == 1 and xp["xp_into_level"] == 35, xp
    names = {b["id"] for b in xp["badges"]}
    assert names == {"first-steps", "decider", "sharer"}, names
    assert store.get_xp("dev")["xp"] == 0  # Dev did not work this turn


def test_leaderboard_orders_by_xp():
    store = _store()
    store.add_xp("dev", 50)
    store.add_xp("qa", 120)
    board = store.leaderboard()
    assert [r["handle"] for r in board] == ["qa", "dev"], board
    assert board[0]["level"] == 2 and board[0]["xp_into_level"] == 20, board[0]
    assert level_for_xp(0)["level"] == 1 and level_for_xp(100)["level"] == 2


def test_goal_progress_and_completion_xp():
    store, alpha, _, goal_a, _ = _two_teams()
    store.create_room("Team Room", room_id="main")
    assert store.get_goal(goal_a["id"])["tasks_done"] == 0
    assert store.get_goal(goal_a["id"])["target_tasks"] == 5
    orchestrator.run_turn(store, "main", "@dana analyse the numbers 10, 20, 30",
                          {"configured": True, "model": "mock", "mock": True},
                          goal_id=goal_a["id"])
    assert store.get_goal(goal_a["id"])["tasks_done"] == 1  # one agent reply
    # marking the goal done pays its members once — and only once
    store.set_goal_status(goal_a["id"], "done")
    dana_xp = store.get_xp("dana")
    assert dana_xp["xp"] == 35 + 25, dana_xp  # turn+lines, then goal bonus
    assert "goal-getter" in {b["id"] for b in dana_xp["badges"]}, dana_xp
    store.set_goal_status(goal_a["id"], "done")  # second time: no double pay
    assert store.get_xp("dana")["xp"] == 60, store.get_xp("dana")


def test_badges_awarded_once():
    store = _store()
    assert store.award_badge("qa", "bug-hunter") is True
    assert store.award_badge("qa", "bug-hunter") is False
    assert len(store.list_badges("qa")) == 1


def test_turso_backend_uses_libsql_and_persists():
    """TURSO_URL switches the store to the libsql client.

    A local file path as TURSO_URL opens a libsql file DB — the same
    client code path as the remote libsql:// URL, without touching the
    network. Proves the store works through libsql: schema, members,
    messages, teams, XP, and the duplicate-handle ValueError.
    """
    tmp = tempfile.mkdtemp()
    db_path = str(Path(tmp) / "turso.db")
    old_url = os.environ.get("TURSO_URL")
    old_token = os.environ.get("TURSO_AUTH_TOKEN")
    os.environ["TURSO_URL"] = db_path
    os.environ.pop("TURSO_AUTH_TOKEN", None)
    try:
        s1 = RoomStore(Path(tmp) / "ignored.db")
        assert s1.backend == "turso", s1.backend
        s1.create_room("Team Room", room_id="main")
        s1.add_member(source="test", **parse_skill_text(SAMPLE_SKILL))
        s1.append_message("main", "you", "user", "hello turso")
        s1.add_xp("dana", 40)
        team = s1.create_team("Alpha", ["dana", "dev"])
        # duplicate handle is a clean ValueError on the libsql path
        try:
            s1.add_member(source="test", **parse_skill_text(SAMPLE_SKILL))
        except ValueError as exc:
            assert "already exists" in str(exc), exc
        else:
            raise AssertionError("expected ValueError for a duplicate handle")
        s2 = RoomStore(Path(tmp) / "ignored.db")  # reopen: nothing lost
        assert s2.backend == "turso"
        assert s2.get_member("dana")["name"] == "Dana"
        assert len(s2.messages("main")) == 1
        assert s2.get_xp("dana")["xp"] == 40
        assert s2.get_team(team["id"])["members"] == ["dana", "dev"]
    finally:
        if old_url is None:
            os.environ.pop("TURSO_URL", None)
        else:
            os.environ["TURSO_URL"] = old_url
        if old_token is not None:
            os.environ["TURSO_AUTH_TOKEN"] = old_token


# --- leader / subagent model ----------------------------------------------

def test_leader_in_roster_but_not_a_delegate():
    store = _store()
    members = agents.list_members(store)
    leader = [m for m in members if m.handle == "leader"]
    assert len(leader) == 1, members
    assert leader[0].name == "Leader" and leader[0].builtin, leader
    assert leader[0].emoji == "🧭", leader
    specialists = agents.list_specialists(store)
    assert all(m.handle != "leader" for m in specialists), specialists
    assert len(specialists) == len(members) - 1


def test_parse_plan_valid_order_and_cap():
    store = _store()
    pool = agents.list_specialists(store)
    plan = ("STEP @researcher: gather the facts.\n"
            "STEP @dev: build it.\n"
            "STEP @qa: verify it.\n"
            "STEP @writer: document it.\n")
    steps = orchestrator.parse_plan(plan, pool)
    assert [m.handle for m, _ in steps] == ["researcher", "dev", "qa"], steps
    assert steps[0][1] == "gather the facts.", steps
    dup = "STEP @dev: first.\nSTEP @dev: second.\nSTEP @qa: check.\n"
    assert [m.handle for m, _ in orchestrator.parse_plan(dup, pool)] == ["dev", "qa"]


def test_parse_plan_filters_to_pool():
    store = _store()
    pool = [m for m in agents.list_specialists(store) if m.handle in ("dev", "qa")]
    plan = ("STEP @researcher: gather facts.\n"
            "STEP @dev: build it.\n"
            "STEP @qa: verify it.\n")
    steps = orchestrator.parse_plan(plan, pool)
    assert [m.handle for m, _ in steps] == ["dev", "qa"], steps
    assert orchestrator.plan_skipped_handles(plan, pool) == ["researcher"]


def test_parse_plan_unparseable():
    store = _store()
    pool = agents.list_specialists(store)
    assert orchestrator.parse_plan("Just do the thing, no steps here.", pool) == []
    assert orchestrator.parse_plan("", pool) == []
    assert orchestrator.parse_plan("STEP @nobody: do it.", pool) == []


def test_leader_turn_posts_plan_delegates_integrates():
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "build a fizzbuzz script and test it",
                          {"configured": True, "model": "mock", "mock": True})
    msgs = store.messages("main")
    leader_msgs = [m for m in msgs if m["handle"] == "leader"]
    # planning status + plan + 3 delegating statuses + integrating
    # status + integrated result
    assert len(leader_msgs) == 7, msgs
    assert "planning how to handle this" in leader_msgs[0]["content"], leader_msgs
    assert "Leader's plan" in leader_msgs[1]["content"], leader_msgs
    assert "Step 1 of 3: delegating to @dev" in leader_msgs[2]["content"], leader_msgs
    assert "Step 2 of 3: delegating to @qa" in leader_msgs[3]["content"], leader_msgs
    assert "Step 3 of 3: delegating to @researcher" in leader_msgs[4]["content"], leader_msgs
    assert "integrating the results" in leader_msgs[5]["content"], leader_msgs
    assert "integrated" in leader_msgs[6]["content"].lower(), leader_msgs
    # every leader message is coordination, never goal progress
    assert all(m["author_type"] == "orchestrator" for m in leader_msgs), leader_msgs
    # the specialists ran in plan order (mock plan: dev, qa, researcher)
    agent_handles = [m["handle"] for m in msgs if m["author_type"] == "agent"]
    assert agent_handles == ["dev", "qa", "researcher"], agent_handles
    # no parallel-path routing: the leader planned instead
    assert not any(m["author"] == "Room" and "Routing to" in (m["content"] or "")
                   for m in msgs), msgs


def test_leader_status_updates_in_order():
    """The Leader narrates the turn as it goes: planning, one delegating
    status per step, integrating — in that order, interleaved with the
    plan, the specialists' work, and the integrated result."""
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "build a fizzbuzz script and test it",
                          {"configured": True, "model": "mock", "mock": True})
    msgs = store.messages("main")
    contents = [m["content"] or "" for m in msgs]
    # order: planning status < plan < step 1 < dev's reply < step 2
    # < qa's reply < step 3 < researcher's reply < integrating < result
    idx = {}
    for needle in ("Leader is planning how to handle this", "Leader's plan:",
                   "Step 1 of 3: delegating to @dev",
                   "Step 2 of 3: delegating to @qa",
                   "Step 3 of 3: delegating to @researcher",
                   "All 3 steps done — integrating the results."):
        hits = [i for i, c in enumerate(contents) if needle in c]
        assert len(hits) == 1, (needle, contents)
        idx[needle] = hits[0]
    ordered = sorted(idx, key=idx.get)
    assert ordered == ["Leader is planning how to handle this", "Leader's plan:",
                       "Step 1 of 3: delegating to @dev",
                       "Step 2 of 3: delegating to @qa",
                       "Step 3 of 3: delegating to @researcher",
                       "All 3 steps done — integrating the results."], ordered
    # each delegating status lands right before that specialist's reply
    for step_no, handle in ((1, "dev"), (2, "qa"), (3, "researcher")):
        d = idx[f"Step {step_no} of 3: delegating to @{handle}"]
        agent = [i for i, m in enumerate(msgs)
                 if m["author_type"] == "agent" and m["handle"] == handle]
        assert agent and d < agent[0], (step_no, handle, msgs)


def test_leader_delegation_order_follows_plan():
    """Execution order must match the order of the plan's steps."""
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "research the options, build it, test it",
                          {"configured": True, "model": "mock", "mock": True})
    msgs = store.messages("main")
    plan = [m for m in msgs if m["handle"] == "leader"
            and "Leader's plan:" in (m["content"] or "")][0]["content"]
    plan_handles = orchestrator.MENTION_RE.findall(plan)
    agent_handles = [m["handle"] for m in msgs if m["author_type"] == "agent"]
    assert agent_handles == plan_handles, (plan_handles, agent_handles)


def test_leader_goal_scoping():
    store, alpha, beta, goal_a, goal_b = _two_teams()
    store.create_room("Team Room", room_id="main")
    # goal_a members are dev + dana; the leader must delegate only to them
    orchestrator.run_turn(store, "main", "build it and analyse the numbers",
                          {"configured": True, "model": "mock", "mock": True},
                          goal_id=goal_a["id"])
    msgs = store.messages("main")
    agent_handles = {m["handle"] for m in msgs if m["author_type"] == "agent"}
    assert agent_handles == {"dev", "dana"}, agent_handles
    plan = [m for m in msgs if m["handle"] == "leader"
            and "Leader's plan:" in (m["content"] or "")][0]["content"]
    assert "@dev" in plan and "@dana" in plan, plan
    assert "@researcher" not in plan and "@qa" not in plan, plan


def test_leader_mention_respect():
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "@qa check the login flow",
                          {"configured": True, "model": "mock", "mock": True})
    msgs = store.messages("main")
    agent_handles = [m["handle"] for m in msgs if m["author_type"] == "agent"]
    assert agent_handles == ["qa"], agent_handles
    leader_msgs = [m for m in msgs if m["handle"] == "leader"]
    # planning status + plan + 1 delegating status + integrating status
    # + integrated result
    assert len(leader_msgs) == 5, msgs
    assert "Step 1 of 1: delegating to @qa" in leader_msgs[2]["content"], leader_msgs


def test_leader_fallback_to_parallel_when_no_plan():
    store = _store()
    store.create_room("Team Room", room_id="main")
    orig = orchestrator.parse_plan
    orchestrator.parse_plan = lambda text, pool: []  # leader makes no plan
    try:
        orchestrator.run_turn(store, "main", "build a fizzbuzz script and test it",
                              {"configured": True, "model": "mock", "mock": True})
    finally:
        orchestrator.parse_plan = orig
    msgs = store.messages("main")
    assert any("couldn't make a plan" in (m["content"] or "") for m in msgs), msgs
    assert any(m["author"] == "Room" and "Routing to" in (m["content"] or "")
               for m in msgs), msgs
    assert any(m["author_type"] == "orchestrator" and m["author"] == "Room"
               for m in msgs), msgs  # Room synthesis still posted
    # the Leader only posted its planning status before failing to plan —
    # no plan, no delegating statuses, no integration
    leader_msgs = [m for m in msgs if m["handle"] == "leader"]
    assert len(leader_msgs) == 1, msgs
    assert "planning how to handle this" in leader_msgs[0]["content"], leader_msgs


def test_leader_explicit_mention():
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "@leader build a fizzbuzz script and test it",
                          {"configured": True, "model": "mock", "mock": True})
    msgs = store.messages("main")
    leader_msgs = [m for m in msgs if m["handle"] == "leader"]
    # planning + plan + 3 delegating + integrating + integration
    assert len(leader_msgs) == 7, msgs
    assert not any("no member with that handle" in (m["content"] or "")
                   for m in msgs), msgs  # @leader is a real member


def test_leader_plan_does_not_count_toward_goal_progress():
    """Goal progress counts specialist work only — the Leader's plan and
    integration are coordination, not tasks done."""
    store, alpha, _, goal_a, _ = _two_teams()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "@dana analyse the numbers 10, 20, 30",
                          {"configured": True, "model": "mock", "mock": True},
                          goal_id=goal_a["id"])
    assert store.get_goal(goal_a["id"])["tasks_done"] == 1  # Dana only


# --- status roundup ---------------------------------------------------------

def test_status_request_matcher():
    """The matcher fires on status asks and stays quiet on work that
    merely mentions 'status' — when in doubt, it is work."""
    yes = ["Status", "status", "status update", "What's the status?",
           "whats the status", "give me a status", "team status",
           "progress", "where are we", "where are we at",
           "how are we doing", "how's it going",
           "what is the status of the build",
           "whats the status of the nvidia goal", "@dev status",
           "status please", "check the status", "status of the goal"]
    no = ["build a status page", "add a status field",
          "show the status in the UI", "build a fizzbuzz script and test it",
          "update the status", "status page", "progress bar",
          "fix the progress bar", "add a status bar to the dashboard",
          "hello test", "?", "write a status report for the client"]
    for text in yes:
        assert orchestrator._is_status_request(text), text
    for text in no:
        assert not orchestrator._is_status_request(text), text


def test_status_roundup_one_consolidated_leader_message():
    """A "Status" message produces exactly ONE new message — the
    Leader's consolidated status naming every goal member — and the
    goal's tasks_done does not move (a status is not work)."""
    store, alpha, _, goal_a, _ = _two_teams()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "Status",
                          {"configured": True, "model": "mock", "mock": True},
                          goal_id=goal_a["id"])
    msgs = store.messages("main")
    assert len(msgs) == 1, msgs  # the single consolidated status
    status = msgs[0]
    assert status["author"] == "Leader" and status["handle"] == "leader", msgs
    assert status["author_type"] == "orchestrator", msgs
    assert status["goal_id"] == goal_a["id"], msgs
    assert "@dev" in status["content"] and "@dana" in status["content"], msgs
    # no work happened: no agent messages, no leader plan, no XP
    assert not any(m["author_type"] == "agent" for m in msgs), msgs
    assert "Leader's plan" not in (status["content"] or ""), msgs
    assert store.get_goal(goal_a["id"])["tasks_done"] == 0
    assert store.leaderboard() == [], store.leaderboard()


def test_status_roundup_without_goal_covers_all_specialists():
    """With no goal active the roundup covers every specialist, still
    in exactly one Leader message."""
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "how are we doing",
                          {"configured": True, "model": "mock", "mock": True})
    msgs = store.messages("main")
    assert len(msgs) == 1, msgs
    for handle in ("dev", "qa", "researcher", "writer", "reviewer"):
        assert f"@{handle}" in msgs[0]["content"], msgs


def test_status_word_in_work_instruction_is_not_a_roundup():
    """'build a status page' is work, not a status ask: no consolidated
    status, the Leader plans instead."""
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "build a status page",
                          {"configured": True, "model": "mock", "mock": True})
    msgs = store.messages("main")
    assert not any("Team status" in (m["content"] or "") for m in msgs), msgs
    assert any("Leader's plan" in (m["content"] or "") for m in msgs), msgs


# --- projects -----------------------------------------------------------

def test_project_create_get_list_and_goal_link():
    store = _store()
    team = store.create_team("Alpha", ["dev", "qa"])
    goal = store.create_goal(team["id"], "Ship v1", "", ["dev"])
    p = store.create_project("nvidia-quotes-api", "FastAPI quotes service",
                             goal_id=goal["id"])
    assert p["name"] == "nvidia-quotes-api", p
    assert p["goal_id"] == goal["id"] and p["team_id"] == team["id"], p
    assert p["file_count"] == 0 and p["files"] == [], p
    assert store.get_project(p["id"])["id"] == p["id"]
    assert store.get_project_for_goal(goal["id"])["id"] == p["id"]
    assert [x["id"] for x in store.list_projects()] == [p["id"]]
    assert [x["id"] for x in store.list_projects(goal_id=goal["id"])] == [p["id"]]
    assert store.list_projects(goal_id="nope") == []
    assert store.get_project("nope") is None
    assert store.get_project_for_goal("nope") is None
    try:
        store.create_project("bad", goal_id="nope")
    except ValueError as exc:
        assert "no such goal" in str(exc), exc
    else:
        raise AssertionError("expected ValueError for an unknown goal")
    try:
        store.create_project("  ")
    except ValueError as exc:
        assert "needs a name" in str(exc), exc
    else:
        raise AssertionError("expected ValueError for an empty name")


def test_project_files_upsert_read_list():
    store = _store()
    p = store.create_project("demo")
    store.upsert_project_file(p["id"], "main.py", "print('hi')")
    store.upsert_project_file(p["id"], "src/app.py", "x = 1")
    row = store.get_project_file(p["id"], "main.py")
    assert row["content"] == "print('hi')" and row["path"] == "main.py", row
    assert [f["path"] for f in store.list_project_files(p["id"])] == [
        "main.py", "src/app.py"]
    assert store.get_project(p["id"])["file_count"] == 2
    # overwrite keeps one file, new content
    store.upsert_project_file(p["id"], "main.py", "print('yo')")
    assert store.get_project_file(p["id"], "main.py")["content"] == "print('yo')"
    assert store.get_project(p["id"])["file_count"] == 2
    assert store.get_project_file(p["id"], "missing.py") is None
    # unsafe paths are rejected, never written
    for bad in ("../escape.py", "/abs.py", "", "a/../../b.py"):
        try:
            store.upsert_project_file(p["id"], bad, "x")
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")
    assert store.get_project(p["id"])["file_count"] == 2
    try:
        store.upsert_project_file("nope", "a.py", "x")
    except ValueError as exc:
        assert "no such project" in str(exc), exc
    else:
        raise AssertionError("expected ValueError for an unknown project")


def test_project_file_tools_use_the_store():
    """With a project active, the same list_files / read_file /
    write_file tools the model knows write into the project store —
    not the local disk. With no project active they fall back to the
    filesystem as before."""
    import json
    from looper.registry import registry
    from teamroom import projects as project_mod
    store = _store()
    project = store.create_project("nvidia-quotes-api", "FastAPI quotes service")
    ws = project_mod.ProjectWorkspace(store, project["id"], project["name"])
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        with project_mod.active_project(ws):
            out = json.loads(registry.dispatch(
                "write_file", {"path": "main.py",
                               "content": "from fastapi import FastAPI\n"}))
            assert out["path"] == "main.py" and out["bytes"] > 0, out
            json.loads(registry.dispatch(
                "write_file", {"path": "requirements.txt",
                               "content": "fastapi\nuvicorn\n"}))
            out = json.loads(registry.dispatch("read_file", {"path": "main.py"}))
            assert "FastAPI" in out["content"] and not out["truncated"], out
            out = json.loads(registry.dispatch("list_files", {"path": "."}))
            assert out["files"] == ["main.py", "requirements.txt"], out
            out = json.loads(registry.dispatch("read_file", {"path": "nope.py"}))
            assert "error" in out, out
        # the writes went to the store, not the disk
        assert store.get_project_file(
            project["id"], "main.py")["content"].startswith("from fastapi")
        assert not (Path(tmp) / "main.py").exists()
        assert not (Path(tmp) / "requirements.txt").exists()
        # no project active: the same tools fall back to the filesystem
        out = json.loads(registry.dispatch(
            "write_file", {"path": "plain.txt", "content": "disk"}))
        assert out["bytes"] == 4, out
        assert (Path(tmp) / "plain.txt").read_text() == "disk"
        assert store.get_project_file(project["id"], "plain.txt") is None
    finally:
        os.chdir(old_cwd)


class _ScriptedClient:
    """Fake provider that pops scripted responses in order (same shape
    as the looper's own test client)."""
    def __init__(self, script):
        self.script = list(script)

    def call(self, messages, tools, deadline=None):
        if not self.script:
            return {"content": "done", "tool_calls": []}
        return self.script.pop(0)


def test_looper_tool_dispatch_writes_into_active_project():
    """A real Looper run whose model emits a write_file call lands the
    file in the active project — proving the ContextVar reaches the
    tool dispatch pool (see looper/loop.py) — not on the local disk."""
    from looper import Looper
    from looper.state import SessionStore as LooperSessionStore
    from teamroom import projects as project_mod
    store = _store()
    project = store.create_project("nvidia-quotes-api")
    ws = project_mod.ProjectWorkspace(store, project["id"], project["name"])
    script = [
        {"content": "", "tool_calls": [
            {"id": "c1", "name": "write_file",
             "arguments": {"path": "main.py",
                           "content": "from fastapi import FastAPI\n"}}]},
        {"content": "Wrote main.py to the project.", "tool_calls": []},
    ]
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        looper = Looper(client=_ScriptedClient(script),
                        store=LooperSessionStore(":memory:"),
                        home=Path(tmp), auto_approve=True)
        with project_mod.active_project(ws):
            out = looper.run_conversation(
                "write main.py for the FastAPI service")
        assert "Wrote main.py" in out.get("final_response", ""), out
        assert store.get_project_file(
            project["id"], "main.py")["content"].startswith("from fastapi")
        assert not (Path(tmp) / "main.py").exists()
    finally:
        os.chdir(old_cwd)


def test_project_resolution_prefers_explicit_then_goal():
    from teamroom import projects as project_mod
    store, alpha, _, goal_a, _ = _two_teams()
    p_goal = store.create_project("goal-proj", goal_id=goal_a["id"])
    p_other = store.create_project("other-proj")
    assert project_mod.resolve_project(
        store, p_other["id"], goal_a["id"]).project_id == p_other["id"]
    assert project_mod.resolve_project(
        store, None, goal_a["id"]).project_id == p_goal["id"]
    assert project_mod.resolve_project(store, None, None) is None
    assert project_mod.resolve_project(store, "nope", goal_a["id"]) is None


def test_agent_context_mentions_active_project():
    from teamroom import projects as project_mod
    store = _store()
    project = store.create_project("nvidia-quotes-api")
    dev = agents.member_by_handle(store, "dev")
    ws = project_mod.ProjectWorkspace(store, project["id"], project["name"])
    ctx = orchestrator.build_agent_context(store, dev, None, ws)
    assert "nvidia-quotes-api" in ctx and "write_file" in ctx, ctx
    assert "Project:" not in orchestrator.build_agent_context(store, dev, None)


def test_run_turn_with_project_active_completes():
    """A turn with a project active runs end to end (mock model). The
    mock never emits tool calls, so no file lands from the model —
    file landing is proven at the tool level above — but the turn
    runs with the project active and nothing crashes."""
    store = _store()
    store.create_room("Team Room", room_id="main")
    project = store.create_project("nvidia-quotes-api")
    orchestrator.run_turn(store, "main", "build a fizzbuzz script and test it",
                          {"configured": True, "model": "mock", "mock": True},
                          project_id=project["id"])
    msgs = store.messages("main")
    assert any(m["author_type"] == "agent" for m in msgs), msgs
    assert any(m["handle"] == "leader" for m in msgs), msgs
    assert store.get_project(project["id"])["file_count"] == 0


def test_github_missing_token_error():
    from teamroom import github
    try:
        github.create_repo_and_push(None, "demo", "", [])
    except github.GitHubError as exc:
        assert "Add a repo-scoped GitHub token as GITHUB_TOKEN in Vercel" in str(exc), exc
    else:
        raise AssertionError("expected GitHubError for a missing token")
    assert github.sanitize_repo_name("My Cool Project!") == "my-cool-project"
    assert github.sanitize_repo_name("  ") == "teamroom-project"
    assert github.sanitize_repo_name("UPPER_case.name") == "upper-case.name"


def test_github_endpoint_missing_token_is_clean_400():
    """The POST /api/projects/{id}/github endpoint without a
    GITHUB_TOKEN returns a clean 400 with the setup message — not a
    500, not a traceback. The server is imported with TEAMROOM_HOME
    pointed at a temp dir so the test never touches the real store."""
    from fastapi import HTTPException
    tmp_home = tempfile.mkdtemp()
    old_home = os.environ.get("TEAMROOM_HOME")
    old_token = os.environ.get("GITHUB_TOKEN")
    os.environ["TEAMROOM_HOME"] = tmp_home
    os.environ.pop("GITHUB_TOKEN", None)
    try:
        import teamroom.server as server
        project = server.store.create_project("demo")
        try:
            server.create_github_repo(project["id"], {})
        except HTTPException as exc:
            assert exc.status_code == 400, exc.status_code
            assert "GITHUB_TOKEN" in str(exc.detail), exc.detail
            assert "Add a repo-scoped GitHub token" in str(exc.detail), exc.detail
        else:
            raise AssertionError("expected HTTPException for a missing token")
        try:
            server.create_github_repo("nope", {})
        except HTTPException as exc:
            assert exc.status_code == 404, exc.status_code
        else:
            raise AssertionError("expected HTTPException 404")
    finally:
        if old_home is None:
            os.environ.pop("TEAMROOM_HOME", None)
        else:
            os.environ["TEAMROOM_HOME"] = old_home
        if old_token is not None:
            os.environ["GITHUB_TOKEN"] = old_token


def test_store_retries_transient_select():
    """A transient Hrana error on a read is retried; a non-transient
    error raises at once; a persistently transient read raises after
    3 attempts."""
    import threading
    from teamroom.store import _LockedConnection

    class _FakeCursor:
        def fetchone(self):
            return ("x",)
        def fetchall(self):
            return [("x",)]

    class _FlakyConn:
        def __init__(self, failures):
            self.failures = failures
            self.calls = 0
        def execute(self, sql, params=()):
            self.calls += 1
            if self.calls <= self.failures:
                raise ValueError("Hrana: `http error: `connection closed "
                                 "before message completed``")
            return _FakeCursor()

    flaky = _FlakyConn(failures=1)
    locked = _LockedConnection(flaky, threading.RLock())
    assert locked.execute("SELECT 1").fetchone() == ("x",)
    assert flaky.calls == 2, flaky.calls

    class _BadConn:
        def __init__(self):
            self.calls = 0
        def execute(self, sql, params=()):
            self.calls += 1
            raise ValueError("UNIQUE constraint failed: members.handle")
    bad = _BadConn()
    locked = _LockedConnection(bad, threading.RLock())
    try:
        locked.execute("SELECT 1")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    assert bad.calls == 1, bad.calls

    flaky3 = _FlakyConn(failures=10)
    locked = _LockedConnection(flaky3, threading.RLock())
    try:
        locked.execute("SELECT 1")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError after retries")
    assert flaky3.calls == 3, flaky3.calls


def test_turn_deadline_stops_before_work():
    """A deadline already past when the turn starts: no plan, no
    specialist work, no parallel fallback — just the timeout note."""
    import time as _time
    store = _store()
    store.create_room("Team Room", room_id="main")
    orchestrator.run_turn(store, "main", "build a fizzbuzz script and test it",
                          {"configured": True, "model": "mock", "mock": True},
                          deadline=_time.monotonic() - 1)
    msgs = store.messages("main")
    assert not any(m["author_type"] == "agent" for m in msgs), msgs
    assert not any("Routing to" in (m["content"] or "") for m in msgs), msgs
    assert not any("Leader's plan" in (m["content"] or "") for m in msgs), msgs
    assert any("time limit" in (m["content"] or "") for m in msgs), msgs


def test_member_run_past_deadline_returns_timeout_note():
    """A member run past its deadline makes no provider call and
    returns a short timeout note instead of an answer."""
    import time as _time
    store = _store()
    dev = agents.member_by_handle(store, "dev")
    out = agents.run_member(dev, "build a fizzbuzz script", "",
                            deadline=_time.monotonic() - 1)
    assert "ran out of time" in out, out


def test_record_outcome_failure_does_not_crash_turn():
    """A failed non-critical write (memory/XP) must not fail the turn:
    the agents' replies are still posted and the turn completes."""
    store = _store()
    store.create_room("Team Room", room_id="main")

    def _boom(handle):
        raise ValueError("simulated store failure")
    store.record_turn = _boom
    store.add_xp = _boom
    orchestrator.run_turn(store, "main", "build a fizzbuzz script and test it",
                          {"configured": True, "model": "mock", "mock": True})
    msgs = store.messages("main")
    assert any(m["author_type"] == "agent" for m in msgs), msgs
    assert any(m["handle"] == "leader" for m in msgs), msgs


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all teamroom tests passed")
