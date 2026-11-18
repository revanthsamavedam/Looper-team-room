# Looper Team Room

A web chat room where a team of looper specialists works on one
instruction together. A Leader agent (@leader, 🧭) plans the work,
delegates it to the right specialists in order — research before
building, building before testing — and integrates their results into
one final answer. The plan, the specialists' work, and the integrated
result are all posted to the room. The user can reply to redirect, or
`@`-mention one agent (`@qa`, `@dev`, `@dana`) to have the Leader
delegate only to them. If the Leader cannot make a plan the room falls
back to the original path: route by @-mention/keyword, run the
selected specialists in parallel, plus a short synthesis.

## Start

```bash
./run-teamroom.sh            # http://127.0.0.1:8765/ (local only)
# or: .venv-teamroom/bin/python -m teamroom --port 8765
```

First run creates `.venv-teamroom` and installs `fastapi` + `uvicorn`
there; the looper itself stays stdlib-only. Data lives in
`~/.looper-teamroom/teamroom.db` (or `TEAMROOM_HOME`).

## Model

The room uses the looper's provider resolution (`LOOPER_MODEL`,
`LOOPER_API_KEY` / `OPENAI_API_KEY`, `ollama/...` for a local model).
If no model is configured the server still starts, the UI loads, the
roster renders — and posting an instruction gets one plain message
saying no model is configured. Nothing is faked.

`TEAMROOM_MOCK_MODEL=1` swaps in a scripted client whose replies are
labelled `[mock model]` — verification only, never a live answer.

## Bring your own key (BYOK)

Anyone using the room can paste their own OpenAI key in the sidebar
("Your OpenAI key") and pick a model (default `gpt-4o-mini`). The key
lives in that browser's `localStorage` only and is sent as an
`X-OpenAI-Key` / `X-OpenAI-Model` header with each instruction. The
server builds a provider from it in memory for that one turn and
never writes it to the DB, logs, or disk. No key is ever needed on
the server itself. A wrong key fails honestly: OpenAI's 401 is posted
to the room as the agent's result.

## Gamification

Visible in the room, not a separate page. XP is earned only from
real completed work — never invented:

| event | XP |
| --- | --- |
| member posts a reply in a finished turn | +10 |
| reply states a `Decision:` line | +5 |
| reply states a `Learned:` line | +5 |
| reply states a `Best practice:` line | +15 |
| reply states a `Bug:` line (usually QA) | +15 |
| member is on a goal when it is marked done | +25 |

Levels: 1 + (XP // 100) — level 2 at 100 XP, level 3 at 200, and so
on. The roster shows each member's level, XP and progress to the
next level; the leaderboard ranks everyone by XP.

Badges (awarded once each): First Steps (first turn), Bug Hunter
(first `Bug:` line), Decider (first `Decision:`), Sharer (first
`Best practice:`), Workhorse (10 turns), Goal Getter (a goal
completed), Centurion (100 XP), Team Player (work in 2 teams).

Goal progress: each completed agent reply on a goal counts as one
task done. The goal's target (default 5, set when creating the goal)
is the denominator — the bar shows tasks done / target.

State lives in the same SQLite store (`member_xp`, `member_badges`,
`target_tasks` on goals), so it survives local restarts. On Vercel
it is ephemeral like everything else in `/tmp` — see the deploy
limits below.

## Deploying to Vercel

```bash
vercel --prod          # from this directory (project: looper-team-room)
```

`api/index.py` serves the same FastAPI app; `vercel.json` sets
`TEAMROOM_HOME=/tmp/looper-teamroom` and a 60s function timeout.
The local run (`./run-teamroom.sh`) is unchanged.

**State on Vercel is persistent when Turso is configured.** Set
`TURSO_URL` (a `libsql://` URL) and `TURSO_AUTH_TOKEN` as production
env vars on the project (`vercel env add TURSO_URL production`, same
for the token) and the store talks to Turso instead of the local file
— teams, goals, members, memory and XP survive restarts and cold
starts. Without those env vars the store falls back to a SQLite file
in `/tmp`, which is ephemeral per instance (a cold start or a
different instance may show an empty room). The Turso token is a
secret: it lives only in the Vercel env vars, never in the repo.

**Mobile.** The UI is responsive: on a narrow screen the sidebar
becomes a slide-in drawer (☰ Team button in the header, tap the
backdrop to close) and the composer wraps (goal picker on its own
row, then input + Send). Inputs are 16px so iOS does not auto-zoom,
tap targets are 44px. Verified at a 390px viewport with headless
Chromium (see `/tmp/mobile-test/`, not part of the repo).
- **Turns are capped at ~45s.** The message POST streams the turn
  (serverless functions freeze after the response, so a background
  thread would never finish it). One deadline (`TEAMROOM_TURN_DEADLINE_S`,
  default 45) is shared by the stream and the work: the Leader,
  each specialist and the integrate step stop starting new provider
  calls once it is past, and the stream ends with `turn_timeout` —
  partial results stay visible, nothing is faked. A single failed
  non-critical write (memory/XP) never fails the turn, and transient
  Turso hiccups are retried on reads (writes get one attempt, then a
  clear 503 instead of a 500). Long turns (several agents, many tool
  calls) will hit the cap; keep instructions focused or run them
  locally, where there is no cap. The persistent `/stream` feed ends
  cleanly after 45s and the browser reconnects, so an open room no
  longer logs a 60s hosting timeout every minute.

## The team

Built-ins (each a `Looper` with its own system prompt and the looper's
tools — files, terminal, todo, memory, session search, delegate):

| handle | member | specialty |
| --- | --- | --- |
| `@dev` | Dev 💻 | Writes and fixes code |
| `@qa` | QA 🧪 | Tests, finds bugs, verifies fixes |
| `@researcher` | Researcher 🔍 | Finds facts, compares options |
| `@writer` | Writer ✍️ | Writes docs, drafts and summaries |
| `@reviewer` | Reviewer 👀 | Reviews work for quality, risk and gaps |

## Custom members via skill file

Add a member in the UI (sidebar) — upload a `.md` file or paste the
text. The file defines the member, SKILL.md-style:

```markdown
---
name: Dana
specialty: Data analysis — turns raw numbers into clear findings
---
# Dana — Data Analyst

## System prompt
You are Dana, the Data Analyst in a team chat room. ...
```

Parsing is tolerant (`name:` / `#` heading, `## System prompt` /
`## Instructions` / `## Prompt`), but validation is strict: a file
that yields no name or no system prompt is rejected with a plain
error, no defaults invented. Custom members persist in SQLite, appear
in the roster with a "custom" badge, and are `@`-mentionable exactly
like the built-ins. Keyword routing also matches them on words from
their specialty and name. A sample is in `examples/dana-skill.md`.

## Teams and goals

- **Teams** (`POST /api/teams`, sidebar): a named group of members —
  built-ins, custom members, or both. Add/remove members any time.
- **Goals** (`POST /api/goals`, sidebar): a shared objective that
  belongs to one team. When creating a goal you pick which of the
  team's members work on it; members must be drawn from that team
  (anything else is rejected with a plain error). Any number of goals
  can exist at once.
- **Working on a goal**: pick it in the goal dropdown above the chat
  input. The room then routes only among that goal's members, tags
  every message in the turn with the goal (🎯 chip), and dims the
  roster members who are not on it. `@`-mentioning someone off-goal
  is reported in the room instead of silently running them.

## Projects: real files, real repos

A project is a real workspace for a goal's work. Its files live in
the same store as everything else (Turso on Vercel, SQLite locally),
so they survive restarts and cold starts — unlike anything an agent
writes to the ephemeral Vercel filesystem, which vanishes after the
turn.

- **Create one** in the sidebar (Projects) or `POST /api/projects`
  (`{"name", "description", "goal_id"}` — linking it to a goal makes
  it the default project for that goal's turns). Pick it in the chat
  bar's project dropdown, or let the goal pick it for you.
- **With a project active**, the Leader's plan includes a step that
  writes the deliverable files (e.g. `main.py`, `requirements.txt`,
  `README.md`), and the specialists' file tools (`list_files`,
  `read_file`, `write_file`) operate on the project's files in the
  store instead of the local disk. Code only pasted in chat does not
  persist; files written with `write_file` do. The `terminal` tool is
  deliberately not project-scoped — project files live in the store,
  not on disk, so there is nothing for a shell command to run against
  on Vercel.
- **View / download**: the Projects panel lists each project's files
  (click one to read it) and offers a `.zip` download
  (`GET /api/projects/{id}/zip`).
- **Push to GitHub**: each project has a "Create GitHub repo" button
  (`POST /api/projects/{id}/github`). It creates a repo on your
  GitHub account and pushes the project's files via the plain GitHub
  REST API — no MCP involved. The repo is **private by default**
  (pass `{"private": false}` to make it public).

### GitHub setup (one-time)

The repo-creation action needs a token the server can use. It is
read from the environment for that one call only and never stored.

1. Create a token at github.com → Settings → Developer settings →
   Personal access tokens → Tokens (classic) → Generate new token.
   Give it the **`repo` scope** (full control of private
   repositories). A fine-grained token works too, with repository
   Administration (read and write) + Contents (read and write).
2. Add it as a production env var on the Vercel project:
   `vercel env add GITHUB_TOKEN production` (paste the token), then
   redeploy so the serverless function picks it up.
3. Without it, the button (and the endpoint) returns a plain error:
   "Add a repo-scoped GitHub token as GITHUB_TOKEN in Vercel" — the
   room itself works fine without it; only repo creation is blocked.

## Memory: team-scoped, best practices: shared

Two separate layers — do not confuse them:

- **Member memory** is scoped to `(team, member)`. After every turn,
  what a member did is recorded automatically; `Decision:` and
  `Learned:` lines in an agent's reply are recorded as decided /
  learned memory for that team. Before a member's next run in the
  same team, its last entries are fed back as context. Memory from
  one team is never fed to the same member in another team.
  Inspect it at `GET /api/teams/{id}/memory?handle=dana`.
- **Best practices** are shared across all teams: one table, visible
  everywhere. A `Best practice:` line in an agent's reply lands here
  (identical lines recorded once), and the list is fed to every agent
  run as context. Add your own in the sidebar or
  `POST /api/best-practices`; list at `GET /api/best-practices`.

## How routing works

1. `@`-mentions win — only the mentioned members run.
2. Otherwise keyword match — members whose trigger words appear run.
3. Otherwise the Researcher answers alone as the generalist.
4. Unknown `@`-handles are reported in the room, not ignored silently.

One instruction at a time per server: a second message while the room
is working gets a plain 409. The room log is SQLite, so history
survives restarts.

## Live web and the browser

The Researcher can read live pages two ways:

- `web_fetch` — plain HTTP GET (stdlib only), works everywhere including
  Vercel. Use it for readable pages.
- The browser tools — `browser_navigate`, `browser_read`, `browser_click`,
  `browser_type`, `browser_screenshot` — drive a real Chromium via
  Playwright, for pages that need JavaScript or interaction. Each call
  has a 15s timeout so a slow page cannot eat the 45s turn deadline.

The browser needs a host with Chromium — a persistent VM, not Vercel
serverless. The tools register only when Playwright is importable, so
on Vercel they are hidden and `web_fetch` is the only live-web tool;
on a VM with Chromium installed they appear. If Chromium cannot
launch, each call returns a clear error instead of crashing the turn.
On a VM: `pip install playwright && python -m playwright install
chromium` (deploy-teamroom.sh does this in the room's venv).

Screenshots are how you see what the browser sees: `browser_screenshot`
saves a PNG and returns its URL, the agent's reply includes the URL,
and the room renders it as an image in the message. Screenshots are
served at `GET /api/screenshots/{name}`. `/api/status` reports whether
the browser is available on the current host.

## API

- `GET /` — the chat UI
- `GET /api/status` — model status, browser availability, counts
- `GET /api/agents` — roster (built-ins + customs)
- `GET /api/rooms`, `POST /api/rooms`
- `GET /api/rooms/{id}/messages?after=N`
- `POST /api/rooms/{id}/messages` — `{content}`; runs the turn in the background
- `GET /api/rooms/{id}/stream?after=N` — SSE message stream
- `POST /api/members` — multipart `file` or form `text` (skill file)
- `DELETE /api/members/{handle}` — custom members only
- `GET /api/teams`, `POST /api/teams`, `GET /api/teams/{id}`
- `POST /api/teams/{id}/members`, `DELETE /api/teams/{id}/members/{handle}`
- `GET /api/goals`, `POST /api/goals`, `GET /api/goals/{id}`,
  `PATCH /api/goals/{id}` (`{"status": "done"}`), `DELETE /api/goals/{id}`
- `GET /api/projects`, `POST /api/projects`, `GET /api/projects/{id}`
- `GET /api/projects/{id}/files`, `GET /api/projects/{id}/files/{path}`,
  `PUT /api/projects/{id}/files/{path}` (`{"content": "..."}`)
- `GET /api/projects/{id}/zip` — download the project as a `.zip`
- `POST /api/projects/{id}/github` — create a GitHub repo and push the
  files (needs `GITHUB_TOKEN`; private by default, `{"private": false}`
  for public)
- `GET /api/teams/{id}/memory?handle=`, `POST /api/teams/{id}/memory`
- `GET /api/best-practices`, `POST /api/best-practices`,
  `DELETE /api/best-practices/{id}`
- `GET /api/screenshots/{name}` — a browser screenshot PNG

## Tests

```bash
python tests/test_teamroom.py   # no key, no network, no server
python tests/test_browser_tools.py  # web_fetch + browser tools (fake Playwright)
python tests/test_looper.py     # the looper's own 6 tests, untouched
```
