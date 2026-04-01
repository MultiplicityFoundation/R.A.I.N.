# James and the R.A.I.N. Lab

**The local-first autonomous coding agent runtime for Rust, Python, and hardware-adjacent teams.**

<p align="center">
  <img src="assets/rain_lab.png" alt="R.A.I.N. Lab logo" width="800" />
</p>

<p align="center">🌐
  <a href="README.zh-CN.md">简体中文</a> •
  <a href="README.ja.md">日本語</a> •
  <a href="README.ru.md">Русский</a> •
  <a href="README.fr.md">Français</a> •
  <a href="README.vi.md">Tiếng Việt</a>
</p>

---

## What This Is

`james_library` is a local-first runtime for autonomous coding agents that need to operate inside real engineering environments, not toy prompt loops.

It is built for work that crosses:
- Rust services and tooling
- Python automation and orchestration
- firmware, peripherals, and hardware control paths

Most coding agents look good in short demos and break under real conditions. They lose context in long loops, miss code relationships, use tools too loosely, and can silently burn money while running unattended.

R.A.I.N. Lab is the opposite bet. Give James a real engineering task and the system is designed to keep context, understand the codebase, stay within tool boundaries, and make the loop auditable.

All on your own computer. Private by default.

---

## What You Can Do

| Use case | What happens |
|----------|--------------|
| **Understand a codebase faster** | LSP-aware agents can inspect symbols, definitions, references, and dependency context |
| **Run longer autonomous tasks** | Context compaction keeps active history useful instead of letting the loop bloat |
| **Keep spending under control** | Budget tracking and circuit breakers stop runaway autonomous sessions |
| **Coordinate specialists** | Multi-agent orchestration lets role-specific workers attack the same task from different angles |
| **Work locally with private code** | Tool execution, memory, and orchestration stay on your machine |

**The result:** better coding loops, fewer blind spots, and a system you can leave running without trusting it blindly.

---

## Try It Now

No setup required for the demo:

```
python rain_lab.py
```

Press Enter. James will walk you through the rest.

On Windows, you can also double-click `INSTALL_RAIN.cmd` to create shortcuts.
On macOS/Linux, run `./install.sh` for a one-click setup.

---

## Getting Started

### Step 1: Try the Demo

```bash
python rain_lab.py
```

Press Enter for instant demo mode. No model, no config — just see how it works.

### Step 2: Run Your First Task

```bash
python rain_lab.py --mode beginner --topic "inspect src/agent/history.rs and explain how long-loop memory is handled"
```

This opens a guided flow where James helps you work through one task end to end.

### Step 3: Set Up Your AI (Optional)

Want to use your own AI model (runs locally, stays private)?

```bash
python rain_lab.py --mode first-run
```

The installer helps you connect to LM Studio or Ollama. Both run on your machine — no data leaves your computer.

---

## Why It's Different

| Regular AI chat | R.A.I.N. Lab |
|-----------------|--------------|
| One answer, right or wrong | Agents can coordinate, challenge, and verify |
| Weak code awareness | LSP-backed code intelligence and dependency context |
| Context degrades over long tasks | Token-aware compaction keeps loops usable |
| Costs are easy to ignore until too late | Real-time cost tracking and hard budget limits |
| Everything in the cloud | Everything stays on your machine |

---

## Features

- **LSP-aware code intelligence** — definitions, references, document symbols, and dependency-prefetch context
- **Context management for long loops** — summarize-then-prune history compaction with exact preservation for dangerous and hardware-critical outputs
- **Hard cost controls** — per-session spend tracking, budget enforcement, and human-in-the-loop reset prompts
- **Multi-agent orchestration** — James plus specialist workers for deeper task decomposition
- **Persistent memory** — session state and resumable histories
- **Local-first execution** — private code stays local, tools stay inspectable
- **Developer-focused runtime** — Python + Rust stack with room to extend the agent surface
- **Available in 6 languages** — 中文, 日本語, Русский, Français, Tiếng Việt, English

---

## Requirements

- **Python 3.10+** (free download for Windows/Mac/Linux)
- **Optional:** LM Studio or Ollama for local AI models
- **Optional:** Rust toolchain for the fast runtime layer

Python works without any of the optional parts. The more you add, the faster and more powerful it gets.

---

## Documentation

- [Start Here](START_HERE.md) — Guided walkthrough
- [Beginner Guide](docs/getting-started/README.md)
- [One-Click Install](docs/one-click-bootstrap.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Research Papers](https://topherchris420.github.io/research/)

---

## For Developers

<details>
<summary>Click to expand</summary>

R.A.I.N. Lab is built in Python + Rust. If you want to hack on the runtime:

```bash
git clone https://github.com/topherchris420/james_library.git
cd james_library

# Python setup
uv python install 3.12
uv venv .venv --python 3.12
uv pip sync --python .venv/bin/python requirements-dev-pinned.txt

# Rust setup (optional, for the fast runtime)
cargo build --release --locked

# Run
uv run --python .venv/bin/python rain_lab.py --mode first-run
```

Recommended mental model:
- Python handles launcher flows and orchestration
- Rust handles the fast runtime, tool surface, and lower-level infrastructure
- The product wedge is a local-first autonomous coding agent for Rust, Python, and hardware-adjacent workflows

**Testing:**
```bash
ruff check .
pytest -q
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

See [ARCHITECTURE.md](ARCHITECTURE.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for details.

</details>

---

## License

MIT. Built by [Vers3Dynamics](https://vers3dynamics.com/), special thanks to ZeroClaw

<a href="https://star-history.com/#topherchris420/james_library&type=date">
  <img src="https://api.star-history.com/image?repos=topherchris420/james_library&type=date&theme=dark" alt="Star History" width="200" />
</a>
