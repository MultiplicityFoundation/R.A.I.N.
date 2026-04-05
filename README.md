# R.A.I.N. Lab

**A private-by-default expert panel in a box for researchers, independent thinkers, and R&D teams.**

Ask a raw research question. The R.A.I.N. Lab assembles multiple expert perspectives, grounds strong claims in papers or explicit evidence, and returns the strongest explanations, disagreements, and next moves.

Most tools help you find papers. R.A.I.N. Lab helps you think with a room full of experts.

James is the assistant inside the R.A.I.N. Lab.

<p align="center">
  <img alt="R.A.I.N. Lab logo" src="assets/rain_lab.png" class="hero" />
</p>

<p align="center">🌐
  <a href="README.zh-CN.md">简体中文</a> •
  <a href="README.ja.md">日本語</a> •
  <a href="README.ru.md">Русский</a> •
  <a href="README.fr.md">Français</a> •
  <a href="README.vi.md">Tiếng Việt</a>
</p>

---

## What It Does

The R.A.I.N. Lab turns one question into a structured research conversation.

- It frames the problem from multiple expert angles.
- It separates strong evidence from weak speculation.
- It shows agreements, disagreements, and open questions instead of forcing false certainty.
- It helps you decide what to read next, test next, or ask next.

This is built for work that starts messy: early-stage research, technical due diligence, strategy exploration, independent investigation, and R&D planning.

---

## Try It Now

### Public Web Experience

Start with the example hosted experience:

- [Here](https://rainlabteam.vercel.app/)

If you want the fastest way to feel the product, start there first.

### Local and Private

Run the local experience on your own machine:

```bash
python rain_lab.py
```

Press Enter for demo mode, or continue into guided setup.

On Windows, you can also double-click `INSTALL_RAIN.cmd` to create shortcuts.
On macOS/Linux, run `./install.sh` for a one-click setup.

---

## Who It Is For

- Researchers working through ambiguous questions
- Independent thinkers building evidence-backed views
- R&D teams comparing explanations, risks, and next moves
- Technical operators who want private workflows and inspectable reasoning

---

## What You Can Do

| Use case | What R.A.I.N. Lab helps you do |
|----------|--------------------------------|
| **Pressure-test a research claim** | Compare competing explanations and inspect where the evidence is thin |
| **Map a new topic fast** | Turn a vague question into viewpoints, sources, disagreements, and next steps |
| **Prepare decisions** | Surface trade-offs, unresolved risks, and what would change the conclusion |
| **Stay private** | Keep your local workflow and model setup on your own machine when needed |

---

## Why It Is Different

Most research tools optimize for retrieval. Most agent frameworks stop at "LLM plus a tool or two." R.A.I.N. Lab is designed for synthesis, challenge, judgment, and real-world action.

| Typical tool / agent framework | R.A.I.N. Lab |
|---------------------------------|--------------|
| Returns a list of papers or links | Returns competing interpretations and strongest next moves |
| Treats the first plausible answer as good enough | Preserves disagreements and uncertainty where it matters |
| Hides reasoning behind one-shot summaries | Makes evidence, gaps, and confidence easier to inspect |
| Assumes cloud-first workflows | Supports local and private usage paths |
| Talks to one model, on one platform | 10+ model providers, 25+ messaging channels, 60+ tools |
| Software only | Controls hardware — Arduino, STM32, Raspberry Pi |
| Generic agent loop | Neuroscience models, knowledge graphs, semantic memory, multi-agent swarms |

---

## Local and Private Workflow

If you want the product to run from your machine with your own setup:

1. Launch the app:

```bash
python rain_lab.py
```

2. For guided setup, run:

```bash
python rain_lab.py --mode first-run
```

3. For a first structured prompt, try:

```bash
python rain_lab.py --mode beginner --topic "compare the strongest arguments for and against a local-first research workflow"
```

The guided flow can connect to LM Studio or Ollama so your model traffic stays local.

---

## See It In Action

Ask a research question. Watch four expert agents — James (lead scientist), Jasmine (post-doc), Luca (geometer), and Elena (logician) — debate it in real time.

```
TOPIC: Could a "Phononic Morphogenetic Field" — precise acoustic interference patterns
guiding matter assembly like DNA guides cell growth — actually work?

**James:** ...phononic frequency combs could act like an acoustic blueprint for
molecular organization. The missing link between quantum coherence and biological
assembly?

**Jasmine:** Hold on. Cymatic patterns are *static* — they don't adapt to errors
or material changes the way DNA does. And the energy density needed exceeds
current acoustic levitation by *orders of magnitude*. Where's the thermal
dissipation analysis?

**Luca:** The geometry is compelling, though. Wavelength spacing in phononic
combs (ωₙ₊₁ - ωₙ = 2πc/λ) parallels scalar field gradients in relic field
tomography. But macroscopic assembly requires E > 10⁴⁵ J — far beyond reach.

**Elena:** The math is elegant, but the premise has a fatal flaw. The energy
density violates the Landauer limit by multiple orders of magnitude. Current
systems operate ~10³ times lower. Without experimental validation at that
scale, this remains speculation.

[Meeting continues — James responds, Jasmine pushes back, consensus forms...]
```

Join a research meeting, explore disagreements, and leave with next steps — not just links.

## Features

- Multi-perspective research synthesis
- Evidence-aware reasoning with explicit uncertainty
- Guided next steps for reading, testing, and follow-up questions
- Private local workflow options
- Available in 6 languages: 中文, 日本語, Русский, Français, Tiếng Việt, English

---

## What James Can Actually Do

Most multi-agent systems stop at "chat with an LLM." James is a full research operating system with real-world reach.

<details>
<summary><b>Talk to any major AI model</b></summary>

James connects to 10+ model providers out of the box — OpenAI, Anthropic, Google Gemini, Ollama (local), Azure, AWS Bedrock, OpenRouter, and more. Switch models mid-conversation or let James route automatically based on the question.

</details>

<details>
<summary><b>Reach you wherever you are</b></summary>

James can live inside Telegram, Discord, Slack, WhatsApp, email, Matrix, Signal, iMessage, IRC, Bluesky, Reddit, Twitter/X, Nostr, DingTalk, Lark/Feishu, WeChat/WeCom, and more. Your research assistant meets you where you already work.

</details>

<details>
<summary><b>Remember what matters</b></summary>

Built-in memory with multiple backends — local SQLite, Markdown files, Postgres, Qdrant vector search, and semantic recall. James remembers your preferences, past research, and evolving context across sessions.

</details>

<details>
<summary><b>Use real tools, not just words</b></summary>

60+ built-in tools: shell commands, file operations, git, web search, web browsing, HTTP APIs, PDF reading, screenshots, calculators, cron scheduling, Jira, Notion, Google Workspace, Microsoft 365, LinkedIn, and more. James does not just suggest actions — he takes them.

</details>

<details>
<summary><b>Connect to the physical world</b></summary>

Hardware peripheral support for Arduino, STM32 Nucleo boards, and Raspberry Pi GPIO. Flash firmware, read sensors, and control hardware directly from a research conversation. This is not typical for an AI assistant.

</details>

<details>
<summary><b>Think with your brain (literally)</b></summary>

New: TRIBE v2 integration predicts fMRI brain activation patterns from video, audio, or text using Facebook Research's brain-encoding model. Run neuroscience experiments from inside a chat. (CC-BY-NC 4.0, non-commercial.)

</details>

<details>
<summary><b>Coordinate teams of agents</b></summary>

Multi-agent delegation and swarm orchestration let James spin up specialized sub-agents for parallel research tracks, then synthesize their findings. Not a chatbot — a research team.

</details>

<details>
<summary><b>Stay secure by default</b></summary>

Deny-by-default security policy, sandboxed execution, encrypted secrets, domain allowlists, rate limiting, and audit logging. Built for people who care about what runs on their machine.

</details>

---

## Outcome Quality and Trust

### Outcome Quality (Benchmarked)

R.A.I.N. Lab tracks engineering quality in CI and publishes explicit metric definitions, baselines, and targets (for example: panic count, unwrap count, flaky test rate, and critical-path coverage).

- Quality metrics contract: [docs/project/quality-metrics.md](docs/project/quality-metrics.md)
- Quality report generator: [scripts/ci/quality_metrics_report.py](scripts/ci/quality_metrics_report.py)

For research-outcome benchmarking, we recommend publishing reproducible before/after evaluation artifacts (task set, baseline, rubric, and result files) alongside these quality reports.

### Trust + Privacy Story

R.A.I.N. Lab is designed local-first with secure defaults:

- local/private workflow paths and local model routing options
- gateway defaults to localhost with pairing enabled and public bind disabled
- deny-by-default allowlist posture for channel access
- encrypted-at-rest secret handling for high-value keys

Current-behavior security docs:

- [docs/security/README.md](docs/security/README.md)
- [docs/reference/api/config-reference.md](docs/reference/api/config-reference.md)

---

## Requirements

- **Python 3.10+**
- **Optional:** LM Studio or Ollama for local AI models
- **Optional:** ZeroClaw/Rust toolchain for the fast runtime layer

Python works without the optional pieces. Adding them expands the local/private path.

---

## Documentation

- [Start Here](START_HERE.md)
- [Beginner Guide](docs/getting-started/README.md)
- [One-Click Install](docs/one-click-bootstrap.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Research Papers](https://topherchris420.github.io/research/)

---

## For Developers

<details>
<summary>Click to expand</summary>

If you want to contribute to R.A.I.N. Lab or run the developer setup locally:

```bash
git clone https://github.com/topherchris420/james_library.git
cd james_library

# Python setup
uv python install 3.12
uv venv .venv --python 3.12
uv pip sync --python .venv/bin/python requirements-dev-pinned.txt

# Rust setup (optional, for the fast runtime layer)
cargo build --release --locked

# Run
uv run --python .venv/bin/python rain_lab.py --mode first-run
```

Recommended mental model:

- R.A.I.N. Lab is the experience.
- James is the assistant you interact with inside the lab.
- Python handles launcher flows and orchestration.
- ZeroClaw/Rust handles the fast runtime, tool surface, and lower-level infrastructure.

**Testing:**

```bash
ruff check .
pytest -q
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

See [ARCHITECTURE.md](ARCHITECTURE.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for contributor details.

</details>

---

## License

MIT. Built by [Vers3Dynamics](https://vers3dynamics.com/), special thanks to ZeroClaw.

<a href="https://star-history.com/#topherchris420/james_library&type=date">
  <img src="https://api.star-history.com/image?repos=topherchris420/james_library&type=date&theme=dark" alt="Star History" width="200" />
</a>
