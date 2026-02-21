"""
R.A.I.N. Lab - Full System Diagnostic
Checks all components needed for rain_lab to work.
"""
import os
import sys
import time
import importlib
from pathlib import Path

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
WARN = "\033[93m⚠️  WARN\033[0m"
INFO = "\033[96mℹ️  INFO\033[0m"

results = []

def check(name, passed, detail="", warn=False):
    status = PASS if passed else (WARN if warn else FAIL)
    results.append((name, passed, warn))
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")

def section(title):
    print(f"\n\033[1m{'='*60}\033[0m")
    print(f"\033[1m  {title}\033[0m")
    print(f"\033[1m{'='*60}\033[0m")

# ── 1. PYTHON ENVIRONMENT ──
section("1. PYTHON ENVIRONMENT")
check("Python version", sys.version_info >= (3, 8), f"Python {sys.version.split()[0]}")
check("Platform", True, sys.platform)

# ── 2. REQUIRED PACKAGES ──
section("2. REQUIRED PACKAGES")
required = ["openai", "httpx", "dataclasses", "argparse"]
for pkg in required:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "ok")
        check(f"import {pkg}", True, f"v{ver}" if ver != "ok" else "")
    except ImportError as e:
        check(f"import {pkg}", False, str(e))

# ── 3. OPTIONAL PACKAGES ──
section("3. OPTIONAL PACKAGES (not required)")
optional = {
    "ddgs": "DuckDuckGo web search",
    "pyttsx3": "Text-to-speech voices",
    "msvcrt": "Windows keyboard input",
}
for pkg, desc in optional.items():
    try:
        importlib.import_module(pkg)
        check(f"{pkg} ({desc})", True, warn=False)
    except ImportError:
        check(f"{pkg} ({desc})", False, "Not installed — feature disabled", warn=True)

# ── 4. FILE STRUCTURE ──
section("4. FILE STRUCTURE")
lib_path = Path(__file__).resolve().parent
check("Library path exists", lib_path.exists(), str(lib_path))

scripts = [
    "rain_lab.py",
    "rain_lab_meeting.py",
    "rain_lab_meeting_chat_version.py",
]
for s in scripts:
    p = lib_path / s
    check(f"{s}", p.exists(), f"{p.stat().st_size:,} bytes" if p.exists() else "MISSING")

# ── 5. RESEARCH PAPERS ──
section("5. RESEARCH PAPERS / LIBRARY")
paper_exts = {".md", ".txt", ".pdf"}
papers = []
for f in lib_path.rglob("*"):
    if f.suffix.lower() in paper_exts and f.is_file() and "meeting_archives" not in str(f):
        # Skip known non-paper files
        if f.name.startswith("RAIN_LAB_MEETING_LOG"):
            continue
        if f.name in ("README.md", "diagnostic.py", "task.md", "implementation_plan.md", "walkthrough.md"):
            continue
        papers.append(f)

check(f"Papers/documents found", len(papers) > 0, f"{len(papers)} files")
if papers:
    # Show first 8 papers
    for p in papers[:8]:
        rel = p.relative_to(lib_path)
        size = p.stat().st_size
        print(f"         📄 {rel} ({size:,} bytes)")
    if len(papers) > 8:
        print(f"         ... and {len(papers) - 8} more")

# ── 6. SOUL FILES ──
section("6. AGENT SOUL FILES")
soul_names = ["james_soul.md", "jasmine_soul.md", "luca_soul.md", "elena_soul.md"]
for sn in soul_names:
    p = lib_path / sn
    if p.exists():
        check(sn, True, f"{p.stat().st_size:,} bytes")
    else:
        check(sn, False, "Missing — will use generated fallback", warn=True)

# ── 7. COMMUNICATION / DIPLOMAT ──
section("7. DIPLOMAT MAILBOX")
comm_dir = lib_path / "communication"
for sub in ["inbox", "outbox", "processed"]:
    d = comm_dir / sub
    check(f"communication/{sub}/", d.exists(), "exists" if d.exists() else "will be created on first run", warn=not d.exists())

# ── 8. LM STUDIO SERVER ──
section("8. LM STUDIO SERVER")
base_url = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
print(f"  {INFO}  Target: {base_url}")

# Test /v1/models endpoint
import urllib.request
import json

models_ok = False
loaded_models = []
try:
    req = urllib.request.Request(f"{base_url}/models", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
        loaded_models = [m["id"] for m in data.get("data", [])]
        models_ok = True
        check("Server responding", True, f"{len(loaded_models)} model(s) loaded")
        for m in loaded_models:
            print(f"         🤖 {m}")
except Exception as e:
    check("Server responding", False, str(e))

# Test actual inference
if models_ok and loaded_models:
    model = os.environ.get("LM_STUDIO_MODEL", "qwen2.5-coder-7b-instruct")
    if model not in loaded_models:
        model = loaded_models[0]
    print(f"\n  {INFO}  Testing inference with model: {model}")
    print(f"         Sending 'say hi' (max_tokens=5, timeout=15s)...")

    try:
        import openai as oai
        import httpx
        client = oai.OpenAI(
            base_url=base_url,
            api_key="lm-studio",
            timeout=httpx.Timeout(15.0),
        )
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "say hi"}],
            max_tokens=5,
        )
        elapsed = time.time() - t0
        content = resp.choices[0].message.content.strip()
        check("Inference working", True, f"Response: '{content}' ({elapsed:.1f}s)")
    except Exception as e:
        err = str(e)
        if "timed out" in err.lower() or "timeout" in err.lower():
            check("Inference working", False,
                  "Request TIMED OUT — model loaded but not generating. "
                  "Try: Reload Model in LM Studio, or restart the server.")
        else:
            check("Inference working", False, err[:200])

# ── 9. CONFIG VALIDATION ──
section("9. CONFIGURATION")
try:
    sys.path.insert(0, str(lib_path))
    # Quick import of Config
    import importlib.util
    spec = importlib.util.spec_from_file_location("chat_version", lib_path / "rain_lab_meeting_chat_version.py")
    mod = importlib.util.module_from_spec(spec)
    # Don't execute the full module, just read the defaults
    check("Default model", True, os.environ.get("LM_STUDIO_MODEL", "qwen2.5-coder-7b-instruct"))
    check("Default base_url", True, base_url)
    timeout_val = os.environ.get("RAIN_LM_TIMEOUT", "600")
    check("Default timeout", True, f"{timeout_val}s")
except Exception as e:
    check("Config parse", False, str(e), warn=True)

# ── SUMMARY ──
section("SUMMARY")
total = len(results)
passed = sum(1 for _, p, w in results if p)
warned = sum(1 for _, p, w in results if not p and w)
failed = sum(1 for _, p, w in results if not p and not w)

print(f"  Total checks: {total}")
print(f"  \033[92mPassed: {passed}\033[0m")
if warned:
    print(f"  \033[93mWarnings: {warned}\033[0m (optional features)")
if failed:
    print(f"  \033[91mFailed: {failed}\033[0m")
    print(f"\n  \033[91m⚡ Fix the FAIL items above before running rain_lab.\033[0m")
else:
    print(f"\n  \033[92m🚀 All critical checks passed! rain_lab should work.\033[0m")

print()
