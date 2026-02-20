"""
test_context_manager.py — Unit tests for ContextManager._discover_files().

Run:  python -m pytest test_context_manager.py -v
  or: python test_context_manager.py
"""

import sys
import os
import tempfile
import shutil
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# We only need Config + ContextManager, but the module also tries to import
# openai at module level.  Provide a lightweight stub so the import succeeds
# even when openai is not installed.
if "openai" not in sys.modules:
    import types
    sys.modules["openai"] = types.ModuleType("openai")

from rain_lab_meeting_chat_version import Config, ContextManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _create_file(directory: Path, name: str, content: str = "test") -> Path:
    """Create a small text file and return its Path."""
    fp = directory / name
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8")
    return fp


class TestDiscoverFiles(unittest.TestCase):
    """Tests for ContextManager._discover_files()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="rain_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ---- 1. Only .md / .txt files are included ---------------------------
    def test_only_md_and_txt_included(self):
        _create_file(self.tmpdir, "paper.md")
        _create_file(self.tmpdir, "notes.txt")
        _create_file(self.tmpdir, "image.png")
        _create_file(self.tmpdir, "script.py")

        config = Config(library_path=str(self.tmpdir), recursive_library_scan=False)
        cm = ContextManager(config)
        files = cm._discover_files()
        names = {f.name for f in files}

        self.assertEqual(names, {"paper.md", "notes.txt"})

    # ---- 2. SOUL / LOG / MEETING files are excluded ----------------------
    def test_soul_log_meeting_excluded(self):
        _create_file(self.tmpdir, "research.md")
        _create_file(self.tmpdir, "JAMES_SOUL.md")
        _create_file(self.tmpdir, "RAIN_LAB_MEETING_LOG.md")
        _create_file(self.tmpdir, "meeting_notes.txt")

        config = Config(library_path=str(self.tmpdir), recursive_library_scan=False)
        cm = ContextManager(config)
        files = cm._discover_files()
        names = {f.name for f in files}

        self.assertEqual(names, {"research.md"})

    # ---- 3. Recursive scanning picks up nested files ---------------------
    def test_recursive_scan(self):
        _create_file(self.tmpdir, "top_level.md")
        _create_file(self.tmpdir / "subdir", "nested.md")
        _create_file(self.tmpdir / "subdir" / "deep", "deep.txt")

        config = Config(library_path=str(self.tmpdir), recursive_library_scan=True)
        cm = ContextManager(config)
        files = cm._discover_files()
        names = {f.name for f in files}

        self.assertEqual(names, {"top_level.md", "nested.md", "deep.txt"})

    # ---- 4. Non-recursive scan ignores nested files ----------------------
    def test_non_recursive_scan(self):
        _create_file(self.tmpdir, "top_level.md")
        _create_file(self.tmpdir / "subdir", "nested.md")

        config = Config(library_path=str(self.tmpdir), recursive_library_scan=False)
        cm = ContextManager(config)
        files = cm._discover_files()
        names = {f.name for f in files}

        self.assertEqual(names, {"top_level.md"})

    # ---- 5. max_library_files cap is honoured ----------------------------
    def test_max_library_files_cap(self):
        for i in range(10):
            _create_file(self.tmpdir, f"paper_{i:02d}.md")

        config = Config(
            library_path=str(self.tmpdir),
            recursive_library_scan=False,
            max_library_files=3,
        )
        cm = ContextManager(config)
        files = cm._discover_files()

        self.assertEqual(len(files), 3)

    # ---- 6. Skipped directories (__pycache__, .git, etc.) ----------------
    def test_skip_dirs_in_recursive_mode(self):
        _create_file(self.tmpdir, "good.md")
        _create_file(self.tmpdir / "__pycache__", "cached.md")
        _create_file(self.tmpdir / ".git", "internal.txt")
        _create_file(self.tmpdir / "venv", "dep.md")

        config = Config(library_path=str(self.tmpdir), recursive_library_scan=True)
        cm = ContextManager(config)
        files = cm._discover_files()
        names = {f.name for f in files}

        self.assertEqual(names, {"good.md"})

    # ---- 7. Empty directory returns empty list ---------------------------
    def test_empty_directory(self):
        config = Config(library_path=str(self.tmpdir), recursive_library_scan=False)
        cm = ContextManager(config)
        files = cm._discover_files()

        self.assertEqual(files, [])

    # ---- 8. Case-insensitive SOUL/LOG exclusion --------------------------
    def test_case_insensitive_exclusion(self):
        _create_file(self.tmpdir, "luca_soul.md")
        _create_file(self.tmpdir, "debug_log.txt")
        _create_file(self.tmpdir, "SoUl_file.md")
        _create_file(self.tmpdir, "actual_research.md")

        config = Config(library_path=str(self.tmpdir), recursive_library_scan=False)
        cm = ContextManager(config)
        files = cm._discover_files()
        names = {f.name for f in files}

        self.assertEqual(names, {"actual_research.md"})


if __name__ == "__main__":
    unittest.main()
