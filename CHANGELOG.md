# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows semantic-style tags for local releases.

## [Unreleased]

### Added
- Reproducible dependency lock files: `requirements-pinned.txt` and `requirements-dev-pinned.txt`.
- Cross-platform local bootstrap script: `bootstrap_local.py`.
- Cleaner PowerShell installer wrapper: `INSTALL_RAIN.ps1`.
- Operational documentation:
  - `docs/TROUBLESHOOTING.md`
  - `docs/BACKUP_RESTORE.md`
  - `RELEASE_CHECKLIST.md`

### Changed
- CI workflows now prefer pinned dependency files when available.
- README now includes reproducible setup instructions and operational docs links.

## [2026.02.22] - 2026-02-22

### Added
- Runtime reliability hardening, strict grounding controls, runtime healthcheck.
- Launcher `preflight` and `backup` modes.
- Workspace-safe trace path and backup path defaults.

### Changed
- Windows-safe launcher banner/spinner rendering.
- README polish and project logo integration.
