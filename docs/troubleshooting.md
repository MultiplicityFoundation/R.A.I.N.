# R.A.I.N. Lab Troubleshooting

This lowercase path is the canonical runtime-contract entry for troubleshooting.

Last verified: **February 20, 2026**.

## Installation / Bootstrap

### `curl` or `uv` setup fails

Symptom:

- `install.sh` cannot download or locate `uv`

Fix:

```bash
curl --version
./install.sh
```

If `curl` is missing, install it with your platform package manager first.
If you already have `uv`, make sure it is in `PATH` or at `~/.local/bin/uv`.

### Reset the local Python environment

Symptom:

- `.venv` is broken or dependencies look out of sync

Fix:

```bash
./install.sh --recreate-venv
```

### Prebuilt runtime fetch fails

Symptoms:

- `bootstrap_local.py` fails while calling the GitHub Releases API
- installer completes Python setup but cannot fetch the Rust runtime

Fixes:

```bash
python bootstrap_local.py --release-tag latest
python rain_lab.py --mode validate
```

If the network is restricted, retry later or download the matching release asset manually into `bin/`.

## Runtime / Gateway

### Gateway unreachable

Checks:

```bash
R.A.I.N. status
R.A.I.N. doctor
```

Verify `~/.R.A.I.N./config.toml`:

- `[gateway].host` (default `127.0.0.1`)
- `[gateway].port` (default `42617`)
- `allow_public_bind` only when intentionally exposing LAN/public interfaces

### Pairing / auth failures on webhook

Checks:

1. Ensure pairing completed (`/pair` flow)
1. Ensure bearer token is current
1. Re-run diagnostics:

```bash
R.A.I.N. doctor
```

## Channel Issues

### Telegram conflict: `terminated by other getUpdates request`

Cause:

- multiple pollers using same bot token

Fix:

- keep only one active runtime for that token
- stop extra `R.A.I.N. daemon` / `R.A.I.N. channel start` processes

### Channel unhealthy in `channel doctor`

Checks:

```bash
R.A.I.N. channel doctor
```

Then verify channel-specific credentials + allowlist fields in config.

## Service Mode

### Service installed but not running

Checks:

```bash
R.A.I.N. service status
```

Recovery:

```bash
R.A.I.N. service stop
R.A.I.N. service start
```

Linux logs:

```bash
journalctl --user -u R.A.I.N..service -f
```

## Installer URL

```bash
./install.sh
```

For Windows, use `.\INSTALL_RAIN.cmd`.
