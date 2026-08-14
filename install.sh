#!/usr/bin/env bash

set -euo pipefail

REPO="${OMNIAGENT_REPO:-YeQing17-2026/OmniAgent}"
REF="${OMNIAGENT_REF:-main}"
INSTALL_BASE="${OMNIAGENT_INSTALL_DIR:-$HOME/.local/share/omniagent}"
BIN_DIR="${OMNIAGENT_BIN_DIR:-$HOME/.local/bin}"
TARBALL_URL="${OMNIAGENT_TARBALL_URL:-https://github.com/${REPO}/archive/refs/heads/${REF}.tar.gz}"

log() {
  printf '[OmniAgent installer] %s\n' "$1"
}

fail() {
  printf '[OmniAgent installer] ERROR: %s\n' "$1" >&2
  exit 1
}

find_python() {
  if [ -n "${OMNIAGENT_PYTHON:-}" ] && command -v "${OMNIAGENT_PYTHON}" >/dev/null 2>&1; then
    printf '%s' "${OMNIAGENT_PYTHON}"
    return 0
  fi

  for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return 0
    fi
  done

  return 1
}

require_python_311() {
  local python_bin="$1"

  "$python_bin" -c '
import sys
sys.exit(0 if sys.version_info >= (3, 11) else 1)
' || fail "Python 3.11+ is required. Set OMNIAGENT_PYTHON to a compatible interpreter and retry."
}

if ! command -v curl >/dev/null 2>&1; then
  fail "curl is required."
fi

if ! command -v tar >/dev/null 2>&1; then
  fail "tar is required."
fi

PYTHON_BIN="$(find_python || true)"
[ -n "$PYTHON_BIN" ] || fail "Python 3.11+ was not found."
require_python_311 "$PYTHON_BIN"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

ARCHIVE_PATH="$TMP_DIR/omniagent.tar.gz"
EXTRACT_DIR="$TMP_DIR/extract"
APP_DIR="$INSTALL_BASE/app"
VENV_DIR="$INSTALL_BASE/venv"
LAUNCHER_PATH="$BIN_DIR/omniagent"

mkdir -p "$EXTRACT_DIR" "$INSTALL_BASE" "$BIN_DIR"

log "Downloading source archive from $TARBALL_URL"
curl -fsSL "$TARBALL_URL" -o "$ARCHIVE_PATH"

log "Extracting archive"
tar -xzf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"

PYPROJECT_PATH="$(find "$EXTRACT_DIR" -mindepth 0 -maxdepth 2 -type f -name pyproject.toml | head -n 1)"
[ -n "$PYPROJECT_PATH" ] || fail "Failed to locate pyproject.toml in the source archive."

SRC_DIR="$(dirname "$PYPROJECT_PATH")"
[ -n "$SRC_DIR" ] || fail "Failed to unpack source archive."

rm -rf "$APP_DIR" "$VENV_DIR"
mv "$SRC_DIR" "$APP_DIR"

log "Creating virtual environment with $PYTHON_BIN"
"$PYTHON_BIN" -m venv "$VENV_DIR"

log "Installing OmniAgent into $VENV_DIR"
PIP_DISABLE_PIP_VERSION_CHECK=1 "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null
PIP_DISABLE_PIP_VERSION_CHECK=1 "$VENV_DIR/bin/python" -m pip install --no-build-isolation "$APP_DIR"

cat > "$LAUNCHER_PATH" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/omniagent" "\$@"
EOF
chmod +x "$LAUNCHER_PATH"

log "Installation complete."
log "Binary: $LAUNCHER_PATH"

case ":$PATH:" in
  *":$BIN_DIR:"*)
    ;;
  *)
    log "Add $BIN_DIR to PATH if 'omniagent' is not found in a new shell."
    ;;
esac

log "Next steps:"
log "  omniagent onboard"
log "  omniagent serve"
