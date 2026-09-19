#!/usr/bin/env bash
# Sudo-less rocDecode install for the zero-copy decode backend.
#
# Why not `apt install rocdecode`: it depends on mesa-amdgpu-va-drivers
# (AMD's pro VA driver, from the separate amdgpu apt repo), which would
# REPLACE the system Mesa VA driver our validated VAAPI decode path uses.
# rocDecode runs fine against system Mesa (>= 25.x) — the dependency is
# packaging strictness — so we stage the library locally instead:
# download the debs, extract to ~/.local/opt/rocdecode, and patch the
# library rpath so no LD_LIBRARY_PATH is ever needed. The avap build
# auto-detects this location; rebuild avap after running this.
set -euo pipefail

DEST="${1:-$HOME/.local/opt/rocdecode}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cd "$TMP"
apt-get download rocdecode rocdecode-dev
for f in *.deb; do dpkg-deb -x "$f" x/; done
ROOT=$(echo x/opt/rocm-*)

mkdir -p "$DEST"
cp -r "$ROOT/lib" "$ROOT/include" "$DEST/"

# patch the real .so so its own deps (hip, rocprofiler-register) resolve
PATCHELF=$(command -v patchelf || echo "$HOME/.local/bin/patchelf")
if ! command -v "$PATCHELF" >/dev/null 2>&1; then
    pip install --user -q patchelf
fi
"$PATCHELF" --set-rpath /opt/rocm/lib "$DEST"/lib/librocdecode.so.*.*.*

echo "rocDecode staged at $DEST"
echo "now rebuild avap:  pip install -e . --no-build-isolation"
