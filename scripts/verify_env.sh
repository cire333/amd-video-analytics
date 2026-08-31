#!/usr/bin/env bash
# Verify the environment is ready for the pipeline. No sudo needed.
set -uo pipefail

pass() { echo "  [ok]   $1"; }
fail() { echo "  [FAIL] $1"; RC=1; }
RC=0

echo "== groups =="
id -nG | grep -qw render && pass "user in render group" || fail "not in render group (re-login after setup_system.sh?)"

echo "== ROCm / HIP =="
if command -v rocminfo >/dev/null; then
    GFX=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort -u | tr '\n' ' ')
    [ -n "$GFX" ] && pass "rocminfo sees: $GFX" || fail "rocminfo runs but reports no GPU agents"
    echo "$GFX" | grep -q gfx12 && pass "gfx12xx (R9700) visible" || fail "R9700 (gfx12xx) not visible to ROCm"
else
    fail "rocminfo not installed"
fi
command -v hipcc >/dev/null || [ -x /opt/rocm/bin/hipcc ] && pass "hipcc present" || fail "hipcc missing"

echo "== VAAPI decode =="
if command -v vainfo >/dev/null; then
    # Pick the AMD render node explicitly (NVIDIA card is also in this box)
    for node in /dev/dri/renderD*; do
        vendor=$(cat /sys/class/drm/$(basename "$node")/device/vendor 2>/dev/null)
        if [ "$vendor" = "0x1002" ]; then
            if vainfo --display drm --device "$node" 2>/dev/null | grep -q VAProfileH264; then
                pass "H.264 decode on $node (AMD)"
            else
                fail "vainfo shows no H264 profile on $node"
            fi
        fi
    done
else
    fail "vainfo not installed"
fi

echo "== build deps =="
pkg-config --exists libva libva-drm libavcodec libavformat libavutil libdrm \
    && pass "libva + ffmpeg + libdrm dev headers" || fail "missing dev headers (pkg-config)"

exit $RC
