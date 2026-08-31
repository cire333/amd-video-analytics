#!/usr/bin/env bash
# System setup for the AMD video analytics pipeline (Ubuntu 24.04 "noble").
# Requires sudo. Run once, then LOG OUT AND BACK IN (group membership).
#
#   ./scripts/setup_system.sh
#
# After re-login, verify with: ./scripts/verify_env.sh
set -euo pipefail

echo "==> 1/4 Adding $USER to render + video groups (needed for /dev/kfd and /dev/dri)"
sudo usermod -aG render,video "$USER"

echo "==> 2/4 Installing decode/build dev packages"
sudo apt-get update
sudo apt-get install -y \
    vainfo libva-dev libdrm-dev \
    libavcodec-dev libavformat-dev libavutil-dev \
    gstreamer1.0-plugins-bad \
    cmake ninja-build g++ pkg-config python3-dev python3-venv

echo "==> 3/4 Adding AMD ROCm apt repository"
sudo mkdir -p /etc/apt/keyrings
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
# 'latest' tracks the newest ROCm release; gfx1201 (R9700) needs >= 6.4.
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/latest noble main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | sudo tee /etc/apt/preferences.d/rocm-pin-600

echo "==> 4/4 Installing ROCm (hip runtime, compiler, rocminfo, MIGraphX)"
sudo apt-get update
sudo apt-get install -y rocm-hip-sdk rocminfo migraphx

echo
echo "Done. LOG OUT AND BACK IN so the render/video group membership takes effect,"
echo "then run ./scripts/verify_env.sh"
