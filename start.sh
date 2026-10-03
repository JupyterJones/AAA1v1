#!/usr/bin/env bash
# ==========================================================
#  AAA1v1: Lightweight Diffusers-Powered A1111 Interface
# ==========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

# Detect Python environment with diffusers & flask
if [[ -f "${SCRIPT_DIR}/venv/bin/python" ]]; then
    PYTHON_CMD="${SCRIPT_DIR}/venv/bin/python"
elif [[ -f "/home/jack/Desktop/Kokoro-TTS-Pause/venv/bin/python" ]]; then
    PYTHON_CMD="/home/jack/Desktop/Kokoro-TTS-Pause/venv/bin/python"
elif [[ -f "/home/jack/Desktop/Media_Studio/venv/bin/python" ]]; then
    PYTHON_CMD="/home/jack/Desktop/Media_Studio/venv/bin/python"
else
    PYTHON_CMD="python3"
fi

# CPU optimizations
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export OPENBLAS_NUM_THREADS="4"

PORT=7865

echo "=========================================================="
echo "  🚀 Starting AAA1v1 WebUI (Fast Diffusers Dashboard)"
echo "  ⚡ Python: ${PYTHON_CMD}"
echo "  ⚡ Port:   http://127.0.0.1:${PORT}"
echo "=========================================================="

"${PYTHON_CMD}" app.py
