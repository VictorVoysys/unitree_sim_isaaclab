#!/bin/bash
set -e
set -o pipefail

# ============================================================================
# Unitree G1 Simulation Environment - Full Installer
#
# Installs everything needed to run the Isaac Lab simulation:
#   - Miniforge (conda) if not present
#   - Conda environment with Python 3.11
#   - PyTorch with CUDA support
#   - Isaac Sim 5.1 + Isaac Lab
#   - CycloneDDS + unitree_sdk2_python
#   - Teleimager, project requirements, and assets
#
# Usage:
#   bash install.sh [env_name]    (default env name: unitree_sim)
#
# Re-running is safe — each step checks if it's already done.
# ============================================================================

ENV_NAME="${1:-unitree_sim}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="$SCRIPT_DIR/deps"

# Accept NVIDIA Omniverse EULA (required for non-interactive Isaac Sim install)
export OMNI_KIT_ACCEPT_EULA=YES

PYTHON_VER="3.11"
ISAAC_SIM_PKG="isaacsim[all,extscache]==5.1.0"
TORCH_PKG="torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

step() { echo -e "\n${BLUE}==== $1 ====${NC}"; }
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}  ! $1${NC}"; }
fail() { echo -e "${RED}  ✗ $1${NC}"; exit 1; }

# ---- Sanity check: run from project root ----
if [ ! -f "$SCRIPT_DIR/requirements.txt" ] || [ ! -f "$SCRIPT_DIR/sim_main.py" ]; then
    fail "This script must live in the unitree_sim_isaaclab project root."
fi
cd "$SCRIPT_DIR"

# ---- Check NVIDIA GPU ----
step "Checking GPU"
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found. An NVIDIA GPU with drivers is required."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "Found GPU: $GPU_NAME"

# ============================================================================
# PHASE 1: System packages
# ============================================================================
step "Phase 1: System packages"
REQUIRED_PKGS="cmake build-essential openssl git-lfs unzip curl"
MISSING_PKGS=""
for pkg in $REQUIRED_PKGS; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
done

if [ -z "$MISSING_PKGS" ]; then
    ok "All system packages already installed"
else
    echo "  Installing missing packages:$MISSING_PKGS"
    sudo apt-get update -qq
    sudo apt-get install -y -qq $MISSING_PKGS > /dev/null
    ok "System packages installed"
fi

# ============================================================================
# PHASE 2: Conda (Miniforge)
# ============================================================================
step "Phase 2: Conda"

# Try to find conda
CONDA_EXE=""
if command -v conda &>/dev/null; then
    CONDA_EXE="$(command -v conda)"
elif [ -f "$HOME/miniforge3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/anaconda3/bin/conda"
fi

if [ -z "$CONDA_EXE" ]; then
    echo "  Conda not found. Installing Miniforge..."
    MINIFORGE_INSTALLER="/tmp/Miniforge3-Linux-x86_64.sh"
    curl -fsSL -o "$MINIFORGE_INSTALLER" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
    rm -f "$MINIFORGE_INSTALLER"
    CONDA_EXE="$HOME/miniforge3/bin/conda"
    ok "Miniforge installed at $HOME/miniforge3"
else
    ok "Conda found at $CONDA_EXE"
fi

# Activate conda in this shell
CONDA_BASE="$("$CONDA_EXE" info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ============================================================================
# PHASE 3: Conda environment
# ============================================================================
step "Phase 3: Conda environment '$ENV_NAME'"

if conda env list | grep -qw "$ENV_NAME"; then
    ok "Environment '$ENV_NAME' already exists"
else
    echo "  Creating conda environment (Python $PYTHON_VER)..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VER" > /dev/null
    ok "Environment created"
fi

conda activate "$ENV_NAME"
ok "Activated '$ENV_NAME' ($(python --version))"

# ============================================================================
# PHASE 4: PyTorch
# ============================================================================
step "Phase 4: PyTorch"

if python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.7.0"; then
    ok "PyTorch 2.7.0 already installed"
else
    echo "  Installing PyTorch (this may take a few minutes)..."
    pip install $TORCH_PKG --index-url "$TORCH_INDEX" -q || fail "PyTorch install failed"
    ok "PyTorch installed"
fi

# ============================================================================
# PHASE 5: Isaac Sim
# ============================================================================
step "Phase 5: Isaac Sim 5.1"

if python -c "import isaacsim" 2>/dev/null; then
    ok "Isaac Sim already installed"
else
    echo "  Installing Isaac Sim 5.1 (this will take a while)..."
    pip install --upgrade pip -q
    pip install "$ISAAC_SIM_PKG" --extra-index-url https://pypi.nvidia.com -q || fail "Isaac Sim install failed"
    ok "Isaac Sim installed"
fi

# ============================================================================
# PHASE 6: Clone dependencies
# ============================================================================
step "Phase 6: Clone dependencies"

mkdir -p "$DEPS_DIR"
cd "$DEPS_DIR"

if [ -d "IsaacLab" ]; then
    ok "IsaacLab already cloned"
else
    echo "  Cloning IsaacLab..."
    git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git
    ok "IsaacLab cloned"
fi

if [ -d "cyclonedds" ]; then
    ok "CycloneDDS already cloned"
else
    echo "  Cloning CycloneDDS..."
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
    ok "CycloneDDS cloned"
fi

if [ -d "unitree_sdk2_python" ]; then
    ok "unitree_sdk2_python already cloned"
else
    echo "  Cloning unitree_sdk2_python..."
    git clone https://github.com/unitreerobotics/unitree_sdk2_python
    ok "unitree_sdk2_python cloned"
fi

cd "$SCRIPT_DIR"

# ============================================================================
# PHASE 7: Build CycloneDDS
# ============================================================================
step "Phase 7: Build CycloneDDS"

CYCLONE_DIR="$DEPS_DIR/cyclonedds"
if [ -d "$CYCLONE_DIR/install/lib" ]; then
    ok "CycloneDDS already built"
else
    echo "  Building CycloneDDS..."
    cd "$CYCLONE_DIR"
    mkdir -p build install
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install > /dev/null 2>&1
    cmake --build . --target install -j"$(nproc)" > /dev/null 2>&1
    ok "CycloneDDS built"
fi

export CYCLONEDDS_HOME="$CYCLONE_DIR/install"
cd "$SCRIPT_DIR"

# ============================================================================
# PHASE 8: Install Isaac Lab
# ============================================================================
step "Phase 8: Isaac Lab"

ISAACLAB_DIR="$DEPS_DIR/IsaacLab"
if python -c "import isaaclab" 2>/dev/null; then
    ok "Isaac Lab already installed"
else
    echo "  Installing Isaac Lab (this will take a while)..."
    cd "$ISAACLAB_DIR"
    ./isaaclab.sh --install || fail "Isaac Lab install failed"
    ok "Isaac Lab installed"
fi
cd "$SCRIPT_DIR"

# ============================================================================
# PHASE 9: Install unitree_sdk2_python
# ============================================================================
step "Phase 9: unitree_sdk2_python"

if python -c "import unitree_sdk2py" 2>/dev/null; then
    ok "unitree_sdk2_python already installed"
else
    echo "  Installing unitree_sdk2_python..."
    cd "$DEPS_DIR/unitree_sdk2_python"
    pip install -e . -q || fail "unitree_sdk2_python install failed"
    ok "unitree_sdk2_python installed"
fi
cd "$SCRIPT_DIR"

# ============================================================================
# PHASE 10: Submodules and teleimager
# ============================================================================
step "Phase 10: Submodules and teleimager"

if [ -f "teleimager/setup.py" ] || [ -f "teleimager/pyproject.toml" ]; then
    ok "Submodules already initialized"
else
    echo "  Initializing git submodules..."
    git submodule update --init --depth 1
    ok "Submodules initialized"
fi

# Configure teleimager for Isaac Sim
if [ -f "teleimager/cam_config_server.yaml" ]; then
    sed -i 's/type:.*/type: isaacsim/' teleimager/cam_config_server.yaml
    sed -i 's/image_shape:.*/image_shape: [480, 640]/' teleimager/cam_config_server.yaml
    ok "Teleimager config updated"
fi

if python -c "from teleimager.image_server import run_isaacsim_server" 2>/dev/null; then
    ok "Teleimager already installed"
else
    echo "  Installing teleimager with server dependencies..."
    cd teleimager
    pip install -e ".[server]" || fail "Teleimager install failed"
    cd "$SCRIPT_DIR"
    python -c "from teleimager.image_server import run_isaacsim_server" || fail "Teleimager import check failed after install"
    ok "Teleimager installed"
fi

# ============================================================================
# PHASE 11: Project requirements
# ============================================================================
step "Phase 11: Project requirements"
echo "  Installing requirements.txt..."
pip install -r requirements.txt -q || fail "requirements.txt install failed"
ok "Requirements installed"

# ============================================================================
# PHASE 12: libstdc++ patch
# ============================================================================
step "Phase 12: libstdc++ compatibility"
conda install -y -c conda-forge libstdcxx-ng > /dev/null 2>&1
ok "libstdc++ updated"

# ============================================================================
# PHASE 13: SSL certificates (for WebRTC)
# ============================================================================
step "Phase 13: SSL certificates"

CERT_DIR="$HOME/.config/xr_teleoperate"
if [ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ]; then
    ok "Certificates already exist"
else
    echo "  Generating self-signed certificates..."
    mkdir -p "$CERT_DIR"
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
        -subj "/CN=localhost" 2>/dev/null
    ok "Certificates generated at $CERT_DIR"
fi

# ============================================================================
# PHASE 14: Download assets
# ============================================================================
step "Phase 14: Simulation assets"

if [ -d "$SCRIPT_DIR/assets" ]; then
    ok "Assets already downloaded"
else
    echo "  Downloading assets from HuggingFace (this is a large download)..."
    chmod +x fetch_assets.sh
    bash fetch_assets.sh
    ok "Assets downloaded"
fi

# ============================================================================
# PHASE 15: Verify installation
# ============================================================================
step "Phase 15: Verify installation"

export LD_LIBRARY_PATH="$CYCLONE_DIR/install/lib:$LD_LIBRARY_PATH"
VERIFY_FAILED=0
for mod in torch isaacsim isaaclab unitree_sdk2py; do
    if python -c "import $mod" 2>/dev/null; then
        ok "$mod"
    else
        warn "$mod FAILED to import"
        VERIFY_FAILED=1
    fi
done

if python -c "from teleimager.image_server import run_isaacsim_server" 2>/dev/null; then
    ok "teleimager.image_server"
else
    warn "teleimager.image_server FAILED to import"
    VERIFY_FAILED=1
fi

if [ "$VERIFY_FAILED" -eq 1 ]; then
    fail "Some packages failed verification. Check the warnings above."
fi

# ============================================================================
# Done
# ============================================================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  To use the environment:"
echo ""
echo "    conda activate $ENV_NAME"
echo "    export CYCLONEDDS_HOME=$CYCLONE_DIR/install"
echo "    export LD_LIBRARY_PATH=\$CYCLONEDDS_HOME/lib:\$LD_LIBRARY_PATH"
echo "    export OMNI_KIT_ACCEPT_EULA=YES"
echo ""
echo "  To run the simulator:"
echo ""
echo "    python sim_main.py --device cuda:0 --enable_cameras \\"
echo "      --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \\"
echo "      --enable_dex1_dds --robot_type g129"
echo ""
