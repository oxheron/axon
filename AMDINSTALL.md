First make sure you install ROCM and the correct pytorch version (rocm pytorch)
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.2 (or whatever version)
I dont think you need special triton, should come with above
clone vllm 
cd vllm 
pip install --upgrade pip

# Build & install AMD SMI
pip
 install /opt/rocm/share/amd_smi

# Install dependencies
pip
 install --upgrade numba \
    scipy \
    huggingface-hub[cli,hf_transfer] \
    setuptools_scm
pip
 install -r requirements/rocm.txt

# To build for a single architecture (e.g., MI300) for faster installation (recommended):
export PYTORCH_ROCM_ARCH="gfx942"
notes, may need to export HSA_OVERRIDE_GFX_VERSION=11.0.0 as trick


# To build vLLM for multiple arch MI210/MI250/MI300, use this instead
# export PYTORCH_ROCM_ARCH="gfx90a;gfx942"

python3
 setup.py develop

 then in stall requirements-base for additional reqs. 