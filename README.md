# svd-webui


> You can also use ComfyUI for only 8 vram
> try my workflow https://github.com/hylarucoder/comfyui-workflow/blob/main/svd/svd-image-to-video.json

## Installation

```bash
git clone https://github.com/hylarucoder/svd-webui
cd svd-webui
py3.10 -m venv venv
# linux/macOS
source ./venv/bin/activate
# windows
venv\Scripts\activate.bat
pip install pdm 
pdm install
# linux/windows
python -m pip install torch==2.1.0 torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
python -m pip install onnxruntime-gpu
# macos
python -m pip install torch torchvision torchaudio 
python -m pip install onnxruntime-silicon

# 
python -m pip install pytorch_lightning torchdata webdataset transformers kornia open-clip-torch

```

## Online GPU

### AutoDL

https://twitter.com/hylarucoder/status/1727922785828417805

### Colab

https://github.com/hylarucoder/svd-webui/blob/main/svd_webui_colab.ipynb

## Credit

- https://github.com/Stability-AI/generative-models
