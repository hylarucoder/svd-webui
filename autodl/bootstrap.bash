echo "export SVD_CKPT_PATH=/root/autodl-tmp/models/checkpoints" >> ~/.bashrc
python -m pip install torch==2.1.0 torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
python -m pip install pytorch_lightning torchdata webdataset transformers kornia open-clip-torch
git clone https://github.com/hylarucoder/svd-webui
cd svd-webui
pip install pdm
pdm install --global


