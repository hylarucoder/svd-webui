echo "export SVD_CKPT_PATH=/root/autodl-tmp/models/checkpoints" >> ~/.bashrc
echo "export SVD_PORT=6006" >> ~/.bashrc
echo "source /etc/network_turbo" >> ~/.bashrc
apt update && apt install git-lfs nethogs neovim tmux htop
python -m pip install torch==2.1.0 torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
python -m pip install pytorch_lightning torchdata webdataset transformers kornia open-clip-torch einops omegaconf
git clone https://github.com/hylarucoder/svd-webui

cd svd-webui
pip install pdm
pdm install --global


