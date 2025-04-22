# 3D Human Texture Inpainting from a Single Image (imgTOtex)

```sh
# Clone the repo
git clone --recursive https://github.com/wangziyao318/imgTOtex.git
cd img2tex

# Download SGHM pretrained weight for human matting
# https://drive.google.com/drive/folders/15mGzPJQFEchaZHt9vgbmyOy46XxWtEOZ?usp=sharing
# Place it in data/human_mat/pretrained_weight/ folder

# Install requirements
# You may need to install torch and torchvision first, and run pip install several times
conda create -n img2tex python=3.12
conda activate img2tex
pip install -r requirements.txt

# login to huggingface so that you can download stabilityai/stable-diffusion-2-inpainting
huggingface-cli login

# Config accelerate if you want to use torch dynamo inductor or numa topology
accelerate config

# Run the program
./main.py
```
