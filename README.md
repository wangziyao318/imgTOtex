# img2tex

```sh
# Clone the repo
git clone --recursive https://github.com/wangziyao318/img2tex.git
cd img2tex

# Install requirements
conda create -n img2tex python=3.12
conda activate img2tex
pip install -r requirements.txt

# Another way to install pytorch3d in case pip build failed
# conda install pytorch3d -c pytorch3d

# Config accelerate if you want to use torch dynamo inductor or numa topology
accelerate config

# Run the program
./main.py
```
