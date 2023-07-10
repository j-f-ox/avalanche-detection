# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

pip install -e .

# Install PyTorch
# The exact version required is system-dependent:
# see https://pytorch.org/
pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113