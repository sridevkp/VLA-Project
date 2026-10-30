## Initialize
conda create -n lerobot python=3.12  -y && conda activate lerobot
conda install ffmpeg -c conda-forge
git clone https://github.com/huggingface/lerobot.git
pip install -e "./lerobot[smolvla]"
python -c "import lerobot; print(lerobot.__file__)"

## Run
conda activate lerobot
python main.py