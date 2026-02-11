# Setup and Execution Guide for Multi-Task Learning (MTL)

This guide explains how to set up the environment and run the MTL project using Python directly, as the provided Ruby scripts are orchestration wrappers.

## Environment Setup

We use `uv` for fast environment management and dependency installation.

### 1. Initialize the Environment
```bash
uv venv --python 3.10
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
uv pip install -r requirements.txt
```

## Running the Code

### Training (2D Example)
To start a training run on the provided 2D example data:
```bash
python src/main2_infer.py \
    --ntrain=1 --nvalid=1 --n1=360 --n2=256 \
    --batch_train=1 --batch_valid=1 --epochs=100 \
    --gpus_per_node=1 \
    --dir_data_train=train/dataset2/data_train \
    --dir_data_valid=train/dataset2/data_valid \
    --dir_target_train=train/dataset2/target_train \
    --dir_target_valid=train/dataset2/target_valid \
    --dir_output=result2_infer
```

### Inference (2D Example)
Once you have a trained model (e.g., `result2_infer/last.ckpt`):
```bash
python src/main2_infer.py \
    --gpus_per_node=1 \
    --model=result2_infer/last.ckpt \
    --n1=256 --n2=1024 \
    --input=test/opunake2 \
    --output=test/opunake2.iter0
```

## Project Structure
- `src/`: Core PyTorch Lightning implementation and models.
- `train/`: Training scripts (Ruby) and example datasets.
- `test/`: Inference scripts (Ruby) and test data.
- `reference/`: Scientific documentation for the project.
