#!/bin/bash
set -e
cd /home/roderickperez/DataScienceProjects/multiTaskLearningSeismic
uv sync
uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
