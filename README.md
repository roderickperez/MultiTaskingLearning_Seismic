# Description
**Multi-task learning and inference from seismic images**

The machine learning model infers and refines a denoised, higher-resolution image (DHR for short), a relative geological time (RGT for short) volume and geological fault attributes (including location, dip, strike; Fault for short) from a noisy, low-resolution seismic image. The code implements the training, validation, and test for two functionalities: (i) multi-task inference (MTI), which infers DHR, RGT and Fault, and (ii) multi-task refinement (MTR), which refines DHR, RGT and Fault output from MTI.

The work was supported by an FY23 Rapid Response project (GRR3KGAO) of Center for Space and Earth Science (CSES), Los Alamos National Laboratory (LANL). LANL is operated by Triad National Security, LLC, for the National Nuclear Security Administration (NNSA) of the U.S. Department of Energy (DOE) under Contract No. 89233218CNA000001. The research used high-performance computing resources provided by LANL's Institutional Computing program.

LANL open source approval reference O4656.

# Reference
LA-UR-23-20649: Gao, 2024, Iterative multi-task learning and inference from seismic images, _Geophysical Journal International_, doi: [doi.org/10.1093/gji/ggad424](https://doi.org/10.1093/gji/ggad424)

# Requirement
The code is implemented with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/).

To set up the environment with GPU support and all dependencies, please refer to the [SETUP_GUIDE.md](SETUP_GUIDE.md).

# Use
The project uses Python scripts for training and inference. While Ruby scripts are provided as wrappers, they are not required.

### Training (3D Example)
To train the model on provided example data:
```bash
python src/main3_infer.py \
    --n1=64 --n2=64 --n3=64 \
    --ntrain=1 --nvalid=1 \
    --batch_train=1 --batch_valid=1 \
    --epochs=100 \
    --gpus_per_node=1 \
    --dir_data_train=train/dataset3/data_train \
    --dir_data_valid=train/dataset3/data_valid \
    --dir_target_train=train/dataset3/target_train \
    --dir_target_valid=train/dataset3/target_valid \
    --dir_output=result3_infer
```

### Inference
To run inference on the field data (`test/opunake3`):
```bash
python src/main3_infer.py \
    --gpus_per_node=1 \
    --model=result3_infer/last.ckpt \
    --n1=256 --n2=512 --n3=256 \
    --input=test/opunake3 \
    --output=test/opunake3.predict
```

### Model Saving & Logs
*   **Models**: Saved in the output directory specified by `--dir_output` (e.g., `result3_infer/`). The final model is saved as `last.ckpt`.
*   **Training Curves**: Logs are saved in CSV and TensorBoard formats within the `lightning_logs/` subdirectory of your output folder.
    *   To view results: `tensorboard --logdir result3_infer/lightning_logs`

For large 3D images, GPU may run out of memory. In such a case, you can use CPU by setting ```--gpus_per_node=0```.

# License
&copy; 2023. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# Author
Kai Gao, <kaigao@lanl.gov>
