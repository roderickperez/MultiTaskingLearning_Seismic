# Seismic Model Workflow Guide: Training, Storage, and Inference

This guide provides technical details on how to retrain the seismic fault segmentation models, manage model storage, and apply the models to new datasets.

## 1. How to Retrain the Model

Training is handled by `src/main3_refine.py` (for the 3D refinement model) using **PyTorch Lightning**.

### Execution Command
Run the training script with the following structure:

```bash
uv run python src/main3_refine.py \
    --dir_data_train ./train/dataset3/data \
    --dir_target_train ./train/dataset3/target \
    --dir_data_valid ./train/dataset3/data_valid \
    --dir_target_valid ./train/dataset3/target_valid \
    --dir_output ./checkpoints \
    --epochs 100 \
    --batch_train 4 \
    --n1 128 --n2 128 --n3 128
```

---

## 2. Interactive Setup (NEW)

Instead of manually typing long CLI commands, you can use the new interactive terminal interface (TUI). This provides a guided setup with detailed parameter descriptions from the reference paper.


### How to Run
#### Option 1: Using `uv run` (Recommended)
This automatically handles environment activation for a single command.
```bash
uv run python interactive_train.py
```

#### Option 2: Manual Activation
If you prefer to activate the environment once and run commands normally:
```bash
# Activate the environment
source .venv/bin/activate

# Run the script directly
python interactive_train.py
```

### Features
- **MTLS Banner**: A stylized, colorful ASCII header.
- **Guided Selection**: Interactive prompts for all training parameters.
- **Contextual Help**: Detailed descriptions of RGT, DHR, and Fault attributes as you select them.
- **Safety Checks**: Shows you the final command and asks for confirmation before starting.

### Key Parameters:
- `--ntrain`: Number of training samples to use.
- `--epochs`: Total training cycles.
- `--resume`: Path to a `.ckpt` file to restart training from a specific point.
- `--n1, --n2, --n3`: Dimensions of the input 3D cubes.

---

## 2. Model Storage and Formats

### Where are models stored?
Models are stored in the directory specified by `--dir_output` during training. PyTorch Lightning automatically creates:
- **Checkpoints**: Stored during training (e.g., `epoch=10.ckpt`).
- **Last Version**: Usually saved as `last.ckpt`.
- **Logs**: TensorBoard logs are saved in the same directory for monitoring performance.

### File Extensions (.pth vs .ckpt vs .hdf5)
- **.ckpt (PyTorch Lightning)**: This contains the model weight (`state_dict`), optimizer state, and training metadata.
- **.pth (Standard PyTorch)**: **Exported automatically after training** as `model.pth`. It contains only the trained weights, making it easier to load in custom scripts.
- **.hdf5**: Not standard for PyTorch architectures.

---

## 3. Training Analytics (New)

After training completes, the script automatically generates analytics to help you evaluate performance.

### Loss Plots
A file named `loss_plot.png` is saved in your `--dir_output` folder. This plot shows the evolution of both training and validation loss, helping you identify overfitting or convergence.

### Accuracy
While accuracy is currently a placeholder in the core loss function, future updates that implement `custom_accuracy` will automatically be reflected in these plots.

---

## 3. Applying the Model (Inference)

To apply a trained model to a test dataset or volume, use `src/main3_infer.py`.

### Inference Command
```bash
uv run python src/main3_infer.py \
    --input ./data/test_volume \
    --model ./checkpoints/last.ckpt \
    --output ./test_output/result \
    --n1 128 --n2 128 --n3 128 \
    --rgt True --fault True --dhr True
```

### Output Files
The inference script will generate binary files (`.bin` or with specific suffixes like `.fsem`, `.rgt`, `.dhr`) in the `--output` path.

---

## 4. Training on a New "Real" Dataset

### Input Requirements
The scripts expect **Raw Binary Files** (float32).
If your real data is in SEGY format, you must first convert it to binary patches or cubes.

### Data Format
Each sample consists of 3D cubes for:
1. **Input**: `rgt`, `dhr`, `fsem`, `fdip`, `fstrike` (depending on the multi-task targets).
2. **Targets**: The ground truth masks for faults or horizons.

### Steps for Real Data:
1. **Export to Binary**: Convert your SEGY volumes to flattened binary files.
2. **Patching**: If your volumes are very large, slice them into smaller cubes (e.g., 128x128x128) as the model cannot process an entire full-scale seismic volume at once in GPU memory.
3. **Naming**: Samples should be numbered (0, 1, 2...) if using the `BasicDataset` class in the provided scripts.

---

## 5. Visualization of Results

Since the outputs are raw binary data, you can visualize them using:
1. **Python**: Use `numpy.fromfile` and `matplotlib` (similar to the logic in `utility.py`).
2. **Seismic Software**: Convert the `.bin` output back to SEGY using `segyio` to load into Petrel, OpendTect, or Kingdom.
3. **Paraview**: Create a `.vgi` or `.raw` header file to import the binary data as a 3D volume.

> [!IMPORTANT]
> Always ensure the dimensions (`n1`, `n2`, `n3`) used during visualization match exactly the dimensions used during inference.
