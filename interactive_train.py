import sys
import os
import argparse
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
import questionary

import glob

console = Console()

def get_default_dataset_info(data_dir, default_n=64, default_count=1):
    """
    Attempts to detect the number of samples and their dimensions
    by looking at files in the specified directory.
    """
    if not os.path.exists(data_dir):
        return default_count, default_n

    # Count files matching *_rgt.bin (assuming standard naming)
    files = glob.glob(os.path.join(data_dir, "*_rgt.bin"))
    count = len(files) if files else default_count

    # Try to detect dimensions from the first file
    n = default_n
    if files:
        try:
            size_bytes = os.path.getsize(files[0])
            # Assume float32 (4 bytes) and cubic shape
            # size = n^3 * 4
            # n^3 = size / 4
            # n = (size / 4)^(1/3)
            elements = size_bytes / 4
            detected_n = int(round(elements**(1/3)))
            if detected_n**3 == elements:
                n = detected_n
        except Exception:
            pass
            
    return count, n

def display_banner():
    banner_text = """
[bold cyan]
███╗   ███╗████████╗██╗     ███████╗
████╗ ████║╚══██╔══╝██║     ██╔════╝
██╔████╔██║   ██║   ██║     ███████╗
██║╚██╔╝██║   ██║   ██║     ╚════██║
██║ ╚═╝ ██║   ██║   ███████╗███████║
╚═╝     ╚═╝   ╚═╝   ╚══════╝╚══════╝
[/bold cyan]
[bold white]Multi-task learning and inference from seismic images[/bold white]
"""
    console.print(Panel(banner_text, expand=False, border_style="green"))

def get_interactive_params():
    # Auto-detect defaults based on likely paths
    train_dir = "train/dataset3/data_train"
    valid_dir = "train/dataset3/data_valid"
    
    ntrain_default, n_default = get_default_dataset_info(train_dir, default_n=64, default_count=1)
    nvalid_default, _ = get_default_dataset_info(valid_dir, default_n=n_default, default_count=1)
    
    # If we detected just 1 sample, maybe default to 1 for safety, otherwise use the count
    
    questions = [
        {
            "type": "text",
            "name": "dir_data_train",
            "message": "Training data directory:",
            "default": train_dir,
            "instruction": "Path to the binary cubes for training input."
        },
        {
            "type": "text",
            "name": "dir_target_train",
            "message": "Training target directory:",
            "default": "train/dataset3/target_train",
            "instruction": "Path to the ground truth binary cubes."
        },
        {
            "type": "text",
            "name": "dir_data_valid",
            "message": "Validation data directory:",
            "default": valid_dir,
            "instruction": "Path to the binary cubes for validation."
        },
        {
            "type": "text",
            "name": "dir_target_valid",
            "message": "Validation target directory:",
            "default": "train/dataset3/target_valid",
            "instruction": "Path to the ground truth binary cubes for validation."
        },
        {
            "type": "text",
            "name": "dir_output",
            "message": "Output directory for models/plots:",
            "default": "result3_infer",
            "instruction": "Where to store .ckpt, .pth, and loss plots."
        },
        # ... (rest of the parameters remain the same) 
        {
            "type": "text",
            "name": "n1",
            "message": "Dimension n1 (Z):",
            "default": str(n_default),
            "instruction": "Vertical sampling points."
        },
        {
            "type": "text",
            "name": "n2",
            "message": "Dimension n2 (Y):",
            "default": str(n_default),
            "instruction": "Cross-line sampling points."
        },
        {
            "type": "text",
            "name": "n3",
            "message": "Dimension n3 (X):",
            "default": str(n_default),
            "instruction": "In-line sampling points."
        },
        {
            "type": "text",
            "name": "ntrain",
            "message": "Training set size:",
            "default": str(ntrain_default),
            "instruction": "Number of training samples."
        },
        {
            "type": "text",
            "name": "nvalid",
            "message": "Validation set size:",
            "default": str(nvalid_default),
            "instruction": "Number of validation samples."
        },
        {
            "type": "text",
            "name": "epochs",
            "message": "Number of epochs:",
            "default": "100",
            "instruction": "Total training cycles."
        },
        {
            "type": "text",
            "name": "batch_train",
            "message": "Batch size:",
            "default": "4",
            "instruction": "Number of samples per training step."
        },
        {
            "type": "confirm",
            "name": "use_gpu",
            "message": "Use GPU if available?",
            "default": True,
            "instruction": "Highly recommended for 3D seismic data."
        },
        {
            "type": "confirm",
            "name": "rgt",
            "message": "Enable Relative Geological Time (RGT)?",
            "default": True,
            "instruction": "Indicates relative stratigraphic time of reflectors."
        },
        {
            "type": "confirm",
            "name": "dhr",
            "message": "Enable Denoised Higher-Resolution (DHR)?",
            "default": True,
            "instruction": "Enhances image resolution and suppresses noise."
        },
        {
            "type": "confirm",
            "name": "fault",
            "message": "Enable Fault Attributes?",
            "default": True,
            "instruction": "Infers probability, dip, and strike (for 3D)."
        }
    ]
    
    rprint("\n[bold yellow]Description of Parameters (Reference: Gao, 2024)[/bold yellow]")
    rprint("[italic cyan]RGT:[/italic cyan] Relative stratigraphic time indicating horizon continuity.")
    rprint("[italic cyan]DHR:[/italic cyan] Inferred high-frequency seismic with reduced noise.")
    rprint("[italic cyan]Fault:[/italic cyan] Geometrical characterization of discontinuities.")
    rprint("-" * 50)
    
    answers = questionary.prompt(questions)
    return answers

def run_training(params):
    if not params:
        rprint("[bold red]Training cancelled.[/bold red]")
        return

    cmd = [
        "python", "src/main3_refine.py",
        "--dir_data_train", params["dir_data_train"],
        "--dir_target_train", params["dir_target_train"],
        "--dir_data_valid", params["dir_data_valid"],
        "--dir_target_valid", params["dir_target_valid"],
        "--dir_output", params["dir_output"],
        "--n1", params["n1"],
        "--n2", params["n2"],
        "--n3", params["n3"],
        "--ntrain", params["ntrain"],
        "--nvalid", params["nvalid"],
        "--epochs", params["epochs"],
        "--batch_train", params["batch_train"],
        "--gpus_per_node", "1" if params["use_gpu"] else "0",
        "--rgt", "y" if params["rgt"] else "n",
        "--dhr", "y" if params["dhr"] else "n",
        "--fault", "y" if params["fault"] else "n"
    ]
    
    rprint("\n[bold green]Ready to start training with the following command:[/bold green]")
    rprint(f"[blue]{' '.join(cmd)}[/blue]\n")
    
    if questionary.confirm("Start now?").ask():
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            rprint(f"[bold red]Training failed with exit code {e.returncode}[/bold red]")
    else:
        rprint("[bold yellow]Command shown above. You can run it manually later.[/bold yellow]")

if __name__ == "__main__":
    display_banner()
    try:
        if questionary.confirm("Proceed with interactive setup?").ask():
            params = get_interactive_params()
            run_training(params)
        else:
            rprint("[bold yellow]Running with default parameters as requested...[/bold yellow]")
            # Default parameters matching the guide
            
            # Detect defaults again for non-interactive mode
            train_dir = "train/dataset3/data_train"
            valid_dir = "train/dataset3/data_valid"
            ntrain_default, n_default = get_default_dataset_info(train_dir, default_n=64, default_count=1)
            nvalid_default, _ = get_default_dataset_info(valid_dir, default_n=n_default, default_count=1)

            default_params = {
                "dir_data_train": train_dir,
                "dir_target_train": "train/dataset3/target_train",
                "dir_data_valid": valid_dir,
                "dir_target_valid": "train/dataset3/target_valid",
                "dir_output": "result3_infer",
                "n1": str(n_default), "n2": str(n_default), "n3": str(n_default),
                "ntrain": str(ntrain_default),
                "nvalid": str(nvalid_default),
                "epochs": "100",
                "batch_train": "4",
                "use_gpu": True,
                "rgt": True,
                "dhr": True,
                "fault": True
            }
            run_training(default_params)
    except KeyboardInterrupt:
        rprint("\n[bold red]Interrupted by user.[/bold red]")
        sys.exit(0)
