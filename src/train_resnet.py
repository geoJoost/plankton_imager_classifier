""" Script for training the PlanktoShare models based on Resnet18/ResNet50 architecture """

# PlanktoShare: Classification of zooplankton from the Plankton Imager (pi-10) using ResNet 
# The label names for groups are extracted from the folder names.

import fastai
from fastai.vision.all import *
import torch
import numpy as np
from pathlib import Path
import time

# Custom imports
from src.utils import save_data_visualizations,  save_evaluation_visualizations

## Hyperparameters ##
STAGE1_PARAMS = [ # Manually define grid search values
    # (lr_slice, epochs, suffix)
    (slice(9e-3), 1, '_stage1_TEST')
    (slice(9e-3), 20, 'stage1_run01'),
    (slice(9e-2), 20, 'stage1_run02'),
    (slice(6e-2), 20, 'stage1_run03'),
    (slice(5e-3), 20, 'stage1_run04'),
    (slice(10e-3), 20, 'stage1_run05'),
    (slice(9e-3), 50, 'stage1_run06'),
    (slice(9e-2), 50, 'stage1_run07'),
    (slice(6e-2), 50, 'stage1_run08'),
    (slice(4e-4), 20, 'stage1_run09'),
    (slice(7e-4), 20, 'stage1_run10'),
]

STAGE2_PARAMS = [
    # (lr_slice, epochs, suffix)
    (slice(1e-6, 1e-4), 20, 'stage2_01'),
    (slice(3e-6, 3e-4), 20, 'stage2_02'),
    (slice(3e-5, 3e-3), 20, 'stage2_03'),
    (slice(3e-7, 3e-5), 20, 'stage2_04'),
    (slice(10e-4, 10e-3), 10, 'stage2_05'),
    (slice(10e-4, 10e-3), 20, 'stage2_06'),
    (slice(10e-4, 10e-3), 10, 'stage2_07'),
    (slice(10e-4, 10e-3), 50, 'stage2_08'),
    (slice(10e-4, 10e-3), 20, 'stage2_09'),
    (slice(3e-7, 3e-5), 50, 'stage2_10')
]

## Setup ##
def create_run_dirs(MODEL_NAME: str) -> tuple[Path, Path]:
    """Create timestamped output directories for models and training images."""
    # Create new folder in /models/ to save .pth files
    # FastAI hard-codes the model part, so have to seperate this for re-use down the line
    timestamp = datetime.today().strftime('%Y%m%d_%H%M')
    run_name = f"{timestamp}_{MODEL_NAME}"

    models_dir = Path('models') / run_name
    images_dir = Path('train') / run_name
    models_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    return models_dir, images_dir

def build_dataloaders(TRAIN_DATASET: str, BATCH_SIZE: int):
    """Create the FastAI DataLoaders with augmentation and normalization."""
    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=Resize(300, ResizeMethod.Pad, pad_mode='zeros'),
        batch_tfms=[
            *aug_transforms(
                mult=1.0,
                do_flip=True,
                flip_vert=True,
                max_rotate=0.2,
                min_zoom=1.0,
                max_zoom=1.1,
                max_lighting=0.3,
                max_warp=0.1,
                p_affine=0.5,
                p_lighting=0.5,
                pad_mode='zeros'
            ),
            Normalize.from_stats(*imagenet_stats)
        ]
    )
    # num_workers must be 0 on Windows: https://github.com/fastai/fastai/issues/2899
    return block.dataloaders(TRAIN_DATASET, bs=BATCH_SIZE, num_workers=0)

def build_learner(dls, MODEL_TYPE: str, models_dir: Path):
    """Create and return a FastAI vision learner for the given architecture."""
    # FastAI loss default: FlattenedLoss of CrossEntropyLoss()
    arch = resnet18 if MODEL_TYPE == "ResNet18" else resnet50
    learn = vision_learner(dls, arch, metrics=error_rate, model_dir=models_dir)
    return learn


## Training ##
def train_model(learn, model_file: str, lr_slice, epochs: int, save_file: str,
                images_dir: Path, unfreeze: bool = False):
    """
    Train the model for one stage with given parameters.

    Args:
        learn:       FastAI Learner object.
        model_file:  Model filename to load as starting point.
        lr_slice:    Learning rate slice for fit_one_cycle.
        epochs:      Number of training epochs.
        save_file:   Filename to save the best model to.
        images_dir:  Directory to save loss plots.
        unfreeze:    Whether to unfreeze all layers before training (stage 2).
    """
    print(f"[INFO] Training | lr: {lr_slice} | epochs: {epochs} | output: {save_file}")
    start_time = time.time()

    learn.load(model_file)

    if unfreeze:
        learn.unfreeze()

    learn.fit_one_cycle(
        epochs,
        lr_slice,
        cbs=SaveModelCallback(monitor='valid_loss', with_opt=True, fname='TempBestModel')
    )

    learn.load('TempBestModel')
    learn.save(save_file)

    learn.recorder.plot_loss()
    plt.savefig(images_dir / f"{save_file}_losses.png")
    plt.close()

    elapsed = (time.time() - start_time) / 60
    print(f"[INFO] Completed {save_file} in {elapsed:.2f} minutes")


def run_stage(
        learn,
        stage_params: list, 
        start_model: str,
        MODEL_TYPE: str,
        MODEL_NAME: str, 
        images_dir: Path, 
        unfreeze: bool
    ) -> str:
    """
    Run a full grid search stage and return the best model filename.

    Args:
        learn:        FastAI Learner object.
        stage_params: List of (lr_slice, epochs, suffix) tuples.
        start_model:  Model filename to use as the starting point for each run.
        MODEL_TYPE:   Architecture name, used in output filenames.
        MODEL_NAME:   Model name, used in output filenames.
        images_dir:   Directory to save loss plots.
        unfreeze:     Whether to unfreeze layers (True for stage 2).

    Returns:
        Filename of the best model based on validation loss.
    """
    stage_models = {}

    for lr_slice, epochs, suffix in stage_params:
        model_file = f"{MODEL_TYPE}_{MODEL_NAME}{suffix}"
        train_model(learn, start_model, lr_slice, epochs, model_file, images_dir, unfreeze=unfreeze)

        # Read validation loss from last epoch: [last_epoch][train_loss, val_loss, err_rate]
        train_loss, val_loss, err_rate = learn.recorder.values[-1]
        print(f"[INFO] Model: {model_file} (val_loss: {val_loss:.4f})")
        stage_models[model_file] = val_loss

    best_model = min(stage_models, key=stage_models.get)
    print(f"[INFO] Best model: {best_model} (val_loss: {stage_models[best_model]:.4f})")
    return best_model


## Entry point ##
def train_resnet(MODEL_NAME: str, MODEL_TYPE: str, TRAIN_DATASET: str, BATCH_SIZE: int):
    np.random.seed(3)

    # Set the device to use GPU if available, else fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device used: {device}")
    print(f'[INFO] FastAI version: {fastai.__version__}')
    
    # Directories
    models_dir, images_dir = create_run_dirs(MODEL_NAME)

    # Data
    dls = build_dataloaders(TRAIN_DATASET, BATCH_SIZE)
    save_data_visualizations(dls, images_dir)

    # Learner
    learn = build_learner(dls, MODEL_TYPE, models_dir)
    print(f"[INFO] Model: {MODEL_NAME}")
    print(f"[INFO] Arch: {MODEL_TYPE}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] Loss: {learn.loss_func}")

    # Save pretrained weights as stage 1 starting point
    model_default = f'{MODEL_TYPE}_{MODEL_NAME}_pretrained'
    learn.save(model_default)

    # Learning rate finder for frozen model
    learn.lr_find()
    plt.savefig(models_dir / "lr_find_frozen.png")
    plt.close()

    # Stage 1: frozen backbone
    print("\n[INFO] Starting Stage 1 (frozen)...")
    best_stage1 = run_stage(learn, STAGE1_PARAMS, model_default, MODEL_TYPE, MODEL_NAME, images_dir, unfreeze=False)
    learn.load(best_stage1)
    learn.save(f"{best_stage1}_final")

    # Stage 2: unfrozen backbone
    print("\n[INFO] Starting Stage 2 (unfrozen)...")
    best_stage2 = run_stage(learn, STAGE2_PARAMS, best_stage1, MODEL_TYPE, MODEL_NAME, images_dir, unfreeze=True)
    learn.load(best_stage2)
    learn.save(f"{best_stage2}_final")

    # Evaluation
    save_evaluation_visualizations(learn, images_dir)