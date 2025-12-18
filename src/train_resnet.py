# Plankton Identifier - Classification from plankton imager
# This script provides code to train and run a CNN to classify plankton from images from the plankton imager.
# It treats this problem as a classification task. The labels are extracted from the folder names.

# Import modules and set parameters
import fastai
from fastai.vision.all import *
import torch
import numpy as np
from pathlib import Path
import time

# Custom imports
from src.utils import save_data_visualizations,  save_evaluation_visualizations

def train_resnet50(MODEL_NAME, MODEL_TYPE, TRAIN_DATASET, BATCH_SIZE):
    np.random.seed(3)

    # Create new folder in /models/ to save .pth files
    # FastAI hard-codes the model part, so have to seperate this for re-use down the line
    timestamp = datetime.today().strftime('%Y%m%d_%H%M')
    models_root = f"{timestamp}_{MODEL_NAME}" # Use today's date for future note keeping
    os.makedirs(os.path.join('models', models_root), exist_ok=True)

    images_root = os.path.join('train', models_root)
    os.makedirs(images_root, exist_ok=True)

    # Set the device to use GPU if available, else fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device used: {device}")
    print(f'[INFO] You are using FastAI version: {fastai.__version__}')

    # Create dataset
    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # for regression, change this CategoryBlock
        splitter=RandomSplitter(valid_pct=0.2, seed=42), # 80-20 split; added seed
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=Resize(300, ResizeMethod.Pad, pad_mode='zeros'),  # see page 73 book
        batch_tfms=[*aug_transforms(
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
    dls = block.dataloaders(TRAIN_DATASET, bs=BATCH_SIZE, num_workers=0) # num_workers NEEDS to be zero for the code to work on Windows; see https://github.com/fastai/fastai/issues/2899

    # Create various data visualizations for context
    save_data_visualizations(dls, images_root)

    # Create Learner
    if MODEL_TYPE == "ResNet18":
        learn = vision_learner(dls,resnet18, metrics=error_rate)  # creates pretrained model
    else:
        # Defaults to ResNet50 architecture
        learn = vision_learner(dls, 
                               resnet50, 
                               metrics=error_rate,
                               loss_func=CrossEntropyLossFlat,#Flat(weight=weights_tensor)
                               )  # creates pretrained model

    # learn.model.to(device)
    print(f'[INFO] This is Plankton Identifier version: {MODEL_NAME}')
    print(f'[INFO] Training new models using {MODEL_TYPE} architecture')
    print(f'[INFO] The batchsize is set at: {BATCH_SIZE}')
    print(f'[INFO] The loss function is: {learn.loss_func}')  # Double check current loss func

    # Save pretrained model
    model_default = os.path.join(models_root, f'{MODEL_TYPE}_{MODEL_NAME}_stage-1_00')
    learn.save(model_default) # Saves pretrained model, for repetitive trials

    # LR finder for frozen model
    learn.lr_find()
    plt.savefig(os.path.join("models", models_root, "lr_find_frozen.png"))
    plt.close()

    def train_model(model_file, lr_slice, epochs, save_file, images_root, unfreeze=False):
        """
        Train a model with given parameters.

        Args:
            model_file (str): Path to the model file to load.
            lr_slice (slice): Learning rate slice.
            epochs (int): Number of epochs to train.
            save_file (str): Path to save the trained model.
            unfreeze (bool): Whether to unfreeze the model before training.
        """
        print(f"[INFO] Started training with settings: \nLearning rate: {lr_slice}\nNumber of epochs: {epochs}\nOutput: {save_file}")
        start_time = time.time()

        # Load pre-trained (phase-1) OR best model from phase-1
        learn.load(model_file)

        if unfreeze: # For phase-2
            learn.unfreeze()

        # Perform one cycle learning
        learn.fit_one_cycle(epochs, lr_slice, cbs=SaveModelCallback(monitor='valid_loss', with_opt=True, fname='TempBestModel'))
        
        # Update the current best performing model
        learn.load('TempBestModel')
        learn.save(save_file)

        # Save losses as figure
        learn.recorder.plot_loss()
        plt.savefig(os.path.join(images_root, f"{save_file}_losses.png"))
        plt.close()

        print(f"[INFO] Model {save_file} training completed in {(time.time() - start_time) / 60:.2f} minutes")

    # Stage 1 training loops
    print("[INFO] Starting Stage 1 training...\n")
    stage1_params = [ # Manually define grid search values
        # (slice(9e-3), 1, '_stage1_TEST'),
        (slice(9e-3), 20, '_stage1_run01'),
        (slice(9e-2), 20, '_stage1_run02'),
        (slice(6e-2), 20, '_stage1_run03'),
        (slice(5e-3), 20, '_stage1_run04'),
        (slice(10e-3), 20, '_stage1_run05'),
        (slice(9e-3), 50, '_stage1_run06'),
        (slice(9e-2), 50, '_stage1_run07'),
        (slice(6e-2), 50, '_stage1_run08'),
        (slice(4e-4), 20, '_stage1_run09'),
        (slice(7e-4), 20, '_stage1_run10'),
    ]
    stage1_models = {} # Save losses to find the most suited model

    for lr_slice, epochs, suffix in stage1_params:
        model_file = os.path.join(models_root, f"{MODEL_TYPE}_{MODEL_NAME}_{suffix}")
        train_model(model_default, lr_slice, epochs, model_file, images_root, unfreeze=False)

        # Load the model to get its validation loss
        learn.load(model_file)
        train_loss, val_loss, err_rate = learn.recorder.values[0] # Selects only final values
        stage1_models[model_file] = val_loss

    # Select the best stage-1 model based on validation loss
    best_stage1_model = min(stage1_models, key=stage1_models.get)
    print(f"[INFO] From stage-1, the best model is: {best_stage1_model}")

    # Load the best performing stage-1 model, rename, and save it
    learn.load(best_stage1_model)
    learn.save(f"{best_stage1_model}_final")

    # Stage 2 training loops
    # Again, manually define hyperparameters
    stage2_params = [
        # (slice(9e-3), 1, '_stage2_TEST'),
        (slice(1e-6, 1e-4), 20, '_stage2_01'),
        (slice(3e-6, 3e-4), 20, '_stage2_02'),
        (slice(3e-5, 3e-3), 20, '_stage2_03'),
        (slice(3e-7, 3e-5), 20, '_stage2_04'),
        (slice(10e-4, 10e-3), 10, '_stage2_05'),
        (slice(10e-4, 10e-3), 20, '_stage2_06'),
        (slice(10e-4, 10e-3), 10, '_stage2_07'),
        (slice(10e-4, 10e-3), 50, '_stage2_08'),
        (slice(10e-4, 10e-3), 20, '_stage2_09'),
        (slice(3e-7, 3e-5), 50, '_stage2_10')
    ]

    stage2_models = {} # Save losses to find the final model

    for lr_slice, epochs, suffix in stage2_params:
        model_file = MODEL_TYPE + MODEL_NAME + suffix
        # Note1: Default model to start from is now the best performing model from first stage
        # Note2: Model is unfrozen now
        train_model(best_stage1_model, lr_slice, epochs, model_file, images_root, unfreeze=True)

        # Load the model to get its validation loss
        learn.load(model_file)
        train_loss, val_loss, err_rate = learn.recorder.values[0]
        stage2_models[model_file] = val_loss

    # Select the best model based on validation loss
    best_stage2_model = min(stage2_models, key=stage2_models.get)
    print(f"[INFO] From stage-2, the best model is: {best_stage2_model}")

    # Load the most suited model, rename, and save it
    learn.load(best_stage2_model)
    learn.save(f"{best_stage2_model}_final")

    # Evaluation
    save_evaluation_visualizations(learn, images_root)