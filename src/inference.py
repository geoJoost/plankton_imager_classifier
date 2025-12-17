
import fastai
from fastai.vision.all import *
import pandas as pd
import os
from pathlib import Path
import torch
import numpy as np
import glob
import time
from datetime import timedelta
import sys
import tarfile, tempfile
import random

# Custom modules
from src.remove_corrupted_files import process_corrupted_files
from src.utils import process_predictions_to_dataframe

def conduct_plankton_inference(SOURCE_BASE_DIR, MODEL_NAME, model_weights, TRAIN_DATASET, CRUISE_NAME, BATCH_SIZE, VOLUME, CLASSIFICATION_SUBSAMPLE=100):
    print(f"[INFO] Started inference...", flush=True)
    start_time = time.time()
    np.random.seed(42)

    # Set the device to use GPU if available, else fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device used: {device}")
    print(f'[INFO] FastAI version: {fastai.__version__}')

    # Define the model using FastAI
    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=RandomSplitter(),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=Resize(300, ResizeMethod.Pad, pad_mode='zeros'),
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
            pad_mode='zeros'),
            Normalize.from_stats(*imagenet_stats)]
    )
    dls = block.dataloaders(TRAIN_DATASET, bs=BATCH_SIZE, num_workers=1) # Note: on Windows set to 0; can silently fail on HPC systems
    learn = vision_learner(dls, resnet50, metrics=error_rate, pretrained=False)
    
    # Check for multiple GPUs and use DataParallel if available
    # NOTE: Not debugged; not recommended to use multiple GPU's without coding expertise
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Number of GPU's available: {torch.cuda.device_count()}")
        learn.model = torch.nn.DataParallel(learn.model)

    # Move the model to the appropriate device
    learn.model.to(device)
    learn.load(model_weights, weights_only=False)

    print(f"[INFO] Inference is conducted using the {MODEL_NAME} classifier")
    print(f'[INFO] This is Plankton Identifier version: \t{model_weights}')
    print(f'[INFO] The batch size is set at: \t\t{BATCH_SIZE}')
    print(f'[INFO] The loss function is: \t\t\t{learn.loss_func}')
    print(f"[INFO] Number of categories available: \t\t{len(learn.dls.vocab)}")
    print(f"[INFO] Available labels: {learn.dls.vocab}")

    # Define output locations for results and logs
    results_root = Path(f"data/{CRUISE_NAME}_results")
    results_dir = results_root / "raw"
    processed_dir = results_root / "processed"

    # Create all directories (parents=True ensures parent directories are created if needed)
    results_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Location for logs
    log_file_path = results_root / "error_log.txt"

    # In previous versions, iteration was done per date but created large output CSV's (>10GB pre day)
    # Hence, this 'uglier' code iterates over each timestamp
    # os.walk() is not used to prevent loading millions of filepaths into memory
    for date_dir in os.listdir(SOURCE_BASE_DIR):
        date_dir_path = os.path.join(SOURCE_BASE_DIR, date_dir)
        if not os.path.isdir(date_dir_path): # See README.md for folder structure
            continue

        for tar_file in sorted(os.listdir(date_dir_path)):
            if not tar_file.endswith('.tar'):
                continue

            tar_file_path = os.path.join(date_dir_path, tar_file)
            timestamp = tar_file.split('_')[-1].split('.')[0]  # Extract timestamp from tar file name
            date_str = date_dir

            # Check if the last digit is zero (i.e., ends with 0)
            if not timestamp.endswith('0'):  # e.g., "1554" would continue to the next iteration
                print(f"[INFO] Timestamp {timestamp} not a full 10-minute bin, skipping processing.")
                print("=================================================")
                continue

            # Check if the CSV already exists
            csv_filename = results_dir / f"{CRUISE_NAME}_{date_dir}_{timestamp}.csv"
            if csv_filename.exists():
                print(f"[INFO] CSV file already exists for {csv_filename}, skipping processing.")
                continue
            
            # Create temporary location to untar data into
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(tar_file_path, 'r') as tar:
                    try:
                        tar.extractall(path=temp_dir)
                        print(f"[INFO] Extracted {tar_file} to temporary directory")
                    except Exception as e:
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(f"[ERROR] Could not extract {tar_file}: {e}\n")
                        print(f"[ERROR] Could not extract {tar_file}: {e}")
                        print("=================================================")
                        continue

                    timestamp_path = temp_dir
                    print(f"[DEBUG] timestamp_path: {timestamp_path}")

                    # Retrieve all available images within the folder (including Background.tif files)
                    imgs = get_image_files(timestamp_path)
                    imgs.sort()

                    if len(imgs) == 0:
                        print(f"[WARNING] No images found in {tar_file}")
                        print("=================================================")
                        continue
                    print(f"[INFO] {len(imgs):,} images in {tar_file}")

                    # Subsample the images if subsample_percent is not 100%
                    if CLASSIFICATION_SUBSAMPLE < 100:
                        random.seed(42)

                        # Seperate Background.tif from original list
                        # This file is used as a 'hack' for downstream processes, like getting geodata and timestamps
                        background_img = [img for img in imgs if img.name == "Background.tif"]
                        plankton_imgs = [img for img in imgs if img.name != "Background.tif"]

                        # Randomly sample % of images
                        num_images_to_sample = int(len(imgs) * CLASSIFICATION_SUBSAMPLE / 100)
                        plankton_imgs = random.sample(plankton_imgs, num_images_to_sample)
                        imgs = background_img + plankton_imgs # Combine lists
                        print(f"[INFO] Randomly selected {CLASSIFICATION_SUBSAMPLE}% of images: {len(imgs):,} images")

                    try:
                        try:
                            # First try and make predictions on the given batch
                            dl = learn.dls.test_dl(imgs)
                            preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
                            print("[INFO] Made predictions")
                        except:
                            # If corrupted files are found, try to remove these files and re-try the predictions
                            print(f"\n\n[WARNING] Corrupted files found in {timestamp_path}")
                            
                            # Remove corrupted files
                            process_corrupted_files(imgs, timestamp, CRUISE_NAME)

                            # Since we removed several files, we have to reload the available images
                            imgs = get_image_files(timestamp_path)
                            imgs.sort()

                            # Remove any files already marked corrupted
                            corrupt_folder = Path(f"data/{CRUISE_NAME}_corrupted")
                            imgs = [f for f in imgs if not (corrupt_folder / f.name).exists()]

                            # Repeat steps, without corrupted files
                            dl = learn.dls.test_dl(imgs)
                            preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
                            print("[INFO] Made predictions")

                        # Create filename for the detailed CSV
                        csv_filename = results_dir / f"{CRUISE_NAME}_{date_str}_{timestamp}.csv"

                        # Process predictions into dataframes and save to CSV files
                        saved_files = process_predictions_to_dataframe(
                            imgs=imgs,
                            preds=preds,
                            label_numeric=label_numeric,
                            vocab=learn.dls.vocab,
                            cruise_name=CRUISE_NAME,
                            date_str=date_str,
                            time_str=timestamp,
                            timestamp_path=timestamp_path,
                            results_dir=results_dir,
                            processed_dir=processed_dir,
                            volume=VOLUME,
                            csv_filename=csv_filename,
                            tar_file_path=tar_file_path # Path to original .tar file
                        )

                    except Exception as e:
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(f"[ERROR] Error processing {tar_file}: {e}\n")
                        print(f"[ERROR] Error processing {tar_file}: {e}")
                        sys.exit(1) # Force code to stop

                    print(f"[INFO] Finished processing: {tar_file}")
                    print("=================================================")

    inference_time = time.time()
    elapsed_inference_time = timedelta(seconds=inference_time - start_time).total_seconds()
    print(f"[INFO] Inference completed in {elapsed_inference_time / 3600:.2f} hours.")

    return results_dir, processed_dir