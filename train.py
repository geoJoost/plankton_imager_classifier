"""New script for setting up the training of new models"""
import argparse

# Custom imports
from src.utils import analyze_tif_files, plot_category_examples
from src.train_resnet import train_resnet

def train(MODEL_NAME, MODEL_TYPE, TRAIN_DATASET, BATCH_SIZE):
    # Plot simple figure describing the training set
    # TODO: Add print statement of dataframe
    analyze_tif_files(TRAIN_DATASET)

    # Plot examples of each class
    plot_category_examples(TRAIN_DATASET, output_path='doc/train_example.png')

    # Execute training for ResNet18/ResNet50 (default)
    train_resnet(MODEL_NAME, MODEL_TYPE, TRAIN_DATASET, BATCH_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a ResNet50 model for plankton classification."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="PlanktoShare",
        help="Name of the model"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ResNet50",
        choices=["ResNet18", "ResNet50"],
        help="Model architecture to use (default: ResNet50)"
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="data/DETAILED_merged_sample",
        help="Path to the training dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )

    args = parser.parse_args()

    # Print run configuration
    kwargs = vars(args)
    print("\n" + "=" * 60)
    print("[INFO] Run configuration")
    print("=" * 60)
    for key, value in kwargs.items():
        print(f"{key:<30} {value}")
    print("=" * 60 + "\n")

    # Train model
    train(
        MODEL_NAME=args.model_name,
        MODEL_TYPE=args.model_type,
        TRAIN_DATASET=args.train_dataset,
        BATCH_SIZE=args.batch_size,
    )

    print(f"[INFO] Finished training...")
