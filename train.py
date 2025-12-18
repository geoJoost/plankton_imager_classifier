"""New script for setting up the training of new models"""

# Custom imports
from src.utils import analyze_tif_files, plot_category_examples
from src.train_resnet import train_resnet50

def train(MODEL_NAME, MODEL_TYPE, TRAIN_DATASET, BATCH_SIZE):
    # Plot simple figure describing the training set
    # TODO: Add print statement of dataframe
    analyze_tif_files(TRAIN_DATASET)

    # Plot examples of each class
    plot_category_examples(TRAIN_DATASET, output_path='doc/train_example.png')

    # Execute training for ResNet50
    train_resnet50(MODEL_NAME, MODEL_TYPE, TRAIN_DATASET, BATCH_SIZE)

    # Space for other model implementations


if __name__ == "__main__":
    MODEL_NAME = "PlanktoFAIR_WeightedLoss"
    MODEL_TYPE = "ResNet50"
    TRAIN_DATASET = "data/DETAILED_merged"
    BATCH_SIZE = 256 # 128 for 16G

    # Execute training regime
    train(MODEL_NAME, MODEL_TYPE, TRAIN_DATASET, BATCH_SIZE)
    print(f"[INFO] Finished training...")
