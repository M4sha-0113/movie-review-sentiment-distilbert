import sys
from pathlib import Path

# Get the absolute path of the directory containing the current file
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent

sys.path.append(str(parent_directory))

from src.using_model import use_model

if __name__ == "__main__":
    input = input("Enter the name of the review file (with extension: .txt): ")
    use_model(model_path="./stanford_dataset_train_results/checkpoint-4221", review_file=input)