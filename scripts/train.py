import sys
from pathlib import Path

# Get the absolute path of the directory containing the current file
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent

sys.path.append(str(parent_directory))


from src.training import train_model

if __name__ == "__main__":

    ###initial training on 5k subset from stanford dataset for quick testing and debugging, at initial stages of development. Comment this line later when we want to train on the full dataset.
    #train_model("../datasets/stanford/train/clean_data/pos_data.csv", "../datasets/stanford/train/clean_data/neg_data.csv", output_dir="./stanford_5k_dataset_train_results", text_column="review", type="train")

    ### full training on stanford dataset.
    train_model("../datasets/stanford/train/clean_data/pos_data.csv", "../datasets/stanford/train/clean_data/neg_data.csv", output_dir="./stanford_dataset_train_results", text_column="review", type="train")
