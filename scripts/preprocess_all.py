import sys
from pathlib import Path

# Get the absolute path of the directory containing the current file
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent

sys.path.append(str(parent_directory))

from src.preprocessing import preprocess_data


if __name__ == "__main__":
    # Preprocess all datasets

    ###5k subset from stanford dataset for quick testing and debugging, at initial stages of development. Comment this line later when we want to preprocess the full dataset.
    #preprocess_data("../datasets/stanford/train/raw_data/combined_data_slice.csv", "review", "sentiment", "positive", "negative")
    #preprocess_data("../datasets/stanford/test/raw_data/combined_data_slice.csv", "review", "sentiment", "positive", "negative")

    ### full datasets from stanford, imdb, rotten tomatoes and amazon.
    preprocess_data("../datasets/stanford/test/raw_data/combined_data.csv", "review", "sentiment", "positive", "negative")
    preprocess_data("../datasets/stanford/train/raw_data/combined_data.csv", "review", "sentiment", "positive", "negative")
    
    preprocess_data("../datasets/imdb/raw_data/IMDB Dataset.csv", "review", "sentiment", "positive", "negative")
    preprocess_data("../datasets/rotten_tomatoes/raw_data/rotten_tomatoes_critic_reviews_subset.csv", "review_content", "review_type", "Fresh", "Rotten")
    preprocess_data("../datasets/amazon/raw_data/amazon_test.csv", 2, 0, 2, 1)