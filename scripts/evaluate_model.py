import sys
from pathlib import Path

# Get the absolute path of the directory containing the current file
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent

sys.path.append(str(parent_directory))


from src.evaluation import evaluate_model

if __name__ == "__main__":
    ###initial evaluation on 5k subset from stanford dataset for quick testing and debugging, at initial stages of development. Comment this line later when we want to evaluate on the full dataset.
    #evaluate_model("../datasets/stanford/test/clean_data/pos_data.csv", "../datasets/stanford/test/clean_data/neg_data.csv", text_column="review", model_dir="./stanford_5k_dataset_train_results/checkpoint-846", output_file="stanford_5k_test_results.txt")

    ### full evaluation on stanford dataset.
    #evaluate_model("../datasets/stanford/test/clean_data/pos_data.csv", "../datasets/stanford/test/clean_data/neg_data.csv", text_column="review", model_dir="./stanford_dataset_train_results/checkpoint-4221", output_file="stanford_test_results.txt")
     #evaluate_model("../datasets/stanford/test/clean_data/pos_data.csv", "../datasets/stanford/test/clean_data/neg_data.csv", text_column="review", model_dir="distilbert-base-uncased", output_file="pre-trained-model_test_results.txt")
    ###other datasets
    # evaluate_model("../datasets/imdb/clean_data/pos_data.csv", "../datasets/imdb/clean_data/neg_data.csv", text_column="review", model_dir="./stanford_dataset_train_results/checkpoint-4221", output_file="imdb_test_results.txt")
    # evaluate_model("../datasets/rotten_tomatoes/clean_data/pos_data.csv", "../datasets/rotten_tomatoes/clean_data/neg_data.csv", text_column="review_content", model_dir="./stanford_dataset_train_results/checkpoint-4221", output_file="25k_rotten_tomatoes_test_results.txt")
    # evaluate_model("../datasets/amazon/clean_data/pos_data.csv", "../datasets/amazon/clean_data/neg_data.csv", text_column="2", model_dir="./stanford_dataset_train_results/checkpoint-4221", output_file="25k_amazon_test_results.txt")

    ### then comment what we just ran, make the preprocessing of other datasets and uncomment the lines below to evaluate on the full datasets from rotten tomatoes and amazon.
    
    evaluate_model("../datasets/rotten_tomatoes/clean_data/pos_data.csv", "../datasets/rotten_tomatoes/clean_data/neg_data.csv", text_column="review_content", model_dir="./stanford_dataset_train_results/checkpoint-4221", output_file="otten_tomatoes_test_results.txt")
    evaluate_model("../datasets/amazon/clean_data/pos_data.csv", "../datasets/amazon/clean_data/neg_data.csv", text_column="2", model_dir="./stanford_dataset_train_results/checkpoint-4221", output_file="amazon_test_results.txt")

