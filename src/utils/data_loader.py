from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from .model_utils import load_tokenizer

def load_data(path_pos, path_neg, type="test", text_column="review_content"):
    """
    Loads and prepares the data for training or evaluation.

    Args:
        path_pos (str): The file path to the positive reviews CSV.
        path_neg (str): The file path to the negative reviews CSV.
        type (str): The type of dataset to prepare ("train" or "test"). Default is "test".
        text_column (str): The name of the column containing the review text. Default is "review_content".

    Returns:
        dict: A dictionary containing the tokenized datasets. For "train", it includes "train" and "test" splits. For "test", it includes only the "test" split.

    """
    # Load the files
    data_pos = load_dataset("csv", data_files={type: path_pos})
    data_neg = load_dataset("csv", data_files={type: path_neg})

    # Add labels
    data_pos = data_pos.map(lambda x: {"label": 1})
    data_neg = data_neg.map(lambda x: {"label": 0})

    # Combine into one flat dataset
    combined_dataset = concatenate_datasets([data_pos[type], data_neg[type]])

    if type == "train":
        #Shuffle and Split (creating a 10% validation set)
        combined_dataset = combined_dataset.shuffle(seed=42).train_test_split(test_size=0.1) 
        # Now I can access them as combined_dataset["train"] and combined_dataset["test"]
    
    # Tokenize the data
    tokenizer = load_tokenizer("distilbert-base-uncased")
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding="max_length")
    # Map the tokenizer over the data
    tokenized_data = combined_dataset.map(tokenize_function, batched=True)
    return tokenized_data

