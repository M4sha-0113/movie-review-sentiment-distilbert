#### Changing this code to universal 
import pandas as pd
import os
from pathlib import Path
from .utils.text_cleaning import clean_text

def preprocess_data(path_to_csv, review_column, label_column, pos_label, neg_label):
    """
    Cleans and splits review data into positive and negative sets.

    Args:
        path_to_csv (str): The file path to the raw CSV data.
        review_column (str): The name of the column containing the text.
        label_column (str): The name of the column containing the labels(negative or positive meaning).
        pos_label (str): The value in the label column that indicates a positive review.
        neg_label (str): The value in the label column that indicates a negative review.


    Returns:
        None: Saves files (positive and negative CSV) directly to the 'clean_data' directory.
    """
    if type (review_column) == int: 
        df = pd.read_csv(path_to_csv, header=None).dropna(subset=[review_column]) #if there is no header in the csv file, we need to specify that
    else:
        df = pd.read_csv(path_to_csv).dropna(subset=[review_column]) #removing rows with NaN review content

    # 1. Create a filter for positive and negative reviews
    pos_reviews = df[df[label_column] == pos_label].copy()
    neg_reviews = df[df[label_column] == neg_label].copy()

    #2. Clean the review content.
    pos_reviews[review_column] = pos_reviews[review_column].apply(clean_text)
    neg_reviews[review_column] = neg_reviews[review_column].apply(clean_text)

    # 3. Save them all in the right folder
    original_path = Path(path_to_csv)
    new_path = original_path.parent.parent / "clean_data"
    new_path.mkdir(parents=True, exist_ok=True)
    file1_to_save = new_path / "pos_data.csv"
    file2_to_save = new_path / "neg_data.csv"
    # If appending, use header=False
    pos_reviews.to_csv(file1_to_save, index=False)
    neg_reviews.to_csv(file2_to_save, index=False)