from transformers import AutoModelForSequenceClassification, Trainer
import numpy as np
from .utils.data_loader import load_data
from .utils.model_utils import load_saved_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(path_pos, path_neg, text_column="review_content", model_dir="./stanford_dataset_train_results/checkpoint-4221",output_file="test_results.txt"):
    """
    Evaluates the DistilBERT model on a chosen test dataset and saves the results to a file.

    Args:
        path_pos (str): The file path to the positive reviews CSV.
        path_neg (str): The file path to the negative reviews CSV.
        text_column (str): The name of the column containing the review text. Default is "review_content".
        model_dir (str): The directory path to the saved model checkpoint. Default is "./stanford_dataset_train_results/checkpoint-4221".
        output_file (str): The name of the output file where results are saved. Default is "test_results.txt".

    Returns:
        None: Saves the evaluation results to the specified output file.
    """

    # Load the model from the final checkpoint
    model = load_saved_model(model_dir)[0]
    
    # Load and prepare the data
    tokenized_data = load_data(path_pos, path_neg, type="test", text_column=text_column)
    
    # Define metrics calculation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        acc = accuracy_score(labels, predictions)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    
    # Initialize a Trainer just for evaluation
    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    
    # Run evaluation
    metrics = trainer.evaluate(tokenized_data)
    print("Test Results:", metrics)
    with open(output_file, "w") as f:
        f.write(str(metrics))
