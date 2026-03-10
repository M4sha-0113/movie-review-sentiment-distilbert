from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from .utils.data_loader import load_data
from .utils.model_utils import load_model

def train_model(path_pos, path_neg, output_dir="./init_results",text_column="review", type="train"):
    """
    Trains a DistilBERT model for sequence classification using the provided positive and negative review datasets.
    Args:
        path_pos (str): The file path to the positive reviews CSV.
        path_neg (str): The file path to the negative reviews CSV.
        type (str): The type of dataset to load ("train" or "test"). Default is "train".
        text_column (str): The name of the column containing the review text. Default is "review_content".
        output_dir (str): The directory where training outputs and checkpoints will be saved. Default is "./init_results".

    Returns:
        None: Saves training outputs and checkpoints to the specified output directory.
    """
    # Load and prepare the data
    tokenized_data = load_data(path_pos, path_neg, type=type, text_column=text_column)

    # Load the DistilBERT model for sequence classification
    model = load_model("distilbert-base-uncased", 2)[0] # We only need the model, not the device
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",      # Evaluate at the end of every epoch
        learning_rate=2e-5,          # Standard BERT fine-tuning rate
       per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
       model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()
