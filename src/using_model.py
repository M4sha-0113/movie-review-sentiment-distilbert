import torch
from .utils.model_utils import load_model, load_tokenizer

def use_model(model_path="./stanford_dataset_train_results/checkpoint-4221", review_file="film_review.txt"):
    # 1. Load the model and tokenizer 
    model, device = load_model(model_path)
    tokenizer = load_tokenizer("distilbert-base-uncased")

    # 3. Test it on a movie review
    with open(review_file, "r") as f:
        review_text = f.read()

    inputs = tokenizer(review_text, truncation=True, padding="max_length",return_tensors="pt")

    # 4. Get the prediction
    with torch.no_grad():
        inputs = {key: value.to(device) for key, value in inputs.items()}
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()

    if predicted_class_id == 1:
        print("The system identified this review as POSITIVE.")
    else:
        print("The system identified this review as NEGATIVE.")