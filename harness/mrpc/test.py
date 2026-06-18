import json
import torch
from model import get_model, get_tokenizer


def test_model(model, tokenizer, dataset, device="cpu"):
    model.eval()
    correct = 0
    total = len(dataset)

    with torch.no_grad():
        for row in dataset:
            encoding = tokenizer(
                row["sentence1"],
                row["sentence2"],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}
            logits = model(**encoding).logits[0]
            predicted = 0 if logits[0].item() > logits[1].item() else 1
            if predicted == row["label"]:
                correct += 1

    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f"Accuracy on test data: {accuracy:.2f}%")
    return accuracy


def predict(samples_file, predictions_file="predictions.txt", device="cpu"):
    """
    Load the fine-tuned MRPC model and make predictions on a JSONL sentence-pair file.

    Args:
        samples_file (str): Path to JSONL file with sentence1 and sentence2 fields.
        predictions_file (str): Output file for predictions (one integer per line).
        device (str): Device to run inference on ('cpu' or 'cuda').

    Returns:
        list: Predicted integer labels.
    """
    tokenizer = get_tokenizer()
    model = get_model(device=device)
    model.eval()

    samples = []
    with open(samples_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    predictions = []
    with torch.no_grad():
        for sample in samples:
            encoding = tokenizer(
                sample["sentence1"],
                sample["sentence2"],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}
            logits = model(**encoding).logits[0]
            label = 0 if logits[0].item() > logits[1].item() else 1
            predictions.append(label)

    with open(predictions_file, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    return predictions
