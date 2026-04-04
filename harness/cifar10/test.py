import os
import torch
from resnet20 import ResNet20
import numpy as np


def test_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f'Accuracy on test data: {accuracy:.2f}%')
    return accuracy


def predict(pixels_file, model_path="cifar10_resnet20_model.pth", predictions_file='predictions.txt', device='cpu'):
    """
    Load a trained ResNet20 and make predictions on pixel data from a file.

    Each line in `pixels_file` should contain 3072 float values (3*32*32), space-separated.
    """
    device = torch.device(device)

    channel_values = [16, 32, 64]
    num_classes = 10
    model = ResNet20(channel_values, num_classes).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # CIFAR normalization
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

    pixel_data = []
    with open(pixels_file, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split() if x]
            if len(vals) != 3 * 32 * 32:
                raise ValueError(f"Each line must contain exactly 3072 pixel values, got {len(vals)}")
            pixel_data.append(vals)

    if not pixel_data:
        return []

    tensors = []
    for vals in pixel_data:
        arr = np.asarray(vals, dtype=np.float32).reshape((3, 32, 32))
        for c in range(3):
            arr[c] = (arr[c] - mean[c]) / std[c]
        tensors.append(torch.from_numpy(arr))

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        outputs = model(batch)
        _, predicted = torch.max(outputs, 1)
        predictions = predicted.cpu().numpy().tolist()

    with open(predictions_file, 'w') as out:
        for p in predictions:
            out.write(f"{p}\n")

    return predictions
