import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from absl import app, flags
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))
from resnet20 import ResNet20
import train
import test

FLAGS = flags.FLAGS

# 1. Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 350 # Increased epochs for potentially better accuracy
MODEL_PATH = './harness/cifar10/cifar10_resnet20_model.pth'
RNG_SEED = 42 # for reproducibility
DATA_DIR='./harness/cifar10/data'

# Define command line flags safely to allow importing this module from other apps
try:
    flags.DEFINE_string('model_path', MODEL_PATH, 'Path to save/load the model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size for training and evaluation')
    flags.DEFINE_float('learning_rate', LEARNING_RATE, 'Learning rate for optimizer')
    flags.DEFINE_float('weight_decay', WEIGHT_DECAY, 'Weight decay for optimizer')
    flags.DEFINE_integer('epochs', EPOCHS, 'Number of training epochs')
    flags.DEFINE_string('data_dir', './harness/cifar10/data', 'Directory to store/load CIFAR10 dataset')
    flags.DEFINE_boolean('no_cuda', False, 'Disable CUDA even if available')
    flags.DEFINE_integer('seed', RNG_SEED, 'Random seed for reproducibility')

    flags.DEFINE_boolean('export_test_data', False, 'Export test dataset to file and exit')
    flags.DEFINE_string('test_data_output', 'cifar10_test.txt', 'Output file for exported test data')
    flags.DEFINE_integer('num_samples', -1, 'Number of samples to export (-1 for all samples)')

    flags.DEFINE_boolean('predict', False, 'Run prediction on pixels file and exit')
    flags.DEFINE_string('pixels_file', '', 'Path to file containing pixel data for prediction')
    flags.DEFINE_string('predictions_file', 'predictions.txt', 'Output file for predictions')
except flags.DuplicateFlagError:
    pass

# Ensure reproducibility
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)

# 2. Data Loading and Preprocessing
def get_cifar10_transform(transform_type="validation"):
    """
    Get the standard CIFAR10 transform for preprocessing.
    
    Returns:
        transforms.Compose: Transform pipeline for CIFAR10 data
    """
    if transform_type == "train":
        return transforms.Compose(
                [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])     
            ])
    else:
        return transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])     
            ])

def load_and_preprocess_data(batch_size=BATCH_SIZE, data_dir=DATA_DIR):
    """
    Load and preprocess CIFAR10 dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        data_dir (str): Directory to store/load dataset
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform = get_cifar10_transform(transform_type="train")
    test_transform = get_cifar10_transform()

    # Download CIFAR10 dataset. Suppress numpy VisibleDeprecationWarning
    # originating from torchvision's CIFAR pickle loader on some numpy versions.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=getattr(np, 'VisibleDeprecationWarning', DeprecationWarning))
        full_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    # Split training data into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
    

# 4. Training Function: See train.py
def train_model(model_path, batch_size, learning_rate, weight_decay, epochs, train_loader, val_loader, data_dir, device):
    """
    Build or load a ResNet20 model, train if necessary, and return the model.
    Mirrors the `mnist.train_model` signature and behavior.
    """
    channel_values = [16, 32, 64]
    num_classes = 10

    model = ResNet20(channel_values, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # If model exists, load it; otherwise train and save
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Train using the helper in harness/cifar10/train.py
        train.train_model_function(model, train_loader, criterion, optimizer, num_epochs=epochs, device=device)
        # Ensure directory exists when saving
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model
    

# 5. Testing Function: See test.py

def run_predict(model_path, pixels_file, predictions_file, device="cpu"):
    """
    Run prediction on the given pixel file using the specified model.
    """
    # If model file doesn't exist, train and save it
    if not os.path.exists(model_path):
        train_loader, val_loader, test_loader = load_and_preprocess_data(batch_size=BATCH_SIZE, data_dir=DATA_DIR)
        _ = train_model(model_path, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                    epochs=EPOCHS, train_loader=train_loader, val_loader=val_loader,
                    data_dir=DATA_DIR, device=device)

    # Determine saved model path
    saved_model_path = model_path
    if os.path.isdir(model_path):
        saved_model_path = os.path.join(model_path, 'cifar10_resnet20_model.pth') if not model_path.endswith('.pth') else model_path

    test.predict(pixels_file, saved_model_path, predictions_file, device=device)


def export_test_pixels_labels(data_dir=DATA_DIR, pixels_file="cifar10_pixels.txt", labels_file="cifar10_labels.txt", num_samples=-1, seed=None):
    """
    Export CIFAR10 test dataset to separate label and pixel files using random sampling.
    
    Args:
        data_dir (str): Directory to download dataset temporarily.
        pixels_file (str): Path to the output file for pixel values
        labels_file (str): Path to the output file for labels
        num_samples (int): Number of samples to export (-1 for all)
    """
    if seed is not None:
        np.random.seed(seed)

    print("Loading CIFAR-10 test data via torchvision...")
    transform = transforms.ToTensor()
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    total_samples = len(test_dataset)
    samples_to_export = total_samples if num_samples == -1 else min(num_samples, total_samples)

    # Use sample_test_data to get random samples (but without normalization for export)
    if samples_to_export == total_samples:

        with open(labels_file, 'w') as label_f, open(pixels_file, 'w') as pixel_f:
            for image, label in test_dataset:
                flattened_image = image.view(-1).numpy()
                label_f.write(f"{label}\n")
                pixel_values = " ".join(f"{pixel:.6f}" for pixel in flattened_image)
                pixel_f.write(f"{pixel_values}\n")
    else:
        # Generate random indices and Create a subset dataset using the random indices
        random_indices = torch.randperm(total_samples)[:samples_to_export]
        subset_dataset = torch.utils.data.Subset(test_dataset, random_indices)
        subset_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False)
        
        with open(labels_file, 'w') as label_f, open(pixels_file, 'w') as pixel_f:
            for batch_images, batch_labels in subset_loader:
                for image, label in zip(batch_images, batch_labels):
                    flattened_image = image.view(-1).numpy()
                    label_f.write(f"{label.item()}\n")
                    pixel_values = " ".join(f"{pixel:.6f}" for pixel in flattened_image)
                    pixel_f.write(f"{pixel_values}\n")


def export_test_data(data_dir=DATA_DIR, output_file='cifar10_test.txt', num_samples=-1, seed=None):
    """
    Export CIFAR10 test dataset to separate label and pixel files using random sampling.
    
    Args:
        data_dir (str): Directory to load dataset from
        output_file (str): Base output file path (will create .labels and .pixels files)
        num_samples (int): Number of samples to export (-1 for all)
    """
    # Create separate file names for labels and pixels
    base_name = str(output_file).rsplit('.', 1)[0] if '.' in str(output_file) else str(output_file)
    labels_file = f"{base_name}_labels.txt"
    pixels_file = f"{base_name}_pixels.txt"
    export_test_pixels_labels(data_dir=data_dir, pixels_file=pixels_file, labels_file=labels_file, num_samples=num_samples, seed=seed)



def main(argv):
    # Check if we should just export test data and exit
    if FLAGS.export_test_data:
        print("Export mode: Loading and exporting test data...")
        export_test_data(data_dir=FLAGS.data_dir, output_file=FLAGS.test_data_output, num_samples=FLAGS.num_samples)
        print("Export completed. Exiting.")
        return
    
    use_cuda = not FLAGS.no_cuda and torch.cuda.is_available()
    random_seed = FLAGS.seed
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)

    if use_cuda:
        torch.cuda.manual_seed_all(random_seed)
    device = "cuda" if use_cuda else "cpu"
    # Train the model.
    train_loader, val_loader, test_loader = load_and_preprocess_data(batch_size=FLAGS.batch_size, data_dir=FLAGS.data_dir)
    model = train_model(FLAGS.model_path, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.weight_decay,
            FLAGS.epochs, train_loader, val_loader, data_dir=FLAGS.data_dir, device=device)

    # Check if we should run prediction and exit
    if FLAGS.predict:
        if not FLAGS.pixels_file:
            print("Error: pixels_file must be specified when using --predict flag")
            return
        print("Prediction mode: Running inference on provided pixel data...")
        run_predict(FLAGS.model_path, FLAGS.pixels_file, FLAGS.predictions_file, device=device)
        print("Prediction completed. Exiting.")
        return
    else:
        # Testing the model
        print(f"\nEvaluating model on test data...")
        test.test_model(model, test_loader, device)

if __name__ == '__main__':
    app.run(main)
