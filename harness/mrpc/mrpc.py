import json
import os
import random
import sys
import torch
from pathlib import Path
from datasets import load_dataset
from absl import app, flags

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

import test
from model import get_model, get_tokenizer

FLAGS = flags.FLAGS

# 1. Configuration
DATA_DIR = './harness/mrpc/data'
MRPC_VALIDATION_SIZE = 408
RNG_SEED = 42  # for reproducibility

# Define command line flags safely to allow importing this module from other apps
try:
    flags.DEFINE_boolean('no_cuda', False, 'Disable CUDA even if available')
    flags.DEFINE_integer('seed', RNG_SEED, 'Random seed for reproducibility')
    flags.DEFINE_string('data_dir', DATA_DIR, 'Cache directory for GLUE MRPC dataset')

    flags.DEFINE_boolean('export_test_data', False, 'Export validation dataset to file and exit')
    flags.DEFINE_string('test_data_output', 'mrpc_test.txt', 'Output file for exported test data')
    flags.DEFINE_integer('num_samples', -1, 'Number of samples to export (-1 for all)')

    flags.DEFINE_boolean('predict', False, 'Run prediction on samples file and exit')
    flags.DEFINE_string('samples_file', '', 'Path to JSONL file containing sentence pairs for prediction')
    flags.DEFINE_string('predictions_file', 'predictions.txt', 'Output file for predictions')
except flags.DuplicateFlagError:
    pass


# Ensure reproducibility
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)


# 2. Data Loading and Preprocessing
def load_and_preprocess_data(data_dir=DATA_DIR):
    """
    Load all MRPC splits from GLUE.

    Args:
        data_dir (str): Cache directory for the GLUE MRPC dataset.

    Returns:
        tuple: (train_dataset, validation_dataset, test_dataset)
    """
    dataset = load_dataset("nyu-mll/glue", "mrpc", cache_dir=str(data_dir))
    return dataset["train"], dataset["validation"], dataset["test"]


# 3. Model Definition: See model.py

# 4. Testing Function: See test.py


# Function to export test data to separate files.
def export_test_sentence_pairs(data_dir=DATA_DIR, samples_file="mrpc_samples.jsonl", labels_file="mrpc_labels.txt", num_samples=-1, seed=None):
    """
    Export MRPC validation samples to separate input and label files.

    Args:
        data_dir (str): Cache directory for downloading the GLUE MRPC dataset.
        samples_file (str): Path to the output JSONL file for sentence pairs.
        labels_file (str): Path to the output file for labels.
        num_samples (int): Number of samples to export (-1 for all).
        seed (int): Random seed for reproducible sampling.
    """
    _, validation_dataset, _ = load_and_preprocess_data(data_dir)
    total = len(validation_dataset)

    samples_to_export = total if (num_samples == -1 or num_samples >= total) else num_samples

    if samples_to_export == total:
        indices = list(range(total))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(total), samples_to_export))

    with open(labels_file, 'w') as lf, open(samples_file, 'w') as sf:
        for i in indices:
            row = validation_dataset[i]
            sf.write(json.dumps({
                "target_idx": row["idx"],
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
            }) + "\n")
            lf.write(f"{row['label']}\n")


def export_test_data(data_dir=DATA_DIR, output_file='mrpc_test.txt', num_samples=-1, seed=None):
    """
    Export MRPC validation dataset to JSONL and label files.

    Args:
        data_dir (str): Cache directory for downloading the GLUE MRPC dataset.
        output_file (str): Base output file path (will create _samples.jsonl and _labels.txt files).
        num_samples (int): Number of samples to export (-1 for all).
        seed (int): Random seed for reproducible sampling.
    """
    base_name = str(output_file).rsplit('.', 1)[0] if '.' in str(output_file) else str(output_file)
    labels_file = f"{base_name}_labels.txt"
    samples_file = f"{base_name}_samples.jsonl"
    export_test_sentence_pairs(
        data_dir=data_dir,
        samples_file=samples_file,
        labels_file=labels_file,
        num_samples=num_samples,
        seed=seed,
    )


def run_predict(input_file, predictions_file, device="cpu"):
    """
    Run prediction on the given JSONL input file using the fine-tuned MRPC model.

    Args:
        input_file (str): Path to JSONL file containing sentence pairs (sentence1, sentence2).
        predictions_file (str): Output file for predictions.
        device (str): Device to run inference on (default: cpu).
    """
    output_path = Path(predictions_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test.predict(input_file, predictions_file, device=device)


def main(argv):
    # Check if we should just export test data and exit
    if FLAGS.export_test_data:
        print("Export mode: Loading and exporting validation data...")
        export_test_data(
            data_dir=FLAGS.data_dir,
            output_file=FLAGS.test_data_output,
            num_samples=FLAGS.num_samples,
            seed=FLAGS.seed,
        )
        print("Export completed. Exiting.")
        return

    use_cuda = not FLAGS.no_cuda and torch.cuda.is_available()
    random_seed = FLAGS.seed
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(random_seed)
    device = "cuda" if use_cuda else "cpu"

    # Check if we should run prediction and exit
    if FLAGS.predict:
        if not FLAGS.samples_file:
            print("Error: samples_file must be specified when using --predict flag")
            return
        print("Prediction mode: Running inference on provided samples file...")
        run_predict(FLAGS.samples_file, FLAGS.predictions_file, device=device)
        print("Prediction completed. Exiting.")
        return
    else:
        # Evaluate model on validation data
        print("Evaluation mode: Running inference on full MRPC validation set...")
        _, validation_dataset, _ = load_and_preprocess_data(FLAGS.data_dir)
        tokenizer = get_tokenizer()
        model = get_model(device=device)
        test.test_model(model, tokenizer, validation_dataset, device=device)


if __name__ == '__main__':
    app.run(main)
