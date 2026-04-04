#!/usr/bin/env python3
"""
Cleartext reference for the “ML Inference” workload.
For each test case:
    - Reads the input pixels from the dataset intermediate directory
    - Writes the predicted labels to output_labels path.
"""
# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
from utils import parse_submission_arguments
from mnist import mnist
from cifar10 import cifar10

def main():
    """
    Usage:  python3 cleartext_impl.py  <input_pixels_path>  <output_labels_path>
    """

    if len(sys.argv) != 4:
        sys.exit("Usage: cleartext_impl.py <input_pixels_path> <output_labels_path> <dataset_name>")

    # Paths to trained models
    minst_path = "harness/mnist/mnist_ffnn_model.pth"
    cifar10_path = "harness/cifar10/cifar10_resnet20_model.pth"

    INPUT_PATH = Path(sys.argv[1])
    OUTPUT_PATH = Path(sys.argv[2])
    DATASET_NAME = sys.argv[3]

    match DATASET_NAME:
        case "mnist": 
            return mnist.run_predict(
                model_path=minst_path, 
                pixels_file=INPUT_PATH, 
                predictions_file=OUTPUT_PATH)
        case "cifar10": 
            return cifar10.run_predict(
                model_path=cifar10_path, 
                pixels_file=INPUT_PATH, 
                predictions_file=OUTPUT_PATH)
        case _:
            raise ValueError(f"Unsupported dataset name: {DATASET_NAME}")

if __name__ == "__main__":
    main()