#!/usr/bin/env python3
"""
If the datasets are too large to include, generate them here or pull them 
from a storage source.
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
from mnist import mnist
from cifar10 import cifar10
from mrpc import mrpc

def main():
    """
    Usage:  python3 generate_dataset.py  <output_file>
    """

    if len(sys.argv) != 3:
        sys.exit("Usage: generate_dataset.py <output_file> [dataset_name]")

    DATASET_PATH = Path(sys.argv[1])
    DATASET_NAME = sys.argv[2]
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    match DATASET_NAME:
        case "mnist": 
            return mnist.export_test_data(
                output_file=DATASET_PATH, 
                num_samples=10000, 
                seed=None)
        case "cifar10":
            return cifar10.export_test_data(
                output_file=DATASET_PATH,
                num_samples=10000,
                seed=None)
        case "mrpc":
            return mrpc.export_test_data(
                output_file=DATASET_PATH,
                num_samples=-1,
                seed=None)
        case _:
            raise ValueError(f"Unsupported dataset name: {DATASET_NAME}")

if __name__ == "__main__":
    main()
