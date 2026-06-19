# FHE Benchmarking Suite - ML Inference
This repository contains the harness for the ML-inference workload of the FHE benchmarking suite of [HomomorphicEncryption.org](https://www.HomomorphicEncryption.org).
The harness currently supports encrypted BERT inference on the MRPC task as specified in `harness/mrpc` directory.
The `main` branch contains a reference implementation of this workload, under the `submissions` subdirectory.

Submitters should clone this repository and add their content as a subdirectory within the `submissions` directory.
They also may need to change `requirements.txt` to account for dependencies of their submission.
Submitters are expected to document any changes made to the model architecture `harness/mrpc/model.py` in the `submissions/[--submission]/README.md` file. Submitters have the option to generate an `io/[--size]/server_reported_steps.json` file, which contains fine grained metrics reported by the server in addition to the metrics reported by the harness.

## Execution Modes

All steps are executed on a single machine:
- Cryptographic context setup and model preprocessing
- Key generation
- Input preprocessing and encryption
- Homomorphic inference
- Decryption and postprocessing

This corresponds to every reference submission in `submissions/`.

## Running the ML-inference workload

#### Dependencies
- Python 3.10+
- The Python packages listed in `requirements.txt` (`numpy`, `torch`, `transformers`, `datasets`, `desilofhe`). The first run downloads the BERT model and the MRPC dataset from the Hugging Face Hub, so an internet connection is required for the initial setup.

#### Execution
To run the workload, clone and install dependencies:
```console
git clone https://github.com/fhe-benchmarking/BERT.git
cd BERT

python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt

python3 harness/run_submission.py -h  # Information about command-line options
```

An example run is provided below.


```console
$ python3 harness/run_submission.py -h
usage: run_submission.py [-h] [--num_runs NUM_RUNS] [--seed SEED]
                         [--clrtxt CLRTXT] [--submission SUBMISSION]
                         [--dataset DATASET] [--thread_count THREAD_COUNT]
                         [--parallel_sample_count PARALLEL_SAMPLE_COUNT]
                         {0,1,2,3}

Run ML Inference FHE benchmark.

positional arguments:
  {0,1,2,3}             Instance size (0-single/1-small/2-medium/3-large)

options:
  -h, --help            show this help message and exit
  --num_runs NUM_RUNS   Number of times to run steps 4-9 (default: 1)
  --seed SEED           Random seed for dataset and query generation
  --clrtxt CLRTXT       Specify with 1 if to rerun the cleartext computation
  --submission SUBMISSION
                        Submission subdirectory under submissions/ (default:
                        run the reference implementation at submissions/)
  --dataset DATASET     Pick a dataset run (default: mrpc)
  --thread_count THREAD_COUNT
                        Number of threads for the FHE engine (default: 16)
  --parallel_sample_count PARALLEL_SAMPLE_COUNT
                        Number of samples to process in parallel during server
                        computation (default: 1)
```

The single instance runs the inference for a single input and verifies the correctness of the obtained label compared to the ground-truth label.

```console
$ python3 ./harness/run_submission.py 0 --seed 3 --num_runs 2 --dataset mnist --model mlp
 

[harness] Running submission for single inference
[get_openfhe] Found OpenFHE at .../ml-inference/third_party/openfhe (use --force to rebuild).
-- FOUND PACKAGE OpenFHE
-- OpenFHE Version: 1.4.0
-- OpenFHE installed as shared libraries: ON
-- OpenFHE include files location: .../ml-inference/third_party/openfhe/include/openfhe
-- OpenFHE lib files location: .../ml-inference/third_party/openfhe/lib
-- OpenFHE Native Backend size: 64
-- FOUND PACKAGE Torch
-- Torch include dirs: .../ml-inference/third_party/libtorch/include;.../ml-inference/third_party/libtorch/include/torch/csrc/api/include
-- Torch libraries: torch;torch_library;.../ml-inference/third_party/libtorch/lib/libc10.so;.../ml-inference/third_party/libtorch/lib/libkineto.a
-- Configuring done
-- Generating done
-- Build files have been written to: .../ml-inference/submission/build
[ 11%] Built target mlp_encryption_utils
[ 33%] Built target client_key_generation
[ 33%] Built target server_preprocess_model
[ 44%] Built target mlp_openfhe
[ 55%] Built target client_encode_encrypt_input
[ 66%] Built target client_decrypt_decode
[ 77%] Built target client_preprocess_input
[ 88%] Built target client_postprocess
[100%] Built target server_encrypted_compute
13:21:55 [harness] 1: Harness: MNIST Test dataset generation completed (elapsed: 8.8735s)
13:21:57 [harness] 2.2: Client: Key Generation completed (elapsed: 2.3535s)
         [harness] Client: Public and evaluation keys size: 1.0G
13:21:57 [harness] 3: Server: (Encrypted) model preprocessing completed (elapsed: 0.198s)

         [harness] Run 1 of 2
13:22:01 [harness] 4: Harness: Input generation for MNIST completed (elapsed: 3.8631s)
13:22:01 [harness] 5: Client: Input preprocessing completed (elapsed: 0.1061s)
13:22:01 [harness] 6: Client: Input encryption completed (elapsed: 0.201s)
         [harness] Client: Encrypted input size: 5.0M
         [server] Loading keys
         [server] PyTorch model weights loaded successfully!
         [server] run encrypted MNIST inference
         [server] Execution time for ciphertext 0 : 12 seconds
13:22:15 [harness] 7: Server: Encrypted ML Inference computation completed (elapsed: 13.4429s)
         [harness] Client: Encrypted results size: 1.0M
13:22:15 [harness] 8: Client: Result decryption completed (elapsed: 0.2832s)
13:22:15 [harness] 9: Client: Result postprocessing completed (elapsed: 0.118s)
[harness] PASS  (expected=7, got=7)
[total latency] 29.4393s

         [harness] Run 2 of 2
13:22:21 [harness] 4: Harness: Input generation for MNIST completed (elapsed: 5.3879s)
13:22:21 [harness] 5: Client: Input preprocessing completed (elapsed: 0.0852s)
13:22:21 [harness] 6: Client: Input encryption completed (elapsed: 0.2011s)
         [harness] Client: Encrypted input size: 5.0M
         [server] Loading keys
         [server] PyTorch model weights loaded successfully!
         [server] run encrypted MNIST inference
         [server] Execution time for ciphertext 0 : 13 seconds
13:22:36 [harness] 7: Server: Encrypted ML Inference computation completed (elapsed: 15.0731s)
         [harness] Client: Encrypted results size: 1.0M
13:22:36 [harness] 8: Client: Result decryption completed (elapsed: 0.2518s)
13:22:36 [harness] 9: Client: Result postprocessing completed (elapsed: 0.1047s)
[harness] PASS  (expected=7, got=7)
[total latency] 32.5287s

All steps completed for the single inference!
```

The batch inference cases run the inference for a batch of inputs of varying sizes. The accuracy (with respect to the ground truth labels) is compared between the decrypted results and the results obtained using the harness model.

After finishing the run, deactivate the virtual environment.
```console
deactivate
```

## Directory structure

The directory structure of this reposiroty is as follows:
```
├─ README.md       # This file
├─ LICENSE.md      # Harness software license (Apache v2)
├─ requirements.txt
├─ harness/        # Scripts to drive the workload implementation
|   ├─ run_submission.py
|   ├─ generate_dataset.py
|   ├─ generate_input.py
|   ├─ cleartext_impl.py
|   ├─ verify_result.py
|   └─ mrpc/        # MRPC dataset and BERT reference model
├─ datasets/       # The harness scripts create and populate this directory
├─ io/             # This directory is used for client<->server communication
├─ measurements/   # Holds logs with performance numbers
└─ submissions/    # This is where the workload implementations live
    ├─ client_*.py / server_*.py   # Reference (CPU) implementation
    ├─ he.py, encode_weights.py, ...
    └─ thor_gpu/   # GPU submission (see its README.md)
```
Submitters add a subdirectory to `submissions/` (selected with `--submission`), or replace the
reference implementation directly in `submissions/`.

## Description of stages

A submitter can copy and edit any of the `client_*` / `server_*` sources in `/submissions`. 
Moreover, for the particular parameters related to a workload, the submitter can modify the params files.
If the current description of the files are inaccurate, the stage names in `run_submission` can be also 
modified.

The current stages are the following, targeted to a client-server scenario.
The order in which they are happening in `run_submission` assumes an initialization step which is 
database-dependent and run only once, and potentially multiple runs for multiple queries.
Each file can take as argument the test case size.


| Stage executables                | Description |
|----------------------------------|-------------|
| `client_key_generation`          | Generate all key material and cryptographic context at the client.           
| `server_preprocess_model`        | (Optional) Any in the clear or encrypted computations the server wants to apply over the model.
| `client_preprocess_input`        | (Optional) Any in the clear computations the client wants to apply over the input.
| `client_encode_encrypt_input`    | Plaintext encoding and encryption of the input at the client.
| `server_encrypted_compute`       | The computation the server applies to achieve the workload solution over encrypted data.
| `client_decrypt_decode`          | Decryption and plaintext decoding of the result at the client.
| `client_postprocess`             | Any in the clear computation that the client wants to apply on the decrypted result.


The outer python script measures the runtime of each stage.
The current stage separation structure requires reading and writing to files more times than minimally necessary.
For a more granular runtime measuring, which would account for the extra overhead described above, we encourage
submitters to separate and print in a log the individual times for reads/writes and computations inside each stage. 
