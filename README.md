# FHE Benchmarking Suite - BERT Inference

This repository contains the harness for the BERT inference workload of the FHE benchmarking suite of [HomomorphicEncryption.org](https://www.HomomorphicEncryption.org).
The harness currently supports [BERT-Base (110M)](https://huggingface.co/google-bert/bert-base-cased-finetuned-mrpc) inference on the MRPC task in the [GLUE benchmark](https://gluebenchmark.com/) as specified in `harness/mrpc` directory.
The `main` branch contains a reference implementation of this workload, under the `submissions` subdirectory.

Submitters should clone this repository and add their content as a subdirectory within the `submissions` directory.
They also may need to change `requirements.txt` to account for dependencies of their submission.
Submitters are expected to document any changes made to the model architecture `harness/mrpc/model.py` in the `submissions/{submission}/README.md` file. Submitters have the option to generate an `io/{size}/server_reported_steps.json` file, which contains fine grained metrics reported by the server in addition to the metrics reported by the harness.

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
- [`vmtouch`](https://github.com/hoytech/vmtouch) (optional, recommended). It is used to warm the page cache with the precomputed light plaintexts, which lowers inference latency. If it is not installed the step is skipped with a warning. Install it via your package manager, e.g. `apt install vmtouch` (Debian/Ubuntu), `brew install vmtouch` (macOS).

#### Execution
To run the workload, clone and install dependencies:
```console
git clone https://github.com/fhe-benchmarking/BERT.git
cd BERT

python3 -m venv .venv
source ./.venv/bin/activate
pip3 install -r requirements.txt

python3 harness/run_submission.py -h  # Information about command-line options
```

An example run is provided below.


```console
$ python3 harness/run_submission.py -h
usage: run_submission.py [-h] [--num_runs NUM_RUNS] [--seed SEED]
                         [--clrtxt CLRTXT] {0,1,2,3}

Run ML Inference FHE benchmark.

positional arguments:
  {0,1,2,3}             Instance size (0-single/1-small/2-medium/3-large)

options:
  -h, --help            show this help message and exit
  --num_runs NUM_RUNS   Number of times to run steps 4-9 (default: 1)
  --seed SEED           Random seed for dataset and query generation
  --clrtxt CLRTXT       Specify with 1 if to rerun the cleartext computation
```

The single instance runs the inference for a single input and verifies the correctness of the obtained label compared to the ground-truth label.

```console
$ python3 harness/run_submission.py 0 --seed 3

[harness] Running submission for single inference
14:19:12 [harness] 1: Test dataset generation completed (elapsed: 8.8415s)
         [submission] compact=False  bootstrap_key_size=large
         [submission] Generating secret key...
         [submission] Generating conjugation key...
         [submission] Generating relinearization key...
         [submission] Generating bootstrap key (size=large)...
         [submission] Generated 200 fixed rotation keys.
         [submission] Generating public key...
         [submission] Keys written to .../io/single/public_keys
14:20:44 [harness] 2: Key Generation completed (elapsed: 91.912s)
         [harness] Public and evaluation keys size: 21.4G
         [submission] Generating light plaintexts...
         [submission] Warming page cache...
14:38:32 [harness] 3: Encrypted model preprocessing completed (elapsed: 1067.17s)
14:38:41 [harness] 4: Input generation completed (elapsed: 9.6484s)
         [submission] Preprocessed 1 records -> .../io/single/intermediate/client_preprocessed_input
14:38:47 [harness] 5: Input preprocessing completed (elapsed: 5.9343s)
         [submission] Encrypting sample 1/1 (target_idx=25)...
14:38:55 [harness] 6: Input encryption completed (elapsed: 7.6941s)
         [harness] Encrypted input size: 46.0M
         [submission] Loading keys and weights...
         [submission] Sample 1 - Compute: 5777.403s, I/O: 1356.036s, Total: 7133.439s
         [submission] Total across all samples - Compute: 5777.403s, I/O: 1356.036s, Total: 7133.439s
16:55:43 [harness] 7: Encrypted computation completed (elapsed: 8207.7512s)
         [harness] Encrypted results size: 30.0M
         [submission] Decrypting sample 1/1 (target_idx=2656)...
         [submission] Decrypted 1 samples -> .../io/single/intermediate/decrypted_results.jsonl
16:55:44 [harness] 8: Result decryption completed (elapsed: 1.8956s)
         [submission] Wrote 1 predictions -> .../io/single/encrypted_model_predictions.txt
16:55:44 [harness] 9: Result postprocessing completed (elapsed: 0.0456s)
         [harness] PASS  (expected=1, got=1)
         [submission] Server reported steps: {'Encrypted computation': 5777.4029, 'I/O': 1356.036, 'Total': 7133.4389}
         [submission] Encrypted computation: 5777.4029s
         [submission] I/O: 1356.036s
         [submission] Total: 7133.4389s
[total latency] 9400.8927s

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
|   ├─ verify_result.py
|   ├─ [...]
|   └─ mrpc/       # MRPC dataset and BERT reference model
├─ datasets/       # The harness scripts create and populate this directory
├─ io/             # This directory is used for client<->server communication
├─ measurements/   # Holds logs with performance numbers
└─ submissions/    # This is where the workload implementations live
    ├─ client_*.py / server_*.py   # Reference (CPU) implementation
    └─ he.py, encode_weights.py, ...
```
Submitters must overwrite the contents of the `scripts` and `submissions` subdirectories.

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
