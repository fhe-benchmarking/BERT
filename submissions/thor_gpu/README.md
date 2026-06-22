# thor_gpu submission

GPU-accelerated variant of the reference BERT/MRPC submission. It mirrors the stages of the CPU
reference implementation in `submissions/`, but initializes the `desilofhe` engine in `async gpu`
mode to run the encrypted inference on a CUDA GPU.

Run it by passing `--submission thor_gpu` to the harness:

```console
python3 harness/run_submission.py 0 --submission thor_gpu
```

Notes:
- The engine runs with a fixed `thread_count` of 1024, so the harness `--thread_count` argument is
  ignored for this submission.
- `--parallel_sample_count` is not supported; samples are processed sequentially.

## Requirements

Same as the top-level `requirements.txt`:

```
numpy>=1.20
torch==2.9.1
transformers>=4.40
datasets>=2.19
desilofhe>=1.14.1
```

> **Important:** for GPU execution you must install a **CUDA build of `desilofhe`** instead of the
> CPU `desilofhe` package, and the build must match your local CUDA environment. The CUDA wheels are
> named `desilofhe-cu<XXX>`, where `<XXX>` is the CUDA version (e.g. `desilofhe-cu130` for CUDA 13.0,
> `desilofhe-cu121` for CUDA 12.1). Installing a wheel that does not match your CUDA toolkit / driver
> will fail to load or run.

Check your CUDA version first:

```console
nvidia-smi   # see the "CUDA Version" field in the top-right of the output
```

Then install the matching wheel instead of plain `desilofhe`, e.g. for CUDA 13.0:

```console
pip install desilofhe-cu130
```
