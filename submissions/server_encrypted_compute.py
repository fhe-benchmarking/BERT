import argparse
import json
import os
import sys
import math
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

import numpy as np

from params import InstanceParams
from he import HE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('size', type=int)
    args = parser.parse_args()

    params = InstanceParams(args.size, dataset="mrpc")
    io_dir = params.iodir()
    batch_size = params.get_batch_size()

    with open(io_dir / "thor_config.json") as f:
        config = json.load(f)
    compact = config["compact"]
    bootstrap_key_size = config["bootstrap_key_size"]

    # All sizes are in GiB
    virtual_memory = psutil.virtual_memory().available // 1024**3
    light_plaintexts_size = 105
    compute_memory = 40

    if virtual_memory > light_plaintexts_size:
        worker_count = max((virtual_memory - light_plaintexts_size) // compute_memory, 1)
    else:
        # Ignore the light plaintext cache and just use the available memory for workers
        worker_count = max(virtual_memory // compute_memory, 1)

    # For small batch sizes, we don't need more workers than samples
    worker_count = min(worker_count, batch_size)

    cpu_count = max(os.cpu_count() or 1 // worker_count, 1)
    thread_count = min(16, cpu_count)

    print("         [submission] Loading keys and weights...")

    he_pool = queue.Queue()
    for _ in range(worker_count):
        he_pool.put(HE(params, compact, bootstrap_key_size, thread_count=thread_count))

    upload_dir = io_dir / "ciphertexts_upload"
    download_dir = io_dir / "ciphertexts_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    total_compute_seconds = 0.0
    total_paused_seconds = 0.0
    total_elapsed_seconds = 0.0
    lock = threading.Lock()

    def process_sample(idx):
        nonlocal total_compute_seconds, total_paused_seconds, total_elapsed_seconds
        he = he_pool.get()
        try:
            sample_dir = upload_dir / str(idx)
            x = np.empty((4,), dtype=object)

            for i in range(4):
                x[i] = he.engine.read_ciphertext(str(sample_dir / f"embedding_ct_{i}"))

            clear_attention_mask = [
                np.load(sample_dir / f"attention_mask_{i}.npy")
                for i in range(8)
            ]

            he.timer.reset()

            for layer_idx in range(12):
                x = he.forward_layer(x, layer_idx, clear_attention_mask)

            x = he.stage_17_pooler(x)
            x = he.stage_18_classifier(x)

            compute = he.timer.compute_elapsed
            paused = he.timer.paused_time
            elapsed = he.timer.elapsed

            with lock:
                total_compute_seconds += compute
                total_paused_seconds += paused
                total_elapsed_seconds += elapsed

            print(f"         [submission] Sample {idx + 1} - Compute: {compute:.3f}s, I/O: {paused:.3f}s, Total: {elapsed:.3f}s")

            out_dir = download_dir / str(idx)
            out_dir.mkdir(exist_ok=True)
            for i in range(len(x)):
                he.engine.write_ciphertext(x[i], str(out_dir / f"output_ct_{i}"))
        finally:
            he_pool.put(he)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(process_sample, range(batch_size)))

    steps = {
        "Encrypted computation": round(total_compute_seconds, 4),
        "I/O": round(total_paused_seconds, 4),
        "Total": round(total_elapsed_seconds, 4),
    }
    with open(io_dir / "server_reported_steps.json", "w") as f:
        json.dump(steps, f, indent=2)

    print(f"         [submission] Total across all samples - Compute: {total_compute_seconds:.3f}s, I/O: {total_paused_seconds:.3f}s, Total: {total_elapsed_seconds:.3f}s")


if __name__ == "__main__":
    main()
