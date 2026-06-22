import json
import sys

import numpy as np

from params import InstanceParams
from he import HE


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <size>", file=sys.stderr)
        sys.exit(1)

    params = InstanceParams(int(sys.argv[1]), dataset="mrpc")
    io_dir = params.iodir()
    batch_size = params.get_batch_size()

    with open(io_dir / "thor_config.json") as f:
        config = json.load(f)
    compact = config["compact"]
    bootstrap_key_size = config["bootstrap_key_size"]

    print("Loading keys and weights...")
    he = HE(io_dir, compact, bootstrap_key_size)

    upload_dir = io_dir / "ciphertexts_upload"
    download_dir = io_dir / "ciphertexts_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    total_compute_seconds = 0.0
    total_paused_seconds = 0.0
    total_elapsed_seconds = 0.0

    for idx in range(batch_size):
        sample_dir = upload_dir / str(idx)
        print(f"Processing sample {idx + 1}/{batch_size}...")

        x = np.empty((4,), dtype=object)
        for i in range(4):
            x[i] = he.engine.read_ciphertext(str(sample_dir / f"embedding_ct_{i}"))

        clear_attention_mask = [
            np.load(sample_dir / f"attention_mask_{i}.npy")
            for i in range(8)
        ]

        he.timer.reset()

        for layer_idx in range(12):
            print(f"  Layer {layer_idx}...")
            x = he.forward_layer(x, layer_idx, clear_attention_mask)

        print("  Pooler + classifier...")
        x = he.stage_17_pooler(x)
        x = he.stage_18_classifier(x)

        compute = he.timer.compute_elapsed
        paused = he.timer.paused_time
        elapsed = he.timer.elapsed

        total_compute_seconds += compute
        total_paused_seconds += paused
        total_elapsed_seconds += elapsed
        print(f"  Compute: {compute:.3f}s, I/O: {paused:.3f}s, Total: {elapsed:.3f}s")s

        out_dir = download_dir / str(idx)
        out_dir.mkdir(exist_ok=True)
        for i in range(len(x)):
            he.engine.write_ciphertext(x[i], str(out_dir / f"output_ct_{i}"))

    steps = {
        "Encrypted computation": round(total_compute_seconds, 4),
        "I/O": round(total_paused_seconds, 4),
        "Total": round(total_elapsed_seconds, 4),
    }
    with open(io_dir / "server_reported_steps.json", "w") as f:
        json.dump(steps, f, indent=2)

    print(f"Compute: {total_compute_seconds:.3f}s, Total: {total_elapsed_seconds:.3f}s")


if __name__ == "__main__":
    main()
