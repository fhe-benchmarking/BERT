import json
import sys
from pathlib import Path

import numpy as np

from he import HE

INSTANCE_NAMES = ["single", "small", "medium", "large"]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <size>", file=sys.stderr)
        sys.exit(1)

    size = int(sys.argv[1])
    io_dir = Path("io") / INSTANCE_NAMES[size]

    with open(io_dir / "thor_config.json") as f:
        config = json.load(f)
    compact = config["compact"]
    bootstrap_key_size = config["bootstrap_key_size"]

    print("Loading keys and light plaintexts...")
    he = HE(io_dir, compact, bootstrap_key_size)

    upload_dir = io_dir / "ciphertexts_upload"
    download_dir = io_dir / "ciphertexts_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    with open(upload_dir / "manifest.json") as f:
        manifest = json.load(f)

    result_manifest = []
    total_compute_seconds = 0.0
    total_elapsed_seconds = 0.0

    for entry in manifest:
        idx = entry["idx"]
        target_idx = entry["target_idx"]
        sample_dir = upload_dir / entry["dir"]

        print(f"Processing sample {idx + 1}/{len(manifest)} (target_idx={target_idx})...")

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

        elapsed = he.timer.elapsed
        compute = he.timer.compute_elapsed
        total_elapsed_seconds += elapsed
        total_compute_seconds += compute
        print(f"  Compute: {compute:.3f}s, Total: {elapsed:.3f}s")

        out_dir = download_dir / str(idx)
        out_dir.mkdir(exist_ok=True)
        for i in range(len(x)):
            he.engine.write_ciphertext(x[i], str(out_dir / f"output_ct_{i}"))

        result_manifest.append({
            "idx": idx,
            "target_idx": target_idx,
            "dir": str(idx),
            "n_outputs": len(x),
        })

    with open(download_dir / "manifest.json", "w") as f:
        json.dump(result_manifest, f, indent=2)

    steps = {
        "Encrypted computation": round(total_compute_seconds, 4),
        "Total": round(total_elapsed_seconds, 4),
    }
    with open(io_dir / "server_reported_steps.json", "w") as f:
        json.dump(steps, f, indent=2)

    print(f"Results written to {download_dir}")
    print(f"Compute: {total_compute_seconds:.3f}s, Total: {total_elapsed_seconds:.3f}s")


if __name__ == "__main__":
    main()
