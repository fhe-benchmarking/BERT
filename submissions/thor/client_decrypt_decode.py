import json
import sys

import numpy as np
from desilofhe import Engine

from params import InstanceParams


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <size>", file=sys.stderr)
        sys.exit(1)

    params = InstanceParams(int(sys.argv[1]), dataset="mrpc")
    io_dir = params.iodir()
    download_dir = io_dir / "ciphertexts_download"
    intermediate_dir = params.io_intermediate_dir()

    config_path = io_dir / "thor_config.json"
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)
    compact = config["compact"]

    secret_key_path = io_dir / "secret_key"
    if not secret_key_path.exists():
        print(f"Error: secret key not found: {secret_key_path}", file=sys.stderr)
        sys.exit(1)

    manifest_path = download_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: download manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    engine = Engine(
        use_bootstrap_to_14_levels=True,
        mode="parallel",
        thread_count=16,
        compact=compact,
    )
    secret_key = engine.read_secret_key(secret_key_path)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Carry labels from upload manifest if available
    label_by_idx = {}
    upload_manifest_path = io_dir / "ciphertexts_upload" / "manifest.json"
    if upload_manifest_path.exists():
        with open(upload_manifest_path) as f:
            for entry in json.load(f):
                if "label" in entry:
                    label_by_idx[entry["idx"]] = entry["label"]

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    output_path = intermediate_dir / "decrypted_results.jsonl"

    with open(output_path, "w") as out:
        for entry in manifest:
            idx = entry["idx"]
            target_idx = entry["target_idx"]
            n_outputs = entry["n_outputs"]
            sample_dir = download_dir / entry["dir"]

            print(f"Decrypting sample {idx + 1}/{len(manifest)} (target_idx={target_idx})...")

            logits = []
            for i in range(n_outputs):
                ct_path = sample_dir / f"output_ct_{i}"
                if not ct_path.exists():
                    print(f"Error: ciphertext not found: {ct_path}", file=sys.stderr)
                    sys.exit(1)
                ct = engine.read_ciphertext(str(ct_path))
                decrypted = engine.decrypt(ct, secret_key)
                logits.append(float(decrypted[0].real))

            pred = int(np.argmax(logits))

            record = {"target_idx": target_idx, "he_logits": logits, "pred": pred}
            if idx in label_by_idx:
                record["label"] = label_by_idx[idx]

            out.write(json.dumps(record) + "\n")

    print(f"Decrypted {len(manifest)} samples -> {output_path}")


if __name__ == "__main__":
    main()
