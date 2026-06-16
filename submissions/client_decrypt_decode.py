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
    preprocessed_path = intermediate_dir / "client_preprocessed_input"

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

    engine = Engine(
        use_bootstrap_to_14_levels=True,
        mode="parallel",
        thread_count=16,
        compact=compact,
    )
    secret_key = engine.read_secret_key(secret_key_path)

    records = []
    with open(preprocessed_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    decrypted_path = intermediate_dir / "decrypted_results.jsonl"

    with open(decrypted_path, "w") as out:
        for idx, src in enumerate(records):
            target_idx = src["target_idx"]
            sample_dir = download_dir / str(idx)

            print(f"Decrypting sample {idx + 1}/{len(records)} (target_idx={target_idx})...")

            ct_files = sorted(
                sample_dir.glob("output_ct_*"),
                key=lambda p: int(p.name.split("_")[-1]),
            )
            logits = [
                float(engine.decrypt(engine.read_ciphertext(str(p)), secret_key)[0].real)
                for p in ct_files
            ]

            pred = int(np.argmax(logits))

            result = {"target_idx": target_idx, "he_logits": logits, "pred": pred}
            if "label" in src:
                result["label"] = src["label"]

            out.write(json.dumps(result) + "\n")

    print(f"Decrypted {len(records)} samples -> {decrypted_path}")


if __name__ == "__main__":
    main()
