import json
import sys

from params import InstanceParams


def main():
    if len(sys.argv) < 2:
        print(f"         [submission] Usage: {sys.argv[0]} <size>", file=sys.stderr)
        sys.exit(1)

    params = InstanceParams(int(sys.argv[1]), dataset="mrpc")
    intermediate_dir = params.io_intermediate_dir()

    decrypted_path = intermediate_dir / "decrypted_results.jsonl"
    if not decrypted_path.exists():
        print(f"         [submission] Error: decrypted results not found: {decrypted_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(decrypted_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records.sort(key=lambda r: r["target_idx"])

    predictions = []
    for r in records:
        logits = r["he_logits"]
        label = 0 if logits[0] > logits[1] else 1
        predictions.append(label)

    encrypted_model_preds = params.get_encrypted_model_predictions_file()
    with open(encrypted_model_preds, "w") as f:
        for label in predictions:
            f.write(f"{label}\n")

    print(f"         [submission] Wrote {len(predictions)} predictions -> {encrypted_model_preds}")


if __name__ == "__main__":
    main()
