import json
import sys

from params import InstanceParams
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

# Hide transformers' loading progress bars
hf_logging.disable_progress_bar()

MODEL_ID = "google-bert/bert-base-cased-finetuned-mrpc"
MAX_LENGTH = 128


def main():
    if len(sys.argv) < 2:
        print(f"         [submission] Usage: {sys.argv[0]} <size>", file=sys.stderr)
        sys.exit(1)

    params = InstanceParams(int(sys.argv[1]), dataset="mrpc")

    input_path = params.get_test_input_file()
    output_dir = params.io_intermediate_dir()
    output_path = output_dir / "client_preprocessed_input"

    if not input_path.exists():
        print(f"         [submission] Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        for record in records:
            encoding = tokenizer(
                record["sentence1"],
                record["sentence2"],
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
            )
            out.write(json.dumps({
                "target_idx": record["target_idx"],
                "input_ids": encoding["input_ids"],
                "token_type_ids": encoding["token_type_ids"],
                "attention_mask": encoding["attention_mask"],
            }) + "\n")

    print(f"         [submission] Preprocessed {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()
