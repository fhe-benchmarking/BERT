import argparse
import json
import os
import sys

import numpy as np
import torch
from desilofhe import Engine
from transformers import BertForSequenceClassification

from params import InstanceParams

EMBED_MODEL_ID = "google-bert/bert-base-cased-finetuned-mrpc"
EMBED_LEVEL = 9


def _ld_entry(matrix, l, i):  # noqa: E741
    b, c = matrix.shape
    return matrix[(l + i) % b, i % c]


def encrypt_embedding(engine, secret_key, embedding):
    if embedding.shape != (128, 768):
        raise ValueError(f"Expected embedding shape (128, 768), got {embedding.shape}")
    x_T = np.transpose(embedding)
    x_blocks = np.vsplit(x_T, 6)
    cts = []
    for i in range(4):
        msg = np.zeros((2**15,), dtype=complex)
        for j in range(16):
            temp = j * (2**11)
            l = i * 16 + j  # noqa: E741
            for t in range(128):
                for b in range(12):
                    x_b = x_blocks[b % 6]
                    msg[temp + t * 16 + b] = complex(
                        _ld_entry(x_b, l, t), _ld_entry(x_b, l + 64, t)
                    )
        cts.append(engine.encrypt(msg, secret_key, EMBED_LEVEL))
    return cts


def encode_attention_mask(attention_mask):
    n_tokens = int(np.count_nonzero(attention_mask))
    clear_masks = []
    for i in range(8):
        msg = np.zeros((2**15,), dtype=float)
        for j in range(16):
            temp = j * (2**11)
            diag_index = i * 16 + j
            for t in range(128):
                col_index = (diag_index + t) % 128
                is_token = 1.0 if col_index < n_tokens else 0.0
                for head in range(12):
                    msg[temp + t * 16 + head] = is_token
        clear_masks.append(msg)
    return clear_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('size', type=int)
    args = parser.parse_args()

    params = InstanceParams(args.size, dataset="mrpc")
    io_dir = params.iodir()
    preprocessed_path = params.io_intermediate_dir() / "client_preprocessed_input"
    upload_dir = io_dir / "ciphertexts_upload"

    if not preprocessed_path.exists():
        print(f"Error: preprocessed input not found: {preprocessed_path}", file=sys.stderr)
        sys.exit(1)

    with open(io_dir / "thor_config.json") as f:
        config = json.load(f)
    compact = config["compact"]
    thread_count = min(16, os.cpu_count() or 1)

    if thread_count == 1:
        engine = Engine(use_bootstrap_to_14_levels=True, compact=compact)
    else:
        engine = Engine(
            use_bootstrap_to_14_levels=True,
            mode="parallel",
            thread_count=thread_count,
            compact=compact,
        )
    secret_key = engine.read_secret_key(io_dir / "secret_key")

    embedding_model = BertForSequenceClassification.from_pretrained(EMBED_MODEL_ID).bert.embeddings
    embedding_model.eval()

    records = []
    with open(preprocessed_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    upload_dir.mkdir(parents=True, exist_ok=True)

    for idx, record in enumerate(records):
        print(f"Encrypting sample {idx + 1}/{len(records)} (target_idx={record['target_idx']})...")
        sample_dir = upload_dir / str(idx)
        sample_dir.mkdir(exist_ok=True)

        with torch.no_grad():
            embedding = embedding_model(
                input_ids=torch.tensor([record["input_ids"]]),
                token_type_ids=torch.tensor([record["token_type_ids"]]),
            ).numpy().squeeze()

        for i, ct in enumerate(encrypt_embedding(engine, secret_key, embedding)):
            engine.write_ciphertext(ct, sample_dir / f"embedding_ct_{i}")

        attention_mask = np.array(record["attention_mask"], dtype=float)
        for i, mask in enumerate(encode_attention_mask(attention_mask)):
            np.save(sample_dir / f"attention_mask_{i}.npy", mask)


if __name__ == "__main__":
    main()
