import json
import mmap
import sys
from pathlib import Path

from desilofhe import Engine
from transformers import BertForSequenceClassification

from params import InstanceParams
from encode_weights import (
    pre_encode_masks,
    pre_encode_stage_03,
    pre_encode_stage_04,
    pre_encode_stage_05,
    pre_encode_stage_10,
    pre_encode_stage_11,
    pre_encode_stage_12,
    pre_encode_stage_14,
    pre_encode_stage_16,
    pre_encode_stage_17,
    pre_encode_stage_18,
)

MODEL_ID = "google-bert/bert-base-cased-finetuned-mrpc"
PER_LAYER_STAGES = [
    "stage_03", "stage_04", "stage_05",
    "stage_10", "stage_11", "stage_12",
    "stage_14", "stage_16",
]


def warm_cache(path: Path):
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        with open(f, "rb") as fh:
            length = f.stat().st_size
            if length == 0:
                continue
            with mmap.mmap(fh.fileno(), length, access=mmap.ACCESS_READ) as m:
                m.madvise(mmap.MADV_SEQUENTIAL)
                m.read(length)


def light_plaintext_path(server_data_dir, compact):
    return server_data_dir / "light_plaintexts" / ("compact" if compact else "default")


def is_complete(lp_path):
    for stage in ("masks", "stage_17", "stage_18"):
        if not (lp_path / stage).is_dir():
            return False
    for stage in PER_LAYER_STAGES:
        for layer in range(12):
            if not (lp_path / stage / f"layer_{layer}").is_dir():
                return False
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <size>", file=sys.stderr)
        sys.exit(1)

    params = InstanceParams(int(sys.argv[1]), dataset="mrpc")
    io_dir = params.iodir()

    with open(io_dir / "thor_config.json") as f:
        config = json.load(f)
    compact = config["compact"]

    lp_path = light_plaintext_path(params.server_data_dir(), compact)

    if is_complete(lp_path):
        print(f"Light plaintexts already exist at {lp_path} — skipping generation.")
        print("Warming page cache...")
        warm_cache(lp_path)
        return

    model = BertForSequenceClassification.from_pretrained(MODEL_ID, output_hidden_states=True)
    model.eval()

    weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    del model

    lp_path.mkdir(parents=True, exist_ok=True)
    engine = Engine(use_bootstrap_to_14_levels=True, compact=compact)

    print("Encoding masks...")
    pre_encode_masks(engine, lp_path)

    for layer_index in range(12):
        print(f"Encoding layer {layer_index}...")
        pre_encode_stage_03(engine, weights, layer_index, lp_path)
        pre_encode_stage_04(engine, weights, layer_index, lp_path)
        pre_encode_stage_05(engine, weights, layer_index, lp_path)
        pre_encode_stage_10(engine, weights, layer_index, lp_path)
        pre_encode_stage_11(engine, weights, layer_index, lp_path)
        pre_encode_stage_12(engine, weights, layer_index, lp_path)
        pre_encode_stage_14(engine, weights, layer_index, lp_path)
        pre_encode_stage_16(engine, weights, layer_index, lp_path)

    print("Encoding pooler and classifier...")
    pre_encode_stage_17(engine, weights, lp_path)
    pre_encode_stage_18(engine, weights, lp_path)

    # This reduces the latency during the inference.
    print("Warming page cache...")
    warm_cache(lp_path)


if __name__ == "__main__":
    main()
