import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import psutil

from desilofhe import Engine
from transformers import BertForSequenceClassification
from transformers.utils import logging as hf_logging

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

# Hide transformers' loading progress bars
hf_logging.disable_progress_bar()

MODEL_ID = "google-bert/bert-base-cased-finetuned-mrpc"
PER_LAYER_STAGES = [
    "stage_03", "stage_04", "stage_05",
    "stage_10", "stage_11", "stage_12",
    "stage_14", "stage_16",
]

# Per-process state. The engine is single threaded, so we fan out across
# processes; each worker holds its own engine. Weights are read-only and
# inherited by workers via fork copy-on-write (never pickled).
_engine = None
_weights = None


def _init_worker(compact):
    global _engine
    _engine = Engine(use_bootstrap_to_14_levels=True, compact=compact)


def _run_task(task):
    fn, uses_weights, args = task
    if uses_weights:
        fn(_engine, _weights, *args)
    else:
        fn(_engine, *args)


def warm_cache(path: Path):
    # All sizes are in GiB.
    virtual_memory = psutil.virtual_memory().available // 1024**3
    light_plaintexts_size = 105
    compute_memory = 40

    if virtual_memory - compute_memory < light_plaintexts_size:
        print("Warning: System memory has not enough space to hold the light plaintexts in memory."
              "This may lead to low performance.")
        return

    if shutil.which("vmtouch") is None:
        print("Warning: vmtouch not found; skipping page cache warming. "
              "Install it (e.g. `apt install vmtouch`) for lower inference latency.")
        return

    subprocess.run(["vmtouch", "-qt", str(path)], check=True)


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
        print(f"         [submission] Usage: {sys.argv[0]} <size>", file=sys.stderr)
        sys.exit(1)

    params = InstanceParams(int(sys.argv[1]), dataset="mrpc")
    io_dir = params.iodir()

    with open(io_dir / "thor_config.json") as f:
        config = json.load(f)
    compact = config["compact"]

    lp_path = light_plaintext_path(params.server_data_dir(), compact)

    if is_complete(lp_path):
        print(f"         [submission] Light plaintexts already exist at {lp_path} — skipping generation.")
        print("         [submission] Warming page cache...")
        warm_cache(lp_path)
        return

    print("         [submission] Generating light plaintexts...")
    model = BertForSequenceClassification.from_pretrained(MODEL_ID, output_hidden_states=True)
    model.eval()

    global _weights
    _weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    del model

    lp_path.mkdir(parents=True, exist_ok=True)

    engine_count = min(16, os.cpu_count() or 1)

    encoding_functions = [
        pre_encode_stage_03, pre_encode_stage_04, pre_encode_stage_05,
        pre_encode_stage_10, pre_encode_stage_11, pre_encode_stage_12,
        pre_encode_stage_14, pre_encode_stage_16,
    ]

    tasks = [(pre_encode_masks, False, (lp_path,))]
    for layer_index in range(12):
        for fn in encoding_functions:
            tasks.append((fn, True, (layer_index, lp_path)))
    tasks.append((pre_encode_stage_17, True, (lp_path,)))
    tasks.append((pre_encode_stage_18, True, (lp_path,)))

    # fork so workers inherit the weights via copy-on-write.
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=engine_count,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(compact,),
    ) as executor:
        list(executor.map(_run_task, tasks))

    # This reduces the latency during the inference.
    print("         [submission] Warming page cache...")
    warm_cache(lp_path)


if __name__ == "__main__":
    main()
