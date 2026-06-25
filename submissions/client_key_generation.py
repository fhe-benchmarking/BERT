import argparse
import json
import os
import sys

from desilofhe import Engine

from params import InstanceParams

# Set COMPACT = True to use compact mode, which reduces overall memory footprint.
COMPACT = False

# Bootstrap deltas covered by the bootstrap key itself (no fixed rotation key needed).
# Copied from src/thor/he.py to stay in sync with server evaluation requirements.
# fmt: off
_COMPACT_BOOTSTRAP_DELTAS = frozenset([
    1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 64, 96, 128, 160, 192, 224, 256, 512, 768,
    1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 16384, 24576, 31744, 32000, 32256,
    32512, 32736, 32744, 32752, 32760,
])
_DEFAULT_BOOTSTRAP_DELTAS = frozenset([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 96, 128, 160, 192,
    224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 1024, 2048, 3072, 4096, 5120,
    6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 31744,
    32256, 32736, 32752,
])

# (delta, level) pairs for fixed rotation keys used during server evaluation.
# Copied verbatim from src/thor/he.py rotation_contexts.
_ROTATION_CONTEXTS = [
    (0, 9), (1, 14), (2, 14), (3, 12), (4, 14), (5, 12), (6, 11), (8, 7), (16, 10),
    (32, 10), (64, 10), (128, 10), (240, 9), (256, 10), (496, 9), (512, 10), (752, 9),
    (1008, 9), (1024, 10), (1264, 9), (1520, 9), (1776, 9), (2032, 9), (2048, 8),
    (2272, 9), (2528, 9), (2784, 9), (3040, 9), (3184, 9), (3296, 9), (3312, 9),
    (3440, 9), (3552, 9), (3568, 9), (3696, 9), (3808, 9), (3824, 9), (3952, 9),
    (4064, 9), (4080, 9), (4304, 9), (4560, 9), (4816, 9), (5072, 9), (5328, 9),
    (5584, 9), (5840, 9), (6096, 9), (6336, 9), (6592, 9), (6848, 9), (7104, 9),
    (7264, 9), (7360, 9), (7392, 9), (7520, 9), (7616, 9), (7648, 9), (7776, 9),
    (7872, 9), (7904, 9), (8032, 9), (8128, 9), (8160, 9), (8368, 9), (8624, 9),
    (8880, 9), (9136, 9), (9392, 9), (9648, 9), (9904, 9), (10160, 9), (10400, 9),
    (10656, 9), (10912, 9), (11168, 9), (11344, 9), (11424, 9), (11472, 9), (11600, 9),
    (11680, 9), (11728, 9), (11856, 9), (11936, 9), (11984, 9), (12112, 9), (12192, 9),
    (12240, 9), (12432, 9), (12688, 9), (12944, 9), (13200, 9), (13456, 9), (13712, 9),
    (13968, 9), (14224, 9), (14464, 9), (14720, 9), (14976, 9), (15232, 9), (15424, 9),
    (15488, 9), (15552, 9), (15680, 9), (15744, 9), (15808, 9), (15936, 9), (16000, 9),
    (16064, 9), (16192, 9), (16256, 9), (16320, 9), (16384, 13), (16496, 9), (16752, 9),
    (17008, 9), (17264, 9), (17520, 9), (17776, 9), (18032, 9), (18288, 9), (18528, 9),
    (18784, 9), (19040, 9), (19296, 9), (19504, 9), (19552, 9), (19632, 9), (19760, 9),
    (19808, 9), (19888, 9), (20016, 9), (20064, 9), (20144, 9), (20272, 9), (20320, 9),
    (20400, 9), (20560, 9), (20816, 9), (21072, 9), (21328, 9), (21584, 9), (21840, 9),
    (22096, 9), (22352, 9), (22592, 9), (22848, 9), (23104, 9), (23360, 9), (23584, 9),
    (23616, 9), (23712, 9), (23840, 9), (23872, 9), (23968, 9), (24096, 9), (24128, 9),
    (24224, 9), (24352, 9), (24384, 9), (24480, 9), (24576, 13), (24624, 9), (24880, 9),
    (25136, 9), (25392, 9), (25648, 9), (25904, 9), (26160, 9), (26416, 9), (26656, 9),
    (26912, 9), (27168, 9), (27424, 9), (27664, 9), (27680, 9), (27792, 9), (27920, 9),
    (27936, 9), (28048, 9), (28176, 9), (28192, 9), (28304, 9), (28432, 9), (28448, 9),
    (28560, 9), (28672, 14), (28688, 9), (28944, 9), (29200, 9), (29456, 9), (29712, 9),
    (29968, 9), (30224, 9), (30480, 9), (30720, 14), (30976, 9), (31232, 9), (31488, 9),
    (31744, 13), (31872, 9), (32000, 9), (32128, 9), (32256, 13), (32384, 9), (32512, 13),
    (32640, 13), (32704, 13), (32736, 13), (32752, 13), (32757, 12), (32758, 12),
    (32759, 12), (32760, 12), (32761, 12), (32762, 11), (32763, 8), (32764, 13),
    (32765, 8), (32766, 13), (32767, 13),
]
# fmt: on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('size', type=int)
    args = parser.parse_args()

    params = InstanceParams(args.size, dataset="mrpc")
    io_dir = params.iodir()
    public_keys_dir = io_dir / "public_keys"
    fixed_rotation_keys_dir = public_keys_dir / "fixed_rotation_keys"

    for d in (public_keys_dir, fixed_rotation_keys_dir):
        d.mkdir(parents=True, exist_ok=True)

    compact = COMPACT
    bootstrap_key_size = "medium" if compact else "large"
    config = {"compact": compact, "bootstrap_key_size": bootstrap_key_size}
    thread_count = min(16, os.cpu_count() or 1)

    with open(io_dir / "thor_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"compact={compact}  bootstrap_key_size={bootstrap_key_size}")

    if thread_count == 1:
        engine = Engine(use_bootstrap_to_14_levels=True, compact=compact)
    else:
        engine = Engine(
            use_bootstrap_to_14_levels=True,
            mode="parallel",
            thread_count=thread_count,
            compact=compact,
        )

    print("Generating secret key...")
    secret_key = engine.create_secret_key()
    engine.write_secret_key(secret_key, io_dir / "secret_key")

    print("Generating conjugation key...")
    conjugation_key = engine.create_conjugation_key(secret_key)
    engine.write_conjugation_key(conjugation_key, public_keys_dir / "conjugation_key")

    print("Generating relinearization key...")
    relinearization_key = engine.create_relinearization_key(secret_key)
    engine.write_relinearization_key(relinearization_key, public_keys_dir / "relinearization_key")

    print(f"Generating bootstrap key (size={bootstrap_key_size})...")
    bootstrap_key = engine.create_bootstrap_key(secret_key, size=bootstrap_key_size)
    engine.write_bootstrap_key(bootstrap_key, public_keys_dir / "bootstrap_key")

    bootstrap_deltas = _COMPACT_BOOTSTRAP_DELTAS if compact else _DEFAULT_BOOTSTRAP_DELTAS
    key_count = 0
    for delta, level in _ROTATION_CONTEXTS:
        if delta == 0 or delta in bootstrap_deltas:
            continue
        key = engine.create_fixed_rotation_key(secret_key, delta, level=level)
        engine.write_fixed_rotation_key(key, fixed_rotation_keys_dir / str(delta))
        key_count += 1

    print(f"Generated {key_count} fixed rotation keys.")

    print("Generating public key...")
    public_key = engine.create_public_key(secret_key)
    engine.write_public_key(public_key, public_keys_dir / "public_key")

    print(f"Keys written to {public_keys_dir}")


if __name__ == "__main__":
    main()
