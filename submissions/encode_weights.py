import numpy as np
from desilofhe import Engine


def to_blocks(matrix, block_shape, diag=True):
    rows, cols = matrix.shape
    block_rows, block_cols = block_shape
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError("Matrix shape should be divisible by block shape")
    vertical = rows // block_rows
    horizontal = cols // block_cols
    blocks = matrix.reshape(vertical, block_rows, horizontal, block_cols).transpose(0, 2, 1, 3)
    if not diag:
        return blocks, (vertical, horizontal)
    diag_rows = min(vertical, horizontal)
    diag_cols = max(vertical, horizontal)
    diag_blocks = np.empty((diag_rows, diag_cols, block_rows, block_cols), dtype=matrix.dtype)
    diag_row_indices = np.arange(diag_rows)
    for diag_col in range(diag_cols):
        diag_blocks[:, diag_col] = blocks[(diag_row_indices + diag_col) % vertical, diag_col % horizontal]
    return diag_blocks, (diag_rows, diag_cols)


SLOT_COUNT = 2**15
GROUP_SIZE = 2**11
PACK = 16
DIM = 128
DIM_RANGE = np.arange(DIM)
PACK_RANGE = np.arange(PACK)
DIM_SLOT_BASE = 16 * DIM_RANGE[:, None]
PACK_GROUP_BASE = GROUP_SIZE * PACK_RANGE[:, None, None]
ATT_SLOT_INDICES = np.arange(12)
FF_SLOT_INDICES = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])


def get_weight_array(weights, name):
    return weights[name]


def get_classifier_prefix(weights):
    return "cls.seq_relationship" if "cls.seq_relationship.weight" in weights else "classifier"


def gather_upper_diagonal_batch(blocks, input_indices, rotations, n_in, n_in_complex):
    offsets = rotations[:, None] + DIM_RANGE[None, :]
    row_indices = offsets % blocks.shape[1]
    real_col_indices = (input_indices[:, None] + offsets) % blocks.shape[2]
    imag_col_indices = (((input_indices + n_in_complex) % n_in)[:, None] + offsets) % blocks.shape[2]
    block_indices = np.arange(blocks.shape[0])[None, :, None]
    real = blocks[block_indices, row_indices[:, None, :], real_col_indices[:, None, :]]
    imag = blocks[block_indices, row_indices[:, None, :], imag_col_indices[:, None, :]]
    return (real - 1j * imag) / 2


def assign_packed_dense_block(msg, values, slot_indices):
    positions = PACK_GROUP_BASE + DIM_SLOT_BASE[None, :, :] + slot_indices[None, None, :]
    msg[positions] = values.transpose(0, 2, 1)


def encode_w_att(w, n_in, n_out, block_shape, scale=1):
    n_in_complex = n_in // 2
    diag_blocks, (diag_count, _) = to_blocks(w, block_shape, diag=True)
    n_out_packed = n_out // PACK
    messages = np.full((n_out_packed, diag_count, n_in_complex), None, dtype=object)
    for diag_index in range(diag_count):
        diagonal = diag_blocks[diag_index]
        for n in range(n_in_complex):
            for out_index in range(n_out_packed):
                msg = np.zeros((SLOT_COUNT,), dtype=complex)
                rotations = out_index * 16 + PACK_RANGE
                input_indices = ((n // 16) * 16 + out_index * 16 + (n + PACK_RANGE) % 16) % n_in_complex
                input_indices = (input_indices - rotations) % n_in
                values = scale * gather_upper_diagonal_batch(diagonal, input_indices, rotations, n_in, n_in_complex)
                assign_packed_dense_block(msg, values, ATT_SLOT_INDICES)
                messages[out_index, diag_index, n] = msg
    return messages


def encode_w_qkv(w, scale=1):
    return encode_w_att(w, n_in=128, n_out=64, block_shape=(64, 128), scale=scale)


def encode_w_ff(w, n_in, n_out, block_shape, vsplit=0, hsplit=0, scale=1):
    n_in_complex = n_in // 2
    if vsplit:
        weight_splits = np.vsplit(w, vsplit)
    elif hsplit:
        weight_splits = np.hsplit(w, hsplit)
    else:
        weight_splits = [w]
    diag_blocks_list = []
    for weight_split in weight_splits:
        diag_blocks, (diag_count, _) = to_blocks(weight_split, block_shape, diag=True)
        diag_blocks_list.append(diag_blocks)
    n_out_packed = n_out // PACK
    messages = np.full((2, n_out_packed, diag_count, n_in_complex), None, dtype=object)
    for rep in range(2):
        for diag_index in range(diag_count):
            combined_diagonal = np.concatenate(
                (diag_blocks_list[rep * 2][diag_index], diag_blocks_list[rep * 2 + 1][diag_index]),
                axis=0,
            )
            for n in range(n_in_complex):
                for out_index in range(n_out_packed):
                    msg = np.zeros((SLOT_COUNT,), dtype=complex)
                    rotations = out_index * 16 + PACK_RANGE
                    input_indices = ((n // 16) * 16 + out_index * 16 + (n + PACK_RANGE) % 16) % n_in_complex
                    input_indices = (input_indices - rotations) % n_in
                    values = scale * gather_upper_diagonal_batch(
                        combined_diagonal, input_indices, rotations, n_in, n_in_complex
                    )
                    assign_packed_dense_block(msg, values, FF_SLOT_INDICES)
                    messages[rep, out_index, diag_index, n] = msg
    return messages


def encode_w_pooler(w, block_shape=(128, 128)):
    diag_blocks, (diag_count, _) = to_blocks(w, block_shape, diag=True)
    messages = np.full((diag_count, 4), None, dtype=object)
    for diag_index in range(diag_count):
        blocks = diag_blocks[diag_index]
        for n in range(4):
            msg = np.zeros((SLOT_COUNT,), dtype=complex)
            for pack_offset in range(PACK):
                input_index = n * 16 + pack_offset
                group_offset = pack_offset * GROUP_SIZE
                values = (blocks[:, :, input_index] - 1j * blocks[:, :, input_index + 64]) / 2
                msg[group_offset + DIM_SLOT_BASE + np.arange(values.shape[0])[None, :]] = values.T
            messages[diag_index, n] = msg
    return messages


def encode_w_cls(w):
    class_count = w.shape[0]
    messages = np.full((class_count,), None, dtype=object)
    positions = DIM_SLOT_BASE + np.arange(6)[None, :]
    for class_index in range(class_count):
        msg = np.zeros((SLOT_COUNT,), dtype=float)
        blocks = w[class_index].reshape(6, DIM)
        msg[positions] = blocks.T
        messages[class_index] = msg
    return messages


def encode_b(b, n_blocks, n_out, pack=16, n_slot=16, pad_index=None, scale=1):
    if pad_index is None:
        pad_index = [slot for slot in range(n_slot) if slot >= n_blocks]
    elif n_blocks + len(pad_index) != n_slot:
        raise ValueError("Parameters do not match")
    if b.shape[0] % n_blocks != 0:
        raise ValueError("Block size does not match")
    blocks = np.stack(np.split(b, n_blocks), axis=0)
    n_out_packed = n_out // pack
    messages = np.full((n_out_packed,), None, dtype=object)
    active_slots = np.array([slot for slot in range(n_slot) if slot not in pad_index])
    output_positions = DIM_SLOT_BASE + active_slots[None, :]
    pack_group_base = GROUP_SIZE * np.arange(pack)[:, None, None]
    for out_index in range(n_out_packed):
        msg = np.zeros((SLOT_COUNT,), dtype=float)
        rotations = out_index * pack + np.arange(pack)
        gather_indices = (rotations[:, None] + DIM_RANGE[None, :]) % blocks.shape[1]
        rotated = np.take_along_axis(blocks[:, None, :], gather_indices[None, :, :], axis=2).transpose(1, 2, 0)
        msg[pack_group_base + output_positions[None, :, :]] = (scale * rotated) / 2
        messages[out_index] = msg
    return messages


def encode_b_pooler(b, n_blocks, n_slot=16, pad_index=None):
    if pad_index is None:
        pad_index = [slot for slot in range(n_slot) if slot >= n_blocks]
    elif n_blocks + len(pad_index) != n_slot:
        raise ValueError("Parameters do not match")
    if b.shape[0] % n_blocks != 0:
        raise ValueError("Block size does not match")
    blocks = np.stack(np.split(b, n_blocks), axis=1)
    msg = np.zeros((GROUP_SIZE,), dtype=float)
    active_slots = np.array([slot for slot in range(n_slot) if slot not in pad_index])
    msg[DIM_SLOT_BASE + active_slots[None, :]] = blocks / 2
    messages = np.full((1,), None, dtype=object)
    messages[0] = np.tile(msg, 2**4)
    return messages


def encode_b_cls(b):
    class_count = b.shape[0]
    messages = np.full((class_count,), None, dtype=object)
    for class_index in range(class_count):
        msg = np.zeros((SLOT_COUNT,), dtype=float)
        msg[0] = b[class_index]
        messages[class_index] = msg
    return messages


def write_light_plaintext_to_file(engine, messages, level, path):
    for index, message in np.ndenumerate(messages):
        postfix = "_".join(map(str, index))
        light_plaintext = engine.encode_to_light_plaintext(message, level)
        engine.write_light_plaintext(light_plaintext, str(path) + postfix)


def pre_encode_masks(engine, light_plaintext_path):
    path = light_plaintext_path / "masks"
    path.mkdir(parents=True, exist_ok=True)

    ccmm_path = path / "ccmm"
    transpose_path = path / "transpose"
    rotate_internal_path = path / "rotate_internal"

    for index in range(4):
        (ccmm_path / str(index)).mkdir(parents=True, exist_ok=True)
        (transpose_path / str(index)).mkdir(parents=True, exist_ok=True)
    for name in ("attention", "block_diag_1", "block_diag_2"):
        (rotate_internal_path / name).mkdir(parents=True, exist_ok=True)

    array = np.full((2**15,), 1)
    array[np.arange(2**15) % (2**12) >= 2**11] = 0
    engine.write_light_plaintext(engine.encode_to_light_plaintext(array), str(path / "make_copies_0"))

    array = np.full((2**15,), 1)
    array[np.arange(2**15) % (2**12) < 2**11] = 0
    engine.write_light_plaintext(engine.encode_to_light_plaintext(array), str(path / "make_copies_1"))

    array = np.ones((2**15,), dtype=int)
    array[np.arange(2**15) % 16 < 6] = 0
    engine.write_light_plaintext(engine.encode_to_light_plaintext(array), str(path / "attention_dense"))

    array = np.full((2**15,), 1, dtype=int)
    array[np.arange(2**15) % 16 >= 6] = 0
    engine.write_light_plaintext(engine.encode_to_light_plaintext(array), str(path / "intermediate_dense"))

    array = np.zeros((2**15,), dtype=int)
    array[np.arange(2**15) % (2**11) < 6] = 1
    engine.write_light_plaintext(engine.encode_to_light_plaintext(array), str(path / "pooler_dense"))

    for i in range(1, 128):
        array = np.ones((2**15,), dtype=int)
        array[np.arange(2**15) % (2**11) >= 16 * i] = 0
        engine.write_light_plaintext(
            engine.encode_to_light_plaintext(array),
            str(rotate_internal_path / "attention" / str(i)),
        )

    for i in range(1, 16):
        array = np.ones((2**15,), dtype=int)
        array[np.arange(2**15) % 16 >= i] = 0
        engine.write_light_plaintext(
            engine.encode_to_light_plaintext(array),
            str(rotate_internal_path / "block_diag_1" / str(i)),
        )

    for i in range(1, 8):
        array = np.ones((2**15,), dtype=int)
        array[np.arange(2**15) % 8 >= i] = 0
        engine.write_light_plaintext(
            engine.encode_to_light_plaintext(array),
            str(rotate_internal_path / "block_diag_2" / str(i)),
        )

    for i in range(8):
        array = np.zeros((2**15,), dtype=int)
        array[2**12 * i : 2**12 * (i + 1)] = 1
        engine.write_light_plaintext(
            engine.encode_to_light_plaintext(array * (1 / 4)),
            str(path / f"make_copies_2_{i}"),
        )

    for i in range(4):
        diag_index = 16 * i
        arr0 = np.array([1] * 16 * (64 + (diag_index - 16) % 64 + 16))
        arr1 = np.array(
            [0] * (2**15 - 16 * (64 - ((diag_index - 16) % 64 + 16))) + [1] * 16 * (64 - ((diag_index - 16) % 64 + 16))
        )
        engine.write_light_plaintext(engine.encode_to_light_plaintext(arr0), str(transpose_path / "0" / str(i)))
        engine.write_light_plaintext(engine.encode_to_light_plaintext(arr1), str(transpose_path / "1" / str(i)))
        for j in range(16 * i + 1, 16 * (i + 1)):
            l = 64 - j  # noqa: E741
            arr2 = np.array([0] * 2**11 * (16 - j % 16) + [1] * (128 - l) * 2**4)
            arr3 = np.array([0] * 2**11 * (16 - j % 16 - 1) + [0] * (128 - l) * 2**4 + [1] * 16 * l)
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr2), str(transpose_path / "2" / str(j)))
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr3), str(transpose_path / "3" / str(j)))

    for n in range(1, 128):
        rot = n
        j = n % 16
        arr0 = np.full((2**15,), 1, dtype=float)
        arr0[np.arange(2**15) % (2**11) >= (2**11 - 16 * rot)] = 0
        arr1 = np.full((2**15,), 0, dtype=float)
        arr1[np.arange(2**15) % (2**11) >= (2**11 - 16 * rot)] = 1
        if j == 0:
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr0), str(ccmm_path / "0" / str(n)))
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr1), str(ccmm_path / "1" / str(n)))
        else:
            arr0[: (2**11) * j] = 0
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr0), str(ccmm_path / "0" / str(n)))
            arr1[-(2**11) :] = 0
            if j > 1:
                arr1[: (2**11) * (j - 1)] = 0
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr1), str(ccmm_path / "1" / str(n)))
            arr2 = np.full((2**15,), 1, dtype=float)
            arr2[np.arange(2**15) % (2**11) >= (2**11 - 16 * rot)] = 0
            arr2[(2**11) * j :] = 0
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr2), str(ccmm_path / "2" / str(n)))
            arr3 = np.full((2**15,), 1, dtype=float)
            arr3 = arr3 - arr0 - arr1 - arr2
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr3), str(ccmm_path / "3" / str(n)))


def pre_encode_stage_03(engine, weights, layer_index, light_plaintext_path):
    level = 8 if layer_index == 0 else 11
    bert_prefix = f"bert.encoder.layer.{layer_index}.attention.self"
    query_weight = encode_w_qkv(get_weight_array(weights, f"{bert_prefix}.query.weight"))
    query_bias = encode_b(get_weight_array(weights, f"{bert_prefix}.query.bias"), n_blocks=12, n_out=64)
    path = light_plaintext_path / "stage_03" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)
    write_light_plaintext_to_file(engine, query_weight, level, path / "w_")
    write_light_plaintext_to_file(engine, query_bias, level - 1, path / "b_")


def pre_encode_stage_04(engine, weights, layer_index, light_plaintext_path):
    level = 8 if layer_index == 0 else 11
    bert_prefix = f"bert.encoder.layer.{layer_index}.attention.self"
    softmax_scale = 1 / 1024 if layer_index == 2 else 1 / 512
    key_weight = encode_w_qkv(get_weight_array(weights, f"{bert_prefix}.key.weight"), scale=softmax_scale)
    key_bias = encode_b(
        get_weight_array(weights, f"{bert_prefix}.key.bias"), n_blocks=12, n_out=64, scale=softmax_scale
    )
    path = light_plaintext_path / "stage_04" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)
    write_light_plaintext_to_file(engine, key_weight, level, path / "w_")
    write_light_plaintext_to_file(engine, key_bias, level - 1, path / "b_")


def pre_encode_stage_05(engine, weights, layer_index, light_plaintext_path):
    level = 8 if layer_index == 0 else 11
    bert_prefix = f"bert.encoder.layer.{layer_index}.attention.self"
    value_weight = encode_w_qkv(get_weight_array(weights, f"{bert_prefix}.value.weight"))
    value_bias = encode_b(get_weight_array(weights, f"{bert_prefix}.value.bias"), n_blocks=12, n_out=64)
    path = light_plaintext_path / "stage_05" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)
    write_light_plaintext_to_file(engine, value_weight, level, path / "w_")
    write_light_plaintext_to_file(engine, value_bias, level - 1, path / "b_")


def pre_encode_stage_10(engine, weights, layer_index, light_plaintext_path):
    level = 14
    bert_prefix = f"bert.encoder.layer.{layer_index}.attention.output.dense"
    weight = encode_w_att(
        get_weight_array(weights, f"{bert_prefix}.weight"), n_in=64, n_out=128, block_shape=(128, 64)
    )
    bias = encode_b(get_weight_array(weights, f"{bert_prefix}.bias"), n_blocks=6, n_out=128)
    path = light_plaintext_path / "stage_10" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)
    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_11(engine, weights, layer_index, light_plaintext_path):
    level = 8 if layer_index == 0 else 11
    bert_prefix = f"bert.encoder.layer.{layer_index}.attention.output.LayerNorm"
    weight = encode_b(get_weight_array(weights, f"{bert_prefix}.weight"), n_blocks=6, n_out=128)
    bias = encode_b(get_weight_array(weights, f"{bert_prefix}.bias"), n_blocks=6, n_out=128)
    path = light_plaintext_path / "stage_11" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)
    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_12(engine, weights, layer_index, light_plaintext_path):
    level = 4 if layer_index == 0 else 7
    bert_prefix = f"bert.encoder.layer.{layer_index}.intermediate.dense"
    weight = encode_w_ff(
        get_weight_array(weights, f"{bert_prefix}.weight"),
        n_in=128, n_out=128, block_shape=(128, 128), vsplit=4, scale=1 / 64,
    )
    bias_chunks = np.split(get_weight_array(weights, f"{bert_prefix}.bias"), 2)
    bias = np.full((2, 8), None, dtype=object)
    for rep in range(2):
        bias[rep] = encode_b(
            bias_chunks[rep], n_blocks=12, n_out=128, pad_index=(6, 7, 14, 15), scale=1 / 64,
        )
    path = light_plaintext_path / "stage_12" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)
    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_14(engine, weights, layer_index, light_plaintext_path):
    level = 3
    bert_prefix = f"bert.encoder.layer.{layer_index}.output.dense"
    weight = encode_w_ff(
        get_weight_array(weights, f"{bert_prefix}.weight"),
        n_in=128, n_out=128, block_shape=(128, 128), hsplit=4,
    )
    bias = encode_b(get_weight_array(weights, f"{bert_prefix}.bias"), n_blocks=6, n_out=128)
    path = light_plaintext_path / "stage_14" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)
    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_16(engine, weights, layer_index, light_plaintext_path):
    level = 14
    bert_prefix = f"bert.encoder.layer.{layer_index}.output.LayerNorm"
    weight = encode_b(get_weight_array(weights, f"{bert_prefix}.weight"), n_blocks=6, n_out=128)
    bias = encode_b(get_weight_array(weights, f"{bert_prefix}.bias"), n_blocks=6, n_out=128)
    path = light_plaintext_path / "stage_16" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)
    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_17(engine, weights, light_plaintext_path):
    level = 14
    weight = encode_w_pooler(get_weight_array(weights, "bert.pooler.dense.weight"))
    bias = encode_b_pooler(get_weight_array(weights, "bert.pooler.dense.bias"), n_blocks=6)
    path = light_plaintext_path / "stage_17"
    path.mkdir(parents=True, exist_ok=True)
    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_18(engine, weights, light_plaintext_path):
    level = 14
    classifier_prefix = get_classifier_prefix(weights)
    weight = encode_w_cls(get_weight_array(weights, f"{classifier_prefix}.weight"))
    bias = encode_b_cls(get_weight_array(weights, f"{classifier_prefix}.bias"))
    path = light_plaintext_path / "stage_18"
    path.mkdir(parents=True, exist_ok=True)
    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")
