from collections import defaultdict
from pathlib import Path

import numpy as np

from timer import Timer
from params import InstanceParams
from timer import Timer
from params import InstanceParams
from desilofhe import (
    BootstrapKey,
    Ciphertext,
    ConjugationKey,
    Engine,
    FixedRotationKey,
    Plaintext,
    PublicKey,
    RelinearizationKey,
)

SLOT_COUNT = 2**15
GROUP_SIZE = 2**11


class DeltaCiphertext:
    def __init__(self, ciphertext: Ciphertext, delta: float):
        self.ciphertext = ciphertext
        self.delta = delta


class HE:
    def __init__(self, params: InstanceParams, compact: bool, bootstrap_key_size: str, thread_count: int = 16):
        self.compact = compact
        self.bootstrap_key_size = bootstrap_key_size
        self.timer = Timer()

        mode = "compact" if compact else "default"
        self.light_plaintext_path = params.server_data_dir() / "light_plaintexts" / mode
        self.mask_path = self.light_plaintext_path / "masks"

        public_keys_dir = params.iodir() / "public_keys"
        fixed_rotation_keys_dir = public_keys_dir / "fixed_rotation_keys"

        if thread_count == 1:
            self.engine = Engine(use_bootstrap_to_14_levels=True, compact=compact)
        else:
            self.engine = Engine(
                use_bootstrap_to_14_levels=True,
                mode="parallel",
                thread_count=thread_count,
                compact=compact,
            )

        self.public_key: PublicKey = self.engine.read_public_key(public_keys_dir / "public_key")
        self.conjugation_key: ConjugationKey = self.engine.read_conjugation_key(
            public_keys_dir / "conjugation_key"
        )
        self.relinearization_key: RelinearizationKey = self.engine.read_relinearization_key(
            public_keys_dir / "relinearization_key"
        )
        self.bootstrap_key: BootstrapKey = self.engine.read_bootstrap_key(
            public_keys_dir / "bootstrap_key"
        )

        # fmt: off
        if compact:
            self.bootstrap_deltas = frozenset([
                1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 64, 96, 128, 160, 192, 224, 256, 512, 768,
                1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 16384, 24576, 31744, 32000, 32256,
                32512, 32736, 32744, 32752, 32760,
            ])
        else:
            self.bootstrap_deltas = frozenset([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 96, 128, 160, 192,
                224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 1024, 2048, 3072, 4096, 5120,
                6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 31744,
                32256, 32736, 32752,
            ])

        _rotation_contexts = [
             (0, 9), (1, 14), (2, 14), (3, 12), (4, 14), (5, 12), (6, 11), (8, 7), (16, 10), (32, 10), (64, 10), (128, 10), (240, 9), (256, 10), (496, 9), (512, 10), (752, 9), (1008, 9), (1024, 10), (1264, 9), (1520, 9), (1776, 9), (2032, 9), (2048, 8), (2272, 9), (2528, 9), (2784, 9), (3040, 9), (3184, 9), (3296, 9), (3312, 9), (3440, 9), (3552, 9), (3568, 9), (3696, 9), (3808, 9), (3824, 9), (3952, 9), (4064, 9), (4080, 9), (4304, 9), (4560, 9), (4816, 9), (5072, 9), (5328, 9), (5584, 9), (5840, 9), (6096, 9), (6336, 9), (6592, 9), (6848, 9), (7104, 9), (7264, 9), (7360, 9), (7392, 9), (7520, 9), (7616, 9), (7648, 9), (7776, 9), (7872, 9), (7904, 9), (8032, 9), (8128, 9), (8160, 9), (8368, 9), (8624, 9), (8880, 9), (9136, 9), (9392, 9), (9648, 9), (9904, 9), (10160, 9), (10400, 9), (10656, 9), (10912, 9), (11168, 9), (11344, 9), (11424, 9), (11472, 9), (11600, 9), (11680, 9), (11728, 9), (11856, 9), (11936, 9), (11984, 9), (12112, 9), (12192, 9), (12240, 9), (12432, 9), (12688, 9), (12944, 9), (13200, 9), (13456, 9), (13712, 9), (13968, 9), (14224, 9), (14464, 9), (14720, 9), (14976, 9), (15232, 9), (15424, 9), (15488, 9), (15552, 9), (15680, 9), (15744, 9), (15808, 9), (15936, 9), (16000, 9), (16064, 9), (16192, 9), (16256, 9), (16320, 9), (16384, 13), (16496, 9), (16752, 9), (17008, 9), (17264, 9), (17520, 9), (17776, 9), (18032, 9), (18288, 9), (18528, 9), (18784, 9), (19040, 9), (19296, 9), (19504, 9), (19552, 9), (19632, 9), (19760, 9), (19808, 9), (19888, 9), (20016, 9), (20064, 9), (20144, 9), (20272, 9), (20320, 9), (20400, 9), (20560, 9), (20816, 9), (21072, 9), (21328, 9), (21584, 9), (21840, 9), (22096, 9), (22352, 9), (22592, 9), (22848, 9), (23104, 9), (23360, 9), (23584, 9), (23616, 9), (23712, 9), (23840, 9), (23872, 9), (23968, 9), (24096, 9), (24128, 9), (24224, 9), (24352, 9), (24384, 9), (24480, 9), (24576, 13), (24624, 9), (24880, 9), (25136, 9), (25392, 9), (25648, 9), (25904, 9), (26160, 9), (26416, 9), (26656, 9), (26912, 9), (27168, 9), (27424, 9), (27664, 9), (27680, 9), (27792, 9), (27920, 9), (27936, 9), (28048, 9), (28176, 9), (28192, 9), (28304, 9), (28432, 9), (28448, 9), (28560, 9), (28672, 14), (28688, 9), (28944, 9), (29200, 9), (29456, 9), (29712, 9), (29968, 9), (30224, 9), (30480, 9), (30720, 14), (30976, 9), (31232, 9), (31488, 9), (31744, 13), (31872, 9), (32000, 9), (32128, 9), (32256, 13), (32384, 9), (32512, 13), (32640, 13), (32704, 13), (32736, 13), (32752, 13), (32757, 12), (32758, 12), (32759, 12), (32760, 12), (32761, 12), (32762, 11), (32763, 8), (32764, 13), (32765, 8), (32766, 13), (32767, 13)  # noqa: E501
        ]
        # fmt: on

        self.fixed_rotation_keys: dict[int, FixedRotationKey] = {}
        for delta, _ in _rotation_contexts:
            if delta == 0 or delta in self.bootstrap_deltas:
                continue
            key_path = fixed_rotation_keys_dir / str(delta)
            self.fixed_rotation_keys[delta] = self.engine.read_fixed_rotation_key(key_path)

        self.rotate_levels: dict[int, int] = defaultdict(int)

        max_level = self.engine.max_level

        self.masks = dict()
        self.masks["rotate_internal"] = dict(attention=dict(), block_diag_1=dict(), block_diag_2=dict())
        self.masks["make_copies"] = dict()
        self.masks["make_copies_2"] = dict()
        self.masks["transpose"] = dict(mask0=dict(), mask1=dict(), mask2=dict(), mask3=dict())
        self.masks["ccmm"] = [dict(), dict(), dict(), dict()]
        self.masks["make_copies"][0] = self.engine.read_light_plaintext(str(self.mask_path / "make_copies_0"))
        self.masks["make_copies"][1] = self.engine.read_light_plaintext(str(self.mask_path / "make_copies_1"))
        self.masks["attention_dense"] = self.engine.read_light_plaintext(str(self.mask_path / "attention_dense"))

        layernorm_mask = np.array(([1] * 1 + [0] * 15) * 2**11)
        self.masks["layernorm"] = layernorm_mask
        self.masks["invsqrt_b"] = self.prepare_for_multiply(
            self.engine.encrypt(layernorm_mask, self.public_key, max_level)
        )
        self.masks["intermediate_dense"] = self.engine.read_light_plaintext(str(self.mask_path / "intermediate_dense"))
        self.masks["pooler_dense"] = self.engine.read_light_plaintext(str(self.mask_path / "pooler_dense"))
        self.masks["inv_a"] = self.engine.encrypt(
            np.array(([1] * 12 + [0.0] * 4) * 2**11), self.public_key, max_level
        )

        for i in range(1, 128):
            self.masks["rotate_internal"]["attention"][i] = self.engine.read_light_plaintext(
                str(self.mask_path / "rotate_internal" / "attention" / str(i))
            )
        for i in range(1, 16):
            self.masks["rotate_internal"]["block_diag_1"][i] = self.engine.read_light_plaintext(
                str(self.mask_path / "rotate_internal" / "block_diag_1" / str(i))
            )
        for i in range(1, 8):
            self.masks["rotate_internal"]["block_diag_2"][i] = self.engine.read_light_plaintext(
                str(self.mask_path / "rotate_internal" / "block_diag_2" / str(i))
            )
        for i in range(8):
            self.masks["make_copies_2"][i] = self.engine.read_light_plaintext(
                str(self.mask_path / f"make_copies_2_{i}")
            )
        for i in range(4):
            self.masks["transpose"]["mask0"][i] = self.engine.read_light_plaintext(
                str(self.mask_path / "transpose" / "0" / str(i))
            )
            self.masks["transpose"]["mask1"][i] = self.engine.read_light_plaintext(
                str(self.mask_path / "transpose" / "1" / str(i))
            )
            for j in range(16 * i + 1, 16 * (i + 1)):
                self.masks["transpose"]["mask2"][j] = self.engine.read_light_plaintext(
                    str(self.mask_path / "transpose" / "2" / str(j))
                )
                self.masks["transpose"]["mask3"][j] = self.engine.read_light_plaintext(
                    str(self.mask_path / "transpose" / "3" / str(j))
                )
        for n in range(1, 128):
            masks = self.masks["ccmm"]
            masks[0][n] = self.engine.read_light_plaintext(str(self.mask_path / "ccmm" / "0" / str(n)))
            masks[1][n] = self.engine.read_light_plaintext(str(self.mask_path / "ccmm" / "1" / str(n)))
            if n % 16 != 0:
                masks[2][n] = self.engine.read_light_plaintext(str(self.mask_path / "ccmm" / "2" / str(n)))
                masks[3][n] = self.engine.read_light_plaintext(str(self.mask_path / "ccmm" / "3" / str(n)))

    def add(self, x, y):
        return self.engine.add(x, y)

    def add_inplace(self, x, y):
        return self.engine.add(x, y, out=x)

    def subtract(self, x, y):
        return self.engine.subtract(x, y)

    def multiply(self, x, y):
        return self.engine.multiply(x, y)

    def square(self, x):
        return self.engine.square(x)

    def multiply_1j(self, x):
        return self.engine.multiply_imaginary_integer(x, 1)

    def conjugate(self, ciphertext):
        return self.engine.conjugate(ciphertext, self.conjugation_key)

    def relinearize(self, ciphertext):
        return self.engine.relinearize(ciphertext, self.relinearization_key)

    def rotate(self, ciphertext, delta):
        normalized_delta = (SLOT_COUNT + delta) % SLOT_COUNT
        self.rotate_levels[normalized_delta] = max(self.rotate_levels[normalized_delta], ciphertext.level)
        if normalized_delta == 0:
            return self.engine.clone(ciphertext)
        if normalized_delta in self.bootstrap_deltas:
            return self.engine.rotate(ciphertext, self.bootstrap_key, normalized_delta)
        fixed_rotation_key = self.fixed_rotation_keys[normalized_delta]
        return self.engine.rotate(ciphertext, fixed_rotation_key)

    def rescale(self, ciphertext):
        return self.engine.rescale(ciphertext)

    def level_down(self, ciphertext, by):
        return self.engine.level_down(ciphertext, ciphertext.level - by)

    def ntt(self, ciphertext):
        return self.engine.ntt(ciphertext)

    def intt(self, ciphertext):
        return self.engine.intt(ciphertext)

    def prepare_for_multiply(self, ciphertext):
        return self.ntt(self.rescale(ciphertext))

    def encode(self, message, level):
        return self.engine.encode(message, level)

    def bootstrap(self, ciphertext):
        return self.engine.bootstrap(
            ciphertext,
            self.relinearization_key,
            self.conjugation_key,
            self.bootstrap_key,
        )

    def get_input_level(self, x):
        return min(x.flat[0].level_available, 14)

    def stage_01_complexify_x(self, x, layer_index):
        x_cplx = np.full((4,), None, dtype=object)
        if x.shape == (8,):
            for i in range(4):
                front = x[i]
                back = x[i + 4]
                back_1j = self.multiply_1j(back)
                x_cplx[i] = self.add(front, back_1j)
            if layer_index != 0:
                for i in range(4):
                    rotated = self.rotate(x_cplx[i], 6)
                    self.add_inplace(x_cplx[i], rotated)
        elif x.shape == (4,):
            x_cplx = np.full((4,), None, dtype=object)
            temp = np.full((8,), None, dtype=object)
            for i in range(4):
                orig = x[i]
                conj = self.conjugate(orig)
                added = self.add(orig, conj)
                subtracted = self.subtract(conj, orig)
                subtracted_1j = self.multiply_1j(subtracted)
                temp[i] = self.multiply(added, 1 / 2)
                temp[i + 4] = self.multiply(subtracted_1j, 1 / 2)
                x_cplx[i] = self.engine.level_down(orig, 6)
            x = temp
        return x, x_cplx

    def stage_02_make_rotated_copies(self, x):
        rotated = np.full((16 * x.shape[0],), None, dtype=object)
        for source_index, ciphertext in np.ndenumerate(x):
            rotated[16 * source_index[0]] = ciphertext
            for rotate_index in range(1, 16):
                index = 16 * source_index[0] + rotate_index
                rotated[index] = self.rotate(rotated[index - 1], -(2**11))
        return rotated

    def parallel_diagonal_pc_mult(self, ws, xs):
        out_dim, diag_dim, in_dim = ws.shape
        output = np.full((out_dim, diag_dim), None, dtype=object)
        prepared = np.full(xs.shape, None, dtype=object)
        for index, x in np.ndenumerate(xs):
            prepared[index] = self.prepare_for_multiply(x)
        for out_index in range(out_dim):
            for diag_index in range(diag_dim):
                temp = self.multiply(ws[out_index, diag_index, 0], prepared[(16 * out_index) % in_dim])
                for in_index in range(1, in_dim):
                    w = ws[out_index, diag_index, in_index]
                    x = prepared[(16 * out_index + in_index) % in_dim]
                    wx = self.multiply(w, x)
                    self.add_inplace(temp, wx)
                output[out_index, diag_index] = self.rescale(temp)
        return output

    def rotate_internal(self, x, delta=0, left_delta=0, right_delta=0, mask=None, mode=None):
        if delta == 0 and left_delta == 0:
            return self.level_down(x, by=1)
        if mask is None:
            mask = self.masks["rotate_internal"][mode][delta]
        match mode:
            case "attention":
                left_delta = 16 * delta
                right_delta = 2**11 - left_delta
            case "block_diag_1":
                left_delta = delta
                right_delta = 12 - left_delta
            case "block_diag_2":
                left_delta = delta
                right_delta = 6 - left_delta
        masked = self.multiply(mask, x)
        subtracted = self.subtract(x, masked)
        right_rotated = self.rotate(masked, right_delta)
        left_rotated = self.rotate(subtracted, -left_delta)
        return self.add(left_rotated, right_rotated)

    def pcmm(self, ws, xs, mode="block_diag_1"):
        out_dim, diag_dim, in_dim = ws.shape
        submatrices = self.parallel_diagonal_pc_mult(ws, xs)
        output = np.full((out_dim,), None, dtype=object)
        for out_index in range(out_dim):
            temp = self.level_down(submatrices[out_index, 0], by=1)
            for diag_index in range(1, diag_dim):
                match mode:
                    case "block_diag_1":
                        rotated = self.rotate_internal(
                            submatrices[out_index, diag_index],
                            delta=12 - diag_index,
                            mode=mode,
                        )
                    case "block_diag_2":
                        rotated = self.rotate_internal(
                            submatrices[out_index, diag_index],
                            delta=6 - diag_index,
                            mode=mode,
                        )
                self.add_inplace(temp, rotated)
            output[out_index] = temp
        return output

    def interval_sum(self, x, interval, need_intt=False):
        count = int(np.log2(2**15 / interval))
        temp = self.intt(x) if need_intt else x
        for index in range(count):
            rotated = self.rotate(temp, -interval * 2**index)
            self.add_inplace(temp, rotated)
        return temp

    def make_copies(self, x, scale=1 / 2):
        copies = np.full((16 * x.shape[0],), None, dtype=object)
        scale = scale / 2
        masks = self.masks["make_copies_2"]
        encoded_mask0 = self.masks["make_copies"][0]
        encoded_mask1 = self.masks["make_copies"][1]
        for index in range(x.shape[0] // 2):
            front = x[index]
            back = x[index + x.shape[0] // 2]
            back_1j = self.multiply_1j(back)
            merged = self.add(front, back_1j)
            prepared = self.prepare_for_multiply(merged)
            for copy_index in range(8):
                masked = self.multiply(masks[copy_index], prepared)
                rescaled = self.rescale(masked)
                copied = self.interval_sum(rescaled, 2**12, need_intt=True)
                ntted = self.ntt(copied)
                temp0 = self.rescale(self.intt(self.multiply(encoded_mask0, ntted)))
                temp1 = self.rescale(self.intt(self.multiply(encoded_mask1, ntted)))
                self.add_inplace(temp0, self.rotate(temp0, 2**11))
                self.add_inplace(temp1, self.rotate(temp1, -(2**11)))
                conjugated0 = self.conjugate(temp0)
                conjugated1 = self.conjugate(temp1)
                a = self.ntt(self.add(temp0, conjugated0))
                b = self.ntt(self.multiply_1j(self.subtract(conjugated0, temp0)))
                c = self.ntt(self.add(temp1, conjugated1))
                d = self.ntt(self.multiply_1j(self.subtract(conjugated1, temp1)))
                copies[index * 16 + 2 * copy_index] = a
                copies[(index + x.shape[0] // 2) * 16 + 2 * copy_index] = b
                copies[index * 16 + 2 * copy_index + 1] = c
                copies[(index + x.shape[0] // 2) * 16 + 2 * copy_index + 1] = d
        return copies

    def stage_03_query(self, x, layer_index):
        return self._apply_qkv_weight_bias(x, layer_index, "stage_03")

    def stage_04_key(self, x, layer_index):
        return self._apply_qkv_weight_bias(x, layer_index, "stage_04")

    def stage_05_value(self, x, layer_index):
        return self._apply_qkv_weight_bias(x, layer_index, "stage_05")

    def _apply_qkv_weight_bias(self, x, layer_index, stage_name):
        weight = np.full((4, 6, 64), None, dtype=object)
        bias = np.full((4,), None, dtype=object)
        path = self.light_plaintext_path / stage_name / f"layer_{layer_index}"
        with self.timer.paused():
            for index in np.ndindex(weight.shape):
                postfix = "_".join(map(str, index))
                weight[index] = self.engine.read_light_plaintext(str(path / f"w_{postfix}"))
            for index in np.ndindex(bias.shape):
                postfix = "_".join(map(str, index))
                bias[index] = self.engine.read_light_plaintext(str(path / f"b_{postfix}"))
        wx = self.pcmm(weight, x)
        output = np.full((4,), None, dtype=object)
        for i in range(4):
            biased = self.add(bias[i], wx[i])
            conj = self.conjugate(biased)
            output[i] = self.add(biased, conj)
        return output

    def transpose_upper_to_lower(self, upper):
        lower_temp = np.ndarray((upper.shape[0], 2), dtype=object)
        for index in range(4):
            diag_index = 16 * index
            rotated = self.rotate(upper[index], -(((64 - diag_index) % 64) * 16) % (2**15))
            prepared = self.prepare_for_multiply(rotated)
            mask0 = self.masks["transpose"]["mask0"][index]
            mask1 = self.masks["transpose"]["mask1"][index]
            temp0 = self.multiply(mask0, prepared)
            temp1 = self.multiply(mask1, prepared)
            lower_temp[(4 - index) % 4][0] = temp0
            lower_temp[(4 - index) % 4][1] = temp1
        for index in range(4):
            for upper_diag in range(16 * index + 1, 16 * (index + 1)):
                lower_diag = 64 - upper_diag
                delta = -(lower_diag * 2**4 + (((upper_diag % 48) * 2) % 16) * 2**11) % (2**15)
                rotated = self.rotate(upper[index], delta)
                prepared = self.prepare_for_multiply(rotated)
                mask2 = self.masks["transpose"]["mask2"][upper_diag]
                mask3 = self.masks["transpose"]["mask3"][upper_diag]
                temp2 = self.multiply(mask2, prepared)
                temp3 = self.multiply(mask3, prepared)
                self.add_inplace(lower_temp[(3 - index) % 4][0], temp2)
                self.add_inplace(lower_temp[(3 - index) % 4][1], temp3)
        lower = np.full((4,), None, dtype=object)
        for index in range(4):
            rotated = self.rotate(lower_temp[index][1], 2**11)
            lower[index] = self.add(lower_temp[index][0], rotated)
        return lower

    def stage_06_attention_score(self, q, k):
        k_lower = self.transpose_upper_to_lower(k)
        k_lower_complex = np.full((4,), None, dtype=object)
        for index in range(4):
            rotated = self.rotate_internal(k_lower[index], 64, mode="attention")
            rotated_1j = self.multiply_1j(rotated)
            k_lower_complex[index] = self.rescale(self.add(k_lower[index], rotated_1j))
        q_copies = self.make_copies(q)
        in_dim, out_dim = 64, 4
        ttemp = np.full((out_dim, 4), None, dtype=object)
        masks = self.masks["ccmm"]
        for out_index in range(out_dim):
            multiplied = self.multiply(k_lower_complex[out_index], q_copies[0])
            ttemp[out_index, 0] = self.ntt(self.level_down(multiplied, by=1))
        for in_index in range(1, in_dim):
            q_val = in_index // 16
            j = in_index % 16
            rot = in_index
            l_prime = q_copies[in_index]
            temp = np.full((out_dim, 4), None, dtype=object)
            rrot = (2**11) * j - 16 * rot
            rrot = rrot % (2**15)

            def add_to_ttemp(index1, index2):
                if ttemp[index1] is None:
                    ttemp[index1] = temp[index2]
                else:
                    self.add_inplace(ttemp[index1], temp[index2])

            if j == 0:
                for out_index in range(out_dim):
                    rotated = self.rotate(k_lower_complex[out_index], rrot)
                    multiplied = self.multiply(rotated, l_prime)
                    rescaled = self.rescale(multiplied)
                    temp[out_index, 0] = self.multiply(masks[0][in_index], rescaled)
                    temp[out_index, 1] = self.subtract(multiplied, temp[out_index, 0])
                if q_val == 1:
                    add_to_ttemp((0, 2), (3, 0)); add_to_ttemp((0, 3), (3, 1))
                    add_to_ttemp((1, 0), (0, 0)); add_to_ttemp((1, 1), (0, 1))
                    add_to_ttemp((2, 0), (1, 0)); add_to_ttemp((2, 1), (1, 1))
                    add_to_ttemp((3, 0), (2, 0)); add_to_ttemp((3, 1), (2, 1))
                elif q_val == 2:
                    add_to_ttemp((0, 2), (2, 0)); add_to_ttemp((0, 3), (2, 1))
                    add_to_ttemp((1, 0), (3, 0)); add_to_ttemp((1, 1), (3, 1))
                    add_to_ttemp((2, 0), (0, 0)); add_to_ttemp((2, 1), (0, 1))
                    add_to_ttemp((3, 0), (1, 0)); add_to_ttemp((3, 1), (1, 1))
                elif q_val == 3:
                    add_to_ttemp((0, 2), (1, 0)); add_to_ttemp((0, 3), (1, 1))
                    add_to_ttemp((1, 0), (2, 0)); add_to_ttemp((1, 1), (2, 1))
                    add_to_ttemp((2, 0), (3, 0)); add_to_ttemp((2, 1), (3, 1))
                    add_to_ttemp((3, 0), (0, 0)); add_to_ttemp((3, 1), (0, 1))
            else:
                for out_index in range(out_dim):
                    rotated = self.rotate(k_lower_complex[out_index], rrot)
                    multiplied = self.multiply(rotated, l_prime)
                    rescaled = self.rescale(multiplied)
                    temp[out_index, 0] = self.multiply(masks[0][in_index], rescaled)
                    temp[out_index, 1] = self.multiply(masks[1][in_index], rescaled)
                    temp[out_index, 2] = self.multiply(masks[2][in_index], rescaled)
                    add_temp = self.add(self.add(temp[out_index, 0], temp[out_index, 1]), temp[out_index, 2])
                    temp[out_index, 3] = self.subtract(multiplied, add_temp)
                if q_val == 0:
                    add_to_ttemp((0, 2), (3, 2)); add_to_ttemp((0, 3), (3, 3))
                    add_to_ttemp((0, 0), (0, 0)); add_to_ttemp((0, 1), (0, 1))
                    add_to_ttemp((1, 0), (0, 2)); add_to_ttemp((1, 1), (0, 3))
                    add_to_ttemp((1, 0), (1, 0)); add_to_ttemp((1, 1), (1, 1))
                    add_to_ttemp((2, 0), (1, 2)); add_to_ttemp((2, 1), (1, 3))
                    add_to_ttemp((2, 0), (2, 0)); add_to_ttemp((2, 1), (2, 1))
                    add_to_ttemp((3, 0), (2, 2)); add_to_ttemp((3, 1), (2, 3))
                    add_to_ttemp((3, 0), (3, 0)); add_to_ttemp((3, 1), (3, 1))
                elif q_val == 1:
                    add_to_ttemp((0, 2), (2, 2)); add_to_ttemp((0, 3), (2, 3))
                    add_to_ttemp((0, 2), (3, 0)); add_to_ttemp((0, 3), (3, 1))
                    add_to_ttemp((1, 2), (3, 2)); add_to_ttemp((1, 3), (3, 3))
                    add_to_ttemp((1, 0), (0, 0)); add_to_ttemp((1, 1), (0, 1))
                    add_to_ttemp((2, 0), (0, 2)); add_to_ttemp((2, 1), (0, 3))
                    add_to_ttemp((2, 0), (1, 0)); add_to_ttemp((2, 1), (1, 1))
                    add_to_ttemp((3, 0), (1, 2)); add_to_ttemp((3, 1), (1, 3))
                    add_to_ttemp((3, 0), (2, 0)); add_to_ttemp((3, 1), (2, 1))
                elif q_val == 2:
                    add_to_ttemp((0, 2), (1, 2)); add_to_ttemp((0, 3), (1, 3))
                    add_to_ttemp((0, 2), (2, 0)); add_to_ttemp((0, 3), (2, 1))
                    add_to_ttemp((1, 2), (2, 2)); add_to_ttemp((1, 3), (2, 3))
                    add_to_ttemp((1, 2), (3, 0)); add_to_ttemp((1, 3), (3, 1))
                    add_to_ttemp((2, 2), (3, 2)); add_to_ttemp((2, 3), (3, 3))
                    add_to_ttemp((2, 0), (0, 0)); add_to_ttemp((2, 1), (0, 1))
                    add_to_ttemp((3, 0), (0, 2)); add_to_ttemp((3, 1), (0, 3))
                    add_to_ttemp((3, 0), (1, 0)); add_to_ttemp((3, 1), (1, 1))
                elif q_val == 3:
                    add_to_ttemp((0, 2), (0, 2)); add_to_ttemp((0, 3), (0, 3))
                    add_to_ttemp((0, 2), (1, 0)); add_to_ttemp((0, 3), (1, 1))
                    add_to_ttemp((1, 2), (1, 2)); add_to_ttemp((1, 3), (1, 3))
                    add_to_ttemp((1, 2), (2, 0)); add_to_ttemp((1, 3), (2, 1))
                    add_to_ttemp((2, 2), (2, 2)); add_to_ttemp((2, 3), (2, 3))
                    add_to_ttemp((2, 2), (3, 0)); add_to_ttemp((2, 3), (3, 1))
                    add_to_ttemp((3, 2), (3, 2)); add_to_ttemp((3, 3), (3, 3))
                    add_to_ttemp((3, 0), (0, 0)); add_to_ttemp((3, 1), (0, 1))

        output = np.full((8,), None, dtype=object)
        ttemp_relin = np.full((out_dim, 4), None, dtype=object)
        for out_index in range(out_dim):
            for i in range(4):
                ttemp_relin[out_index, i] = self.relinearize(ttemp[out_index, i])
            ttemp_relin[out_index, 1] = self.rotate(ttemp_relin[out_index, 1], 2**11)
            ttemp_relin[out_index, 3] = self.rotate(ttemp_relin[out_index, 3], 2**11)
            ttemp_relin[out_index, 2] = self.multiply_1j(
                self.conjugate(self.add(ttemp_relin[out_index, 2], ttemp_relin[out_index, 3]))
            )
            output_complex = self.add(
                self.add(ttemp_relin[out_index, 0], ttemp_relin[out_index, 1]),
                ttemp_relin[out_index, 2],
            )
            conj = self.conjugate(output_complex)
            output[out_index] = self.add(output_complex, conj)
            output[out_index + out_dim] = self.multiply_1j(self.subtract(conj, output_complex))
        return output

    def evaluate_polynomial_stockmeyer(self, x, coefficients):
        degree = len(coefficients) - 1
        x2 = self.relinearize(self.square(x))
        x3 = self.relinearize(self.multiply(x, x2))
        x_powers = [None, x, x2, x3]

        def evaluate_baby_poly(coefficients):
            if len(coefficients) == 1:
                return coefficients[0]
            result = self.multiply(coefficients[1], x_powers[1])
            if len(coefficients) == 2:
                return self.add(result, coefficients[0])
            elif len(coefficients) == 3:
                result = self.add(result, self.multiply(coefficients[2], x_powers[2]))
                return self.add(result, coefficients[0])
            else:
                result = self.add(result, self.multiply(coefficients[2], x_powers[2]))
                ax3 = self.relinearize(self.multiply(x_powers[2], self.multiply(coefficients[3], x)))
                result = self.add(result, ax3)
                return self.add(result, coefficients[0])

        x4 = self.relinearize(self.square(x2))
        x8 = self.relinearize(self.square(x4))
        if degree >= 16:
            x16 = self.relinearize(self.square(x8))
        if len(coefficients) > 16:
            sections = np.split(coefficients, len(coefficients) // 4)
            baby_results = [evaluate_baby_poly(section) for section in sections]
            giant_steps_1 = []
            for index in range(0, len(baby_results), 2):
                if index + 1 < len(baby_results):
                    giant_steps_1.append(self.add(baby_results[index], self.relinearize(self.multiply(baby_results[index + 1], x4))))
                else:
                    giant_steps_1.append(baby_results[index])
            giant_steps_2 = []
            for index in range(0, len(giant_steps_1), 2):
                if index + 1 < len(giant_steps_1):
                    giant_steps_2.append(self.add(giant_steps_1[index], self.relinearize(self.multiply(giant_steps_1[index + 1], x8))))
                else:
                    giant_steps_2.append(giant_steps_1[index])
            result = self.add(giant_steps_2[0], self.relinearize(self.multiply(giant_steps_2[1], x16)))
        elif len(coefficients) == 16:
            sections = np.split(coefficients, 4)
            baby_results = [evaluate_baby_poly(section) for section in sections]
            giant_steps_1 = []
            for index in range(0, 4, 2):
                giant_steps_1.append(self.add(baby_results[index], self.relinearize(self.multiply(baby_results[index + 1], x4))))
            result = self.add(giant_steps_1[0], self.relinearize(self.multiply(giant_steps_1[1], x8)))
        return result

    def he_exp1(self, x, min_x, max_x, n):
        mid_x = (min_x + max_x) / 2
        x = self.add(x, -mid_x / 32)
        p = [0.032855468333339584, 0.05948672763856172, 0.03881607331549499, 0.0670090353368128, 0.15202099984697098, 0.20618261949210986, 0.23721029007596767, 0.26787311936472025, 0.27220647178765545, 0.2379982262906916, 0.1780344447042791, 0.11128698173597897, 0.05566510463488879, 0.020873931555133732, 0.005218196900295354, 0.0006522770224130905]  # noqa: E501
        p.reverse()
        p = np.array(p)
        exp_x = self.evaluate_polynomial_stockmeyer(x, p)
        for _ in range(int(np.log2(n))):
            exp_x = self.relinearize(self.square(exp_x))
        return exp_x

    def he_exp2(self, x, min_x, max_x, n):
        mid_x = (min_x + max_x) / 2
        x = self.add(x, -mid_x / 64)
        p = [0.008201736399899691, 0.014226972463907047, -0.008386712802267769, -0.009262268572236316, 0.0397324053296174, 0.04817928878801878, 0.016604320800445653, 0.02336452059478656, 0.04217318306400685, 0.03517495704921328, 0.022268203231858744, 0.014216636671807894, 0.00749909544008294, 0.0027930565779849003, 0.0006877176070101981, 8.615994668877663e-05]  # noqa: E501
        p.reverse()
        p = np.array(p)
        exp_x = self.evaluate_polynomial_stockmeyer(x, p)
        for _ in range(int(np.log2(n))):
            exp_x = self.relinearize(self.square(exp_x))
        exp_x = self.multiply(exp_x, 128)
        return exp_x

    def he_inv(self, denominator, epsilon, alpha, delta=1):
        d = 0
        a = DeltaCiphertext(self.masks["inv_a"], delta)
        b = DeltaCiphertext(self.bootstrap(denominator), delta)
        e = epsilon
        while e < 1 - alpha:
            d = d + 1
            k = 2 / (e + 1)
            correction = self.prepare_for_multiply(self.engine.subtract(2 / k * b.delta, b.ciphertext))
            a_delta = a.delta * b.delta / k**2
            a = DeltaCiphertext(self.relinearize(self.multiply(a.ciphertext, correction)), a_delta)
            b_delta = b.delta * b.delta / k**2
            b = DeltaCiphertext(self.relinearize(self.multiply(b.ciphertext, correction)), b_delta)
            e = k * e * (2 - k * e)
            scale_adjust = int(1 / b.delta / 2**8)
            if scale_adjust > 1:
                self.add_inplace(a.ciphertext, self.conjugate(a.ciphertext))
                a.delta *= 2
                self.add_inplace(b.ciphertext, self.conjugate(b.ciphertext))
                b.delta *= 2
                scale_adjust = int(1 / b.delta / 2**8)
            else:
                scale_adjust = 1
            b = DeltaCiphertext(self.multiply(b.ciphertext, scale_adjust), b.delta * scale_adjust)
            a = DeltaCiphertext(self.multiply(a.ciphertext, scale_adjust), a.delta * scale_adjust)
        output_precision = e
        if d > 10:
            a.ciphertext = self.multiply(a.ciphertext, 1 / 128)
            a.ciphertext = self.bootstrap(a.ciphertext)
            a.ciphertext = self.multiply(a.ciphertext, 128)
        return a.ciphertext, a.delta, output_precision

    def update_inv_D(self, exp_u, attention_mask, inv_D, D_delta, output_precision, alpha, final_inv=False):
        exp_2u = np.full((8,), None, dtype=object)
        prepared = self.rescale(inv_D)
        inv_D_list = [self.multiply(prepared, attention_mask[i]) for i in range(8)]
        k = max(int(1 / D_delta / 2), 1)

        def multiply(x, y):
            return self.relinearize(self.multiply(x, y))

        def square(x):
            return self.relinearize(self.square(x))

        for index in range(4):
            partial_softmax0 = square(self.multiply(multiply(self.add(exp_u[index], exp_u[index]), inv_D_list[index]), k))
            partial_softmax1 = square(self.multiply(multiply(self.add(exp_u[index + 4], exp_u[index + 4]), inv_D_list[index + 4]), k))
            exp_2u[index] = partial_softmax0
            exp_2u[index + 4] = partial_softmax1
        summation = self.add(exp_2u[0], exp_2u[1])
        for index in range(2, len(exp_2u)):
            self.add_inplace(summation, exp_2u[index])
        summation = self.interval_sum(summation, interval=2**11)
        epsilon2 = output_precision / 128 / 2
        inv_D, D_delta, output_precision = self.he_inv(summation, epsilon=epsilon2, alpha=alpha / 10)
        if final_inv:
            masking = np.array(([1] * 12 + [0.0] * 4) * 2**7)
        else:
            masking = np.array(([1] * 12 + [0.0] * 4) * 2**11)
        inv_D = self.multiply(inv_D, masking)
        return exp_2u, inv_D, D_delta, output_precision

    def he_softmax(self, u, attention_mask, min_x, max_x, n, l, inv_epsilon, output_alpha):  # noqa: E741
        if max_x < 30:
            exp_u = [self.he_exp1(ct, min_x=min_x, max_x=max_x, n=n) for ct in u]
        else:
            exp_u = [self.he_exp2(ct, min_x=min_x, max_x=max_x, n=n) for ct in u]
        for index in range(8):
            exp_u[index] = self.multiply(exp_u[index], attention_mask[index])
        sigma_exp = self.add(exp_u[0], exp_u[1])
        for index in range(2, len(exp_u)):
            self.add_inplace(sigma_exp, exp_u[index])
        sigma_exp = self.interval_sum(sigma_exp, interval=2**11, need_intt=True)
        internal_alpha = 0.1
        if l > 1:
            inv_D, D_delta, output_precision = self.he_inv(sigma_exp, epsilon=inv_epsilon, alpha=internal_alpha / 10)
            for _ in range(int(np.log2(l) - 1)):
                exp_u, inv_D, D_delta, output_precision = self.update_inv_D(exp_u, attention_mask, inv_D, D_delta, output_precision, alpha=internal_alpha)
            exp_u, inv_D, D_delta, output_precision = self.update_inv_D(exp_u, attention_mask, inv_D, D_delta, output_precision, alpha=output_alpha, final_inv=True)
        cplx_softmax = np.empty((128,), dtype=object)
        rotated_inv_D = [inv_D]
        for index in range(15):
            rotated_inv_D.append(self.rotate(rotated_inv_D[-1], GROUP_SIZE))
        prepared_rotated_inv_D = [self.prepare_for_multiply(rotated) for rotated in rotated_inv_D]
        scale = int(1 / (2 * D_delta)) + 1
        for index in range(4):
            exp_cplx = self.add(exp_u[index], self.multiply_1j(exp_u[index + 4]))
            exp_cplx = self.multiply(exp_cplx, scale)
            prepared = self.rescale(exp_cplx)
            for rotated_index in range(16):
                masked_softmax = self.relinearize(self.multiply(prepared, prepared_rotated_inv_D[rotated_index]))
                softmax_copied = self.interval_sum(masked_softmax, interval=GROUP_SIZE)
                conj = self.conjugate(softmax_copied)
                output_index = index * 16 + rotated_index
                cplx_softmax[output_index] = self.add(softmax_copied, conj)
                cplx_softmax[output_index + 64] = self.multiply_1j(self.subtract(conj, softmax_copied))
        return cplx_softmax

    def he_softmax1(self, x, attention_mask):
        return self.he_softmax(x, attention_mask, min_x=-27.2493, max_x=21.72692, n=2, l=2, inv_epsilon=2**(-11), output_alpha=0.01)

    def he_softmax2(self, x, attention_mask):
        return self.he_softmax(x, attention_mask, min_x=-70, max_x=70, n=2, l=4, inv_epsilon=2**(-18), output_alpha=0.01)

    def stage_07_softmax(self, x, attention_mask, layer_index):
        new_x = np.full((8,), None, dtype=object)
        for index in range(4):
            temp = self.add(x[index], self.multiply_1j(x[index + 4]))
            temp = self.bootstrap(temp)
            if layer_index != 2:
                temp = self.level_down(temp, 3)
            conj = self.conjugate(temp)
            new_x[index] = self.add(temp, conj)
            new_x[index + 4] = self.multiply_1j(self.subtract(conj, temp))
        x = new_x
        if layer_index == 2:
            output = self.he_softmax2(x, attention_mask)
        else:
            output = self.he_softmax1(x, attention_mask)
        return output

    def stage_08_attention_context(self, v, softmax_output):
        v_complex = np.full((2,), None, dtype=object)
        for index in range(2):
            v_complex[index] = self.rescale(self.add(v[index], self.multiply_1j(v[index + 2])))
        prepared_softmax_output = np.full((128,), None, dtype=object)
        for index in np.ndindex(softmax_output.shape):
            prepared_softmax_output[index] = self.prepare_for_multiply(softmax_output[index])
        in_dim, out_dim = 128, 2
        ttemp = np.full((out_dim, 4), None, dtype=object)
        masks = self.masks["ccmm"]
        for out_index in range(out_dim):
            multiplied = self.multiply(v_complex[out_index], prepared_softmax_output[0])
            ttemp[out_index, 0] = self.ntt(self.level_down(multiplied, by=1))
        for in_index in range(1, in_dim):
            q_val = (in_index // 16) % 4
            j = in_index % 16
            rot = in_index
            l_prime = prepared_softmax_output[in_index]
            temp = np.full((out_dim, 4), None, dtype=object)
            rrot = ((2**11) * j - 16 * rot) % (2**15)

            def add_to_ttemp(index1, index2):
                if ttemp[index1] is None:
                    ttemp[index1] = temp[index2]
                else:
                    self.add_inplace(ttemp[index1], temp[index2])

            if j == 0:
                for out_index in range(out_dim):
                    rotated = self.rotate(v_complex[out_index], rrot)
                    multiplied = self.multiply(rotated, l_prime)
                    temp[out_index, 0] = self.multiply(masks[0][in_index], multiplied)
                    temp[out_index, 1] = self.subtract(multiplied, temp[out_index, 0])
                if q_val == 1:
                    add_to_ttemp((0, 2), (1, 0)); add_to_ttemp((0, 3), (1, 1))
                    add_to_ttemp((1, 0), (0, 0)); add_to_ttemp((1, 1), (0, 1))
                elif q_val == 2:
                    add_to_ttemp((0, 2), (0, 0)); add_to_ttemp((0, 3), (0, 1))
                    add_to_ttemp((1, 2), (1, 0)); add_to_ttemp((1, 3), (1, 1))
                elif q_val == 3:
                    add_to_ttemp((0, 0), (1, 0)); add_to_ttemp((0, 1), (1, 1))
                    add_to_ttemp((1, 2), (0, 0)); add_to_ttemp((1, 3), (0, 1))
                elif q_val == 0:
                    add_to_ttemp((0, 0), (0, 0)); add_to_ttemp((0, 1), (0, 1))
                    add_to_ttemp((1, 0), (1, 0)); add_to_ttemp((1, 1), (1, 1))
            else:
                for out_index in range(out_dim):
                    rotated = self.rotate(v_complex[out_index], rrot)
                    multiplied = self.multiply(rotated, l_prime)
                    prepared = self.prepare_for_multiply(multiplied)
                    temp[out_index, 0] = self.multiply(masks[0][in_index], prepared)
                    temp[out_index, 1] = self.multiply(masks[1][in_index], prepared)
                    temp[out_index, 2] = self.multiply(masks[2][in_index], prepared)
                    add_temp = self.add(self.add(temp[out_index, 0], temp[out_index, 1]), temp[out_index, 2])
                    temp[out_index, 3] = self.subtract(multiplied, add_temp)
                if q_val == 0:
                    add_to_ttemp((0, 2), (1, 2)); add_to_ttemp((0, 3), (1, 3))
                    add_to_ttemp((0, 0), (0, 0)); add_to_ttemp((0, 1), (0, 1))
                    add_to_ttemp((1, 0), (0, 2)); add_to_ttemp((1, 1), (0, 3))
                    add_to_ttemp((1, 0), (1, 0)); add_to_ttemp((1, 1), (1, 1))
                elif q_val == 1:
                    add_to_ttemp((0, 2), (0, 2)); add_to_ttemp((0, 3), (0, 3))
                    add_to_ttemp((0, 2), (1, 0)); add_to_ttemp((0, 3), (1, 1))
                    add_to_ttemp((1, 2), (1, 2)); add_to_ttemp((1, 3), (1, 3))
                    add_to_ttemp((1, 0), (0, 0)); add_to_ttemp((1, 1), (0, 1))
                elif q_val == 2:
                    add_to_ttemp((0, 0), (1, 2)); add_to_ttemp((0, 1), (1, 3))
                    add_to_ttemp((0, 2), (0, 0)); add_to_ttemp((0, 3), (0, 1))
                    add_to_ttemp((1, 2), (0, 2)); add_to_ttemp((1, 3), (0, 3))
                    add_to_ttemp((1, 2), (1, 0)); add_to_ttemp((1, 3), (1, 1))
                elif q_val == 3:
                    add_to_ttemp((0, 0), (0, 2)); add_to_ttemp((0, 1), (0, 3))
                    add_to_ttemp((0, 0), (1, 0)); add_to_ttemp((0, 1), (1, 1))
                    add_to_ttemp((1, 0), (1, 2)); add_to_ttemp((1, 1), (1, 3))
                    add_to_ttemp((1, 2), (0, 0)); add_to_ttemp((1, 3), (0, 1))
        output = np.full((2,), None, dtype=object)
        ttemp_relin = np.full((out_dim, 4), None, dtype=object)
        for out_index in range(out_dim):
            for i in range(4):
                ttemp_relin[out_index, i] = self.relinearize(ttemp[out_index, i])
            ttemp_relin[out_index, 1] = self.rotate(ttemp_relin[out_index, 1], 2**11)
            ttemp_relin[out_index, 3] = self.rotate(ttemp_relin[out_index, 3], 2**11)
            ttemp_relin[out_index, 2] = self.multiply_1j(
                self.conjugate(self.add(ttemp_relin[out_index, 2], ttemp_relin[out_index, 3]))
            )
            output[out_index] = self.add(
                self.add(ttemp_relin[out_index, 0], ttemp_relin[out_index, 1]),
                ttemp_relin[out_index, 2],
            )
            output[out_index] = self.bootstrap(output[out_index])
        return output

    def stage_10_attention_dense(self, x, layer_index):
        weight = np.full((8, 6, 32), None, dtype=object)
        bias = np.full((8,), None, dtype=object)
        path = self.light_plaintext_path / "stage_10" / f"layer_{layer_index}"
        with self.timer.paused():
            for index in np.ndindex(weight.shape):
                postfix = "_".join(map(str, index))
                weight[index] = self.engine.read_light_plaintext(str(path / f"w_{postfix}"))
            for index in range(8):
                bias[index] = self.engine.read_light_plaintext(str(path / f"b_{index}"))
        wx = self.pcmm(weight, x)
        mask = self.masks["attention_dense"]
        output = np.full((8,), None, dtype=object)
        for index in range(8):
            temp = self.rotate(self.multiply(mask, wx[index]), -6)
            biased = self.add(self.add(wx[index], temp), bias[index])
            output[index] = self.add(biased, self.conjugate(biased))
        return output

    def he_invsqrt(self, denominator, e, alpha, mask):
        d = 0
        a = denominator
        a = self.bootstrap(a)
        b = self.masks["invsqrt_b"]
        while e < 1 - alpha:
            d = d + 1
            k = np.roots([1 - e**3, 6 * e**2 - 6, 9 - 9 * e])[1]
            b1 = self.multiply(b, (k ** (3 / 2) / 2) * mask)
            if d > 6:
                a = self.bootstrap(a)
                b1 = self.bootstrap(self.engine.intt(b1))
            b2 = self.subtract((3 / k) * mask, a)
            b = self.relinearize(self.multiply(b1, b2))
            a1 = self.multiply(a, (k**3 / 4) * mask)
            a2 = self.relinearize(self.square(self.subtract((3 / k) * mask, a)))
            a = self.relinearize(self.multiply(a1, a2))
            e = k * e * (3 - k * e) ** 2 / 4
        if d < 7:
            b = self.bootstrap(b)
        return b

    def he_layernorm(self, x, gamma, beta, var_e, min_var, max_var):
        n = 768
        if min_var <= 0.16:
            name = "ln1"
        elif min_var >= 0.74:
            name = "ln3"
        else:
            name = "ln2"
        epsilon_var1 = min_var / max_var
        w_buffer = 1.05
        max_for_denominator = (max_var * w_buffer + var_e) * n**2
        mask = np.array(([1 / max_for_denominator ** (1 / 2)] * 6 + [0] * 10) * 2**11)
        if name != "ln1":
            mask = mask / 2
        masked = [self.multiply(ct, mask) for ct in x]
        sum_x = masked[0]
        for index in range(1, len(masked)):
            sum_x = self.add(sum_x, masked[index])
        sum_x = self.interval_sum(sum_x, 2**11, need_intt=True)
        self.add_inplace(sum_x, self.rotate(sum_x, -1))
        self.add_inplace(sum_x, self.rotate(sum_x, -2))
        self.add_inplace(sum_x, self.rotate(sum_x, -4))
        sum_mask = self.masks["layernorm"]
        sum_x = self.multiply(sum_x, sum_mask)
        sq_sum_x = self.relinearize(self.square(sum_x))
        sum_x = self.intt(sum_x)
        self.add_inplace(sum_x, self.rotate(sum_x, 1))
        self.add_inplace(sum_x, self.rotate(sum_x, 2))
        self.add_inplace(sum_x, self.rotate(sum_x, 4))
        nx = [self.multiply(ct, n) for ct in masked]
        numerator = [self.subtract(ct, sum_x) for ct in nx]
        sigma_x2 = self.relinearize(self.square(masked[0]))
        for index in range(1, len(masked)):
            sigma_x2 = self.add(sigma_x2, self.relinearize(self.square(masked[index])))
        sigma_x2 = self.interval_sum(sigma_x2, 2**11)
        self.add_inplace(sigma_x2, self.rotate(sigma_x2, -1))
        self.add_inplace(sigma_x2, self.rotate(sigma_x2, -2))
        self.add_inplace(sigma_x2, self.rotate(sigma_x2, -4))
        sigma_x2 = self.multiply(sigma_x2, sum_mask)
        n_sigma_x2 = self.multiply(sigma_x2, n)
        variance = self.subtract(n_sigma_x2, sq_sum_x)
        variance = self.add(variance, var_e / max_for_denominator)
        denominator = self.he_invsqrt(variance, epsilon_var1, alpha=0.001, mask=sum_mask)
        if name == "ln2":
            denominator = self.level_down(denominator, 5)
        elif name == "ln3":
            denominator = self.level_down(denominator, 3)
        self.add_inplace(denominator, self.rotate(denominator, 1))
        self.add_inplace(denominator, self.rotate(denominator, 2))
        self.add_inplace(denominator, self.rotate(denominator, 4))
        prepared = self.prepare_for_multiply(denominator)
        output = np.full((8,), None, dtype=object)
        for index in range(8):
            output[index] = self.relinearize(self.multiply(numerator[index], self.multiply(gamma[index], prepared)))
            self.add_inplace(output[index], beta[index])
            output[index] = self.add(output[index], output[index])
        return output

    def he_layernorm1(self, x, gamma, beta, var_e=10**(-5), min_var=0.15, max_var=10):
        return self.he_layernorm(x, gamma, beta, var_e, min_var, max_var)

    def he_layernorm2(self, x, gamma, beta, var_e=10**(-5), min_var=0.2, max_var=150):
        return self.he_layernorm(x, gamma, beta, var_e, min_var, max_var)

    def he_layernorm3(self, x, gamma, beta, var_e=10**(-5), min_var=0.75, max_var=2500):
        return self.he_layernorm(x, gamma, beta, var_e, min_var, max_var)

    def stage_11_attention_layernorm(self, x, dense, layer_index):
        layernorm_input = np.full((8,), None, dtype=object)
        for index in range(8):
            layernorm_input[index] = self.add(x[index], dense[index])
        weight = np.full((8,), None, dtype=object)
        bias = np.full((8,), None, dtype=object)
        path = self.light_plaintext_path / "stage_11" / f"layer_{layer_index}"
        with self.timer.paused():
            for index in range(8):
                weight[index] = self.engine.read_light_plaintext(str(path / f"w_{index}"))
                bias[index] = self.engine.read_light_plaintext(str(path / f"b_{index}"))
        return self.he_layernorm1(layernorm_input, weight, bias)

    def stage_12_intermediate_dense(self, x, layer_index):
        dense_input = np.full((64,), None, dtype=object)
        mask = self.masks["intermediate_dense"]
        for index in range(4):
            temp = self.add(x[index], self.multiply_1j(x[index + 4]))
            temp = self.multiply(mask, temp)
            dense_input[16 * index] = self.add(temp, self.rotate(temp, 8))
            for rotate_index in range(1, 16):
                dense_index = 16 * index + rotate_index
                dense_input[dense_index] = self.rotate(dense_input[dense_index - 1], -(2**11))
        weight = np.full((2, 8, 6, 64), None, dtype=object)
        bias = np.full((2, 8), None, dtype=object)
        path = self.light_plaintext_path / "stage_12" / f"layer_{layer_index}"
        with self.timer.paused():
            for index in np.ndindex(weight.shape):
                postfix = "_".join(map(str, index))
                weight[index] = self.engine.read_light_plaintext(str(path / f"w_{postfix}"))
            for index in np.ndindex(bias.shape):
                postfix = "_".join(map(str, index))
                bias[index] = self.engine.read_light_plaintext(str(path / f"b_{postfix}"))
        output = np.full((2, 8), None, dtype=object)
        output[0] = self.pcmm(weight[0], dense_input, mode="block_diag_2")
        output[1] = self.pcmm(weight[1], dense_input, mode="block_diag_2")
        for row in range(2):
            for col in range(8):
                biased = self.add(bias[row, col], output[row, col])
                output[row, col] = self.add(biased, self.conjugate(biased))
        return output

    def he_tanh_single_for_gelu(self, x):
        p1 = [-1.06240033e-05, 1.64454894e-04, -5.83533517e-04, -3.80912692e-04, 2.24431193e-03, 8.92295204e-03, -1.05277477e-02, -1.91827040e-02, -2.04634786e-01, 4.54014410e-01, -5.40759203e-01, 5.67745523e00, -1.36433727e01, 1.82574621e01, -8.48849601e01, 1.28686741e02, 3.66720281e02, -1.01400159e03, -1.26278856e02, 2.21728878e03, -9.95421415e02, -2.31059465e03, 1.73583957e03, 1.27394360e03, -1.27836230e03, -3.66781716e02, 4.79663919e02, 4.94610178e01, -9.06754761e01, -2.36515790e00, 8.74311855e00, 1.62838703e-02]  # noqa: E501
        p2 = [-1.70270667e02, 6.81076279e01, 1.79197364e03, -6.81621043e02, -8.49256169e03, 3.05629446e03, 2.39579397e04, -8.10435126e03, -4.48145152e04, 1.41297616e04, 5.86197512e04, -1.70371505e04, -5.51326382e04, 1.45532495e04, 3.77866438e04, -8.87673890e03, -1.89514802e04, 3.84972853e03, 6.94169727e03, -1.16901058e03, -1.84658407e03, 2.41693754e02, 3.54452276e02, -3.24499570e01, -4.91918227e01, 2.58122977e00, 5.78392852e00, -9.45171527e-02]  # noqa: E501
        p1.reverse(); p2.reverse()
        p1 = np.array(p1)
        p2 = np.array(p2) * 0.5
        p1_x = self.evaluate_polynomial_stockmeyer(x, p1)
        tanhx = self.evaluate_polynomial_stockmeyer(p1_x, p2)
        return tanhx

    def stage_13_gelu(self, x):
        gelu_input = np.full((2, 8), None, dtype=object)
        for index in range(8):
            temp = self.add(x[0, index], self.multiply_1j(x[1, index]))
            temp = self.multiply(temp, 1 / 2)
            temp = self.bootstrap(temp)
            conj = self.conjugate(temp)
            gelu_input[0, index] = self.add(temp, conj)
            gelu_input[1, index] = self.multiply_1j(self.subtract(conj, temp))
        output = np.full((2, 8), None, dtype=object)
        for row in range(2):
            for col in range(8):
                x_val = gelu_input[row, col]
                tanh_x = self.he_tanh_single_for_gelu(x_val)
                one_plus_tanh_x = self.add(tanh_x, 1 / 2)
                x_scaled = self.multiply(x_val, 64)
                output[row, col] = self.relinearize(self.multiply(x_scaled, one_plus_tanh_x))
        return output

    def stage_14_output_dense(self, x, layer_index):
        weight = np.full((2, 8, 6, 64), None, dtype=object)
        bias = np.full((8,), None, dtype=object)
        path = self.light_plaintext_path / "stage_14" / f"layer_{layer_index}"
        with self.timer.paused():
            for index in np.ndindex(weight.shape):
                postfix = "_".join(map(str, index))
                weight[index] = self.engine.read_light_plaintext(str(path / f"w_{postfix}"))
            for index in range(8):
                bias[index] = self.engine.read_light_plaintext(str(path / f"b_{index}"))
        wx = np.full((2, 8), None, dtype=object)
        for row in range(2):
            rotated = np.full((64,), None, dtype=object)
            for col in range(4):
                rotated[16 * col] = self.add(x[row, col], self.multiply_1j(x[row, col + 4]))
                for rotate_index in range(1, 16):
                    index = 16 * col + rotate_index
                    rotated[index] = self.rotate(rotated[index - 1], -(2**11))
            wx[row] = self.pcmm(weight[row], rotated, mode="block_diag_2")
        mask = np.ones((2**15,), dtype=int)
        mask[np.arange(2**15) % 16 >= 6] = 0
        output = np.full((8,), None, dtype=object)
        for index in range(8):
            temp = self.add(wx[0, index], wx[1, index])
            self.add_inplace(temp, self.rotate(temp, -8))
            self.add_inplace(temp, bias[index])
            output[index] = self.add_inplace(temp, self.conjugate(temp))
        return output

    def stage_15_prepare_layernorm(self, x, y):
        output = np.full((8,), None, dtype=object)
        for index in range(8):
            output[index] = self.add(x[index], y[index])
        for index in range(4):
            temp = self.add(output[index], self.multiply_1j(output[index + 4]))
            temp = self.bootstrap(temp)
            temp = self.level_down(temp, 3)
            conj = self.conjugate(temp)
            output[index] = self.add(temp, conj)
            output[index + 4] = self.multiply_1j(self.subtract(conj, temp))
        return output

    def stage_16_output_layernorm(self, x, layer_index):
        weight = np.full((8,), None, dtype=object)
        bias = np.full((8,), None, dtype=object)
        path = self.light_plaintext_path / "stage_16" / f"layer_{layer_index}"
        with self.timer.paused():
            for index in range(8):
                weight[index] = self.engine.read_light_plaintext(str(path / f"w_{index}"))
                bias[index] = self.engine.read_light_plaintext(str(path / f"b_{index}"))
        if layer_index == 9 or layer_index == 10:
            return self.he_layernorm3(x, weight, bias)
        return self.he_layernorm2(x, weight, bias)

    def pooler_dense(self, x):
        weight = np.full((6, 4), None, dtype=object)
        path = self.light_plaintext_path / "stage_17"
        with self.timer.paused():
            for index in np.ndindex(weight.shape):
                postfix = "_".join(map(str, index))
                weight[index] = self.engine.read_light_plaintext(str(path / f"w_{postfix}"))
            bias = self.engine.read_light_plaintext(str(path / "b_0"))
        mask = self.masks["pooler_dense"]
        for index in range(4):
            masked = self.multiply(mask, x[index])
            for rotate_index in range(7):
                rotated = self.rotate(masked, 16 * 2**rotate_index)
                masked = self.add(masked, rotated)
            x[index] = masked
        prepared = np.full(x.shape, None, dtype=object)
        for index, x_ct in np.ndenumerate(x):
            prepared[index] = self.prepare_for_multiply(x_ct)
        wx_rescaled = np.full((6,), None, dtype=object)
        for row in range(6):
            temp = self.multiply(weight[row, 0], prepared[0])
            for col in range(1, 4):
                temp = self.add(temp, self.multiply(weight[row, col], prepared[col]))
            wx_rescaled[row] = self.rescale(temp)
        temp = wx_rescaled[0]
        for index in range(1, 6):
            rotated = self.rotate_internal(wx_rescaled[index], delta=6 - index, mode="block_diag_2")
            temp = self.add(temp, rotated)
        temp = self.interval_sum(temp, 2**11)
        output = np.full((1,), None, dtype=object)
        self.add_inplace(temp, bias)
        self.add_inplace(temp, self.conjugate(temp))
        output[0] = temp
        return output

    def he_tanh_single_for_pooler(self, x):
        p1 = [-7.14529052e03, -7.76519925e01, 2.74279201e04, 2.45150249e02, -4.25793697e04, -3.01953016e02, 3.42189880e04, 1.82989351e02, -1.51158283e04, -5.64098990e01, 3.58757327e03, 8.17596753e00, -4.13341496e02, -4.29024545e-01, 1.95056729e01, 2.06201784e-03]  # noqa: E501
        p2 = [-9.02573450e-03, -1.12320034e-04, 1.08762008e-01, 7.96793166e-04, -5.41327356e-01, -1.42873183e-03, 1.46476749e00, -2.22416152e-03, -2.43259032e00, 1.17381072e-02, 2.74974898e00, -1.77631073e-02, -2.38934873e00, 1.30194846e-02, 2.02874846e00, -4.08442578e-03]  # noqa: E501
        p1.reverse(); p2.reverse()
        p1 = np.array(p1)
        p2 = np.array(p2)
        x = self.bootstrap(x)
        tanh_x = self.evaluate_polynomial_stockmeyer(x, p1)
        tanh_x = self.evaluate_polynomial_stockmeyer(tanh_x, p2)
        tanh_x = self.bootstrap(tanh_x)
        return tanh_x

    def stage_17_pooler(self, x):
        x_complex = np.full((4,), None, dtype=object)
        for index in range(4):
            x_complex[index] = self.add(x[index], self.multiply_1j(x[index + 4]))
        x = self.pooler_dense(x_complex)
        x[0] = self.multiply(x[0], 1 / 40)
        output = np.full((1,), None, dtype=object)
        output[0] = self.he_tanh_single_for_pooler(x[0])
        return output

    def stage_18_classifier(self, x):
        weight = np.full((2,), None, dtype=object)
        bias = np.full((2,), None, dtype=object)
        path = self.light_plaintext_path / "stage_18"
        with self.timer.paused():
            for index in range(2):
                weight[index] = self.engine.read_light_plaintext(str(path / f"w_{index}"))
                bias[index] = self.engine.read_light_plaintext(str(path / f"b_{index}"))
        class_count = weight.shape[0]
        output = np.full((class_count,), None, dtype=object)
        prepared = self.prepare_for_multiply(x[0])
        for class_index in range(class_count):
            temp = self.multiply(weight[class_index], prepared)
            temp = self.add(temp, self.rotate(temp, -1))
            temp2 = self.add(temp, self.rotate(temp, -2))
            temp = self.add(temp2, self.rotate(temp, -4))
            for index in range(4, 11):
                temp = self.add(temp, self.rotate(temp, -(2**index)))
            output[class_index] = self.add(bias[class_index], temp)
        return output

    def forward_layer(self, x, layer_idx, clear_attention_mask):
        x, x_cplx = self.stage_01_complexify_x(x, layer_idx)
        x_cplx_rots = self.stage_02_make_rotated_copies(x_cplx)
        q_wo_rescale = self.stage_03_query(x_cplx_rots, layer_idx)
        k = self.stage_04_key(x_cplx_rots, layer_idx)
        v = self.stage_05_value(x_cplx_rots, layer_idx)
        sftmx_in = self.stage_06_attention_score(q_wo_rescale, k)
        sftmx_out = self.stage_07_softmax(sftmx_in, clear_attention_mask, layer_idx)
        att_context = self.stage_08_attention_context(v, sftmx_out)
        att_context_rots = self.stage_02_make_rotated_copies(att_context)
        att_dense = self.stage_10_attention_dense(att_context_rots, layer_idx)
        layernorm_1_output = self.stage_11_attention_layernorm(x, att_dense, layer_idx)
        intermediate_dense = self.stage_12_intermediate_dense(layernorm_1_output, layer_idx)
        gelu_output = self.stage_13_gelu(intermediate_dense)
        output_dense = self.stage_14_output_dense(gelu_output, layer_idx)
        layernorm_2_input = self.stage_15_prepare_layernorm(layernorm_1_output, output_dense)
        return self.stage_16_output_layernorm(layernorm_2_input, layer_idx)
