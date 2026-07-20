# BERT Reference Implementation

This submission runs BERT under CKKS using [`desilofhe`](https://pypi.org/project/desilofhe/)
with `use_bootstrap_to_14_levels=True`.

## Parameters and security justification

- **Ring dimension:** `N = 2^16 = 65,536` (packs up to `2^15 = 32,768` slots).
- **Bootstrapping:** sparse-secret encapsulation with Hamming weight **192**.

| Component        | Count | Bits each | Total bits |
| ---------------- | ----- | --------- | ---------- |
| Bootstrap primes | 15    | 54        | 810        |
| Scaling primes   | 15    | 40        | 600        |
| Base primes      | 5     | 60        | 300        |
| **`log QP`**     |       |           | **1710**   |

The total ciphertext modulus (including the auxiliary primes for hybrid key
switching) is `15×54 + 15×40 + 5×60 = 1710 bits`. Per Table 5.2 of
[Bossuat et al., 2024](https://eprint.iacr.org/2024/463), the largest modulus
providing 128-bit security at `N = 2^16` is **1747 bits**. Since `1710 ≤ 1747`,
the parameter set exceeds 128-bit security for a uniform ternary secret.

Bootstrapping uses
[sparse-secret encapsulation](https://eprint.iacr.org/2020/1203): the main
computation runs under a dense (uniform ternary) secret, and only bootstrapping
uses a sparse secret. The Hamming weight of **192** keeps this sparse secret at
the 128-bit level for `N = 2^16` under the same analysis.
