# BERT Reference Implementation

This submission runs BERT under CKKS using [`desilofhe`](https://pypi.org/project/desilofhe/)
with `use_bootstrap_to_14_levels=True`.

## Parameters and security justification

- **Ring dimension:** `N = 2^16 = 65,536` (packs up to `2^15 = 32,768` slots).
- `log Q = 15×54 + 15×40 + 60`
- `log P = 4×60`
- **Bootstrapping:** sparse-secret encapsulation, Hamming weight **128**.

Together these give `log QP = 1710 bits`. According to Table 5.2 of
[Bossuat et al., 2024](https://eprint.iacr.org/2024/463), the largest modulus
that provides 128-bit security at `N = 2^16` is **1747 bits**. Since
`1710 ≤ 1747`, the parameter set exceeds 128-bit security for a uniform ternary
secret.

Bootstrapping uses
[sparse-secret encapsulation](https://eprint.iacr.org/2022/024): the sparse
secret key is used only for key encapsulation, while the main computation runs
under a dense (uniform ternary) secret. The sparse secret has Hamming weight
**128**, and Table 8 [56] of
[Ogilvie, 2026](https://eprint.iacr.org/2026/279.pdf) confirms that this weight
retains 128-bit security at `N = 2^16`.
