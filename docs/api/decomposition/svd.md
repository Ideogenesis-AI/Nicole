# svd

Low-level SVD returning singular values as dictionary.

::: nicole.decomp.svd
    options:
      show_source: false
      heading_level: 2

## Description

Performs block-wise SVD, returning:
- **U**: Left unitary tensor
- **S_dict**: Dictionary mapping block keys to 1D singular value arrays
- **Vh**: Right unitary tensor
- **info** *(optional)*: Dict of auxiliary data, returned only when `requires_info=True`. Currently contains:
  - `"discarded_weight"`: `float` in `[0, 1)`, the relative sum of squared singular values truncated away across all charge sectors, i.e. the fractional squared 2-norm loss `‖T − T_trunc‖²/‖T‖²`.
  - `"engine"`: the engine that ran, echoing the `engine` argument.
  - `"fallbacks"`: dict mapping each charge sector that needed the fallback ladder to the list of steps it took. Empty when nothing failed.

This low-level function provides direct access to singular values for each block before they're combined into a tensor.

## Engines

The `engine` argument selects the solver used on each charge sector.

```python
# Exact: torch.linalg.svd, the complete spectrum (default)
U, S_blocks, Vh = svd(T, axis=0, trunc={"nkeep": 50})

# Randomized: only the leading singular values are computed
U, S_blocks, Vh = svd(T, axis=0, trunc={"nkeep": 50}, engine="random")
```

`engine="random"` requires `trunc["nkeep"]` to set the target rank; a `"thresh"` criterion alone does not bound the rank, so omitting `"nkeep"` raises `ValueError`.

### Tuning the randomized engine

The tuning rides along in the same `trunc` dict, since `"nkeep"` already doubles as the target rank. These three keys are ignored by the exact engine:

| Key | Default | Meaning |
| --- | --- | --- |
| `oversample` | `10` | Extra subspace dimensions beyond the per-sector target rank |
| `niter` | `2` | Subspace (power) iterations |
| `seed` | `None` | Seed making the decomposition reproducible |

The oversampled triplets are returned in full: they cost nothing once computed, and they widen the candidate pool for the global `"nkeep"` selection. Trimming them could not help anyway, since a projection only shrinks singular values, so a trailing estimate never displaces a larger one. Set `oversample=0` for exactly the target rank per sector.

```python
U, S_blocks, Vh = svd(T, axis=0, trunc={"nkeep": 50, "seed": 42}, engine="random")
```

Seeding is necessary for reproducibility because `torch.svd_lowrank` draws its test matrix from the global generator and accepts no `generator` argument. Nicole forks the RNG around the decomposition, so a seeded call leaves the caller's random stream untouched.

### When the randomized engine pays off

Two conditions have to hold together:

**The kept rank must be well below the full rank.** Randomized SVD costs roughly `(2 * niter + 2) * O(m * n * q)` against the exact `O(m * n * min(m, n))`. On a four-sector U(1) tensor of total rank 390, the speedup over the exact engine runs 2.2x at `rho = 0.02`, 1.6x at `rho = 0.1`, and 1.2x at `rho = 0.2`, crossing below 1x at around `rho = 0.3`. Use it below roughly a fifth of the full rank.

**The spectrum must decay.** A randomized method works by finding a dominant subspace, so a flat spectrum leaves it nothing to find. On the same tensor with singular values spanning six decades, the retained values come back to a relative error of `1e-6` or better; with a flat spectrum the error is around 5 percent no matter how much oversampling is thrown at it. Timing is identical in both cases, since it depends only on the shapes.

Oversampling is not optional. At `nkeep = 19` on the decaying spectrum, the maximum relative error falls from `8.9e-2` with no oversampling to `1.4e-3` at 5 columns and `9.5e-6` at the default 10, while the speedup only drops from 2.7x to 2.0x.

## Per-sector rank budget

`"nkeep"` is a global budget, but a randomized SVD has to be told its target rank before it runs. Nicole reduces every sector by the same ratio of its own full rank:

```
rho = nkeep / (sum of full ranks over all sectors)
k_q = max(1, ceil(rho * r_q))
```

The ceiling guarantees `sum(k_q) >= nkeep`, so the global selection that follows never runs short of candidates, and the floor of 1 keeps every charge sector on the bond index, which the budget would otherwise be free to erase.

This allocates by dimension, not by spectral weight. The two correlate in the usual symmetric-MPS case, where the split is exactly optimal. When they do not — a sector that is large but spectrally negligible drawing rank away from a small dominant one — the retained squared weight stays within about 2 percent of the optimum on measured adversarial layouts.

Oversampling recovers even that gap, which is a second reason to leave it on. Each sector returns `k_q + oversample` candidates, so a dominant sector short-changed by the budget can still win back rank in the global selection, which ranks candidates across all sectors at once. At the default `oversample=10` the discarded weight on such a layout matches the exact engine.

## Fallback on non-convergence

`torch.linalg.svd` can fail to converge on badly scaled or near-degenerate input. Instead of aborting, each sector is retried along a short ladder:

1. `torch.linalg.svd`, the direct call.
2. On CUDA only, a retry with the `gesvd` cuSOLVER driver, which is slower but more robust. The `driver` argument rejects CPU input, so this step is skipped there.
3. A full-rank randomized SVD. The random projection rotates the matrix into a fresh orthonormal basis before any driver sees it, which breaks up the near-degeneracy or bad scaling that stalled the iteration. At full rank the rotation preserves the spectrum, so the factorization is still that of the original matrix.
4. Re-raise `torch.linalg.LinAlgError` with the sector, shape, dtype and device named.

Being full rank, step 3 returns a complete spectrum, so a sector recovered that way is treated as exact and feeds the usual truncation and discarded-weight bookkeeping.

Every step is reported at `DEBUG` on the `nicole.decomp` logger. Nicole installs no handler, so the records are inert until the application opts in:

```python
import logging

logging.basicConfig()
logging.getLogger("nicole").setLevel(logging.DEBUG)
```

The `info` dict's `"fallbacks"` entry carries the same information programmatically, with no logging configuration needed.

## Import

```python
from nicole.decomp import svd
```

(Not exported in public API)

## See Also

- [decomp](decomp.md): High-level decomposition
- [qr](qr.md): Low-level QR function
- [eig](eig.md): Eigenvalue decomposition
- [Examples: Decomposition](../../examples/operations/decomposition-examples.md)

## Notes

- Use `decomp()` for most cases. Use `svd()` when you need per-block singular value access before tensor creation.
- Pass `requires_info=True` to obtain the `info` dict containing `"discarded_weight"`. This incurs a small extra computation and is intended for diagnostic use.
- `"discarded_weight"` is a relative squared weight: squared singular values normalized by the full spectrum. For a generic symmetry group each reduced singular value carries a multiplicity factor `irrep_dim(q)`, the number of degenerate Schmidt values it stands for. The factor weights the reported value only, not the `"thresh"` or `"nkeep"` truncation criteria.
- Under `engine="random"` the discarded tail is never computed, so the same quantity comes from the identity `Σ_i s_i² = ‖M‖_F²`: the denominator is the Frobenius norm of each sector's matrix, exact and nearly free, and the numerator is the difference against the retained singular values. The two engines agree to the accuracy of the retained values themselves.
- For Abelian tensors, `Vh` satisfies `Vh @ Vh† = I` directly from the standard block-wise matrix SVD.
- For SU(2) tensors, the intertwiner weight matrix is canonicalized to a scaled unitary form before the SVD, ensuring `Vh` is a physical isometry (`Vh @ Vh† = I`).
