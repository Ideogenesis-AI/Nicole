# Yuzuha Protocol

This page describes the internal structure of SU(2)-symmetric tensors in Nicole and the mathematical framework behind their implementation. Nicole follows the **Yuzuha Protocol** — a specification for non-Abelian tensor algebra — and uses the [Yuzuha](https://ideogenesis-ai.github.io/Yuzuha/) SU(2) recoupling engine as its computational backend.

!!! note "Prerequisites"
    This page assumes familiarity with [Core Concepts](core-concepts.md), particularly the section on non-Abelian groups and irrep dimension.

---

## The Wigner–Eckart Theorem

At the heart of SU(2)-symmetric tensor networks is the **Wigner–Eckart theorem**. For a tensor operator \(\hat{T}\) with definite symmetry properties, the theorem states that all matrix elements within a given charge block factorize as:

\[
\langle j', m' \| \hat{T}^{(k)}_q \| j, m \rangle
= \langle j' \| \hat{T}^{(k)} \| j \rangle \cdot \langle j, m; k, q \mid j', m' \rangle
\]

where:

- \(\langle j' \| \hat{T}^{(k)} \| j \rangle\) is the **reduced matrix element (RME)** — a single number depending only on the spin labels \(j', k, j\), not on the magnetic quantum numbers \(m', q, m\)
- \(\langle j, m; k, q \mid j', m' \rangle\) is the **Clebsch–Gordan (CG) coefficient** — a universal recoupling factor

This factorization extends to arbitrary \(n\)th-order tensors: a block of an \(n\)th-order tensor labeled by spins \((j_1, j_2, \ldots, j_n)\) has \(\prod_\ell (2j_\ell + 1)\) physical components, but **all of them are determined by a small number of reduced tensor elements** and the corresponding CG recoupling tensor.

<div style="text-align:center; margin:1.5em 0;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 148" style="max-width:460px;width:100%;">
  <!-- downward index lines from R̃ -->
  <line x1="75"  y1="88" x2="75"  y2="122" stroke="currentColor" stroke-width="1.5"/>
  <line x1="115" y1="88" x2="115" y2="122" stroke="currentColor" stroke-width="1.5"/>
  <line x1="165" y1="88" x2="165" y2="122" stroke="currentColor" stroke-width="1.5"/>
  <text x="75"  y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">j₁</text>
  <text x="115" y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">j₂</text>
  <text x="141" y="120" text-anchor="middle" fill="currentColor" font-size="16">⋯</text>
  <text x="165" y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">jₙ</text>
  <!-- R̃ box -->
  <rect x="52" y="44" width="136" height="44" fill="none" stroke="currentColor" stroke-width="1.5" rx="6"/>
  <text x="120" y="72" text-anchor="middle" fill="currentColor" font-size="22" font-style="italic">R̃</text>
  <!-- α connection -->
  <line x1="188" y1="66" x2="272" y2="66" stroke="currentColor" stroke-width="1.5"/>
  <text x="230" y="58" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">α</text>
  <!-- C box -->
  <rect x="272" y="44" width="136" height="44" fill="none" stroke="currentColor" stroke-width="1.5" rx="6"/>
  <text x="340" y="72" text-anchor="middle" fill="currentColor" font-size="22" font-style="italic">C</text>
  <!-- downward index lines from C -->
  <line x1="295" y1="88" x2="295" y2="122" stroke="currentColor" stroke-width="1.5"/>
  <line x1="335" y1="88" x2="335" y2="122" stroke="currentColor" stroke-width="1.5"/>
  <line x1="385" y1="88" x2="385" y2="122" stroke="currentColor" stroke-width="1.5"/>
  <text x="295" y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">m₁</text>
  <text x="335" y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">m₂</text>
  <text x="361" y="120" text-anchor="middle" fill="currentColor" font-size="16">⋯</text>
  <text x="385" y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">mₙ</text>
</svg>
</div>

Tensor **$\tilde R$** (left) is the reduced tensor, with one index per tensor position running over the multiplet space (size `Sector.dim`). Tensor **$C$** (right) is the CG basis tensor, with one index per position running over the \(2j_\ell+1\) magnetic substates. The connecting index **$α$** is the **outer multiplicity (OM) index**, enumerating the independent fusion-tree configurations consistent with the charge labels \((j_1, \ldots, j_n)\).

Nicole exploits this structure to store tensors in **reduced form**, gaining both memory compression and computational acceleration proportional to \(\prod_\ell (2j_\ell + 1)\).

---

## Outer Multiplicity

The **outer multiplicity (OM)** counts how many linearly independent ways a given set of spins \((j_1, j_2, \ldots, j_n)\) can be fused to a definite total spin \(J\). Each independent configuration corresponds to a distinct **fusion tree** — a choice of intermediate coupling at each step of the sequential spin addition. The OM index \(\alpha\) runs over these configurations and is the key internal index in the R-W-C decomposition.

| Concept | Dimension | Scales with |
|---------|------|-------------|
| Magnetic substates | \(2j+1\) per index | Spin magnitude — fixed by physics |
| Outer multiplicity (OM) | \(\text{dim}(\alpha)\) | Fusion-tree configurations |

For a fixed charge block \((j_1, j_2, \ldots, j_n)\), the OM is a combinatorial number determined entirely by the spin labels. It counts the number of valid intermediate spin values in the left-associative fusion tree and generally grows with both the tensor order \(n\) and the spin magnitudes \(j_\ell\). For SU(2) tensors:

- A 2nd-order block \((j_1, j_2)\) has OM \(= 1\) — there is no coupling, just an identity
- A 3rd-order block \((j_1, j_2, j_3)\) has OM \(= 1\) — the block is a single CG coefficient
- OM \(> 1\) first appears at 4th order, where multiple intermediate spin paths may be valid
- For larger spin magnitudes, more intermediate values are compatible, further increasing the OM dimension

By operating entirely in the reduced (OM) space, Nicole avoids ever constructing the full \(\prod_\ell (2j_\ell+1)\)-dimensional physical block. All contractions, decompositions, and manipulations are carried out using X-symbols and R-symbols computed by Yuzuha.

---

## Nicole's R-W-C Decomposition

A block labeled by charge \((j_1, \ldots, j_n)\) lives in an OM space of dimension \(\text{dim}(\alpha)\), but the actual physical content of that block may only occupy a **small subspace** of it. This is particularly valuable when the OM space is large (high tensor order or large spins): rather than storing a full vector across the entire OM space, Nicole introduces a **weight matrix** \(W\) with shape \((\text{dim}(r),\, \text{dim}(\alpha))\) where \(\text{dim}(r) \ll \text{dim}(\alpha)\). Each row of \(W\) is a vector in the OM space, and \(\text{dim}(r)\) tracks how many independent directions the physical content spans. This is the key motivation for the R-W-C structure.

For every block of an SU(2)-symmetric tensor, Nicole stores the physical content as a three-factor product:

<div style="text-align:center; margin:1.5em 0;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 148" style="max-width:460px;width:100%;">
  <!-- downward index lines from R -->
  <line x1="55"  y1="88" x2="55"  y2="122" stroke="currentColor" stroke-width="1.5"/>
  <line x1="85"  y1="88" x2="85"  y2="122" stroke="currentColor" stroke-width="1.5"/>
  <line x1="130" y1="88" x2="130" y2="122" stroke="currentColor" stroke-width="1.5"/>
  <text x="55"  y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">j₁</text>
  <text x="85"  y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">j₂</text>
  <text x="108" y="120" text-anchor="middle" fill="currentColor" font-size="16">⋯</text>
  <text x="130" y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">jₙ</text>
  <!-- R box -->
  <rect x="30" y="44" width="120" height="44" fill="none" stroke="currentColor" stroke-width="1.5" rx="6"/>
  <text x="90" y="72" text-anchor="middle" fill="currentColor" font-size="22" font-style="italic">R</text>
  <!-- r connection R-W -->
  <line x1="150" y1="66" x2="190" y2="66" stroke="currentColor" stroke-width="1.5"/>
  <text x="170" y="58" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">r</text>
  <!-- W box -->
  <rect x="190" y="44" width="80" height="44" fill="none" stroke="currentColor" stroke-width="1.5" rx="6"/>
  <text x="230" y="72" text-anchor="middle" fill="currentColor" font-size="22" font-style="italic">W</text>
  <!-- α connection W-C -->
  <line x1="270" y1="66" x2="310" y2="66" stroke="currentColor" stroke-width="1.5"/>
  <text x="290" y="58" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">α</text>
  <!-- C box -->
  <rect x="310" y="44" width="120" height="44" fill="none" stroke="currentColor" stroke-width="1.5" rx="6"/>
  <text x="370" y="72" text-anchor="middle" fill="currentColor" font-size="22" font-style="italic">C</text>
  <!-- downward index lines from C -->
  <line x1="330" y1="88" x2="330" y2="122" stroke="currentColor" stroke-width="1.5"/>
  <line x1="360" y1="88" x2="360" y2="122" stroke="currentColor" stroke-width="1.5"/>
  <line x1="405" y1="88" x2="405" y2="122" stroke="currentColor" stroke-width="1.5"/>
  <text x="330" y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">m₁</text>
  <text x="360" y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">m₂</text>
  <text x="383" y="120" text-anchor="middle" fill="currentColor" font-size="16">⋯</text>
  <text x="405" y="136" text-anchor="middle" fill="currentColor" font-size="14" font-style="italic">mₙ</text>
</svg>
</div>

### R — Reduced Tensor

`R` is the **reduced tensor**, stored in `Tensor.data[key]`. Its shape is over the **multiplet space** — the size along each dimension equals the number of independent multiplets for the corresponding `Index` (i.e. `Sector.dim`, the outer multiplicity per index). `R` contains the physically meaningful content of the tensor: the reduced tensor elements themselves.

For a 2nd-order operator block `(j, j)`, `R` is a scalar (or a small matrix if multiple multiplets are present). For a 3rd-order isometry or operator, `R` is a matrix over the multiplet dimensions of the free indices.

### C — CG Basis

`C` is the **Clebsch–Gordan basis tensor**, computed on demand by the [Yuzuha](https://ideogenesis-ai.github.io/Yuzuha/) recoupling engine from the `CGSpec` stored in the `Bridge`. It has one edge per external `Index` (running over the \(2j+1\) magnetic quantum numbers) plus one axis for the **outer multiplicity (OM) space**. `C` encodes all fusion tree configurations that are consistent with the charge labels and total spin, in a canonical orthonormal basis.

`C` is delegated to a dedicated [Yuzuha](https://ideogenesis-ai.github.io/Yuzuha/) CG engine via the `CGSpec` stored in the `Bridge`.

### W — Weight Matrix

`W` is the **weight matrix**, stored as `Bridge.weights` with shape \((\text{dim}(r),\, \text{dim}(\alpha))\):

- **`om_dimension`** \(= \text{dim}(\alpha)\): the total number of linearly independent canonical basis states in the OM space — i.e. the number of distinct left-associative fusion-tree configurations for the given charge labels. This is computed by Yuzuha from the `CGSpec`.
- **`num_components`** \(= \text{dim}(r)\): the number of independent directions the physical content spans in the OM space. This starts at 1 and can grow when tensors with overlapping OM structure are added or multiplied together.

Each row of `W` is a vector in the OM space, representing the coefficient of one component of the physical content projected onto the canonical CG basis.

---

## Recoupling: X-Symbols and R-Symbols

When two SU(2) tensors are contracted or permuted, their `Bridge` objects must be combined:

- **X-symbol** (`compute_xsymbol`): the recoupling coefficient needed when contracting two tensors along a shared bond. It transforms the product of two OM spaces into the OM space of the result.
- **R-symbol** (`compute_rsymbol`): the recoupling coefficient needed when permuting axes of a tensor. It transforms the OM space under a reordering of the fusion tree.

Both symbols are computed by the [Yuzuha](https://ideogenesis-ai.github.io/Yuzuha/) engine and are cached in its database for efficiency. Nicole calls these internally — users never invoke X- or R-symbols directly.

---

## Nicole's Implementation

Nicole stores the R-W-C structure transparently through two tensor attributes:

| Attribute | Content |
|-----------|---------|
| `Tensor.data[key]` | `R` — reduced tensor (PyTorch tensor, multiplet-space shape) |
| `Tensor.intw[key]` | `Bridge(cgspec, weights)` — contains `W` and the CGSpec for `C` |

The `Bridge` class (from `nicole.symmetry.delegate`) is Nicole's adapter to the Yuzuha engine:

```python
from nicole.symmetry.delegate import Bridge

bridge = T.intw[(1, 1)]          # Bridge for block with charges (j=1/2, j=1/2)
print(bridge.om_dimension)       # OM dimension from yuzuha CGSpec
print(bridge.num_components)     # Number of rows in W
print(bridge.weights.shape)      # (num_components, om_dimension)
```

All operations in Nicole that involve SU(2) tensors (`contract`, `trace`, `decomp`, `permute`, `conj`, etc.) automatically update both `R` and the `Bridge` in a consistent way, using X- and R-symbols from Yuzuha. The user never needs to handle intertwiners directly.

---

## Summary

| Symbol | Name | Where stored | Role |
|--------|------|-------------|------|
| `R` | Reduced tensor | `Tensor.data[key]` | Physically meaningful content; reduced tensor elements |
| `W` | Weight matrix | `Bridge.weights` | Expansion of physical content in the OM canonical basis |
| `C` | CG basis | `Bridge.cgspec` | Universal recoupling factor |

The compression gain of SU(2)-symmetric tensors over explicit Abelian (U(1)-resolved) tensors comes entirely from replacing the full magnetic-substate block with the compact R-W-C representation. A block with spin labels \((j_1, \ldots, j_n)\) has \(\prod_\ell (2j_\ell + 1)\) physical components, but is represented by a single reduced tensor element (times the small W and C factors). The gain therefore grows rapidly with both the spin magnitudes and the tensor order.

## See Also

- [Yuzuha documentation](https://ideogenesis-ai.github.io/Yuzuha/) — the SU(2) recoupling engine
- [Core Concepts](core-concepts.md) — symmetry groups, sectors, and irrep dimension
- [SU2Group API](../api/symmetry/su2-group.md) — charge conventions and fusion rules
- [SU(2) Examples](../examples/symmetries/su2-examples.md) — working with SU(2) tensors in practice
