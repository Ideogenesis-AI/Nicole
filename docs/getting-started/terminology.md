# Terminology

Nicole uses a precise vocabulary to describe tensor structure and operations. This page defines the key terms and distinguishes concepts that are often conflated in the broader tensor network literature.

---

## Axis, Index, and Edge

These three terms all refer to "a index of a tensor" in colloquial usage, but they mean distinct things in Nicole.

**Axis**
:   An integer position or string tag (`itag`) that designates *where* an index sits in a tensor. Axes are ordinal labels — they identify a slot, not the object occupying it. When you permute a tensor or refer to an index by number, you are working with axes.

**Index** (`Index`)
:   The concrete object that occupies an axis. An `Index` carries the full sector structure of the index: the symmetry group, the list of charge–dimension pairs, and the direction (incoming or outgoing). Two tensors can only be contracted along a pair of axes if the `Index` objects at those axes are compatible — same group, complementary sectors, opposite directions.

**Edge**
:   The dedicated term for an index of a CG (Clebsch–Gordan) tensor, as used in the [Yuzuha Protocol](yuzuha-protocol.md). An edge in the CG tensor network corresponds to an index labelled by irrep label \(j\). The term "edge" is reserved for this graph-theoretic context and is not used for ordinary tensor axes.

---

## Sector and Block

**Sector**
:   A single symmetry charge together with its multiplicity (dimension). For example, in a U(1) index, the pair `(charge=1, dim=4)` is a sector — charge \(q = 1\) appearing with degeneracy 4. Sectors are the building blocks of an `Index`.

**Block**
:   A combination of sectors, one per axis of a tensor, that jointly satisfies the charge conservation constraint. A block identifies a dense sub-matrix (or sub-array) in the block-sparse representation of the tensor. While a sector lives on a single index, a block lives on the full tensor and corresponds to an actual data chunk stored in memory.

---

## reverse, flip, and invert

These three words all suggest "turning something around", but they apply at different levels of the data hierarchy.

**`reverse`** — acts on `Direction`
:   Changes the direction of a `Direction` value between incoming and outgoing. This is a low-level primitive on the enum itself.

**`flip`** — acts on `Index`
:   Returns a new `Index` with the direction reversed. The sector structure is unchanged; only the arrow on the index is flipped. Used when constructing compatible pairs of indices for contraction.

**`invert`** — acts on `Tensor`
:   Flips the direction of one or more axes of a tensor by updating the associated `Index` objects in place, and also reverses the corresponding direction in the intertwiner via the `Bridge`. Acts on a single tensor in isolation and does not apply any phase correction. For inverting a bond between two tensors, use [`capcup`](../api/manipulation/capcup.md) instead, which applies the necessary Frobenius–Schur phase for SU(2).

---

## fuse, combine, and merge

These three words all suggest "bringing things together", but they operate at different levels.

**`fuse`** — acts on charges
:   Combines two or more symmetry charges according to the group fusion rules to produce the resulting charge(s). For Abelian groups this yields a single charge (`fuse_unique`); for non-Abelian groups it yields a set of allowed output charges (`fuse_channels`). Fusion is a property of the symmetry group, not of tensors.

**`combine`** — acts on `Index` objects
:   Produces a single `Index` whose charge content is the fusion product of two or more input indices. The result is a new index-level object — it has its own sector structure and an explicitly chosen direction, independent of the input directions. `combine` operates purely on index structure without touching any tensor data.

**`merge`** — acts on tensor axes
:   Collapses multiple axes of a tensor into one, physically restructuring the tensor data. Unlike `combine`, which is a pure index operation, `merge` acts on a live tensor and changes its shape. The inverse operation — splitting the merged axis back — is achieved by contracting with the Hermitian conjugate of the isometry used to perform the merge.
