# API Reference

Welcome to the Nicole API Reference. This comprehensive documentation covers all functions, classes, and methods available in Nicole, organized by functionality for quick access.

**What you'll find here:**

- **Complete function signatures** with all parameters and return types
- **Detailed descriptions** of behavior and algorithms
- **Usage examples** demonstrating each function
- **Cross-references** to related functions and examples

The API is organized into logical categories: Core concepts (Tensor, Index, Sector), Operations (contraction, decomposition, manipulation), Symmetry groups (U(1), Z(2), ProductGroup), and Utilities. Each page provides both high-level understanding and technical details needed for advanced usage.

## Core Classes

The fundamental building blocks:

| Class | Description |
|-------|-------------|
| [Tensor](core/tensor.md) | Block-sparse tensor with symmetry |
| [Index](core/index-class.md) | Tensor index with charge structure |
| [Sector](core/sector.md) | Charge-dimension pair |
| [Direction](core/direction.md) | Index orientation (IN/OUT) |

## Creating Tensors

Functions to create new tensors:

| Function | Description |
|----------|-------------|
| [zeros](creation/zeros.md) | Zero-filled tensor |
| [random](creation/random.md) | Random-filled tensor |
| [identity](creation/identity.md) | 2-index identity tensor |
| [isometry](creation/isometry.md) | 2-to-1 fusion isometry |
| [isometry_n](creation/isometry_n.md) | N-to-1 fusion isometry |

## Manipulation

Transform and rearrange tensors:

| Function | Description |
|----------|-------------|
| [conj](manipulation/conjugate.md) | Conjugate + flip directions |
| [permute](manipulation/permute.md) | Permute axes |
| [transpose](manipulation/transpose.md) | Transpose axes |
| [capcup](manipulation/capcup.md) | Invert bond direction |
| [retag](manipulation/retag.md) | Change index tags |
| [filter_blocks](manipulation/filter_blocks.md) | Extract block subset |
| [merge_axes](manipulation/merge_axes.md) | Merge multiple axes into one |

## Arithmetic

Mathematical operations:

| Topic | Description |
|-------|-------------|
| [Basic Operations](arithmetic/addition.md) | +, -, *, /, negation |
| [allclose](arithmetic/allclose.md) | Numerical equality within tolerance |
| [oplus](arithmetic/oplus.md) | Direct sum |
| [diag](arithmetic/diag.md) | Create diagonal tensor |
| [inv](arithmetic/inv.md) | Matrix inversion |

## Contraction

Contract and trace tensors:

| Function | Description |
|----------|-------------|
| [contract](contraction/contract.md) | General tensor contraction |
| [trace](contraction/trace.md) | Trace over index pairs |
| [einsum](contraction/einsum.md) | Einstein summation notation |

## Decomposition

SVD, QR, eigenvalue decomposition, and related operations:

| Function | Description |
|----------|-------------|
| [decomp](decomposition/decomp.md) | High-level decomposition (SVD/QR) |
| [svd](decomposition/svd.md) | Low-level SVD |
| [qr](decomposition/qr.md) | Low-level QR |
| [eig](decomposition/eig.md) | Eigenvalue decomposition |

## Symmetry Groups

Define and use symmetries:

| Group | Description |
|-------|-------------|
| [Overview](symmetry/overview.md) | Symmetry system introduction |
| [U1Group](symmetry/u1-group.md) | Integer charge symmetry |
| [Z2Group](symmetry/z2-group.md) | Binary symmetry |
| [SU2Group](symmetry/su2-group.md) | SU(2) non-Abelian spin symmetry |
| [ProductGroup](symmetry/product-group.md) | Multiple symmetries |

## Utilities

Supporting functionality:

| Topic | Description |
|-------|-------------|
| [Display](utilities/display.md) | Pretty-printing tensors |
| [Blocks](utilities/blocks.md) | Block structure utilities |
| [load_space](utilities/load_space.md) | Load physical spaces and operators |
| [serialize](utilities/serialize.md) | Convert a Tensor to a serializable dict |
| [deserialize](utilities/deserialize.md) | Reconstruct a Tensor from a serialized dict |

## Usage Patterns

For practical examples and complete working code, see:

- [Examples: Basic Usage](../examples/basic/first-tensor.md)
- [Examples: Symmetries](../examples/symmetries/u1-examples.md)
- [Examples: Operations](../examples/operations/contraction-examples.md)
- [Examples: Advanced](../examples/advanced/build-operators.md)

## Quick Links by Task

### I want to...

**Create a tensor**
→ [zeros](creation/zeros.md), [random](creation/random.md), [identity](creation/identity.md)

**Contract tensors**
→ [contract](contraction/contract.md), [trace](contraction/trace.md), [einsum](contraction/einsum.md)

**Decompose a tensor**
→ [decomp](decomposition/decomp.md), [svd](decomposition/svd.md), [qr](decomposition/qr.md), [eig](decomposition/eig.md)

**Merge tensor axes**
→ [merge_axes](manipulation/merge_axes.md), [isometry_n](creation/isometry_n.md)

**Use multiple symmetries**
→ [ProductGroup](symmetry/product-group.md)

**Build quantum operators**
→ [load_space](utilities/load_space.md)

**Save and load tensors**
→ [serialize](utilities/serialize.md), [deserialize](utilities/deserialize.md)

**See working examples**
→ [Examples](../examples/index.md)

## API Design

Nicole follows these principles:

- **Functional operations** return new tensors (e.g., `conj()`)
- **In-place methods** modify existing tensors (e.g., `Tensor.conj()`)
- **Charge conservation** enforced automatically
- **Array-style** API with familiar syntax (similar to NumPy/PyTorch)
- **Device management** for CPU and GPU computation
- **Autograd control** for optional gradient tracking

## See Also

- [Core Concepts](../getting-started/core-concepts.md): Understanding symmetry-aware tensors
- [Quick Examples](../getting-started/quick-examples.md): Hands-on introduction
- [Examples](../examples/index.md): Practical code examples
