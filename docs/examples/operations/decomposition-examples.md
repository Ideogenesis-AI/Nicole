# Decomposition Examples

Tensor decomposition factorizes a tensor into simpler components, preserving symmetry at each factor. This is crucial for compression, canonical forms, and various many-body algorithms like DMRG and iPEPS.

**Decomposition modes:**

- **SVD**: Full singular value decomposition \(T = U \cdot S \cdot V^\dagger\)
- **UR**: Left-orthogonal form \(T = U \cdot R\), where \(U^\dagger U = I\)
- **LV**: Right-orthogonal form \(T = L \cdot V^\dagger\), where \(V V^\dagger = I\)
- **QR**: QR decomposition \(T = Q \cdot R\), where \(Q^\dagger Q = I\) and \(R\) is upper triangular

Each mode performs the decomposition **block-by-block**: sectors with different charges are decomposed independently. This block structure ensures that the decomposed factors maintain proper charge conservation and can be efficiently contracted in subsequent operations.

**Truncation** allows you to compress tensors by keeping only the most important singular values, controlled by either a maximum count or threshold.

```python exec="1" session="decomposition" result=""
from nicole import decomp, contract, conj, Tensor, Index, Sector, Direction, U1Group
from nicole.decomp import svd, qr
```

## Basic SVD

```python exec="1" session="decomposition" result="console" idprefix="" source="material-block"
group = U1Group()
idx = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))

# Create a tensor
T = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=7)

# Perform SVD: T ≈ U @ S @ Vh
U, S, Vh = decomp(T, axes=0, mode="SVD")

print(f"U:\n{U}\n\nS:\n{S}\n\nVh:\n{Vh}\n")

# Verify reconstruction
US = contract(U, S)  # U @ S
T_reconstructed = contract(US, Vh)  # (U @ S) @ Vh
error = (T - T_reconstructed).norm() / T.norm()
print(f"Reconstruction error: {error:.2e}")
```

## UR Decomposition

```python exec="1" session="decomposition" result="console" idprefix="" source="material-block"
# Get U and R = S @ Vh combined
U_ur, R = decomp(T, axes=0, mode="UR")

print(f"U:\n{U_ur}\n\nR:\n{R}\n")

# Verify
T_reconstructed_ur = contract(U_ur, R)
error = (T - T_reconstructed_ur).norm() / T.norm()
print(f"Reconstruction error: {error:.2e}")
```

## LV Decomposition

```python exec="1" session="decomposition" result="console" idprefix="" source="material-block"
# Get L = U @ S and V combined
L, V = decomp(T, axes=0, mode="LV")

print(f"L:\n{L}\n\nV:\n{V}\n")

# Verify
T_reconstructed_lv = contract(L, V)
error = (T - T_reconstructed_lv).norm() / T.norm()
print(f"Reconstruction error: {error:.2e}")
```

## Truncation: Keep N Values

```python exec="1" session="decomposition" result="console" idprefix="" source="material-block"
# Keep at most 10 singular values globally
U_trunc, S_trunc, Vh_trunc = decomp(T, axes=0, mode="SVD", trunc={"nkeep": 10})

print(f"Original bond dim: {U.indices[1].dim}")
print(f"Truncated bond dim: {U_trunc.indices[1].dim}\n")

# Compare norms
print(f"Original norm: {T.norm():.4f}")
US_trunc = contract(U_trunc, S_trunc)
T_trunc = contract(US_trunc, Vh_trunc)
print(f"Truncated norm: {T_trunc.norm():.4f}\n")

# Truncation error
error = (T - T_trunc).norm() / T.norm()
print(f"Relative error: {error:.4f}")
```

## Truncation: Threshold

```python exec="1" session="decomposition" result="console" idprefix="" source="material-block"
# Keep singular values >= 1e-10
U_thresh, S_thresh, Vh_thresh = decomp(T, axes=0, mode="SVD", trunc={"thresh": 1e-10})

print(f"Threshold truncated bond dim: {U_thresh.indices[1].dim}")
```

## Multi-Index Decomposition

```python exec="1" session="decomposition" result="console" idprefix="" source="material-block"
# Decompose 4-index tensor
idx4 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
T4 = Tensor.random([idx4, idx4.flip(), idx4, idx4.flip()], itags=["a", "b", "c", "d"], seed=42)

# Partition: (a, b) | (c, d)
U4, S4, Vh4 = decomp(T4, axes=[0, 1], mode="SVD")

print(f"U:\n{U4}\n\nVh:\n{Vh4}")
```

## Examining Singular Values

```python exec="1" session="decomposition" result="console" idprefix="" source="material-block"
# Low-level SVD for singular value access
U_sv, S_dict, Vh_sv = svd(T, axis=0)

# S_dict maps block keys to 1D singular value arrays
for key, s_values in S_dict.items():
    print(f"Block {key}:")
    print(f"  Number of singular values: {len(s_values)}")
    print(f"  Largest: {s_values[0]:.4f}")
    print(f"  Smallest: {s_values[-1]:.4f}")
    print(f"  Ratio: {s_values[0] / s_values[-1]:.2e}")
```

To monitor truncation loss, pass `requires_info=True`:

```python exec="1" session="decomposition" result="console" idprefix="" source="material-block"
U_sv, S_dict, Vh_sv, info = svd(T, axis=0, trunc={"nkeep": 2}, requires_info=True)
print(f"Discarded weight: {info['discarded_weight']:.4f}")
```

## QR Decomposition

QR decomposition factorizes a tensor into an orthogonal matrix Q and an upper triangular matrix R. Unlike SVD, no truncation is applied, making it useful for obtaining canonical forms without compression.

```python exec="1" session="decomposition" result="console" idprefix="" source="material-block"
# QR decomposition using high-level interface
Q, R = decomp(T, axes=0, mode="QR")

print(f"Q:\n{Q}\n\nR:\n{R}\n")

# Verify reconstruction
T_reconstructed_qr = contract(Q, R)
error_qr = (T - T_reconstructed_qr).norm() / T.norm()
print(f"Reconstruction error: {error_qr:.2e}")

# Verify orthogonality: Q†Q = I
QdagQ = contract(conj(Q), Q, excl=((1,),()))
print(f"\nOrthogonality check (Q†Q should be identity-like):\n{QdagQ}")
```

Alternatively, use the low-level `qr()` function directly:

```python exec="1" session="decomposition" result="console" idprefix="" source="material-block"
# Low-level QR function
Q_low, R_low = qr(T, axis=0)

print(f"Q (low-level):\n{Q_low}\n\nR (low-level):\n{R_low}")
```

## See Also

- API Reference: [decomp](../../api/decomposition/decomp.md)
- API Reference: [svd](../../api/decomposition/svd.md)
- API Reference: [qr](../../api/decomposition/qr.md)
