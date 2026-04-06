# Contraction Examples

Tensor contraction is the fundamental operation for connecting tensors in a network. It generalizes matrix multiplication to higher-order tensors while automatically enforcing charge conservation at every contracted bond.

**Key concepts:**

- **Automatic contraction**: Nicole matches indices by tags and opposite directions
- **Manual specification**: Control exactly which indices to contract via position or exclusion
- **Charge conservation**: Only blocks with matching charges on contracted indices contribute
- **Trace**: Special case where a tensor contracts with itself

Contractions in Nicole are **symmetry-aware**: they only compute blocks that satisfy charge conservation, making them significantly more efficient than dense tensor operations for systems with symmetry.

```python exec="1" session="contraction" result=""
from nicole import contract, trace, Tensor, Index, Sector, Direction, U1Group
```

## Basic Contraction

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
group = U1Group()
idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))

# Create two tensors
A = Tensor.random([idx_out, idx_out], itags=["i", "mid"], seed=10)
B = Tensor.random([idx_in, idx_out], itags=["mid", "j"], seed=11)

# Automatic contraction (matching tags with opposite directions)
C = contract(A, B)
print(C)
```

## Manual Pair Specification

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
# Specify axes explicitly: (axis_in_A, axis_in_B)
Cp = contract(A, B, axes=(1, 0))
print(Cp)
print()

# Verify they are identical
diff = (C - Cp).norm()
print(f"Difference from automatic: {diff:.2e}")
```

## Matrix-Matrix Multiplication

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
# Two matrices to contract
M1 = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=20)
M2 = Tensor.random([idx_out, idx_in], itags=["j", "k"], seed=21)

# Multiply: M1 @ M2 (contracts on "j")
result_mm = contract(M1, M2)
print(f"Result indices: {result_mm.itags}")
```

## Full Contraction (Scalar Result)

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
# Two tensors that fully contract (need opposite directions for each tag)
T1 = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=1)
T2 = Tensor.random([idx_in, idx_out], itags=["i", "j"], seed=2)

# Full contraction: all indices contract
scalar = contract(T1, T2)
print(f"Scalar result: {scalar.norm():.4f}")
print(f"Number of indices: {len(scalar.indices)}")
```

## MPS-like Contraction

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
# 3-index tensor: (auxiliary_left, auxiliary_right, physical)
from nicole import conj
idx_aux = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
idx_phys = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))

M = Tensor.random([idx_aux, idx_aux.flip(), idx_phys], itags=["left", "right", "phys"], seed=5)
print(f"M tensor:\n{M}\n")

# Contract M with its conjugate on left and phys, excluding right
M_dag = conj(M)
print(f"M† tensor:\n{M_dag}\n")

# Method 1: Specify which axes to contract (axes 0 and 2)
result1 = contract(M, M_dag, axes=((0, 2), (0, 2)))
print(f"Method 1 result (axes=[0,2]):\n{result1}\n")

# Method 2: Exclude right auxiliary index (position 1)
result2 = contract(M, M_dag, excl=((1,),()))
print(f"Method 2 result (excl=1):\n{result2}")
```

## Trace

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
# Square tensor
T_trace = Tensor.random([idx_out, idx_in], itags=["i", "i"], seed=99)

# Automatic trace (finds matching itags with opposite directions)
tr = trace(T_trace)
print(f"Trace (automatic): {tr.norm():.4f}")
print(f"Result is scalar: {len(tr.indices) == 0}\n")

# Manual specification by position
tr2 = trace(T_trace, axes=(0, 1))
print(f"Trace (manual): {tr2.norm():.4f}")
```

## Trace Operations

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
# 4-index tensor with paired itags
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
T_multi = Tensor.random(
    [idx, idx.flip(), idx, idx.flip()],
    itags=["a", "a", "b", "b"],
    seed=77
)

# Automatic mode: trace all matching pairs
result_auto = trace(T_multi)
print(f"Result is scalar: {result_auto.is_scalar()}")

# Manual mode: trace specific pair only
partial = trace(T_multi, axes=(0, 1))
print(f"Remaining indices: {partial.itags}")

# Exclusion mode: trace all except specified
partial2 = trace(T_multi, excl=[0, 1])
print(f"Remaining indices: {partial2.itags}")
```

## Multiple Contractions

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
# Contract three tensors: A-B-C
A_multi = Tensor.random([idx_out, idx_out], itags=["i", "mid1"], seed=1)
B_multi = Tensor.random([idx_in, idx_out], itags=["mid1", "mid2"], seed=2)
C_multi = Tensor.random([idx_in, idx_out], itags=["mid2", "j"], seed=3)

# Contract step by step
AB = contract(A_multi, B_multi)
print(f"After A-B: {AB.itags}")

ABC = contract(AB, C_multi)
print(f"After A-B-C: {ABC.itags}")
```

## Einstein Summation

`einsum` provides a compact subscript-string interface that dispatches to
`contract`, `trace`, and `permute` automatically.  The `->` separator is
required; the left-hand side is a comma-separated list of subscripts (one per
tensor) and the right-hand side is the output subscript.

> **Note:** Contractions are performed strictly left to right with no
> optimisation of the contraction order.

```python exec="1" session="contraction" result=""
from nicole import einsum
```

**Permutation** — reorder axes of a single tensor:

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
T = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=30)
T_T = einsum('ij->ji', T)
print(f"Original indices: {T.itags}")
print(f"Transposed indices: {T_T.itags}")
```

**Trace to scalar** — contract a repeated subscript letter within one tensor:

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
T_sq = Tensor.random([idx_out, idx_in], itags=["i", "i"], seed=31)
scalar = einsum('ii->', T_sq)
print(f"Trace result is scalar: {scalar.is_scalar()}")
print(f"Value: {scalar.norm():.4f}")
```

**Matrix multiply** — contract the shared index between two tensors:

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
A_e = Tensor.random([idx_out, idx_out], itags=["i", "j"], seed=32)
B_e = Tensor.random([idx_in, idx_out], itags=["j", "k"], seed=33)
C_e = einsum('ij,jk->ik', A_e, B_e)
print(f"Result indices: {C_e.itags}")
```

**Chain contraction** — three tensors contracted left to right:

```python exec="1" session="contraction" result="console" idprefix="" source="material-block"
idx_mid = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
X = Tensor.random([idx_out, idx_mid], itags=["i", "j"], seed=40)
Y = Tensor.random([idx_mid.flip(), idx_mid], itags=["j", "k"], seed=41)
Z = Tensor.random([idx_mid.flip(), idx_in], itags=["k", "l"], seed=42)
result = einsum('ij,jk,kl->il', X, Y, Z)
print(f"Chain result indices: {result.itags}")
```

## See Also

- API Reference: [contract](../../api/contraction/contract.md)
- API Reference: [trace](../../api/contraction/trace.md)
- API Reference: [einsum](../../api/contraction/einsum.md)
