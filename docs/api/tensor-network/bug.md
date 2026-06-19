# BUG Integrator

The two-site Basis-Update & Galerkin (BUG) integrator advances a
[TensorTrain](tensor-train.md) by one odd–even sweep of rank-adaptive,
symmetry-faithful local K/L/S updates.

## bug_two_site

::: nicole.bug.bug_two_site
    options:
      show_source: false
      heading_level: 3

## BUGInfo

::: nicole.bug.BUGInfo
    options:
      show_source: false
      heading_level: 3

## Bond Gates

::: nicole.bug.bug_xx_bond_gates
    options:
      show_source: false
      heading_level: 3

::: nicole.bug.bug_heisenberg_bond_gates
    options:
      show_source: false
      heading_level: 3

::: nicole.bug.bug_xx_parity_mpos
    options:
      show_source: false
      heading_level: 3

## Description

A `bug_two_site` step splits the nearest-neighbour Hamiltonian into commuting
odd and even bond parities and applies them in a Lie (`order="lie"`) or Strang
(`order="strang"`) product. Each active bond is updated with a local K/L/S
solve whose basis is optionally augmented (`augment=True`) before a final SVD
truncation to `maxdim`. With a U(1)-symmetric state the updates preserve the
total charge exactly, so quantities such as the total magnetization are
conserved to machine precision.

## Usage Example

```python
import nicole.bug as bug

sites = bug.siteinds(6, d=2)
psi = bug.ttutils.random_tt(sites, maxdim=3, seed=42)
gates = bug.bug_xx_bond_gates(sites, J=1.0)

info = bug.bug_two_site(psi, gates, dt=0.05, order="strang", maxdim=64)
print(info.bond_dims_after)
```

## See Also

- [Tensor-Train States and Operators](tensor-train.md): the state/operator containers
- [Tensor Networks (BUG)](index.md): subpackage overview
