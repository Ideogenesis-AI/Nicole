# trace

Compute trace over index pairs within a tensor.

::: nicole.trace
    options:
      show_source: false
      heading_level: 2

## Description

Computes the trace by summing over matching indices. Index pairs must have:
- Opposite directions (OUT ↔ IN)
- Matching symmetry groups and itags (for automatic mode)
- Compatible charge sectors

The function supports three modes:

1. **Automatic mode** (no parameters): Automatically finds and traces all pairs with matching itags and opposite directions
2. **Manual mode** (`axes` parameter): Explicitly specify which index pairs to trace
3. **Exclusion mode** (`excl` parameter): Automatically trace all matching pairs except specified exclusions

Result has reduced dimensionality (2 indices removed per pair).

## See Also

- [contract](contract.md): General contraction between two tensors
- [Examples: Contraction](../../examples/operations/contraction-examples.md)

## Notes

For a scalar result, trace over all index pairs. The automatic mode provides the most convenient API for tensors with clear itag structure.
