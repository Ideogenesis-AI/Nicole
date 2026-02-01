# contract

Contract two tensors along specified index pairs.

::: nicole.contract
    options:
      show_source: false
      heading_level: 2

## Description

Performs tensor contraction while preserving charge conservation. Can automatically contract indices with matching tags and opposite directions, or manually specify pairs.

## Automatic vs Manual Contraction

**Automatic** (axes=None):
- Finds all indices where tags match
- Requires opposite directions (OUT ↔ IN)
- Contracts all matching pairs

**Manual** (axes specified):
- Contract only specified index pairs
- Still requires opposite directions
- Validates tag matching

## See Also

- [trace](trace.md): Trace over index pairs within a tensor
- [Examples: Contraction](../../examples/operations/contraction-examples.md)

## Notes

Only blocks satisfying charge conservation are computed. For contracted indices, charges must match. Output blocks must conserve total charge.
