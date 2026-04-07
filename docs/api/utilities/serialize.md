# serialize

Convert a `Tensor` to a plain dict of Python primitives and `torch.Tensor`.

::: nicole.serialize.serialize
    options:
      show_source: false
      heading_level: 2

## Description

`serialize` encodes every piece of a `Tensor`'s state — its indices, charge
structure, block data, and (for non-Abelian tensors) intertwiner weights — into
a plain Python `dict` that contains only:

- `str`, `int`, `tuple`, `list`, `dict`, `None`
- `torch.Tensor` (block data and intertwiner weights)

The returned dict is therefore directly compatible with `torch.save` and
`torch.load(..., weights_only=True)`.

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor` | `Tensor` | The tensor to serialize. |

## Returns

`dict` — Serialized representation with the following top-level keys:

| Key | Type | Description |
|-----|------|-------------|
| `"version"` | `int` | Format version (currently `1`). |
| `"label"` | `str` | Tensor label string. |
| `"dtype"` | `str` | PyTorch dtype name, e.g. `"float64"`. |
| `"itags"` | `tuple[str, ...]` | Per-index tag strings. |
| `"indices"` | `list[dict]` | Serialized `Index` objects (direction, group, sectors). |
| `"data"` | `list[dict]` | List of `{"key": tuple, "value": torch.Tensor}` entries. |
| `"intw"` | `list[dict]` or `None` | Intertwiner list for non-Abelian tensors, or `None` for Abelian. |

### Serialized index schema

Each entry in `"indices"` has the form:

```python
{
    "direction": int,       # +1 (IN) or -1 (OUT)
    "group": {
        "type": str,        # "U1" | "Z2" | "SU2" | "Product"
        "components": [...]  # only present for "Product"
    },
    "sectors": [
        {"charge": int | tuple, "dim": int},
        ...
    ]
}
```

### Serialized intertwiner schema

Each entry in `"intw"` has the form:

```python
{
    "key":     tuple,          # BlockKey (one charge per index)
    "edges":   tuple,          # ((two_j, dir_sign), ...) one per tensor index
    "weights": torch.Tensor    # intertwiner weight matrix
}
```

`two_j` is the doubled SU(2) spin (integer), and `dir_sign` is `+1` for
incoming and `-1` for outgoing.

The intertwiner entries are produced by `nicole.symmetry.delegate.serialize`
and consumed by `nicole.symmetry.delegate.deserialize` — the format is fully
owned by the `delegate` module and is independent of which backend package
provides the CG coefficients.

## Notes

- `torch.save` preserves Python `tuple` values natively, so charges and block
  keys remain tuples after a save/load round-trip — no additional conversion is
  needed.
- The `"version"` key allows future format evolution while keeping
  backward-compatible deserialization in `deserialize`.

## Examples

### Save a tensor to disk

```python
import torch
from nicole import Tensor, Index, Sector, Direction
from nicole import U1Group
from nicole import serialize

group = U1Group()
idx_o = Index(Direction.OUT, group, (Sector(-1, 2), Sector(0, 3), Sector(1, 2)))
idx_i = Index(Direction.IN,  group, (Sector(-1, 2), Sector(0, 3), Sector(1, 2)))
t = Tensor.random([idx_o, idx_o, idx_i], seed=0)

payload = serialize(t)
torch.save(payload, "tensor.tnsr")
```

### Inspect the serialized structure

```python
payload = serialize(t)
print(payload["version"])   # 1
print(payload["dtype"])     # "float64"
print(payload["itags"])     # ('_init_', '_init_', '_init_')  (default tags)
print(len(payload["data"])) # number of non-zero blocks
print(payload["intw"])      # None for Abelian tensors
```

## See Also

- [deserialize](deserialize.md): Reconstruct a `Tensor` from a serialized dict
- [Examples: Serialization](../../examples/advanced/serialization-examples.md)
- [Tensor](../core/tensor.md): Main tensor class
