# deserialize

Reconstruct a `Tensor` from a dict previously produced by `serialize`.

::: nicole.deserialize
    options:
      show_source: false
      heading_level: 2

## Description

`deserialize` is the inverse of [`serialize`](serialize.md). It reads the
plain dict produced by `serialize` (or loaded from disk via
`torch.load(..., weights_only=True)`) and reconstructs the original `Tensor`,
including all indices, block data, and — for non-Abelian tensors — intertwiner
weights.

All `torch.Tensor` values in the result are placed on `device`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `dict` | — | Dict previously produced by `serialize`. |
| `device` | `str` or `torch.device` | `"cpu"` | Device to place all block data and intertwiner weights on. |

## Returns

`Tensor` — Fully reconstructed tensor with all blocks on `device`.

## Raises

`ValueError` — If version from `data["version"]` is not supported.

## Notes

- `device` applies to both block data tensors (`"data"`) and intertwiner
  weight tensors (`"intw"`), if present.
- For non-Abelian tensors, reconstruction of the CG data is delegated to
  `nicole.symmetry.delegate.deserialize`.
- The `dtype` is stored in the serialized dict and applied to all tensors on
  reconstruction; no explicit `dtype` argument is needed.

## Examples

### Basic load from disk

```python
import torch
from nicole import deserialize

payload = torch.load("tensor.tnsr", weights_only=True)
t = deserialize(payload)
```

### Load to GPU

```python
payload = torch.load("tensor.tnsr", weights_only=True)
t = deserialize(payload, device="cuda:0")
```

### Round-trip in memory

```python
from nicole import serialize, deserialize

payload = serialize(t)
t2 = deserialize(payload)
```

## See Also

- [serialize](serialize.md): Convert a `Tensor` to a serializable dict
- [Examples: Serialization](../../examples/advanced/serialization-examples.md)
- [Tensor](../core/tensor.md): Main tensor class
