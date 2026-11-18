# deserialize

Reconstruct a `Tensor` from a dict previously produced by `serialize`. Indices, charge sectors, block data and, for a generic group, the intertwiners are rebuilt from the recorded description, so the tensor that comes back is equal to the one that went in. The dtype travels in the dict and is reapplied here, leaving the device as the only thing worth choosing at load time.

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

## Notes

- `data["version"]` is checked before anything is rebuilt, and an unsupported
  version raises `ValueError`.
- `device` applies to both block data tensors (`"data"`) and intertwiner
  weight tensors (`"intw"`), if present.
- For non-Abelian tensors, reconstruction of the CG data is delegated to
  `nicole.symmetry.delegate.deserialize`.
- The `dtype` is stored in the serialized dict and applied to all tensors on
  reconstruction; no explicit `dtype` argument is needed.
