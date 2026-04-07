# Serialization

Tensors can be saved to disk and reloaded using `serialize` and `deserialize`.
The format is a plain Python `dict` of primitives and `torch.Tensor` values,
making it directly compatible with `torch.save` / `torch.load(..., weights_only=True)`.

```python exec="1" session="serial" result=""
import torch
from nicole import Tensor, Index, Sector, Direction
from nicole import U1Group, Z2Group, SU2Group, ProductGroup
from nicole import serialize, deserialize
```

## Abelian Tensors

For Abelian tensors (U(1), Z(2), U(1)×Z(2), …) the serialized dict contains
no intertwiner data; the `"intw"` key is `None`.

```python exec="1" session="serial" result="console" idprefix="" source="material-block"
group = U1Group()
idx_o = Index(Direction.OUT, group, (Sector(-1, 2), Sector(0, 3), Sector(1, 2)))
idx_i = Index(Direction.IN,  group, (Sector(-1, 2), Sector(0, 3), Sector(1, 2)))

t = Tensor.random([idx_o, idx_i], seed=0, itags=["p", "q"])

payload = serialize(t)
print("version:", payload["version"])
print("dtype:  ", payload["dtype"])
print("label:  ", payload["label"])
print("itags:  ", payload["itags"])
print("blocks: ", len(payload["data"]))
print("intw:   ", payload["intw"])
```

Round-trip the tensor through `deserialize`:

```python exec="1" session="serial" result="console" idprefix="" source="material-block"
t2 = deserialize(payload)

print("label matches:", t2.label == t.label)
print("dtype matches:", t2.dtype == t.dtype)
print("itags match:  ", t2.itags == t.itags)

for key in t.data:
    assert torch.equal(t.data[key], t2.data[key])
print("all blocks match: True")
```

## Save and Load with torch.save

```python exec="1" session="serial" result="console" idprefix="" source="material-block"
import tempfile, os

group = ProductGroup([U1Group(), Z2Group()])
idx_o = Index(Direction.OUT, group, (Sector((0, 0), 2), Sector((1, 0), 2),
                                     Sector((0, 1), 2), Sector((1, 1), 2)))
idx_i = Index(Direction.IN,  group, (Sector((0, 0), 2), Sector((1, 0), 2),
                                     Sector((0, 1), 2), Sector((1, 1), 2)))

t = Tensor.random([idx_o, idx_o, idx_i, idx_i], seed=10, itags=["a", "b", "c", "d"])

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "tensor.tnsr")
    torch.save(serialize(t), path)

    payload = torch.load(path, weights_only=True)
    t2 = deserialize(payload)

print("block data: ~%d bytes" % (sum(v.nbytes for e in payload["data"] for v in [e["value"]])))
print("block keys match:", set(t.data.keys()) == set(t2.data.keys()))
for key in t.data:
    assert torch.equal(t.data[key], t2.data[key])
print("all blocks match: True")
```

## Non-Abelian Tensors

Non-Abelian tensors carry intertwiner weights in addition to the reduced blocks.
Both are included in the serialized output, so the round-trip is lossless.
In the example below, the weights are filled with random values before
serializing.

```python exec="1" session="serial" result="console" idprefix="" source="material-block"
group = SU2Group()
idx_i = Index(Direction.IN,  group, (Sector(1, 2), Sector(2, 2)))
idx_o = Index(Direction.OUT, group, (Sector(1, 2), Sector(2, 2)))

t = Tensor.random([idx_i, idx_i, idx_o], seed=20, itags=["a", "b", "c"])

# Fill intertwiner weights with reproducible random values
gen = torch.Generator()
gen.manual_seed(21)
for key in t.intw:
    t.intw[key].weights.copy_(torch.randn(t.intw[key].weights.shape, generator=gen))

payload = serialize(t)
print("intw entries:", len(payload["intw"]))

t2 = deserialize(payload)

for key in t.data:
    assert torch.equal(t.data[key], t2.data[key])
print("data blocks match: True")

for key in t.intw:
    assert torch.allclose(t.intw[key].weights, t2.intw[key].weights)
print("intertwiner weights match: True")
```

## Loading to a Different Device

The `device` argument of `deserialize` places all tensors (data blocks and
intertwiner weights) on the requested device:

```python
# Load to GPU
payload = torch.load("tensor.tnsr", weights_only=True)
t_gpu = deserialize(payload, device="cuda:0")

# Load back to CPU
t_cpu = deserialize(payload, device="cpu")
```

The `device` argument avoids a separate `.to()` call after loading.

## See Also

- API Reference: [serialize](../../api/utilities/serialize.md)
- API Reference: [deserialize](../../api/utilities/deserialize.md)
- Previous: [Manipulation](../operations/manipulation-examples.md)
