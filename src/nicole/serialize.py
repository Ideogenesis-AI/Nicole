# Copyright (C) 2026 Changkai Zhang.
#
# This file is part of Nicole library.
#
# Nicole is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Nicole is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Nicole. If not, see <https://www.gnu.org/licenses/>.


"""Serialize and deserialize Tensor instances to/from plain dicts.

The serialized format uses only Python primitives (str, int, tuple, dict,
None) and torch.Tensor values, making it directly compatible with
`torch.save` / `torch.load(..., weights_only=True)`.

Serialized dict schema (version 1):

    {
        "version": 1,
        "label": str,
        "dtype": str,          # e.g. "float64"
        "itags": tuple[str, ...],
        "indices": [
            {
                "direction": int,       # +1 (IN) or -1 (OUT)
                "group": {
                    "type": str,        # "U1" | "Z2" | "SU2" | "Product"
                    "components": [...] # only present for "Product"
                },
                "sectors": [
                    {"charge": int | tuple, "dim": int},
                    ...
                ]
            },
            ...
        ],
        "data": [
            {"key": tuple, "value": torch.Tensor},
            ...
        ],
        "intw": None | [
            {
                "key": tuple,
                "edges": tuple of (two_j: int, dir_sign: int) per edge,
                "weights": torch.Tensor
            },
            ...
        ]
    }
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch

from .blocks import BlockKey
from .index import Index
from .tensor import Tensor
from .typing import Direction, Sector
from .symmetry import SymmetryGroup
from .symmetry import U1Group, Z2Group
from .symmetry import SU2Group
from .symmetry import ProductGroup


# --- Group helpers ---

_GROUP_TO_TYPE: Dict[type, str] = {
    U1Group: "U1",
    Z2Group: "Z2",
    SU2Group: "SU2",
}

_TYPE_TO_GROUP: Dict[str, type] = {
    "U1": U1Group,
    "Z2": Z2Group,
    "SU2": SU2Group,
}


def _serialize_group(group: SymmetryGroup) -> dict:
    if type(group) in _GROUP_TO_TYPE:
        return {"type": _GROUP_TO_TYPE[type(group)]}
    if isinstance(group, ProductGroup):
        return {
            "type": "Product",
            "components": [_serialize_group(c) for c in group.components],
        }
    raise TypeError(f"Cannot serialize symmetry group of type {type(group).__name__}")


def _deserialize_group(d: dict) -> SymmetryGroup:
    t = d["type"]
    if t in _TYPE_TO_GROUP:
        return _TYPE_TO_GROUP[t]()
    if t == "Product":
        return ProductGroup([_deserialize_group(c) for c in d["components"]])
    raise ValueError(f"Unknown group type in serialized data: {t!r}")


# --- Index helpers ---

def _serialize_index(idx: Index) -> dict:
    return {
        "direction": int(idx.direction),
        "group": _serialize_group(idx.group),
        "sectors": [{"charge": s.charge, "dim": s.dim} for s in idx.sectors],
    }


def _deserialize_index(d: dict) -> Index:
    direction = Direction(d["direction"])
    group = _deserialize_group(d["group"])
    sectors = tuple(Sector(charge=s["charge"], dim=s["dim"]) for s in d["sectors"])
    return Index(direction=direction, group=group, sectors=sectors)


# --- serialize / deserialize ---

def serialize(tensor: Tensor) -> dict:
    """Convert a Tensor to a plain dict of Python primitives and torch.Tensor.

    The returned dict is directly compatible with `torch.save` /
    `torch.load(..., weights_only=True)`.

    Parameters
    ----------
    tensor : Tensor
        The tensor to serialize.

    Returns
    -------
    dict
        Serialized representation. Top-level keys: `"version"`, `"label"`,
        `"dtype"`, `"itags"`, `"indices"`, `"data"`, `"intw"`.

    Examples
    --------
    >>> payload = serialize(t)
    >>> torch.save(payload, "tensor.tnsr")
    """
    data_list = [{"key": k, "value": v} for k, v in tensor.data.items()]

    intw_list: Optional[list] = None
    if tensor.intw is not None:
        from .symmetry import delegate as dg
        intw_list = []
        for k, bridge in tensor.intw.items():
            entry = dg.serialize(bridge)
            entry["key"] = k
            intw_list.append(entry)

    return {
        "version": 1,
        "label": tensor.label,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "itags": tensor.itags,
        "indices": [_serialize_index(idx) for idx in tensor.indices],
        "data": data_list,
        "intw": intw_list,
    }


def deserialize(data: dict, device: Union[str, torch.device] = "cpu") -> Tensor:
    """Reconstruct a Tensor from a dict previously produced by `serialize`.

    Parameters
    ----------
    data : dict
        Dict previously produced by `serialize`.
    device : str or torch.device, optional
        Device to place all tensors (data blocks and intertwiner weights) on.
        Defaults to `"cpu"`.

    Returns
    -------
    Tensor
        Reconstructed Tensor with all blocks on *device*.

    Examples
    --------
    >>> payload = torch.load("tensor.tnsr", weights_only=True)
    >>> t = deserialize(payload, device="cpu")
    """
    version = data.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported serialization version: {version!r}")

    device = torch.device(device)
    dtype: torch.dtype = getattr(torch, data["dtype"])

    indices = tuple(_deserialize_index(d) for d in data["indices"])
    itags = tuple(data["itags"])
    label: str = data["label"]

    block_data: Dict[BlockKey, torch.Tensor] = {
        entry["key"]: entry["value"].to(device=device, dtype=dtype)
        for entry in data["data"]
    }

    intw = None
    if data["intw"] is not None:
        from .symmetry import delegate as dg
        intw = {
            entry["key"]: dg.deserialize(entry, device=device, dtype=dtype)
            for entry in data["intw"]
        }

    return Tensor(
        indices=indices, itags=itags, data=block_data, intw=intw,
        dtype=dtype, label=label,
    )
