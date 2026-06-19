# Copyright (C) 2025-2026 Changkai Zhang.
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


"""Tensor-train state helpers built on Nicole tensors.

The functions in this module are the main state-construction and basic linear
algebra layer for the BUG code. The underlying algorithms are unchanged
from the original port, but the Python surface is intentionally cleaner:

- :class:`TensorTrainBuildOptions` gathers common construction options.
- the public helpers keep legacy keywords working for the test suite;
- every function documents the one-based site/bond conventions used
  throughout the package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch
from nicole import Direction, Tensor

from ..indices import (
    Ix,
    bond_index,
    fresh_itag,
    has_nontrivial_symmetry,
    normalize_symmetry,
    spin_half_site_sectors,
)
from ..linalg import lq, qr, random_unitary, truncate
from ..helpers import dag, flatten_fortran, make_tensor, reshape_fortran, scalar, tcontract, to_dense

__all__ = [
    "TensorTrain",
    "TensorTrainBuildOptions",
    "dot",
    "linkind",
    "linkinds",
    "linkdims",
    "maxlinkdim",
    "normalize_",
    "norm",
    "orthogonalize",
    "orthogonalize_",
    "product_tt",
    "random_tt",
    "replacelinks",
    "replacelinks_",
    "rescale_",
    "siteind",
    "siteinds",
    "tensor_train_from_arrays",
    "tensor_train_from_vector",
    "tensortrain_bond_indices",
    "tensortrain_boundary_link",
    "tensortrain_site_index",
    "vector",
]

_UP = 0
_DOWN = 1
_SPIN_CHARGE = {_UP: 1, _DOWN: -1}


@dataclass(frozen=True)
class TensorTrainBuildOptions:
    """Options shared by tensor-train constructors.

    Parameters
    ----------
    symmetry:
        Symmetry label passed to :func:`normalize_symmetry` when
        building product states. Supported values are whatever the index
        helpers accept, typically ``"trivial"`` and ``"u1"``.
    maxdim:
        Maximum bond dimension kept by SVD-based constructors.
    cutoff:
        Singular-value truncation threshold for dense factorizations.
    seed:
        Optional torch RNG seed for randomized constructors.
    gauge_centre:
        One-based orthogonality center used by random and
        rescaling helpers. ``None`` defaults to the last site.
    """

    symmetry: str = "u1"
    maxdim: int | float = math.inf
    cutoff: float = 0.0
    seed: int | None = None
    gauge_centre: int | None = None


_BUILD_OPTION_FIELDS = {field.name for field in TensorTrainBuildOptions.__dataclass_fields__.values()}


def _coerce_build_options(options: TensorTrainBuildOptions | None = None, **kwargs: Any) -> TensorTrainBuildOptions:
    """Merge legacy keyword arguments into a build-options object.

    Parameters
    ----------
    options:
        Existing options object to start from.
    **kwargs:
        Keyword overrides such as ``maxdim=8`` or ``seed=3``.

    Returns
    -------
    A normalized :class:`TensorTrainBuildOptions` instance.
    """

    overrides = {name: kwargs.pop(name) for name in list(kwargs) if name in _BUILD_OPTION_FIELDS}
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown tensor-train option(s): {unknown}")
    if options is None:
        return TensorTrainBuildOptions(**overrides)
    return replace(options, **overrides)


def _finite_maxdim(maxdim: int | float, *, label: str) -> int:
    """Convert a finite bond-dimension request to an integer.

    Parameters
    ----------
    maxdim:
        User-supplied bond cap.
    label:
        Name of the calling helper, used in error messages.

    Returns
    -------
    ``maxdim`` as an integer.
    """

    if not math.isfinite(float(maxdim)):
        raise ValueError(f"{label} requires a finite 'maxdim'.")
    return int(maxdim)


@dataclass
class TensorTrain:
    """Ordered list of Nicole MPS core tensors.

    The package consistently treats sites and bonds as one-based when a helper
    accepts a site number or bond number. The underlying storage is still a
    Python list, so indexing inside the implementation remains zero-based.

    Parameters
    ----------
    cores:
        Rank-3 Nicole tensors ordered from left to right.
    """

    cores: list[Tensor]

    def __len__(self) -> int:
        """Return the number of sites in the state."""

        return len(self.cores)

    def __getitem__(self, idx: int) -> Tensor:
        """Return the core tensor at a zero-based Python index."""

        return self.cores[idx]

    def __setitem__(self, idx: int, value: Tensor) -> None:
        """Replace one core tensor in-place."""

        self.cores[idx] = value

    def copy(self) -> "TensorTrain":
        """Return a deep tensor copy of the full MPS."""

        return TensorTrain([core.clone() for core in self.cores])

    def deepcopy(self) -> "TensorTrain":
        """Compatibility alias for :meth:`copy`."""

        return self.copy()


def _leg(core: Tensor, itag: str) -> Ix:
    """Extract one named leg from a Nicole core as an :class:`Ix`.

    Parameters
    ----------
    core:
        Tensor carrying the requested leg.
    itag:
        Nicole tag of the desired index.

    Returns
    -------
    The matching :class:`Ix`.
    """

    axis = core.itags.index(itag)
    idx = core.indices[axis]
    return Ix(itag, int(idx.dim), idx.direction, idx.sectors, idx.group)


def _shared_itag(left: Tensor, right: Tensor) -> str:
    """Return the unique link tag shared by two neighboring cores.

    Parameters
    ----------
    left:
        Left neighboring core.
    right:
        Right neighboring core.

    Returns
    -------
    The shared bond tag.
    """

    shared = [tag for tag in left.itags if tag in right.itags]
    if len(shared) != 1:
        raise ValueError(f"Expected exactly one shared itag, got {shared}.")
    return shared[0]


def linkind(psi: TensorTrain, j: int) -> Ix:
    """Return the internal bond index crossing bond ``j``.

    Parameters
    ----------
    psi:
        Tensor-train state.
    j:
        One-based bond number between sites ``j`` and ``j+1``.

    Returns
    -------
    The shared bond :class:`Ix`.
    """

    return _leg(psi[j - 1], _shared_itag(psi[j - 1], psi[j]))


def linkinds(psi: TensorTrain, j: int | None = None):
    """Return boundary and internal bond indices.

    Parameters
    ----------
    psi:
        Tensor-train state.
    j:
        Optional one-based bond number. When supplied, only the shared index
        for that bond is returned inside a single-item list.

    Returns
    -------
    Either the requested bond list or the full ``[left_boundary, ..., right_boundary]``
    bond-index list.
    """

    if j is not None:
        return [_leg(psi[j - 1], tag) for tag in psi[j - 1].itags if tag in psi[j].itags]

    n = len(psi)
    left = _leg(psi[0], psi[0].itags[0])
    interior = [linkind(psi, k) for k in range(1, n)]
    right = _leg(psi[n - 1], psi[n - 1].itags[2])
    return [left, *interior, right]


def siteinds(psi: TensorTrain, j: int | None = None):
    """Return physical site indices.

    Parameters
    ----------
    psi:
        Tensor-train state.
    j:
        Optional one-based site number.

    Returns
    -------
    All site indices, or the requested single site index.
    """

    sites = [_leg(core, core.itags[1]) for core in psi.cores]
    return sites[j - 1] if j is not None else sites


def siteind(psi: TensorTrain, j: int) -> Ix:
    """Convenience wrapper returning one site index.

    Parameters
    ----------
    psi:
        Tensor-train state.
    j:
        One-based site number.

    Returns
    -------
    The requested physical site :class:`Ix`.
    """

    return siteinds(psi, j)


def tensortrain_boundary_link(psi: TensorTrain, side: str) -> Ix:
    """Return the left or right boundary bond index.

    Parameters
    ----------
    psi:
        Tensor-train state with at least two sites.
    side:
        Either ``"left"`` or ``"right"``.

    Returns
    -------
    The requested boundary bond index.
    """

    if len(psi) < 2:
        raise ValueError("requires at least two sites")
    if side == "left":
        return _leg(psi[0], psi[0].itags[0])
    if side == "right":
        return _leg(psi[-1], psi[-1].itags[2])
    raise ValueError("side must be 'left' or 'right'")


def tensortrain_site_index(psi: TensorTrain, k: int) -> Ix:
    """Return the site index at one-based position ``k``.

    Parameters
    ----------
    psi:
        Tensor-train state with at least two sites.
    k:
        One-based site number.

    Returns
    -------
    The requested physical site index.
    """

    if len(psi) < 2:
        raise ValueError("requires at least two sites")
    return siteinds(psi, k)


def tensortrain_bond_indices(psi: TensorTrain, bond: int):
    """Return the five indices surrounding an active two-site bond.

    Parameters
    ----------
    psi:
        Tensor-train state.
    bond:
        One-based bond number between sites ``bond`` and ``bond+1``.

    Returns
    -------
    ``(link_l, link_mid, link_r, site_l, site_r)``.
    """

    left_t = psi[bond - 1]
    right_t = psi[bond]
    link_mid = _leg(left_t, _shared_itag(left_t, right_t))
    link_l = _leg(left_t, left_t.itags[0])
    link_r = _leg(right_t, right_t.itags[2])
    site_l = _leg(left_t, left_t.itags[1])
    site_r = _leg(right_t, right_t.itags[1])
    return link_l, link_mid, link_r, site_l, site_r


def replacelinks(psi: TensorTrain) -> TensorTrain:
    """Clone an MPS and give every internal bond a fresh tag.

    Parameters
    ----------
    psi:
        Input tensor train.

    Returns
    -------
    A copied tensor train with distinct internal bond labels.
    """

    out = psi.copy()
    replacelinks_(out)
    return out


def replacelinks_(psi: TensorTrain) -> TensorTrain:
    """Retag every internal bond of an MPS in-place.

    Parameters
    ----------
    psi:
        Tensor train to modify.

    Returns
    -------
    The same ``psi`` object.
    """

    for j in range(1, len(psi)):
        old = _shared_itag(psi[j - 1], psi[j])
        new = fresh_itag(f"b{j}")
        psi[j - 1].retag({old: new})
        psi[j].retag({old: new})
    return psi


def maxlinkdim(psi: TensorTrain) -> int:
    """Return the largest internal bond dimension.

    Parameters
    ----------
    psi:
        Tensor-train state.

    Returns
    -------
    Maximum internal link dimension, or ``1`` for a single-site state.
    """

    if len(psi) <= 1:
        return 1
    return max(linkind(psi, j).dim for j in range(1, len(psi)))


def rescale_(psi: TensorTrain, c: complex, gauge_centre: int | None = None) -> TensorTrain:
    """Scale one core tensor so the whole state is multiplied by ``c``.

    Parameters
    ----------
    psi:
        Tensor-train state to modify.
    c:
        Scalar multiplier.
    gauge_centre:
        One-based core that should absorb the scale factor.

    Returns
    -------
    The same ``psi`` object.
    """

    target = len(psi) if gauge_centre is None else gauge_centre
    core = psi[target - 1]
    core.data = {key: block * c for key, block in core.data.items()}
    return psi


def tensor_train_from_arrays(sites: Sequence[Ix], arrays: Sequence[torch.Tensor | object]) -> TensorTrain:
    """Construct an MPS from explicit rank-3 arrays.

    Parameters
    ----------
    sites:
        Physical site indices, one per core.
    arrays:
        Rank-3 arrays ordered as ``(bond_left, site, bond_right)``.

    Returns
    -------
    A :class:`TensorTrain` with matching physical indices.
    """

    n = len(sites)
    if len(arrays) != n:
        raise ValueError("Length of sites and arrays must match.")

    cores: list[Tensor] = []
    for k in range(1, n + 1):
        arr = torch.as_tensor(arrays[k - 1], dtype=torch.complex128)
        if arr.ndim != 3:
            raise ValueError(f"Core {k} must be rank-3, got shape {tuple(arr.shape)}.")

        dl, d, dr = (int(dim) for dim in arr.shape)
        if d != sites[k - 1].dim:
            raise ValueError(f"Core {k} physical dimension {d} != site dimension {sites[k - 1].dim}.")

        left = Ix(f"b{k-1}", dl, Direction.IN)
        site = sites[k - 1]
        right = Ix(f"b{k}", dr, Direction.OUT)
        cores.append(make_tensor(arr, [left, site, right], dtype=torch.complex128))

    return TensorTrain(cores)


def product_tt(spins, *, options: TensorTrainBuildOptions | None = None, **kwargs: Any) -> TensorTrain:
    """Build a spin-1/2 product state.

    Parameters
    ----------
    spins:
        Iterable describing local spin states. Entries ``"u"``, ``"up"``
        and ``"0"`` map to ``|up>``; everything else maps to ``|down>``.
    options:
        Optional :class:`TensorTrainBuildOptions`.
    **kwargs:
        Legacy overrides such as ``symmetry="u1"``.

    Returns
    -------
    The requested product state as a tensor train.
    """

    config = _coerce_build_options(options, **kwargs)
    symmetry = normalize_symmetry(config.symmetry)
    site_sectors = spin_half_site_sectors(symmetry)
    codes = [(_UP if str(state).lower() in {"u", "up", "0"} else _DOWN) for state in spins]

    if symmetry == "trivial":
        # Dense product states are simple one-hot rank-3 cores.
        sites = [Ix(f"s{k}", 2, Direction.OUT, site_sectors) for k in range(1, len(codes) + 1)]
        arrays = []
        for code in codes:
            arr = torch.zeros((1, 2, 1), dtype=torch.complex128)
            arr[0, code, 0] = 1.0
            arrays.append(arr)
        return tensor_train_from_arrays(sites, arrays)

    # In the U(1) case each bond records the cumulative outgoing Sz charge.
    cores: list[Tensor] = []
    q_left = 0
    for k, code in enumerate(codes, start=1):
        spin_charge = _SPIN_CHARGE[code]
        q_right = q_left - spin_charge
        left = bond_index(f"b{k-1}", Direction.IN, [(q_left, 1)])
        site = Ix(f"s{k}", 2, Direction.OUT, site_sectors)
        right = bond_index(f"b{k}", Direction.OUT, [(q_right, 1)])
        data = {(q_left, spin_charge, q_right): torch.ones((1, 1, 1), dtype=torch.complex128)}
        cores.append(make_tensor(data, [left, site, right], dtype=torch.complex128))
        q_left = q_right
    return TensorTrain(cores)


def tensor_train_from_vector(
    fx,
    sites: Sequence[Ix],
    *,
    options: TensorTrainBuildOptions | None = None,
    **kwargs: Any,
) -> TensorTrain:
    """Factor a dense state vector into a tensor train using sequential SVDs.

    Parameters
    ----------
    fx:
        Dense state vector in flat Fortran ordering.
    sites:
        Physical site indices defining the target tensor shape.
    options:
        Optional :class:`TensorTrainBuildOptions`.
    **kwargs:
        Legacy overrides such as ``maxdim=8`` and ``cutoff=1e-12``.

    Returns
    -------
    A tensor train whose dense vector matches ``fx`` up to truncation.
    """

    config = _coerce_build_options(options, **kwargs)
    dprod = math.prod(ix.dim for ix in sites)
    vec = torch.as_tensor(fx, dtype=torch.complex128).reshape(-1)
    if int(vec.numel()) != int(dprod):
        raise ValueError(f"Expected vector of length {dprod}, got {vec.numel()}.")

    arrays: list[torch.Tensor] = []
    A = reshape_fortran(vec, (1, dprod))
    left_dim = 1

    # Split off one site at a time while keeping Fortran-order conventions so
    # the tensorized state matches the rest of the codebase.
    for k in range(len(sites) - 1):
        d = sites[k].dim
        right_dim = int(A.numel() // (left_dim * d))
        A = reshape_fortran(A, (left_dim * d, right_dim))
        U, s, Vh = torch.linalg.svd(A, full_matrices=False)
        keep = truncate(s, maxdim=config.maxdim, cutoff=config.cutoff)
        arrays.append(reshape_fortran(U[:, :keep], (left_dim, d, keep)))
        A = torch.diag(s[:keep]) @ Vh[:keep, :]
        left_dim = keep

    arrays.append(reshape_fortran(A, (left_dim, sites[-1].dim, 1)))
    return tensor_train_from_arrays(sites, arrays)


def linkdims(sitedims: Sequence[int], bonddim: int) -> list[int]:
    """Compute the standard capped bond-dimension profile for an MPS.

    Parameters
    ----------
    sitedims:
        Physical dimensions of each site.
    bonddim:
        Requested maximum internal bond dimension.

    Returns
    -------
    A length ``len(sitedims)+1`` list containing boundary and internal link
    dimensions.
    """

    n = len(sitedims)
    dims = [1] * (n + 1)
    left_prod = [1]
    for d in sitedims:
        left_prod.append(left_prod[-1] * int(d))
    right_prod = [1] * (n + 1)
    for k in range(n - 1, -1, -1):
        right_prod[k] = right_prod[k + 1] * int(sitedims[k])
    for k in range(1, n):
        dims[k] = min(left_prod[k], bonddim, right_prod[k])
    return dims


def random_tt(
    sites: Sequence[Ix],
    maxdim: int | None = None,
    *,
    options: TensorTrainBuildOptions | None = None,
    **kwargs: Any,
) -> TensorTrain:
    """Build a random dense tensor train in mixed canonical form.

    Parameters
    ----------
    sites:
        Physical site indices.
    maxdim:
        Optional legacy maximum bond dimension.
    options:
        Optional :class:`TensorTrainBuildOptions`.
    **kwargs:
        Legacy overrides such as ``seed=7`` or ``gauge_centre=3``.

    Returns
    -------
    A random tensor-train state.
    """

    if maxdim is not None:
        kwargs["maxdim"] = maxdim
    config = _coerce_build_options(options, **kwargs)
    if has_nontrivial_symmetry(sites):
        raise NotImplementedError("random_tt currently supports only dense/trivial site indices.")

    bonddim = _finite_maxdim(config.maxdim, label="random_tt")
    gauge_centre = len(sites) if config.gauge_centre is None else config.gauge_centre
    if not (1 <= gauge_centre <= len(sites)):
        raise ValueError(f"gauge_centre must lie in [1, {len(sites)}], got {gauge_centre}.")

    sitedims = [ix.dim for ix in sites]
    ldims = linkdims(sitedims, bonddim)
    arrays: list[torch.Tensor] = []

    with torch.random.fork_rng(devices=[]):
        if config.seed is not None:
            torch.manual_seed(config.seed)

        for k, site in enumerate(sites, start=1):
            left_dim = ldims[k - 1]
            right_dim = ldims[k]
            d = site.dim

            # Left-of-centre cores are column-orthonormal, the centre is dense,
            # and right-of-centre cores are row-orthonormal.
            if k < gauge_centre:
                mat = random_unitary(left_dim * d, right_dim, dtype=torch.complex128)
                arr = mat.reshape(left_dim, d, right_dim)
            elif k == gauge_centre:
                real = torch.randn((left_dim, d, right_dim), dtype=torch.float64)
                imag = torch.randn((left_dim, d, right_dim), dtype=torch.float64)
                arr = (real + 1j * imag).to(torch.complex128)
            else:
                mat = random_unitary(right_dim, left_dim * d, dtype=torch.complex128).conj().transpose(0, 1)
                arr = mat.reshape(left_dim, d, right_dim)
            arrays.append(arr)

    return tensor_train_from_arrays(sites, arrays)


def vector(psi: TensorTrain) -> torch.Tensor:
    """Fully contract an MPS into a flat dense vector.

    Parameters
    ----------
    psi:
        Tensor-train state.

    Returns
    -------
    Dense state vector in Fortran ordering.
    """

    contracted = psi[0]
    for core in psi.cores[1:]:
        contracted = tcontract(contracted, core)

    left = linkinds(psi)[0]
    right = linkinds(psi)[-1]
    site_tags = [ix.itag for ix in siteinds(psi)]
    dense = to_dense(contracted, [left.itag, *site_tags, right.itag])
    dense = dense.squeeze(0).squeeze(-1)
    return flatten_fortran(dense)


def orthogonalize(psi: TensorTrain, gc: int) -> TensorTrain:
    """Return a copy of ``psi`` with orthogonality center ``gc``.

    Parameters
    ----------
    psi:
        Input tensor train.
    gc:
        One-based gauge-center site.

    Returns
    -------
    A copied tensor train in mixed canonical form around ``gc``.
    """

    out = psi.copy()
    orthogonalize_(out, gc)
    return out


def orthogonalize_(
    psi: TensorTrain,
    gc: int,
    *,
    leftlim: int = 0,
    rightlim: int | None = None,
) -> TensorTrain:
    """Move the orthogonality center in-place using QR and LQ sweeps.

    Parameters
    ----------
    psi:
        Tensor train to modify.
    gc:
        One-based target orthogonality center.
    leftlim:
        One-based site just left of the active orthogonalization window.
    rightlim:
        One-based site just right of the active orthogonalization window.

    Returns
    -------
    The same ``psi`` object.
    """

    if rightlim is None:
        rightlim = len(psi) + 1

    # Sweep left-to-right until the centre is reached.
    for k in range(leftlim + 1, gc):
        core_k = psi[k - 1]
        qixs = [_leg(core_k, core_k.itags[0]), _leg(core_k, core_k.itags[1])]
        Q, R, _ = qr(core_k, qixs, tag=f"b{k}", positive=False)
        psi[k - 1] = Q
        psi[k] = tcontract(R, psi[k])

    # Sweep right-to-left to finish the mixed-canonical form.
    for k in range(rightlim - 1, gc, -1):
        core_k = psi[k - 1]
        qixs = [_leg(core_k, core_k.itags[1]), _leg(core_k, core_k.itags[2])]
        L, Q, _ = lq(core_k, qixs, tag=f"b{k-1}")
        psi[k - 1] = Q
        psi[k - 2] = tcontract(psi[k - 2], L)

    return psi


def dot(x: TensorTrain, y: TensorTrain) -> complex:
    """Return the Hilbert-space inner product ``<x|y>``.

    Parameters
    ----------
    x:
        Bra state.
    y:
        Ket state.

    Returns
    -------
    Complex scalar overlap.
    """

    if len(x) != len(y):
        raise ValueError("TensorTrains have different lengths.")
    y_work = replacelinks(y)
    contracted = tcontract(dag(x[0]), y_work[0])
    for k in range(1, len(x)):
        contracted = tcontract(contracted, tcontract(dag(x[k]), y_work[k]))
    return scalar(contracted)


def norm(psi: TensorTrain) -> float:
    """Return the Euclidean norm of an MPS.

    Parameters
    ----------
    psi:
        Tensor-train state.

    Returns
    -------
    Non-negative norm as a Python float.
    """

    val = dot(psi, replacelinks(psi))
    return float(max(val.real, 0.0) ** 0.5)


def normalize_(psi: TensorTrain) -> TensorTrain:
    """Normalize an MPS in-place when its norm is nonzero.

    Parameters
    ----------
    psi:
        Tensor-train state to modify.

    Returns
    -------
    The same ``psi`` object.
    """

    nrm = norm(psi)
    if nrm > 0:
        rescale_(psi, 1.0 / nrm)
    return psi
