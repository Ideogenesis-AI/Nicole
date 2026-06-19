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


"""Shared BUG/KLS bookkeeping and canonical-bond helpers.

The sweep code in :mod:`nicole.bug.sweep.bug_sweep` and :mod:`nicole.bug.sweep.kls`
relies on a small set of stateful utilities collected here:

- :class:`BUGInfo` records sweep diagnostics;
- schedule helpers choose the multi-sweep composition order; and
- canonical bond snapshots package the two-site QR/LQ decomposition used by
  local BUG/KLS updates.

The algorithms are unchanged, but the module now uses small option dataclasses
to keep long helper signatures under control while preserving legacy call
patterns used throughout the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from ..indices import Ix

__all__ = [
    "BUGInfo",
    "record_backward_correction",
    "record_s_step_rank",
]


@dataclass(frozen=True)
class SStepRecordOptions:
    """Inputs for one recorded S-step truncation event.

    Parameters
    ----------
    bond:
        One-based bond number being truncated.
    keep:
        Rank retained after truncation.
    svals:
        Optional singular values used to compute discarded-weight metrics.
    """

    bond: int
    keep: int
    svals: Any = None


@dataclass
class BUGInfo:
    """Diagnostic container populated during BUG/KLS sweeps.

    The fields mirror the Julia implementation closely so existing regression
    tests and future instrumentation can continue to rely on the same names.
    Lists are append-only and are filled as the sweep progresses.
    """

    bond_dims_before: list[int] = field(default_factory=list)
    bond_dims_after: list[int] = field(default_factory=list)
    elapsed: float = 0.0
    rhs_eval_elapsed: float = 0.0
    trial_state_elapsed: float = 0.0
    forward_sweep_elapsed: float = 0.0
    reverse_sweep_elapsed: float = 0.0
    aug_sizes_k: list[int] = field(default_factory=list)
    aug_sizes_l: list[int] = field(default_factory=list)
    aug_dims_k: list[int] = field(default_factory=list)
    aug_dims_l: list[int] = field(default_factory=list)
    lanczos_numops: list[int] = field(default_factory=list)
    backward_correction_calls: int = 0
    s_step_sweeps: list[str] = field(default_factory=list)
    s_step_bonds: list[int] = field(default_factory=list)
    s_step_kept_ranks: list[int] = field(default_factory=list)
    s_step_full_ranks: list[int] = field(default_factory=list)
    s_step_tail_norms: list[float] = field(default_factory=list)
    s_step_max_discarded_svals: list[float] = field(default_factory=list)
    boundary_trace_sweeps: list[str] = field(default_factory=list)
    boundary_trace_bonds: list[int] = field(default_factory=list)
    boundary_trace_old_ranks: list[int] = field(default_factory=list)
    boundary_trace_left_caps: list[int] = field(default_factory=list)
    boundary_trace_right_caps: list[int] = field(default_factory=list)
    boundary_trace_left_frame_ranks: list[int] = field(default_factory=list)
    boundary_trace_right_frame_ranks: list[int] = field(default_factory=list)
    boundary_trace_s_full_ranks: list[int] = field(default_factory=list)
    boundary_trace_n_new_k: list[int] = field(default_factory=list)
    boundary_trace_n_new_l: list[int] = field(default_factory=list)


def _coerce_s_step_record_options(
    args: tuple[Any, ...],
    options: SStepRecordOptions | None = None,
    **kwargs: Any,
) -> SStepRecordOptions:
    """Normalize legacy and modern S-step record arguments.

    Parameters
    ----------
    args:
        Optional legacy positional tail ``(bond, keep[, svals])``.
    options:
        Optional pre-built :class:`SStepRecordOptions`.
    **kwargs:
        Keyword overrides for the options fields.

    Returns
    -------
    A normalized :class:`SStepRecordOptions` instance.
    """

    if args:
        if len(args) not in {2, 3}:
            raise TypeError("record_s_step_rank expects (bond, keep[, svals]) after (info, sweep).")
        if options is not None:
            raise TypeError("Use either an options object or legacy positional arguments, not both.")
        options = SStepRecordOptions(bond=args[0], keep=args[1], svals=args[2] if len(args) == 3 else None)

    overrides = {name: kwargs.pop(name) for name in list(kwargs) if name in SStepRecordOptions.__dataclass_fields__}
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown S-step record option(s): {unknown}")

    if options is None:
        missing = [name for name in ("bond", "keep") if name not in overrides]
        if missing:
            missing_str = ", ".join(missing)
            raise TypeError(f"Missing record_s_step_rank inputs: {missing_str}")
        return SStepRecordOptions(**overrides)
    return replace(options, **overrides)


def record_s_step_rank(info: BUGInfo | None, sweep: str, *args, options: SStepRecordOptions | None = None, **kwargs: Any):
    """Append one S-step truncation record to :class:`BUGInfo`.

    Parameters
    ----------
    info:
        Optional diagnostics object. ``None`` keeps the helper as a no-op.
    sweep:
        Sweep label such as ``"forward"`` or ``"reverse"``.
    *args:
        Optional legacy positional tail ``(bond, keep[, svals])``.
    options:
        Optional :class:`SStepRecordOptions` object.
    **kwargs:
        Keyword overrides for ``bond``, ``keep``, and ``svals``.

    Returns
    -------
    The retained rank ``keep``.
    """

    record = _coerce_s_step_record_options(args, options=options, **kwargs)
    if info is None:
        return record.keep

    info.s_step_sweeps.append(sweep)
    info.s_step_bonds.append(record.bond)
    info.s_step_kept_ranks.append(record.keep)

    if record.svals is None:
        info.s_step_full_ranks.append(record.keep)
        info.s_step_tail_norms.append(0.0)
        info.s_step_max_discarded_svals.append(0.0)
        return record.keep

    vals = torch.as_tensor(record.svals)
    full_rank = int(vals.numel())
    info.s_step_full_ranks.append(full_rank)
    if record.keep < full_rank:
        discarded = torch.abs(vals[record.keep :])
        info.s_step_tail_norms.append(float(torch.linalg.norm(discarded)))
        info.s_step_max_discarded_svals.append(float(torch.max(discarded)))
    else:
        info.s_step_tail_norms.append(0.0)
        info.s_step_max_discarded_svals.append(0.0)
    return record.keep


def record_backward_correction(info: BUGInfo | None):
    """Record that one backward-correction step was taken.

    Parameters
    ----------
    info:
        Optional diagnostics object.

    Returns
    -------
    Updated correction count, or ``0`` when ``info`` is ``None``.
    """

    if info is None:
        return 0
    info.backward_correction_calls += 1
    return info.backward_correction_calls


def _site_bond_index(V_tens, s_next: Ix, link_r: Ix) -> Ix:
    """Return the extra bond index inside a right canonical factor.

    Parameters
    ----------
    V_tens:
        Rank-3 right factor.
    s_next:
        Physical site index that should be ignored.
    link_r:
        Right boundary bond index that should be ignored.

    Returns
    -------
    The remaining bond :class:`Ix`.
    """

    for tag, idx in zip(V_tens.itags, V_tens.indices):
        if tag != s_next.itag and tag != link_r.itag:
            return Ix(tag, int(idx.dim), idx.direction, idx.sectors, idx.group)
    raise ValueError("Could not find site bond index in V tensor")


def _left_site_bond_index(U_tens, link_l: Ix, site_l: Ix) -> Ix:
    """Return the extra bond index inside a left canonical factor.

    Parameters
    ----------
    U_tens:
        Rank-3 left factor.
    link_l:
        Left boundary bond index that should be ignored.
    site_l:
        Physical site index that should be ignored.

    Returns
    -------
    The remaining bond :class:`Ix`.
    """

    for tag, idx in zip(U_tens.itags, U_tens.indices):
        if tag != link_l.itag and tag != site_l.itag:
            return Ix(tag, int(idx.dim), idx.direction, idx.sectors, idx.group)
    raise ValueError("Could not find left-site bond index in U tensor")
