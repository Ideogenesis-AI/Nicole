# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Nicole (TN) library.
#
# Nicole (TN) is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Nicole (TN) is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Nicole (TN). If not, see <https://www.gnu.org/licenses/>.


from __future__ import annotations

"""Pretty-print helpers for block-symmetric tensors.

This module encapsulates the logic required to present Nicole tensors in a compact,
human-friendly textual representation.

Key responsibilities
--------------------
- Compute a symmetry signature and aggregate statistics (order, block count, byte size, norm).
- Iterate over the tensor's block dictionary and produce aligned, padded tables of charges.

Implementation approach
-----------------------
1. Lightweight formatting helpers (_format_bytes, _format_single_value, _format_count_list)
   handle recurring presentation tasks so the main summariser stays readable.
2. `_charge_components` and `_group_signature` normalise charge data regardless of whether
   the tensor uses simple integers or tuple-based non-Abelian multiplet identifiers.
3. `tensor_summary` orchestrates the process: it first builds the heading lines, then
   computes global padding for charges so every printed sector column aligns. Finally it
   assembles per-block information, truncating after a configurable number of lines for brevity.

The output follow symmetry-aware tensor conventions (e.g. listing multiplet counts before
state counts, showing charge conservation per block) to ease adoption for users migrating
from traditional workflows.
"""

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .index import Index
from .symmetry.base import AbelianGroup
from .symmetry.abelian import U1Group
from .symmetry.product import ProductGroup
from .typing import Charge, Direction


def _charge_components(charge: Charge) -> Tuple:
    """Normalise a charge to a tuple of components."""
    if isinstance(charge, tuple):
        return charge
    if isinstance(charge, list):
        return tuple(charge)
    return (charge,)


def _group_signature(indices: Sequence[Index], components_per_charge: int) -> str:
    """Return a symmetry signature string identifying the group type.
    
    U1Group uses 'A', other Abelian groups use their name (e.g., 'Z2'), and
    ProductGroup uses the full product name (e.g., 'U1×Z2').
    """
    if not indices:
        return ""
    group = indices[0].group
    if isinstance(group, ProductGroup):
        # ProductGroup has a descriptive name like "U1×Z2"
        return group.name
    elif isinstance(group, U1Group):
        # U1Group specifically gets 'A' label
        label = "A"
        count = max(components_per_charge, 1)
        return ",".join([label] * count)
    elif isinstance(group, AbelianGroup):
        # Other Abelian groups use their name
        return group.name
    else:
        # Non-Abelian groups
        gname = getattr(group, "name", "")
        label = gname.upper() if gname else "?"
        count = max(components_per_charge, 1)
        return ",".join([label] * count)


def _format_bytes(size: int) -> str:
    """Format a number of bytes with an appropriate unit suffix."""
    units = ["B", "kB", "MB", "GB", "TB"]
    value = float(size)
    for idx, unit in enumerate(units):
        if value < 1024.0 or idx == len(units) - 1:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.3g} {unit}"
        value /= 1024.0
    return f"{value:.3g} TB"


def _format_single_value(arr: np.ndarray) -> str:
    """Return a formatted scalar read-out for a 1x1 block."""
    val = arr.reshape(-1)[0]
    if np.iscomplexobj(val):
        real_part = f"{val.real:.6g}"
        imag_part = f"{abs(val.imag):.6g}"
        sign = "+" if val.imag >= 0 else "-"
        return f"{real_part}{sign}{imag_part}i."
    return f"{val:.6g}."


def _format_count_list(counts: Sequence[int]) -> str:
    """Format elements of a count list joined by 'x' symbols."""
    if not counts:
        return "0"
    return " x ".join(str(count) for count in counts)


def tensor_summary(
    indices: Sequence[Index],
    itags: Sequence[str],
    data: Mapping[Tuple[Charge, ...], np.ndarray],
    dtype: np.dtype,
    label: str,
    norm: float,
    sorted_keys: Sequence[Tuple[Charge, ...]] = None,
    max_lines: Optional[int] = 9,
    block_numbers: Optional[Sequence[int]] = None,
) -> str:
    """Create a multi-line summary for a tensor.

    Parameters
    ----------
    indices:
        Ordered tensor indices (each carrying symmetry information and direction).
    itags:
        Ordered tuple of human-readable labels for each index.
    data:
        Mapping from block keys (one charge per leg) to dense NumPy arrays.
    dtype:
        Data type of the tensor entries.
    label:
        Human-readable tag printed in the header (e.g. "Tensor").
    norm:
        Frobenius norm of the tensor; pre-computed by the caller for efficiency.
    sorted_keys:
        Optional pre-sorted sequence of block keys. If None, keys are sorted
        internally by string representation.
    max_lines:
        Maximum number of blocks to display. If None, displays all blocks.
        Defaults to 9.
    block_numbers:
        Optional sequence of block numbers (1-indexed) to use for display.
        If provided, must match the length of sorted_keys. Used to preserve
        original block numbering when displaying a subset of blocks.

    Returns
    -------
    str
        A formatted multi-line string describing tensor order, block statistics, and
        up to nine individual blocks with aligned charges and sizes.
    """
    # Basic tensor statistics
    num_blocks = len(data)
    total_bytes = sum(int(arr.nbytes) for arr in data.values())
    order = len(indices)

    # Special handling for scalars (0D tensors)
    if order == 0:
        if () in data:
            value = data[()].item()
            # Format value nicely
            if np.iscomplexobj(value):
                real_part = f"{value.real:.6g}"
                imag_part = f"{abs(value.imag):.6g}"
                sign = "+" if value.imag >= 0 else "-"
                value_str = f"{real_part}{sign}{imag_part}i"
            else:
                value_str = f"{value:.6g}"
            dtype_name = np.dtype(dtype).name
            info_line = f"\n  info:  0x {{ 1 x 0 }}   {label}"
            data_line = f"  data:  0-D {dtype_name} ({_format_bytes(total_bytes)})    [ {value_str} ]"
            return info_line + "\n" + data_line
        else:
            # Empty scalar
            dtype_name = np.dtype(dtype).name
            info_line = f"\n  info:  0x {{ 1 x 0 }}   {label}"
            data_line = f"  data:  0-D {dtype_name} (0 B)    [ empty ]"
            return info_line + "\n" + data_line

    # Determine how many charge components to display (e.g. tuples vs scalars)
    sample_components = 0
    if indices:
        first_idx = indices[0]
        if first_idx.sectors:
            sample_components = len(_charge_components(first_idx.sectors[0].charge))
        else:
            sample_components = 1

    # -------------------------------------------------------------------
    # Heading: order, block count, symmetry signature, and index overview
    # -------------------------------------------------------------------
    sym_signature = _group_signature(indices, sample_components)
    itag_list = ", ".join(
        f"{tag}{'*' if idx.direction > 0 else ''}" for tag, idx in zip(itags, indices)
    )
    info_line = (
        f"\n  info:  {order}x {{ {num_blocks} x {sample_components or 1} }}  "
        f"having '{sym_signature}'  {label:>8},  {{ {itag_list} }}"
    )

    # -------------------------------------------------------------------
    # Data line: dtype, total bytes, multiplet counts, state counts, norm
    # -------------------------------------------------------------------
    dtype_name = np.dtype(dtype).name
    multiplet_counts_list = [idx.dim for idx in indices]
    multiplet_counts = _format_count_list(multiplet_counts_list)
    state_counts_list = []
    for idx in indices:
        if isinstance(idx.group, AbelianGroup):
            state_counts_list.append(idx.dim)
        else:
            # TODO: For non-Abelian groups, this should be sector.dim x degeneracy
            # when degeneracy is implemented. For now, use the same as Abelian.
            state_counts_list.append(idx.dim)
    state_counts = _format_count_list(state_counts_list)
    data_line = (
        f"  data:  {order}-D {dtype_name} ({_format_bytes(total_bytes)})    "
        f"{multiplet_counts} => {state_counts}  @ norm = {norm:.6g}\n"
    )

    # -------------------------------------------------------------------
    # Block listings: charge tables, dense shapes, optional single values
    # -------------------------------------------------------------------
    block_lines = []
    if data:
        # Determine padding for charges across all keys and positions.
        components_per_position: List[List[str]] = []
        for key in data:
            # `key` = tuple of charges, one per leg. Collect each component string.
            for pos, comps in enumerate(_charge_components(charge) for charge in key):
                while len(components_per_position) <= pos:
                    components_per_position.append([])
                components_per_position[pos].extend(str(comp) for comp in comps)
        position_widths = [
            max((len(value) for value in values), default=1) for values in components_per_position
        ]

        # Iterate deterministically over blocks; limit display if max_lines is set.
        if sorted_keys is None:
            sorted_keys = sorted(data.keys(), key=str)
        sorted_blocks = [(k, data[k]) for k in sorted_keys]
        blocks_to_show = sorted_blocks if max_lines is None else sorted_blocks[:max_lines]
        
        # Determine block numbers for display
        if block_numbers is not None:
            # Use provided block numbers (preserves original indices)
            display_numbers = list(block_numbers) if max_lines is None else list(block_numbers[:max_lines])
        else:
            # Default: sequential numbering starting from 1
            display_numbers = list(range(1, len(blocks_to_show) + 1))
        
        for idx_num, (key, arr) in zip(display_numbers, blocks_to_show):
            # Dense dims (state space) and trivial CGC placeholder (Abelian => all ones).
            state_dims = "x".join(str(dim) for dim in arr.shape) or "1"
            cgc_dims = "x".join("1" for _ in arr.shape) or "1"

            # Format charges, reusing global padding so columns line up across blocks.
            charge_components = [_charge_components(charge) for charge in key]
            padded_rows = []
            for position, comps in enumerate(charge_components):
                width = position_widths[position] if position < len(position_widths) else 1
                padded_rows.append(" ".join(f"{comp:>{width}}" for comp in comps))
            charges_repr = "[ " + " ; ".join(padded_rows) + " ]"

            # Display block information for each charge sector.
            block_bytes = arr.nbytes
            if arr.size == 1:
                # Scalar block — print the entry itself.
                value_repr = _format_single_value(arr)
                block_lines.append(
                    f"  {idx_num:>4}.  {state_dims:<7} |  {cgc_dims:<7} {charges_repr} {value_repr:>7}"
                )
            else:
                # High-dimensional array — display dims and byte footprint.
                byte_repr = _format_bytes(block_bytes)
                block_lines.append(
                    f"  {idx_num:>4}.  {state_dims:<7} |  {cgc_dims:<7} {charges_repr} {byte_repr:>7}"
                )

        # If more than max_lines blocks, note how many are omitted.
        if max_lines is not None and len(sorted_blocks) > max_lines:
            remaining = len(sorted_blocks) - max_lines
            block_lines.append(f"    ... ({remaining} more)")
    else:
        block_lines.append("  (no sectors)")

    return "\n".join([info_line, data_line, *block_lines])


def index_summary(index: Index) -> str:
    """Create a formatted summary for an Index.
    
    Parameters
    ----------
    index:
        Index to summarize
        
    Returns
    -------
    str
        A formatted multi-line string with:
        - First line: Index having '{group name}' with {direction}
        - Following lines: Table of charge and dimension
    """
    # Get group name
    group = index.group
    if isinstance(group, ProductGroup):
        group_name = group.name
    elif isinstance(group, U1Group):
        group_name = "U1"
    elif isinstance(group, AbelianGroup):
        group_name = group.name
    else:
        group_name = getattr(group, "name", "Unknown")
    
    # Get direction string
    dir_str = "-" if index.direction == Direction.OUT else "+"
    
    # First line
    header = f"\n  Index having '{group_name}' with {dir_str}\n"
    
    # If no sectors, return early
    if not index.sectors:
        return header + "\n  (no sectors)"
    
    # Build the table
    table_lines = []
    
    # Determine charge width based on all charges
    charge_strs = []
    is_product_group = False
    for sector in index.sectors:
        charge = sector.charge
        # Handle tuple charges (product groups)
        if isinstance(charge, tuple):
            charge_str = str(charge)
            is_product_group = True
        else:
            charge_str = str(charge)
        charge_strs.append(charge_str)
    
    # Calculate column widths
    charge_width = max(len(s) for s in charge_strs)
    charge_width = max(charge_width, len("Charge"))
    
    dim_strs = [str(sector.dim) for sector in index.sectors]
    dim_width = max(len(s) for s in dim_strs)
    dim_width = max(dim_width, len("Dims"))
    
    # Adjust indentation based on group type
    # Product groups: 6 spaces, single groups: 4 spaces
    indent = "      " if is_product_group else "    "
    
    # Header row
    table_lines.append(f"      {'Charge':>{charge_width}}  {'Dims':>{dim_width}}")
    
    # Sector rows
    for charge_str, dim_str in zip(charge_strs, dim_strs):
        table_lines.append(f"{indent}{charge_str:>{charge_width}}  {dim_str:>{dim_width}}")
    
    return header + "\n" + "\n".join(table_lines)

