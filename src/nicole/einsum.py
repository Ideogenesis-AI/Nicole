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


"""Einstein summation notation for symmetry-aware tensors.

This module provides `einsum`, which parses a subscript equation string and
dispatches to `contract`, `trace`, and `permute` to carry out the requested
operation.

Three equation types are supported:

1. **Permutation** — a single input tensor whose output subscript is a
   reordering of the input subscript, e.g. `'ij->ji'`.
2. **Trace** — a single input tensor with one repeated subscript letter,
   e.g. `'ii->'`. A permutation of the surviving axes follows if needed.
3. **Sequential contraction** — two or more input tensors contracted from left
   to right, e.g. `'ij,jk->ik'`. Tensors are contracted pairwise in the
   order they appear; no contraction-order optimisation is performed.

Functions
---------
einsum(equation, *tensors)
    Evaluate an Einstein summation equation on symmetry-aware tensors.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from .contract import contract, trace
from .maneuver import permute
from .tensor import Tensor


def _parse_equation(
    equation: str,
    tensors: Tuple[Tensor, ...],
) -> Tuple[List[str], str]:
    """Parse and validate a subscript equation string.

    Parameters
    ----------
    equation:
        Equation of the form `'<lhs>-><rhs>'` where `<lhs>` is a
        comma-separated list of subscript strings (one per input tensor) and
        `<rhs>` is the output subscript.
    tensors:
        Tuple of input tensors.

    Returns
    -------
    Tuple[List[str], str]
        A pair `(input_subs, output_sub)` where `input_subs[i]` is the
        subscript string for `tensors[i]` and `output_sub` is the output
        subscript.

    Raises
    ------
    ValueError
        If the equation does not contain `'->'`, if the number of subscript
        strings does not match the number of tensors, or if any subscript
        length does not match the order of the corresponding tensor.
    """
    if '->' not in equation:
        raise ValueError(
            "einsum equation must contain '->'. "
            "Implicit output subscripts are not supported."
        )

    lhs, output_sub = equation.split('->', 1)
    input_subs = lhs.split(',')

    if len(input_subs) != len(tensors):
        raise ValueError(
            f"einsum equation has {len(input_subs)} subscript(s) but "
            f"{len(tensors)} tensor(s) were provided."
        )

    for i, (sub, tensor) in enumerate(zip(input_subs, tensors)):
        order = len(tensor.indices)
        if len(sub) != order:
            raise ValueError(
                f"Subscript '{sub}' for tensor {i} has length {len(sub)} "
                f"but the tensor has order {order}."
            )

    return input_subs, output_sub


def _apply_within_trace(
    tensor: Tensor,
    sub: List[str],
) -> Tuple[Tensor, List[str]]:
    """Trace any repeated subscript letters within a single tensor.

    Scans `sub` for letters that appear exactly twice and collects them as
    trace pairs by axis position. Letters appearing three or more times are
    rejected immediately. A single call to `trace` with all collected pairs
    is made; errors from that call (e.g. axes with the same direction)
    propagate unchanged.

    Parameters
    ----------
    tensor:
        Input tensor.
    sub:
        List of subscript characters, one per axis, matching
        `tensor.indices`.

    Returns
    -------
    Tuple[Tensor, List[str]]
        The traced tensor and the updated subscript list with the traced
        axes removed.

    Raises
    ------
    ValueError
        If any letter appears three or more times in `sub`.
    """
    counts = Counter(sub)

    pairs: List[Tuple[int, int]] = []
    traced_positions: set = set()

    for letter, count in counts.items():
        if count > 2:
            raise ValueError(
                f"Subscript letter '{letter}' appears {count} times. "
                "At most two occurrences (one trace pair) are supported "
                "per input tensor."
            )
        if count == 2:
            positions = [i for i, c in enumerate(sub) if c == letter]
            pairs.append((positions[0], positions[1]))
            traced_positions.add(positions[0])
            traced_positions.add(positions[1])

    if not pairs:
        return tensor, sub

    # trace accepts a list of (a, b) pairs referring to original axis positions
    traced = trace(tensor, axes=pairs)
    remaining_sub = [sub[i] for i in range(len(sub)) if i not in traced_positions]
    return traced, remaining_sub


def _apply_output_permutation(
    tensor: Tensor,
    current_sub: List[str],
    output_sub: str,
) -> Tensor:
    """Permute axes of `tensor` so that they match `output_sub`.

    Parameters
    ----------
    tensor:
        Tensor whose axes are currently ordered according to `current_sub`.
    current_sub:
        List of subscript characters describing the current axis order.
    output_sub:
        Target subscript string.

    Returns
    -------
    Tensor
        Tensor with axes reordered to match `output_sub`, or `tensor`
        itself if the order already matches or the output is a scalar.
    """
    if not output_sub:
        return tensor

    order = [current_sub.index(c) for c in output_sub]
    if order == list(range(len(current_sub))):
        return tensor
    return permute(tensor, order)


def einsum(equation: str, *tensors: Tensor) -> Tensor:
    """Evaluate an Einstein summation equation on symmetry-aware tensors.

    Parses `equation` and dispatches to `contract`, `trace`, and `permute`.
    Three equation types are supported:

    1. **Permutation** — single tensor, output is a reordering of the input
       subscript:

            einsum('ij->ji', A)

    2. **Trace** — single tensor with a repeated subscript letter. The output
       subscript lists the surviving axes in the desired order:

            einsum('ii->', A)       # full trace → scalar
            einsum('iijk->jk', A)   # partial trace

    3. **Sequential contraction** — two or more tensors contracted from left to
       right. At each step, letters present in both the current result and the
       next tensor are contracted, unless they appear in the output subscript:

            einsum('ij,jk->ik', A, B)       # matrix multiply
            einsum('ij,jk,kl->il', A, B, C) # chain contraction

    Parameters
    ----------
    equation:
        Subscript equation of the form `'<lhs>-><rhs>'`. `<lhs>` is a
        comma-separated list of subscript strings, one per input tensor.
        `<rhs>` is the output subscript. The `->` separator is **required**;
        implicit output subscripts are not supported.

        Each subscript character must be a single ASCII letter. A letter may
        appear at most twice within a single input subscript (forming one
        evaluation pair).

    *tensors:
        Input `Tensor` objects. The number of tensors must match the number
        of comma-separated subscript strings in `equation`, and the order of
        each tensor must equal the length of its subscript.

    Returns
    -------
    Tensor
        Result tensor. For a full trace this is a scalar (order 0). For a
        partial trace or contraction the result has the axes listed in the
        output subscript in that order.

    Raises
    ------
    ValueError
        Raised in any of the following situations:

        - `equation` does not contain `'->'`.
        - The number of comma-separated input subscripts does not match the
          number of tensors supplied.
        - A subscript length does not match the order of the corresponding
          tensor.
        - A letter appears three or more times within a single input subscript.
        - An error is propagated from `trace`, e.g. a trace pair whose two
          axes share the same direction.
        - An error is propagated from `contract`, e.g. contracted axes have
          mismatched itags or the same direction.

    Notes
    -----
    **No contraction-order optimisation.** For multi-tensor equations the
    tensors are contracted strictly from left to right. This may be
    suboptimal for networks where a different pairing would reduce intermediate
    tensor sizes.

    **Vectors (1st order tensors) are not supported** by this library. Any
    equation whose intermediate or final result would have exactly one index
    will raise an error from the underlying `contract` or `trace` call.

    **Hadamard (batch) indices** — a letter that appears in two or more input
    subscripts and also in the output subscript — are not supported. Such a
    letter would need to survive as a free axis in both tensors without being
    summed over, which requires element-wise multiplication semantics that
    `contract` does not provide.

    Examples
    --------
    **Permutation:**

    >>> einsum('ij->ji', A)

    **Trace to scalar:**

    >>> einsum('ii->', A)

    **Matrix multiplication:**

    >>> einsum('ij,jk->ik', A, B)

    **Chain contraction:**

    >>> einsum('ij,jk,kl->il', A, B, C)
    """
    if not tensors:
        raise ValueError("einsum requires at least one input tensor.")

    input_subs_raw, output_sub = _parse_equation(equation, tensors)

    # Pre-process: trace any repeated letters within each individual tensor.
    processed: List[Tuple[Tensor, List[str]]] = []
    for tensor, sub in zip(tensors, input_subs_raw):
        t, s = _apply_within_trace(tensor, list(sub))
        processed.append((t, s))

    # --- Single-tensor path ---
    if len(processed) == 1:
        current, current_sub = processed[0]
        return _apply_output_permutation(current, current_sub, output_sub)

    # --- Multi-tensor path: contract left to right ---
    current, current_sub = processed[0]
    output_set = set(output_sub)

    for next_tensor, next_sub in processed[1:]:
        next_sub_set = set(next_sub)

        # Letters shared between current and next that are absent from the
        # output subscript are contracted at this step.
        seen: set = set()
        contract_letters: List[str] = []
        for letter in current_sub:
            if letter in next_sub_set and letter not in output_set and letter not in seen:
                contract_letters.append(letter)
                seen.add(letter)

        axes_A = [current_sub.index(l) for l in contract_letters]
        axes_B = [next_sub.index(l) for l in contract_letters]

        # axes=([], []) produces an outer product when no letters are shared;
        # errors from mismatched itags or directions propagate unchanged.
        current = contract(current, next_tensor, axes=(axes_A, axes_B))

        contracted_set = set(contract_letters)
        current_sub = (
            [l for l in current_sub if l not in contracted_set]
            + [l for l in next_sub if l not in contracted_set]
        )

    return _apply_output_permutation(current, current_sub, output_sub)
