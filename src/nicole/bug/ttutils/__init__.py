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


"""Tensor-train state, MPO, and local operator helpers."""

from .mpo import (
    TensorTrainOperator,
    contract,
    identity_op,
    matrix,
    tensor_train_operator_from_arrays,
    tto_direct_sum,
)
from .ops import OpSum, mpo_from_opsum, op
from .tensortrain import (
    TensorTrain,
    TensorTrainBuildOptions,
    dot,
    linkind,
    linkinds,
    maxlinkdim,
    normalize_,
    norm,
    orthogonalize,
    orthogonalize_,
    product_tt,
    random_tt,
    replacelinks,
    replacelinks_,
    rescale_,
    siteind,
    siteinds,
    tensor_train_from_arrays,
    tensor_train_from_vector,
    tensortrain_bond_indices,
    tensortrain_boundary_link,
    tensortrain_site_index,
    vector,
)

__all__ = [
    "OpSum",
    "TensorTrain",
    "TensorTrainBuildOptions",
    "TensorTrainOperator",
    "contract",
    "dot",
    "identity_op",
    "linkind",
    "linkinds",
    "matrix",
    "maxlinkdim",
    "mpo_from_opsum",
    "normalize_",
    "norm",
    "op",
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
    "tensor_train_operator_from_arrays",
    "tensortrain_bond_indices",
    "tensortrain_boundary_link",
    "tensortrain_site_index",
    "tto_direct_sum",
    "vector",
]
