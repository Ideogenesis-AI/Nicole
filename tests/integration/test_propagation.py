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


"""Integration tests for end-to-end GPU device propagation.

Strategy
--------
CPU is set as the *ambient default device* for each test via an `autouse`
fixture, while every test call explicitly passes `device=accel_device`
(MPS or CUDA).  The expected result is therefore always on the accelerator.

Any code path that silently falls through to `torch.get_default_device()`
instead of honouring the forwarded `device` argument will produce a tensor
on CPU rather than the accelerator, causing an immediate assertion failure.

This approach avoids MPS float64 incompatibility: intermediate CPU-side
computations (e.g. building data blocks in `load_space`) legitimately use
the default CPU device, and the final `.to(device)` call transfers them to
the accelerator.  This tests that the explicit `device=` argument is
correctly propagated, not that all intermediate tensors land on the
accelerator.

Skip conditions
---------------
- The entire module is skipped when no accelerator (MPS or CUDA) is available.
- `TestDecompDevice.test_qr_abelian` is skipped on MPS because
  `torch.linalg.qr` is not implemented for that backend.
"""

from typing import Dict, List
import pytest
import torch

from nicole import Direction, Index, Sector, Tensor, U1Group, SU2Group
from nicole import identity, isometry, oplus, diag, merge_axes, contract, decomp, trace, load_space
from nicole.decomp import eig
from tests.utils import populate_random_weights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def accel_device() -> torch.device:
    """Return an accelerator device (MPS preferred, then CUDA); skip if none."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("No accelerator device available (MPS or CUDA required)")


@pytest.fixture(autouse=True)
def with_cpu_default(accel_device: torch.device):
    """Set the global default device to CPU for the duration of each test.

    Every test explicitly passes `device=accel_device`; any call that
    silently uses `torch.get_default_device()` will therefore produce a
    tensor on CPU rather than the accelerator, failing the assertion.

    The fixture always restores the original default device on teardown so
    that this module cannot affect other test modules.
    """
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


# ---------------------------------------------------------------------------
# Core assertion helper
# ---------------------------------------------------------------------------

def assert_on_device(T: Tensor, device: torch.device, label: str = "") -> None:
    """Assert that every data block and every Bridge weight lives on *device*.

    Parameters
    ----------
    T:
        Tensor to inspect.
    device:
        Expected device.  Comparison is made on `device.type` so that
        ordinal suffixes (e.g. `cuda:0`) are matched correctly.
    label:
        Optional prefix for assertion failure messages.
    """
    prefix = f"[{label}] " if label else ""
    for key, block in T.data.items():
        assert block.device.type == device.type, (
            f"{prefix}data block {key}: on {block.device}, expected {device}"
        )
    if T.intw is not None:
        for key, bridge in T.intw.items():
            w = bridge.weights
            assert w.device.type == device.type, (
                f"{prefix}Bridge weights for block {key}: on {w.device}, expected {device}"
            )


# ---------------------------------------------------------------------------
# Index factories
# ---------------------------------------------------------------------------

def _u1_bond_index() -> Index:
    """Bond index (IN direction) for U1Group, as SVD typically returns."""
    group = U1Group()
    return Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))

def _su2_out_index() -> Index:
    """Two-sector SU(2) index: j=0 (dim 1) and j=1 (dim 2)."""
    group = SU2Group()
    return Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))

def _su2_bond_index() -> Index:
    group = SU2Group()
    return Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 2)))


# ===========================================================================
# identity() – device forwarded to torch.eye / torch.zeros / Bridge.from_block
# ===========================================================================

class TestIdentityDevice:
    """identity() must create all blocks, and all Bridge weights, on the explicit device."""

    def test_abelian(self, accel_device: torch.device) -> None:
        idx = Index(Direction.OUT, U1Group(), sectors=(Sector(0, 2), Sector(1, 3)))
        I = identity(idx, dtype=torch.float32, device=accel_device)
        assert_on_device(I, accel_device, "identity/U1")

    def test_su2(self, accel_device: torch.device) -> None:
        I = identity(_su2_out_index(), dtype=torch.float32, device=accel_device)
        assert_on_device(I, accel_device, "identity/SU2")
        assert I.intw is not None, "SU(2) identity must carry Bridge weights"


# ===========================================================================
# isometry() – device forwarded to torch.eye / torch.zeros / Bridge.from_block
# ===========================================================================

class TestIsometryDevice:
    """isometry() must place both data blocks and Bridge weights on the explicit device."""

    def test_abelian(self, accel_device: torch.device) -> None:
        group = U1Group()
        idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
        idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(-1, 1)))
        iso = isometry(idx_a, idx_b, dtype=torch.float32, device=accel_device)
        assert_on_device(iso, accel_device, "isometry/U1")

    def test_su2(self, accel_device: torch.device) -> None:
        group = SU2Group()
        idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1)))
        idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
        iso = isometry(idx_a, idx_b, dtype=torch.float32, device=accel_device)
        assert_on_device(iso, accel_device, "isometry/SU2")
        assert iso.intw is not None, "SU(2) isometry must carry Bridge weights"


# ===========================================================================
# oplus() – torch.zeros calls now forward device=A.device / B.device
# ===========================================================================

class TestOplusDevice:
    """oplus() must keep the merged result on the same device as the inputs."""

    def test_abelian(self, accel_device: torch.device) -> None:
        group = U1Group()
        idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
        A = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "b"], device=accel_device)
        B = Tensor.random([idx, idx.flip()], seed=2, itags=["a", "b"], device=accel_device)
        C = oplus(A, B)
        assert_on_device(C, accel_device, "oplus/U1")

    def test_su2(self, accel_device: torch.device) -> None:
        group = SU2Group()
        idx = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
        A = Tensor.random([idx, idx.flip()], seed=3, dtype=torch.float32,
                          itags=["a", "b"], device=accel_device)
        populate_random_weights(A, seed=8)
        B = Tensor.random([idx, idx.flip()], seed=4, dtype=torch.float32,
                          itags=["a", "b"], device=accel_device)
        populate_random_weights(B, seed=9)
        C = oplus(A, B)
        assert_on_device(C, accel_device, "oplus/SU2")
        assert C.intw is not None, "SU(2) oplus result must carry Bridge weights"


# ===========================================================================
# diag() – Bridge.from_block now receives device= for non-Abelian groups
# ===========================================================================

class TestDiagDevice:
    """diag() must place both data blocks and Bridge weights on the explicit device."""

    def test_abelian(self, accel_device: torch.device) -> None:
        bond = _u1_bond_index()
        S_blocks: Dict = {
            (0, 0):   torch.tensor([3.0, 2.0, 1.0], device=accel_device),
            (1, 1):   torch.tensor([0.5, 0.3],       device=accel_device),
            (-1, -1): torch.tensor([1.2, 0.8],       device=accel_device),
        }
        D = diag(S_blocks, bond, device=accel_device)
        assert_on_device(D, accel_device, "diag/U1")

    def test_su2(self, accel_device: torch.device) -> None:
        bond = _su2_bond_index()
        S_blocks: Dict = {
            (0, 0): torch.tensor([1.0],      dtype=torch.float32, device=accel_device),
            (2, 2): torch.tensor([0.8, 0.3], dtype=torch.float32, device=accel_device),
        }
        D = diag(S_blocks, bond, device=accel_device)
        assert_on_device(D, accel_device, "diag/SU2")
        assert D.intw is not None, "SU(2) diag must carry Bridge weights"


# ===========================================================================
# merge_axes() – passes device=tensor.device into isometry_n()
# ===========================================================================

class TestMergeAxesDevice:
    """merge_axes() must propagate device through the internal isometry_n call."""

    def test_abelian(self, accel_device: torch.device) -> None:
        group = U1Group()
        idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
        A = Tensor.random([idx, idx.flip(), idx], seed=3, itags=["a", "b", "c"],
                          device=accel_device)
        merged, iso_conj = merge_axes(A, [0, 2])
        assert_on_device(merged,   accel_device, "merge_axes/merged")
        assert_on_device(iso_conj, accel_device, "merge_axes/iso_conj")

    def test_su2(self, accel_device: torch.device) -> None:
        group = SU2Group()
        idx = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
        A = Tensor.random([idx, idx.flip(), idx], seed=5, dtype=torch.float32,
                          itags=["a", "b", "c"], device=accel_device)
        populate_random_weights(A, seed=10)
        merged, iso_conj = merge_axes(A, [0, 2])
        assert_on_device(merged,   accel_device, "merge_axes/su2/merged")
        assert_on_device(iso_conj, accel_device, "merge_axes/su2/iso_conj")


# ===========================================================================
# decomp() – Bridge.from_block receives device=T.device in svd / qr / eig
# ===========================================================================

class TestDecompDevice:
    """decomp() must keep all output tensors and their Bridge weights on the input device."""

    @staticmethod
    def _u1_matrix(device: torch.device) -> Tensor:
        group = U1Group()
        idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
        idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
        return Tensor.random([idx1, idx2], seed=7, itags=["a", "b"], device=device)

    @staticmethod
    def _su2_matrix(device: torch.device) -> Tensor:
        group = SU2Group()
        idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(2, 1)))
        idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(2, 1)))
        T = Tensor.random([idx1, idx2], seed=9, dtype=torch.float32,
                          itags=["a", "b"], device=device)
        populate_random_weights(T, seed=4)
        return T

    def test_svd_abelian(self, accel_device: torch.device) -> None:
        T = self._u1_matrix(accel_device)
        U, S, Vh = decomp(T, axes=0, mode="SVD")
        assert_on_device(U,  accel_device, "svd/U")
        assert_on_device(S,  accel_device, "svd/S")
        assert_on_device(Vh, accel_device, "svd/Vh")

    def test_svd_su2(self, accel_device: torch.device) -> None:
        T = self._su2_matrix(accel_device)
        U, S, Vh = decomp(T, axes=0, mode="SVD")
        assert_on_device(U,  accel_device, "svd_su2/U")
        assert_on_device(S,  accel_device, "svd_su2/S")
        assert_on_device(Vh, accel_device, "svd_su2/Vh")
        assert U.intw  is not None, "SU(2) U from SVD must carry Bridge weights"
        assert Vh.intw is not None, "SU(2) Vh from SVD must carry Bridge weights"

    def test_qr_abelian(self, accel_device: torch.device) -> None:
        if accel_device.type == "mps":
            pytest.skip("torch.linalg.qr is not implemented for MPS")
        T = self._u1_matrix(accel_device)
        Q, R = decomp(T, axes=0, mode="QR")
        assert_on_device(Q, accel_device, "qr/Q")
        assert_on_device(R, accel_device, "qr/R")

    def test_qr_su2(self, accel_device: torch.device) -> None:
        if accel_device.type == "mps":
            pytest.skip("torch.linalg.qr is not implemented for MPS")
        T = self._su2_matrix(accel_device)
        Q, R = decomp(T, axes=0, mode="QR")
        assert_on_device(Q, accel_device, "qr_su2/Q")
        assert_on_device(R, accel_device, "qr_su2/R")
        assert Q.intw is not None, "SU(2) Q from QR must carry Bridge weights"

    def test_ur_abelian(self, accel_device: torch.device) -> None:
        T = self._u1_matrix(accel_device)
        U, R = decomp(T, axes=0, mode="UR")
        assert_on_device(U, accel_device, "ur/U")
        assert_on_device(R, accel_device, "ur/R")

    def test_eig_abelian(self, accel_device: torch.device) -> None:
        if accel_device.type == "mps":
            pytest.skip("torch.linalg.eig is not implemented for MPS")
        T = self._u1_matrix(accel_device)
        V, D_blocks = eig(T)
        assert_on_device(V, accel_device, "eig/V")
        for key, eigenvalues in D_blocks.items():
            assert eigenvalues.device.type == accel_device.type, (
                f"[eig/D] eigenvalue block {key}: on {eigenvalues.device}, "
                f"expected {accel_device}"
            )

    def test_eig_su2(self, accel_device: torch.device) -> None:
        if accel_device.type == "mps":
            pytest.skip("torch.linalg.eig is not implemented for MPS")
        T = self._su2_matrix(accel_device)
        V, D_blocks = eig(T)
        assert_on_device(V, accel_device, "eig_su2/V")
        assert V.intw is not None, "SU(2) V from eig must carry Bridge weights"
        for key, eigenvalues in D_blocks.items():
            assert eigenvalues.device.type == accel_device.type, (
                f"[eig_su2/D] eigenvalue block {key}: on {eigenvalues.device}, "
                f"expected {accel_device}"
            )


# ===========================================================================
# contract / trace – torch.full now passes device=T.device
# ===========================================================================

class TestContractDevice:
    """contract() and trace() must keep results on the same device as the inputs."""

    # -- contract -----------------------------------------------------------

    def test_contract_abelian(self, accel_device: torch.device) -> None:
        group = U1Group()
        idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
        A = Tensor.random([idx, idx.flip()], seed=10, itags=["a", "b"], device=accel_device)
        B = Tensor.random([idx, idx.flip()], seed=11, itags=["b", "c"], device=accel_device)
        C = contract(A, B)
        assert_on_device(C, accel_device, "contract/U1")

    def test_contract_su2(self, accel_device: torch.device) -> None:
        group = SU2Group()
        idx = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
        A = Tensor.random([idx, idx.flip()], seed=12, dtype=torch.float32,
                          itags=["a", "b"], device=accel_device)
        populate_random_weights(A, seed=5)
        B = Tensor.random([idx, idx.flip()], seed=13, dtype=torch.float32,
                          itags=["b", "c"], device=accel_device)
        populate_random_weights(B, seed=6)
        C = contract(A, B)
        assert_on_device(C, accel_device, "contract/SU2")
        assert C.intw is not None, "SU(2) contract result must carry Bridge weights"

    # -- trace --------------------------------------------------------------

    def test_trace_abelian(self, accel_device: torch.device) -> None:
        group = U1Group()
        idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
        T = Tensor.random([idx, idx.flip()], seed=5, itags=["a", "a"], device=accel_device)
        result = trace(T)
        assert_on_device(result, accel_device, "trace/U1")

    def test_trace_su2(self, accel_device: torch.device) -> None:
        group = SU2Group()
        idx = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
        T = Tensor.random([idx, idx.flip()], seed=14, dtype=torch.float32,
                          itags=["a", "a"], device=accel_device)
        populate_random_weights(T, seed=7)
        result = trace(T)
        assert_on_device(result, accel_device, "trace/SU2")


# ===========================================================================
# load_space() – all 10 presets
# ===========================================================================

_LOAD_SPACE_PRESETS = [
    pytest.param("Spin",  "U1",     {"J": 0.5}, ["Sz", "Sp", "Sm"],                      id="Spin/U1/J=0.5"),
    pytest.param("Spin",  "U1",     {"J": 1.0}, ["Sz", "Sp", "Sm"],                      id="Spin/U1/J=1.0"),
    pytest.param("Spin",  "SU2",    {"J": 0.5}, ["S"],                                   id="Spin/SU2/J=0.5"),
    pytest.param("Spin",  "SU2",    {"J": 1.0}, ["S"],                                   id="Spin/SU2/J=1.0"),
    pytest.param("Ferm",  "U1",     {},         ["F", "Z"],                              id="Ferm/U1"),
    pytest.param("Ferm",  "Z2",     {},         ["F", "Z"],                              id="Ferm/Z2"),
    pytest.param("Band",  "U1,U1",  {},         ["F_up", "F_dn", "Z", "Sz", "Sp", "Sm"], id="Band/U1,U1"),
    pytest.param("Band",  "Z2,U1",  {},         ["F_up", "F_dn", "Z", "Sz", "Sp", "Sm"], id="Band/Z2,U1"),
    pytest.param("Band",  "U1,SU2", {},         ["F", "Z", "S"],                         id="Band/U1,SU2"),
    pytest.param("Band",  "Z2,SU2", {},         ["F", "Z", "S"],                         id="Band/Z2,SU2"),
]


class TestLoadSpaceDevice:
    """load_space() must honour an explicit `device` in the option dict.

    Intermediate CPU-side data construction in each loader is intentional;
    the device check targets the final `.to(device)` call that transfers
    every operator tensor to the accelerator.  On MPS, float64 tensors are
    automatically normalised to float32 during the transfer.
    """

    @pytest.mark.parametrize("preset,preserv,option,op_keys", _LOAD_SPACE_PRESETS)
    def test_preset(
        self,
        accel_device: torch.device,
        preset: str,
        preserv: str,
        option: Dict,
        op_keys: List[str],
    ) -> None:
        opt = dict(option)
        opt["device"] = accel_device
        _, Op = load_space(preset, preserv, opt)
        for key in op_keys:
            T = Op[key]
            assert isinstance(T, Tensor), (
                f"Op['{key}'] for {preset}/{preserv} should be a Tensor, got {type(T)}"
            )
            assert_on_device(T, accel_device, f"{preset}/{preserv}/{key}")

