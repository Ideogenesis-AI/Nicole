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


"""Tests for autograd control."""

import torch
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group, SU2Group
from nicole import contract, trace, permute, decomp
from ..utils import populate_random_weights


def test_autograd_disabled_by_default():
    """Test that autograd is disabled globally by default."""
    # Check global state
    assert not torch.is_grad_enabled()
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    tensor = Tensor.random([idx, idx.flip()], seed=42)
    
    # Tensors should not track gradients by default
    for block in tensor.data.values():
        assert not block.requires_grad


def test_torch_enable_grad_context_manager():
    """Test torch.enable_grad() context manager works with Nicole tensors."""
    # Before context: autograd disabled
    assert not torch.is_grad_enabled()
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    with torch.enable_grad():
        # Inside context: autograd enabled
        assert torch.is_grad_enabled()
        
        # Create tensor with requires_grad=True to track gradients
        tensor = Tensor.random([idx, idx.flip()], seed=42, requires_grad=True)
        
        # Tensors with requires_grad=True track gradients
        for block in tensor.data.values():
            assert block.requires_grad
    
    # After context: autograd disabled again
    assert not torch.is_grad_enabled()


def test_requires_grad_property():
    """Test Tensor.requires_grad property for querying gradient status."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    # Create tensor without gradients
    t1 = Tensor.random([idx, idx.flip()], seed=42)
    assert not t1.requires_grad
    
    # Create tensor with gradients
    t2 = Tensor.random([idx, idx.flip()], seed=43, requires_grad=True)
    assert t2.requires_grad
    
    # Use property setter to enable gradients
    t1.requires_grad = True
    assert t1.requires_grad
    for block in t1.data.values():
        assert block.requires_grad
    
    # Use property setter to disable gradients
    t2.requires_grad = False
    assert not t2.requires_grad
    for block in t2.data.values():
        assert not block.requires_grad


def test_requires_grad_property_setter():
    """Test Tensor.requires_grad property setter for fine-grained control."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    with torch.enable_grad():
        # Create tensors with requires_grad=True
        t1 = Tensor.random([idx, idx.flip()], seed=42, requires_grad=True)
        t2 = Tensor.random([idx, idx.flip()], seed=43, requires_grad=True)
        
        # Both should track gradients
        assert t1.requires_grad
        assert t2.requires_grad
        
        # Disable gradients for t2 using property setter
        t2.requires_grad = False
        assert not t2.requires_grad
        
        # t1 still tracks gradients
        assert t1.requires_grad


def test_gradient_computation():
    """Test that gradients can be computed when enabled."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    with torch.enable_grad():
        # Create tensor with requires_grad=True
        tensor = Tensor.random([idx, idx.flip()], seed=42, requires_grad=True)
        
        # Verify requires_grad is set
        for block in tensor.data.values():
            assert block.requires_grad
        
        # Compute a simple loss (sum of squares)
        loss = sum((block ** 2).sum() for block in tensor.data.values())
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients exist
        for block in tensor.data.values():
            assert block.grad is not None
            assert block.grad.shape == block.shape


def test_no_gradients_outside_context():
    """Test that operations outside context don't track gradients."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42)
    
    # No gradients should be tracked
    for block in tensor.data.values():
        assert not block.requires_grad
    
    # Operations shouldn't create gradient tape
    loss = sum((block ** 2).sum() for block in tensor.data.values())
    
    # This should not raise an error, just no gradients
    # (backward() would raise error if requires_grad=False)


def test_device_and_autograd_interaction():
    """Test that device management and autograd work together."""
    torch.set_default_device('cpu')
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    with torch.enable_grad():
        # Create tensor with requires_grad=True
        tensor_cpu = Tensor.random([idx, idx.flip()], seed=42, requires_grad=True)
        
        # Should have gradients on CPU
        for block in tensor_cpu.data.values():
            assert block.requires_grad
            assert block.device == torch.device('cpu')
        
        # Move to CPU (no-op) should preserve requires_grad
        tensor_cpu2 = tensor_cpu.cpu()
        for block in tensor_cpu2.data.values():
            assert block.requires_grad
            assert block.device == torch.device('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_to_cuda_preserves_gradients():
    """Test that moving to CUDA preserves gradient tracking."""
    torch.set_default_device('cpu')
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    with torch.enable_grad():
        # Create tensor with requires_grad=True
        tensor_cpu = Tensor.random([idx, idx.flip()], seed=42, requires_grad=True)
        
        # Verify gradients on CPU
        for block in tensor_cpu.data.values():
            assert block.requires_grad
        
        # Move to CUDA
        tensor_gpu = tensor_cpu.cuda()
        
        # Should still have gradients on GPU
        for block in tensor_gpu.data.values():
            assert block.requires_grad
            assert block.device.type == 'cuda'


def test_nested_enable_grad():
    """Test that torch.enable_grad() can be nested."""
    assert not torch.is_grad_enabled()
    
    with torch.enable_grad():
        assert torch.is_grad_enabled()
        
        with torch.enable_grad():
            assert torch.is_grad_enabled()
        
        assert torch.is_grad_enabled()
    
    assert not torch.is_grad_enabled()


def test_enable_grad_restores_state_on_exception():
    """Test that torch.enable_grad() restores grad state even on exception."""
    assert not torch.is_grad_enabled()
    
    try:
        with torch.enable_grad():
            assert torch.is_grad_enabled()
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Grad mode should be restored
    assert not torch.is_grad_enabled()


def test_backward_on_scalar():
    """Test that backward() works on scalar tensors."""
    with torch.enable_grad():
        # Create a scalar tensor with gradient tracking
        scalar = Tensor.from_scalar(5.0, requires_grad=True)
        
        # Verify it's a scalar
        assert scalar.is_scalar()
        assert scalar.data[()].numel() == 1
        
        # Compute some operation
        result = scalar * 2.0
        
        # Call backward
        result.backward()
        
        # Check that gradients were computed
        assert scalar.data[()].grad is not None
        # Gradient of 2*x with respect to x is 2
        assert torch.isclose(scalar.data[()].grad, torch.tensor(2.0, dtype=torch.float64))


def test_backward_raises_on_non_scalar():
    """Test that backward() raises error on non-scalar tensors."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    with torch.enable_grad():
        # Create a non-scalar tensor
        tensor = Tensor.random([idx, idx.flip()], seed=42, requires_grad=True)
        
        # Verify it's not a scalar
        assert not tensor.is_scalar()
        
        # backward() should raise ValueError
        with pytest.raises(ValueError, match="backward.*can only be called on scalars"):
            tensor.backward()


def test_backward_validates_numel():
    """Test that backward() raises when the scalar block has numel != 1.

    The constructor enforces shape () for data[()], but data[()] can be
    replaced directly after construction.  The numel check inside backward()
    guards against such cases.
    """
    with torch.enable_grad():
        scalar = Tensor.from_scalar(1.0)
        # Replace the data block with a multi-element tensor, bypassing
        # constructor validation.
        scalar.data[()] = torch.tensor([1.0, 2.0], requires_grad=True)

        with pytest.raises(ValueError, match="backward.*requires a scalar"):
            scalar.backward()


def test_backward_through_operations():
    """Test backward() computes gradients through tensor operations."""
    with torch.enable_grad():
        # Create scalar tensors
        a = Tensor.from_scalar(3.0, requires_grad=True)
        b = Tensor.from_scalar(4.0, requires_grad=True)
        
        # Perform operations: result = a * 2 + b * 3
        result = a * 2.0 + b * 3.0
        
        # Backward from result
        result.backward()
        
        # Check gradients
        # d(2a + 3b)/da = 2
        assert torch.isclose(a.data[()].grad, torch.tensor(2.0, dtype=torch.float64))
        # d(2a + 3b)/db = 3
        assert torch.isclose(b.data[()].grad, torch.tensor(3.0, dtype=torch.float64))


def test_backward_with_multiplication_chain():
    """Test backward() with chained multiplications."""
    with torch.enable_grad():
        x = Tensor.from_scalar(2.0, requires_grad=True)
        
        # Compute x * 3 * 4
        temp = x * 3.0
        result = temp * 4.0
        
        # Backward
        result.backward()
        
        # Gradient should be 3 * 4 = 12
        assert torch.isclose(x.data[()].grad, torch.tensor(12.0, dtype=torch.float64))


def test_backward_preserves_graph():
    """Test backward() with retain_graph behavior."""
    with torch.enable_grad():
        # Create scalar with gradient tracking
        x = Tensor.from_scalar(5.0, requires_grad=True)
        
        # Compute result
        result = x * 2.0
        
        # First backward should work
        result.backward()
        assert x.data[()].grad is not None
        
        # Note: PyTorch doesn't retain graph by default, so a second backward
        # on the same result would fail. We just verify the first one works.


def test_backward_complex_dtype():
    """Test backward() with complex scalar tensors.
    
    Note: PyTorch does not support automatic gradient creation for complex
    scalars (grad must be explicitly provided). This is a PyTorch limitation.
    """
    with torch.enable_grad():
        # Create complex scalar
        scalar = Tensor.from_scalar(3.0 + 4.0j, dtype=torch.complex128, requires_grad=True)
        
        # Multiply by 2
        result = scalar * 2.0
        
        # PyTorch doesn't support backward() on complex scalars without explicit grad
        with pytest.raises(RuntimeError, match="grad can be implicitly created only for real scalar"):
            result.backward()


def test_backward_without_requires_grad():
    """Test backward() behavior when requires_grad is False."""
    # Create scalar without gradient tracking
    scalar = Tensor.from_scalar(5.0, requires_grad=False)
    
    # Multiply by 2
    result = scalar * 2.0
    
    # backward() will raise an error from PyTorch because no graph was built
    with pytest.raises(RuntimeError):
        result.backward()


def test_backward_on_zero_scalar():
    """Test backward() on zero-valued scalar."""
    with torch.enable_grad():
        zero = Tensor.from_scalar(0.0, requires_grad=True)
        
        # Compute result
        result = zero * 5.0
        
        # Backward should work even though value is zero
        result.backward()
        
        # Gradient should be 5.0
        assert torch.isclose(zero.data[()].grad, torch.tensor(5.0, dtype=torch.float64))


# ============================================================================
#   Tests for autograd with tensor operations
# ============================================================================

def test_autograd_with_contraction():
    """Test that gradients flow through tensor contraction."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    with torch.enable_grad():
        # Create tensors with gradient tracking
        A = Tensor.random([idx_a, idx_b], itags=["a", "b"], seed=42, requires_grad=True)
        B = Tensor.random([idx_b.flip(), idx_c], itags=["b", "c"], seed=43, requires_grad=True)
        
        # Verify output has requires_grad and is part of computational graph
        assert A.requires_grad and B.requires_grad
        
        # Contract along matching indices
        C = contract(A, B)
        
        # Verify C blocks are part of computational graph
        assert any(block.requires_grad for block in C.data.values())
        assert any(block.grad_fn is not None for block in C.data.values())
        
        # Compute a scalar loss (sum of all elements) using torch.stack
        loss = torch.stack([block.sum() for block in C.data.values()]).sum()
        
        # Backward
        loss.backward()
        
        # Check that at least some gradients exist (blocks that contributed to output)
        # Not all blocks may have gradients due to charge conservation
        has_grad_A = sum(1 for block in A.data.values() if block.grad is not None)
        has_grad_B = sum(1 for block in B.data.values() if block.grad is not None)
        
        assert has_grad_A > 0, "At least some blocks in A should have gradients"
        assert has_grad_B > 0, "At least some blocks in B should have gradients"
        
        # Check shapes of existing gradients
        for block in A.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape
        
        for block in B.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape


def test_autograd_with_trace():
    """Test that gradients flow through trace operation."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    with torch.enable_grad():
        # Create tensor with gradient tracking (4 indices to avoid single-index result)
        T = Tensor.random([idx_a, idx_b, idx_c, idx_d], itags=["a", "a", "c", "d"], seed=42, requires_grad=True)
        
        # Trace over the first two indices (leaves 2 indices in result)
        result = trace(T, axes=(0, 1))
        
        # Verify result is part of computational graph
        assert any(block.requires_grad for block in result.data.values())
        
        # Compute a scalar loss using torch.stack
        loss = torch.stack([block.sum() for block in result.data.values()]).sum()
        
        # Backward
        loss.backward()
        
        # Check that at least some gradients exist
        has_grad = sum(1 for block in T.data.values() if block.grad is not None)
        assert has_grad > 0, "At least some blocks should have gradients"
        
        # Check shapes of existing gradients
        for block in T.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape


def test_autograd_with_permute():
    """Test that gradients flow through permutation."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 1)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    with torch.enable_grad():
        # Create tensor with gradient tracking
        T = Tensor.random([idx_a, idx_b, idx_c], itags=["a", "b", "c"], seed=42, requires_grad=True)
        
        # Permute axes
        T_perm = permute(T, [2, 0, 1])
        
        # Verify permuted tensor is part of computational graph
        assert any(block.requires_grad for block in T_perm.data.values())
        
        # Compute a scalar loss using torch.stack
        loss = torch.stack([block.sum() for block in T_perm.data.values()]).sum()
        
        # Backward
        loss.backward()
        
        # Permutation should preserve all blocks, so all should have gradients
        for block in T.data.values():
            assert block.grad is not None, "All blocks should have gradients after permute"
            assert block.grad.shape == block.shape


def test_autograd_with_decomposition():
    """Test that gradients flow through SVD decomposition."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    with torch.enable_grad():
        # Create tensor with gradient tracking
        T = Tensor.random([idx_a, idx_b, idx_c], itags=["a", "b", "c"], seed=42, requires_grad=True)
        
        # Decompose (SVD mode returns U, S, Vh)
        result = decomp(T, axes="a", mode="SVD")
        U, S, Vh = result
        
        # Verify U and Vh are part of computational graph
        assert any(block.requires_grad for block in U.data.values())
        assert any(block.requires_grad for block in Vh.data.values())
        
        # Compute a scalar loss from U and Vh using torch.stack
        loss_u = torch.stack([block.sum() for block in U.data.values()]).sum()
        loss_vh = torch.stack([block.sum() for block in Vh.data.values()]).sum()
        loss = loss_u + loss_vh
        
        # Backward
        loss.backward()
        
        # Check that at least some gradients exist for the input tensor
        has_grad = sum(1 for block in T.data.values() if block.grad is not None)
        assert has_grad > 0, "At least some blocks should have gradients"
        
        # Check shapes of existing gradients
        for block in T.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape


def test_autograd_with_chained_operations():
    """Test gradients through multiple chained operations."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    with torch.enable_grad():
        # Create tensors with gradient tracking
        A = Tensor.random([idx_a, idx_b], itags=["a", "b"], seed=42, requires_grad=True)
        B = Tensor.random([idx_b.flip(), idx_c], itags=["b", "c"], seed=43, requires_grad=True)
        
        # Chain operations: contract, then permute, then scale
        C = contract(A, B)  # Result has indices ["a", "c"]
        C_perm = permute(C, [1, 0])  # Swap to ["c", "a"]
        C_scaled = C_perm * 2.0  # Scale by 2
        
        # Verify final result is part of computational graph
        assert any(block.requires_grad for block in C_scaled.data.values())
        
        # Compute a scalar loss using torch.stack
        loss = torch.stack([block.sum() for block in C_scaled.data.values()]).sum()
        
        # Backward
        loss.backward()
        
        # Check gradients exist for both original tensors (at least some blocks)
        has_grad_A = sum(1 for block in A.data.values() if block.grad is not None)
        has_grad_B = sum(1 for block in B.data.values() if block.grad is not None)
        
        assert has_grad_A > 0, "At least some blocks in A should have gradients"
        assert has_grad_B > 0, "At least some blocks in B should have gradients"
        
        # Check shapes
        for block in A.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape
        
        for block in B.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape


def test_autograd_with_trace_to_scalar():
    """Test backward through trace that produces a scalar."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    with torch.enable_grad():
        # Create a tensor where we can trace all indices to get a scalar
        T = Tensor.random([idx, idx.flip()], itags=["a", "a"], seed=42, requires_grad=True)
        
        # Trace to get a scalar
        scalar = trace(T)
        
        # Verify it's a scalar
        assert scalar.is_scalar()
        
        # Backward directly on the scalar
        scalar.backward()
        
        # Check gradients
        for block in T.data.values():
            assert block.grad is not None
            assert block.grad.shape == block.shape


def test_autograd_with_contraction_to_scalar():
    """Test backward through contraction that produces a scalar."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    with torch.enable_grad():
        # Create tensors that contract to a scalar
        # A has ["a", "b"] with directions [OUT, IN]
        # B has ["a", "b"] with directions [IN, OUT] - opposite to A's
        A = Tensor.random([idx, idx.flip()], itags=["a", "b"], seed=42, requires_grad=True)
        B = Tensor.random([idx.flip(), idx], itags=["a", "b"], seed=43, requires_grad=True)
        
        # Contract to scalar (both indices match)
        scalar = contract(A, B)
        
        # Verify it's a scalar
        assert scalar.is_scalar()
        
        # Backward
        scalar.backward()
        
        # Check gradients exist for blocks that contributed
        has_grad_A = sum(1 for block in A.data.values() if block.grad is not None)
        has_grad_B = sum(1 for block in B.data.values() if block.grad is not None)
        
        assert has_grad_A > 0, "At least some blocks in A should have gradients"
        assert has_grad_B > 0, "At least some blocks in B should have gradients"
        
        # Check shapes of existing gradients
        for block in A.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape
        
        for block in B.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape


# ============================================================================
#   Tests for autograd with SU(2) tensors
# ============================================================================

def test_autograd_su2_with_contraction():
    """Test that gradients flow through contraction of SU(2) tensors.

    populate_random_weights() replaces block tensors, so requires_grad must be
    set *after* the SU(2) intertwiner structure is fully initialised.
    """
    group = SU2Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 2)))

    with torch.enable_grad():
        A = Tensor.random([idx_a, idx_b], itags=["p", "q"], seed=42)
        populate_random_weights(A, seed=42)
        A.requires_grad = True  # enable after intw is set up

        B = Tensor.random([idx_b.flip(), idx_c], itags=["q", "r"], seed=43)
        populate_random_weights(B, seed=43)
        B.requires_grad = True

        C = contract(A, B)

        # At least some result blocks must be in the computational graph
        assert any(block.requires_grad for block in C.data.values())

        loss = torch.stack([block.sum() for block in C.data.values()]).sum()
        loss.backward()

        has_grad_A = sum(1 for block in A.data.values() if block.grad is not None)
        has_grad_B = sum(1 for block in B.data.values() if block.grad is not None)
        assert has_grad_A > 0, "At least some blocks in A should have gradients"
        assert has_grad_B > 0, "At least some blocks in B should have gradients"

        for block in A.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape
        for block in B.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape


def test_autograd_su2_with_permute():
    """Test that gradients flow through permutation of SU(2) tensors."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 3), Sector(3, 2)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 3)))

    with torch.enable_grad():
        T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=44)
        populate_random_weights(T, seed=44)
        T.requires_grad = True  # enable after intw is set up

        T_perm = permute(T, [2, 0, 1])

        assert any(block.requires_grad for block in T_perm.data.values())

        loss = torch.stack([block.sum() for block in T_perm.data.values()]).sum()
        loss.backward()

        # Permutation is a bijection on blocks, so all blocks should have gradients
        for block in T.data.values():
            assert block.grad is not None, "All blocks should have gradients after permute"
            assert block.grad.shape == block.shape


def test_autograd_su2_with_decomposition():
    """Test that gradients flow through SVD decomposition of SU(2) tensors."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 3), Sector(3, 2)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 3)))

    with torch.enable_grad():
        T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=45)
        populate_random_weights(T, seed=45)
        T.requires_grad = True  # enable after intw is set up

        U, S, Vh = decomp(T, axes="a", mode="SVD")

        assert any(block.requires_grad for block in U.data.values())
        assert any(block.requires_grad for block in Vh.data.values())

        loss = (torch.stack([block.sum() for block in U.data.values()]).sum()
                + torch.stack([block.sum() for block in Vh.data.values()]).sum())
        loss.backward()

        has_grad = sum(1 for block in T.data.values() if block.grad is not None)
        assert has_grad > 0, "At least some blocks in T should have gradients"

        for block in T.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape


def test_autograd_su2_with_contraction_to_scalar():
    """Test backward through a full SU(2) contraction that produces a scalar."""
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))

    with torch.enable_grad():
        A = Tensor.random([idx, idx.flip()], itags=["a", "b"], seed=46)
        populate_random_weights(A, seed=46)
        A.requires_grad = True  # enable after intw is set up

        B = Tensor.random([idx.flip(), idx], itags=["a", "b"], seed=47)
        populate_random_weights(B, seed=47)
        B.requires_grad = True

        scalar = contract(A, B)
        assert scalar.is_scalar()

        scalar.backward()

        has_grad_A = sum(1 for block in A.data.values() if block.grad is not None)
        has_grad_B = sum(1 for block in B.data.values() if block.grad is not None)
        assert has_grad_A > 0, "At least some blocks in A should have gradients"
        assert has_grad_B > 0, "At least some blocks in B should have gradients"

        for block in A.data.values():
            if block.grad is not None:
                assert block.grad.shape == block.shape


def test_autograd_su2_intw_weights_are_not_trainable():
    """Test that Bridge intertwiner weights do not acquire requires_grad.

    Intertwiner weights are fixed structural constants derived from CG coefficients
    and must never appear as trainable parameters in the computational graph.
    """
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))

    with torch.enable_grad():
        T = Tensor.random([idx, idx.flip()], itags=["a", "b"], seed=48)
        populate_random_weights(T, seed=48)
        T.requires_grad = True  # enable after intw is set up

        # Data blocks should now be in the computational graph
        assert T.requires_grad

        # Intertwiner weight tensors must NOT have requires_grad — they are
        # fixed structural constants, not learnable parameters.
        for key, bridge in T.intw.items():
            assert not bridge.weights.requires_grad, (
                f"Bridge weights for key {key} should not require grad"
            )
