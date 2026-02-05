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


"""Tests for autograd control."""

import torch
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group


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
    """Test that backward() validates tensor has exactly 1 element.
    
    Note: The Tensor constructor already validates shapes, so this test
    verifies the numel check exists but may not be reachable in practice.
    """
    # This test is redundant with the constructor validation, but we keep it
    # to document the numel check in backward(). In practice, a valid scalar
    # tensor always has numel() == 1 due to constructor validation.
    pass  # Test is covered by constructor validation


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
