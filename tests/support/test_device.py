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


"""Tests for device management and GPU support."""

import torch
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group
from ..utils import assert_blocks_equal


def test_default_device_is_cpu():
    """Test that default device is CPU."""
    # Reset to default
    torch.set_default_device('cpu')
    assert torch.get_default_device() == torch.device('cpu')
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    tensor = Tensor.zeros([idx, idx.flip()])
    
    assert tensor.device == torch.device('cpu')
    for block in tensor.data.values():
        assert block.device == torch.device('cpu')


def test_set_default_device():
    """Test changing default device using PyTorch's native functions."""
    original_device = torch.get_default_device()
    
    try:
        # Test with string
        torch.set_default_device('cpu')
        assert torch.get_default_device() == torch.device('cpu')
        
        # Test with torch.device
        torch.set_default_device(torch.device('cpu'))
        assert torch.get_default_device() == torch.device('cpu')
    finally:
        # Restore original device
        torch.set_default_device(original_device)


def test_tensor_device_property():
    """Test Tensor.device property."""
    torch.set_default_device('cpu')
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    tensor = Tensor.zeros([idx, idx.flip()])
    
    assert tensor.device == torch.device('cpu')


def test_tensor_to_cpu():
    """Test Tensor.to() and cpu() methods."""
    torch.set_default_device('cpu')
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    tensor = Tensor.random([idx, idx.flip()], seed=42)
    
    # Move to CPU (should be no-op)
    tensor_cpu = tensor.to('cpu')
    assert tensor_cpu.device == torch.device('cpu')
    assert_blocks_equal(tensor, tensor_cpu)
    
    # Test cpu() method
    tensor_cpu2 = tensor.cpu()
    assert tensor_cpu2.device == torch.device('cpu')
    assert_blocks_equal(tensor, tensor_cpu2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_to_cuda():
    """Test moving tensor to CUDA device."""
    torch.set_default_device('cpu')
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    tensor_cpu = Tensor.random([idx, idx.flip()], seed=42)
    
    # Move to CUDA
    tensor_gpu = tensor_cpu.to('cuda')
    assert tensor_gpu.device.type == 'cuda'
    for block in tensor_gpu.data.values():
        assert block.device.type == 'cuda'
    
    # Test cuda() method
    tensor_gpu2 = tensor_cpu.cuda()
    assert tensor_gpu2.device.type == 'cuda'
    
    # Move back to CPU and check values preserved
    tensor_back = tensor_gpu.cpu()
    assert_blocks_equal(tensor_cpu, tensor_back)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_tensor_to_mps():
    """Test moving tensor to MPS device (Apple Silicon)."""
    torch.set_default_device('cpu')
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    tensor_cpu = Tensor.random([idx, idx.flip()], seed=42)
    
    # Move to MPS (will auto-convert float64 to float32 since MPS doesn't support float64)
    tensor_mps = tensor_cpu.to('mps')
    assert tensor_mps.device.type == 'mps'
    assert tensor_mps.dtype == torch.float32  # MPS doesn't support float64
    for block in tensor_mps.data.values():
        assert block.device.type == 'mps'
        assert block.dtype == torch.float32
    
    # Move back to CPU and check values preserved (with float32 precision)
    tensor_back = tensor_mps.cpu()
    # Convert original to float32 for comparison since MPS conversion changed dtype
    tensor_cpu_f32 = Tensor(
        indices=tensor_cpu.indices,
        itags=tensor_cpu.itags,
        data={k: v.to(dtype=torch.float32) for k, v in tensor_cpu.data.items()},
        dtype=torch.float32,
    )
    assert_blocks_equal(tensor_cpu_f32, tensor_back)


def test_tensor_explicit_device_in_constructor():
    """Test passing device parameter to constructors."""
    torch.set_default_device('cpu')
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    # Test zeros with explicit device
    tensor = Tensor.zeros([idx, idx.flip()], device='cpu')
    assert tensor.device == torch.device('cpu')
    
    # Test random with explicit device
    tensor = Tensor.random([idx, idx.flip()], seed=42, device='cpu')
    assert tensor.device == torch.device('cpu')
    
    # Test from_scalar with explicit device
    scalar = Tensor.from_scalar(3.14, device='cpu')
    assert scalar.device == torch.device('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_creation_on_cuda():
    """Test creating tensors directly on CUDA."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    # Create tensor on CUDA
    tensor = Tensor.zeros([idx, idx.flip()], device='cuda')
    assert tensor.device.type == 'cuda'
    for block in tensor.data.values():
        assert block.device.type == 'cuda'
    
    # Create random tensor on CUDA
    tensor_rand = Tensor.random([idx, idx.flip()], seed=42, device='cuda')
    assert tensor_rand.device.type == 'cuda'


def test_default_device_affects_construction():
    """Test that torch.set_default_device affects new tensor construction."""
    original_device = torch.get_default_device()
    
    try:
        torch.set_default_device('cpu')
        
        group = U1Group()
        idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
        tensor = Tensor.zeros([idx, idx.flip()])
        
        assert tensor.device == torch.device('cpu')
    finally:
        torch.set_default_device(original_device)


def test_operations_preserve_device():
    """Test that tensor operations preserve device placement."""
    torch.set_default_device('cpu')
    
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    t1 = Tensor.random([idx, idx.flip()], seed=42)
    t2 = Tensor.random([idx, idx.flip()], seed=43)
    
    # Test addition preserves device
    result = t1 + t2
    assert result.device == torch.device('cpu')
    
    # Test subtraction preserves device
    result = t1 - t2
    assert result.device == torch.device('cpu')
    
    # Test multiplication preserves device
    result = t1 * 2.0
    assert result.device == torch.device('cpu')
    
    # Test copy preserves device
    result = t1.copy()
    assert result.device == torch.device('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_operations_on_cuda():
    """Test that tensor operations work on CUDA."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    t1 = Tensor.random([idx, idx.flip()], seed=42, device='cuda')
    t2 = Tensor.random([idx, idx.flip()], seed=43, device='cuda')
    
    # Test addition on CUDA
    result = t1 + t2
    assert result.device.type == 'cuda'
    
    # Test subtraction on CUDA
    result = t1 - t2
    assert result.device.type == 'cuda'
    
    # Test multiplication on CUDA
    result = t1 * 2.0
    assert result.device.type == 'cuda'
    
    # Test copy on CUDA
    result = t1.copy()
    assert result.device.type == 'cuda'


def test_empty_tensor_device():
    """Test device property of empty tensor."""
    torch.set_default_device('cpu')
    
    group = U1Group()
    # Create a tensor with no charge-conserving blocks
    idx_out = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    
    # This should create an empty tensor (no blocks conserve charge)
    tensor = Tensor.zeros([idx_out, idx_in])
    
    # Empty tensor should report default device
    assert tensor.device == torch.get_default_device()
