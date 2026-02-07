# Automatic Differentiation (Autograd)

Nicole supports PyTorch's automatic differentiation (autograd) for gradient-based optimization of tensor network states. This guide explains when and how to use autograd effectively with symmetric tensors.

## Overview

By default, Nicole **disables** autograd for performance, as most tensor network algorithms don't require gradients. However, you can enable gradient tracking for:

- Variational optimization of tensor network states
- Optimizing parametric quantum states
- Gradient-based energy minimization
- Differentiable tensor network layers in neural networks

## Enabling Gradient Tracking

PyTorch's autograd system has two components: **per-tensor flags** (`requires_grad`) and **global gradient mode** (context managers). Understanding both is essential for correct usage.

### The `requires_grad` Flag

The `requires_grad` attribute is a **per-tensor property** that marks whether a tensor is a variational parameter that should accumulate gradients. When set to `True`:

- The tensor becomes a "leaf" node in the computation graph
- Operations involving this tensor are recorded for backpropagation
- After calling `.backward()`, gradients accumulate in the tensor's `.grad` attribute

```python exec="1" session="autograd" result=""
import torch
from nicole import Tensor, Index, Sector, Direction, U1Group

group = U1Group()
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
```

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Set at creation
T1 = Tensor.random([idx, idx.flip()], itags=["i", "j"], 
                   requires_grad=True, seed=42)

# Or set after creation
T2 = Tensor.random([idx, idx.flip()], itags=["k", "l"], seed=43)
T2.requires_grad = True

print(f"T1 requires grad: {T1.requires_grad}")  # True
print(f"T2 requires grad: {T2.requires_grad}")  # True
```

!!! note "Storage Location"
    In Nicole, `requires_grad` is propagated to the underlying PyTorch blocks in `Tensor.data`. Each block inherits the gradient tracking status from the parent Tensor.

### Global Gradient Mode

The **global gradient mode** controls whether operations are recorded in the computation graph, regardless of tensor flags. This is controlled by context managers:

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Check current mode
print(torch.is_grad_enabled())  # False (Nicole disables by default)
```

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Disable gradient tracking globally (explicit, though already disabled in Nicole)
with torch.no_grad():
    # No operations are tracked, even if tensors have requires_grad=True
    T = Tensor.random([idx, idx.flip()], itags=["i", "j"], requires_grad=True)
    result = T + T  # result.requires_grad will be False
    print(f"In no_grad: {result.requires_grad}")  # False
```

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Enable gradient tracking explicitly (REQUIRED in Nicole)
with torch.enable_grad():
    # Operations are tracked when explicitly enabled
    T = Tensor.random([idx, idx.flip()], itags=["i", "j"], requires_grad=True)
    result = T + T  # result.requires_grad will be True
    print(f"In enable_grad: {result.requires_grad}")  # True
```

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Set mode programmatically
torch.set_grad_enabled(False)  # Disable (Nicole's default)
print(f"Grad disabled: {not torch.is_grad_enabled()}")

torch.set_grad_enabled(True)   # Enable for gradient tracking
print(f"Grad enabled: {torch.is_grad_enabled()}")
```

```python exec="1" session="autograd" result=""
# Reset to default for next examples
torch.set_grad_enabled(False)
```

### Gradient Propagation Rules

The `requires_grad` flag of output tensors depends on **both** input tensor flags **and** the global gradient mode:

| Global Mode | Input `requires_grad` | Output `requires_grad` |
|------------|----------------------|------------------------|
| Disabled (default in Nicole) | Any value | `False` |
| Enabled (`enable_grad`) | Any input is `True` | `True` |
| Enabled (`enable_grad`) | All inputs are `False` | `False` |

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
from nicole import contract

A = Tensor.random([idx, idx.flip()], itags=["i", "j"], requires_grad=True, seed=10)
B = Tensor.random([idx, idx.flip()], itags=["j", "k"], requires_grad=False, seed=11)

print("Tensor A (requires_grad=True):")
print(A)
print(f"\nTensor B (requires_grad=False):")
print(B)
```

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Rule 1: Default mode in Nicole is DISABLED - gradients NOT tracked
C = contract(A, B)
print(f"C requires grad: {C.requires_grad}")  # False (grad mode disabled by default)
```

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Rule 2: Must explicitly enable grad mode for tracking
with torch.enable_grad():
    D = contract(A, B)  # A has requires_grad=True
    print(f"D requires grad: {D.requires_grad}")  # True (A propagates in enabled mode)
```

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Rule 3: no_grad context also disables (redundant in Nicole, but explicit)
with torch.no_grad():
    E = contract(A, B)
    print(f"E requires grad: {E.requires_grad}")  # False (explicitly disabled)
```

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Rule 4: Nested contexts (inner takes precedence)
with torch.no_grad():
    with torch.enable_grad():
        F = contract(A, B)
        print(f"F requires grad: {F.requires_grad}")  # True (inner context wins)
```

## Basic Gradient Computation

Since Nicole disables autograd by default, you must explicitly enable it for gradient tracking:

```python exec="1" session="autograd" result=""
# Create parameterized tensors
A_grad = Tensor.random([idx, idx.flip()], itags=["i", "mid"], 
                       requires_grad=True, seed=1)
B_grad = Tensor.random([idx, idx.flip()], itags=["mid", "i"], 
                       requires_grad=True, seed=2)
```

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Enable gradient tracking (REQUIRED in Nicole)
with torch.enable_grad():
    # Forward pass: compute contraction
    loss = contract(A_grad, B_grad)
    
    # Backward pass: compute gradients
    loss.backward()

# Access gradients (stored in PyTorch blocks)
for key, block in A_grad.data.items():
    if block.grad is not None:
        print(f"Gradient for block {key}:\n{block.grad}")
```

!!! warning "Remember to Enable Grad Mode"
    Nicole sets `torch.set_grad_enabled(False)` on import. Always wrap gradient computations in `torch.enable_grad()` or call `torch.set_grad_enabled(True)` before variational optimization.

## Performance Considerations

Enabling autograd comes with trade-offs. While it enables powerful gradient-based optimization, it adds both computational overhead and memory costs. Understanding these costs helps you make informed decisions about when to use gradients.

### Computational Overhead

Autograd adds overhead in two ways:

1. **Graph Construction**: Every operation must record its inputs and create nodes in the computation graph
2. **Bookkeeping**: PyTorch tracks tensor metadata (shapes, strides, version counters) to enable backpropagation

The overhead scales with:

- **Depth of computation**: Longer chains of operations accumulate more graph nodes
- **Number of operations**: Each tensor operation adds bookkeeping cost
- **Tensor complexity**: Symmetric tensors with many blocks require more metadata tracking

For tensor networks, this overhead can be significant because contractions often form deep chains. Let's measure the impact on a realistic tensor network computation:

```python exec="1" session="autograd" result=""
import time
from nicole import conj

# Create a larger index for performance testing
idx_large = Index(Direction.OUT, group, sectors=(
    Sector(0, 10), Sector(1, 8), Sector(2, 6)
))
```

```python exec="1" session="autograd" result="console" idprefix="" source="material-block"
# Create a chain of 10 tensors with sequential contractions
num_tensors = 10
tensors = []
for i in range(num_tensors):
    itag_in = f"i{i}"
    itag_out = f"i{i+1}"
    T = Tensor.random([idx_large, idx_large.flip()], 
                      itags=[itag_in, itag_out], seed=100+i)
    tensors.append(T)

print(f"Created chain of {num_tensors} tensors")

# Without gradients - contract all sequentially
start = time.time()
for _ in range(20):
    result = tensors[0]
    for T in tensors[1:]:
        result = contract(result, T)
time_no_grad = time.time() - start

# With gradients - same chain
for T in tensors:
    T.requires_grad = True
torch.set_grad_enabled(True)
start = time.time()
for _ in range(20):
    result = tensors[0]
    for T in tensors[1:]:
        result = contract(result, T)
    if result.data:  # Ensure computation
        pass
time_with_grad = time.time() - start

print(f"\nSequential chain: T0 --[i1]-- T1 --[i2]-- ... --[i{num_tensors}]-- T{num_tensors-1}")
print(f"Without grad: {time_no_grad:.4f}s")
print(f"With grad: {time_with_grad:.4f}s")
print(f"Overhead: {(time_with_grad / time_no_grad - 1) * 100:.1f}%")
```

```python exec="1" session="autograd" result=""
# Reset gradient mode
torch.set_grad_enabled(False)
```

**Key Observations:**

- The overhead is **multiplicative**, not additive: longer chains show proportionally larger slowdowns
- For typical tensor network algorithms (DMRG, TEBD, etc.), this overhead is often **not worth it** since they use specialized update schemes rather than gradient descent
- For variational optimization of small networks, the overhead may be acceptable if gradients provide faster convergence

**When to worry about overhead:**

- ✅ **Don't worry**: Single-shot computations, prototyping, small-scale optimization
- ⚠️ **Be cautious**: Inner loops with thousands of iterations, production DMRG/TEBD
- ❌ **Avoid**: Time-critical code where gradients aren't needed

### Memory Usage

Beyond computation time, autograd also increases memory consumption significantly. PyTorch must store:

1. **Intermediate tensors**: All results from the forward pass needed for backprop
2. **Gradient buffers**: Space to accumulate gradients for each parameter
3. **Computation graph metadata**: Nodes tracking operation types and connections

For tensor networks, this can be problematic because:
- **Large bond dimensions** create big intermediate tensors
- **Deep contraction trees** accumulate many intermediates
- **Iterative algorithms** repeatedly build and destroy graphs

```python
# Memory-efficient: clear gradients regularly
optimizer = torch.optim.Adam([block for block in psi.data.values()])

# Enable gradient tracking for variational optimization
torch.set_grad_enabled(True)

for step in range(100):
    optimizer.zero_grad()  # Clear old gradients (REQUIRED!)
    
    # Compute energy/observable and gradients
    energy = compute_energy(psi)
    energy.backward()
    optimizer.step()
    
    # Free computation graph immediately
    del energy
```

**Additional memory-saving techniques:**

- **Gradient checkpointing**: Recompute intermediate values during backward instead of storing them (trade compute for memory)
- **Detach when possible**: Use `.detach()` on tensors you don't need gradients for to break the computation graph
- **Context managers**: Wrap non-differentiable computations in `torch.no_grad()` to avoid graph construction
- **Chunking**: Break large contractions into smaller pieces to limit peak memory usage

## When to Use Autograd

### Good Use Cases:
- Variational ground state search
- Training parametric tensor networks
- Optimization problems with smooth objectives
- Differentiable physics simulations

### Not Recommended:
- Standard DMRG (uses iterative eigensolver)
- Time evolution (uses analytical formulas)
- Sampling-based methods
- When gradients are not informative

## See Also

- [GPU Acceleration](gpu-acceleration.md): Combine with GPU for faster optimization
- [Performance Tips](performance.md): General optimization strategies
- [API: Tensor.requires_grad](../../api/core/tensor.md): Autograd property documentation
