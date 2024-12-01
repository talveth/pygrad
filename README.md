
### pygrad: A lightweight differentiation engine written in Python.

Documentation: https://baubels.github.io/pygrad/.

This is a lightweight (<300kB) automatic differentiation engine based on NumPy, Numba, and opt_einsum.
Included is a differentiable Tensor class, layers such as Dropout/Linear/Attention, loss functions such as BCE/CCE, optimizers such as SGD/RMSProp/Adam, and an example DNN/CNN/Transformer architecture. This library is a good alternative if you want to do backpropagation on simple and small functions or networks, without much overhead.

The main component is the `Tensor` class supporting many math operations. `Tensor`s have `.value` and `.grad` attributes, gradients being populated by calling `.backward()` on either self or any of its children. They can be used standalone, or for constructing more complex architectures such as a vanilla Transformer.

#### Installation

```bash
pip install pygradproject
# OR
git clone https://github.com/baubels/pygrad.git
pip install . (or .[examples] or .[dev])
```

#### Usage

Tensors accept the same input value as a NumPy array. Create them with Tensor(value) or tensor.array(value).
Run backprop on them with `.backward()`.

A simple usage example:

```python
from pygrad.tensor import Tensor
x = Tensor(1)
(((x**3 + x**2 + x + 1) - 1)**2).backward()
x.value, x.grad  # 1.0, 36.0
```

Since `Tensor` store their value in `.value` and their gradient in `.grad`, it's easy to perform gradient descent.

```python
for _ in range(100):
    (((x**3 + x**2 + x + 1) - 1)**2).backward()     # gradients are automatically reset when called
    x.value = x.value - 0.01*x.grad
```

Tensors can also be operated on with broadcast-friendly NumPy arrays or other Tensors whose value is broadcast friendly.
Internally, a Tensor will always cast it's set value to a NumPy array.

```python
import numpy as np
x  = Tensor(np.ones((10,20)))
y  = Tensor(np.ones((20,10)))
z1 = x@y
z2 = x@np.ones((20,10))       
np.all(z1.value == z2.value)  # True
```

There are enough expressions defined to be able to create many different models.
For example usage and in-depth descriptions of each component of pygrad, 
check out [the docs](https://baubels.github.io/pygrad/).

#### Citation/Contribution

If you find this project helpful in your research or work, I kindly ask that you cite it: [View Citation](./CITATION.cff). Thank you! 

If there are issues with the project, please submit an issue. 
Otherwise, please read [the current status for contributors](https://baubels.github.io/pygrad/contrib.html).
