
### pygrad: Lightweight automatic differentiation engine in NumPy/Numba.

This is a lightweight automatic differentiation engine based on NumPy and Numba. Included is a differentiable Tensor class, layers such as Dropout/Linear/Attention, loss functions such as BCE/CCE, optimizers such as SGD/RMSProp/Adam, and a DNN/CNN/Transformer architecture. Documentation can be found at: https://baubels.github.io/pygrad/.

The main component is the `Tensor` class supporting many math operations. `Tensor`s have `.value` and `.grad` attributes, gradients being populated by calling `.backward()` on either self or any of its children. They can be used standalone, or for constructing more complex architectures such as a vanilla Transformer.

#### Usage

Tensors accept the same input value as a NumPy array. Create them with Tensor(value) or tensor.array(value).

A simple usage example:

```python
from pygrad.tensor import Tensor
x = Tensor(1)
(((x**3 + x**2 + x + 1) - 1)**2).backward()
x.grad  # 36.
```



**Supported Tensor Methods**

| Tensor Ops       | `pygrad.tensor`                      | Dependencies |
| ---------------  | -------------                              | ------------ |
| Magic methods    |  + - * / ** @                              | NumPy        |
| Other math ops   | sum, reshape, transpose, mean, std, conv2D | NumPy        |
| Common activations | relu, tanh, sigmoid                      | NumPy        |
| Loss              | softmax                                   | NumPy, Numba  |

**Supported Objects**

| Extraneous Ops   | Created Classes                                            | Dependencies |
| ---------------  | -------------                                              | ------------ |
| Common layers    |  ReLU, Dropout, AddNorm, Linear, Softmax, Flatten, Conv2D  | NumPy, Numba |
| Losses           | BCELoss, CCELoss                                           | NumPy, Numba |
| Optimizers       | SGD, SGD_Momentum, RMSProp, Adam                           | NumPy        |
| Architectures    | DNN, CNN, Vanila Transformer                               | NumPy/Numba  |


This calculates `y.grad = 1` and `x.grad = 6`.

To view the computational graph, use with `y.topo` or `y.weights`, depending on whether or not to view leaf tensors.
Additional examples can be found under `examples/`.

#### Citation

If you find this project helpful in your research or work, I kindly ask that you cite it: [View Citation](./CITATION.cff). Thank you! 
