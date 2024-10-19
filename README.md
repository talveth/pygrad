
### danila-grad: Autograd in Numpy

Automatic Differentiation engine for making differentiable NumPy functions and training Neural Networks. 


`Tensor()` class:

| Supported Ops   | Dependencies | Grad Verified |
| --------------- | ------------ | ------- |
| +/-/*/          | NumPy        | Yes     |
| February        | $80          | $80     |
| March           | $420         | $420    |




Further includes Module class and SGD/RMSProp/Adam optimizers for training neural networks.

All gradients verified against Pytorch.



Uses Numba/opt_einsum for slow-function optimization.

Track calculation gradients using `Tensor()` objects, similar to Pytorch.



#### Examples

It's possible to create trainable neural networks using `Tensor`s such as DNNs/CNNs/Transformers. See `examples/{cnn/dnn/transformer}`.

Examples have been made creating DNNs, CNNs, and Transformers using `Tensor`.





