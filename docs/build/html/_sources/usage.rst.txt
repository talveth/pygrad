Usage
=====

.. _installation:

Installation
------------

To use danila-grad, clone the repository and then install with pip:

.. code-block:: console

   $ pip install danila-grad

This will install danila-grad with the Python importable name ``autograd``.


Basic Usage
------------

All library functionality can be found in ``autograd``. 
The main differentiable object in the library is the ``Tensor`` class.
The below shows performing backprop on a function ``y``.

.. code-block:: Python

   from autograd.cpu.tensor import Tensor
   x = Tensor(1)
   y = x**2 + 1
   y.backward()
   print(y.grad, x.grad) 
   # -> 1.0, 2.0


*Common deep learning layers* can be found in ``autograd.cpu.basics``. The below shows finding the gradient of a Dense linear layer.

.. code-block:: Python

    import numpy as np
    from autograd.cpu.tensor import Tensor
    from autograd.cpu.basics import Linear
    x  = Tensor(np.ones((1,1,2)))
    l1 = Linear(2,1)
    y  = l1(x)
    y.backward()
    print(y.shape, y.grad.shape, l1.W.grad.shape, l1.B.grad.shape, x.shape) 
    # -> (1, 1, 1) (1, 1, 1) (1, 2, 1) (1, 1, 1) (1, 1, 2)


*Gradient Descent schemes* can be found in ``autograd.cpu.optims``. 
The below shows an example of minimizing the L2 loss between ``y`` and ``1.5`` using the ``SGD`` class.

.. code-block:: Python

    from autograd.cpu.tensor import Tensor
    from autograd.cpu.optims import SGD

    x     = Tensor([1])
    y     = x**2 + 1
    optim = SGD(y.create_graph()[1], lr=0.01)

    for _ in range(100):
        optim.zero_grad()
        y    = x**2 + 1
        loss = (y-1.5)**2
        loss.backward()
        optim.step(loss)

    print(x.value, y.value, loss.value) 
    # -> 0.7100436 1.50433688 1.88085134e-05
