Usage
=====

.. _installation:

Installation
------------

To use pygrad, either do ``pip install pygradproject`` or
 clone `the repository <https://github.com/baubels/pygrad>`_ and install:

.. code-block:: console

   $ pip install .               # normal install
   $ pip install .[examples]     # normal + examples (for dnn/cnn/transformer training)
   $ pip install .[dev]          # normal + dev      (for development purposes)

This will install pygrad with the Python importable name ``pygrad``.

If your installing with ``[examples]`` and want to use the examples, 
download the missing datasets in the repo's `/examples <https://github.com/baubels/pygrad/tree/master/examples>`_.


Basic Usage
------------

All library functionality can be found in ``pygrad``. 
The main differentiable object in the library is the ``Tensor`` class.
The below shows performing backprop on a function ``y``.

.. code-block:: Python

   from pygrad.tensor import Tensor
   x = Tensor(1)
   y = x**2 + 1
   y.backward()
   print(y.grad, x.grad) 
   # -> 1.0, 2.0

For more details, see :ref:`tensor`.

*Common deep learning layers* can be found in ``pygrad.basics`` (:ref:`basics`). 
The below finds the gradient of a Dense linear layer.

.. code-block:: Python

    import numpy as np
    from pygrad.tensor import Tensor
    from pygrad.basics import Linear
    x  = Tensor(np.ones((1,1,2)))
    l1 = Linear(2,1)
    y  = l1(x)
    y.backward()
    print(y.shape, y.grad.shape, l1.W.grad.shape, l1.B.grad.shape, x.shape) 
    # -> (1, 1, 1) (1, 1, 1) (1, 2, 1) (1, 1, 1) (1, 1, 2)


*Gradient Descent schemes* can be found in ``pygrad.optims`` (:ref:`optims`).
The below shows an example of minimizing the L2 loss between ``y`` and ``1.5`` using the ``SGD`` class.

.. code-block:: Python

    from pygrad.tensor import Tensor
    from pygrad.optims import SGD

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

For more details on the modules see :ref:`modules`, for API, see :ref:`api`.
