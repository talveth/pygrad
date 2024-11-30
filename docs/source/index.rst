
.. pygrad documentation master file, created by
   sphinx-quickstart on Wed Nov 20 23:06:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pygrad documentation
=========================

**pygrad** is a lightweight automatic differentiation engine:

   * written entirely in Python,
   * relying only on NumPy, Numba, and opt_einsum,
   * verified against Pytorch^*,
   * and less than 300kB in size.

Pygrad will be useful to you if you are looking to compute gradients and/or perform 
gradient descent for models with less than 1 million parameters.

Pygrad's ``Tensor`` object operates like a NumPy array, additionally storing gradients.

``Tensors``:
   * Store operations performed on them, with support for broadcasting
   * Perform backpropagation with ``.backward()``
   * Store gradients in ``.grad``
   * Support np.float16 to np.float128 data types

A simple example performing gradient descent on a Tensor:

.. code-block:: Python

      from pygrad.tensor import Tensor

      loss_fn = lambda y, yh: (y-yh)**2   # L2 norm
      x       = Tensor(1)                 # Tensor
      y       = x**2 + 0.25               # model
      yh      = 0.5                       # float

      for _ in range(1000):
         y    = x**2 + 0.25               # fwd pass
         loss_fn(y,yh).backward()         # populates x.grad
         x.value = x.value - 0.01*x.grad  # gradient descent

      x.value, loss_fn(y,yh).value        # 0.5, 0

This documentation includes examples using Tensors to perform gradient descent on the very 
simplest of functions to training a Vaswani Transformer with Adam.

For installation instructions and a quick glance at usage, see :doc:`usage`.
All classes and functions can be found in :doc:`api`.
For in-depth module descriptions, check out :doc:`modules`.

If you are interesting in contributing, :ref:`please click here <contribs>`. If you are wondering who I am, :ref:`click here <aboutme>`

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   modules
   examples
   api
   methods
   contrib
   about

.. note::
   *All operations are verified against Pytorch, except for Conv2D gradients 
   when performing strictly more than 1 backwards pass when ``reset_grad=False``.
