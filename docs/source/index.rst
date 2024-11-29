.. danila-grad documentation master file, created by
   sphinx-quickstart on Wed Nov 20 23:06:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

danila-grad documentation
=========================

**danila-grad** is a lightweight automatic differentiation (autograd) engine written entirely in Python, 
relying only on NumPy and Numba, verified against Pytorch*, and less than 300 KBs in size.
It automatically tracks derivatives, being able to both natively compute derivatives and apply gradient descent to any produced function. 
Any np.floating data type is supported, thus allowing for 16-bit to 128-bit values and gradients if required. 
This documentation includes examples performing gradient descent on the very 
simplest of functions to training a Vaswani Transformer with Adam.

For installation instructions and a quick glance at usage, see :doc:`usage`.
All classes and functions can be found in :doc:`api`.
For in-depth module descriptions, check out :doc:`modules`.

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

.. note::
   All operations are verified against Pytorch, except for Conv2D gradients 
   when performing strictly more than 1 backwards pass.
