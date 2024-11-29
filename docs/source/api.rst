
.. _API:

API
========


autograd.cpu
-------------

``autograd.cpu`` holds all the base modules for the differentiation engine, for CPU-based runtimes.

.. autosummary::
   :toctree: generated
   :recursive:
   
   autograd.cpu


architectures
-------------

``architectures`` holds three example architectures defined solely using this library: A DNN, a CNN, and a Vanila Vaswani Transformer.

.. autosummary::
   :toctree: generated
   :recursive:
   
   architectures


examples
-------------

``examples`` shows training pipelines for each of the architectures defined in ``architectures``.
Running these will require installing the library with ``.[examples]``.

.. autosummary::
   :toctree: generated
   :recursive:
   
   examples


