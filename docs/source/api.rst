
.. _API:

API
========


pygrad
-------------

``pygrad`` holds all the base modules for the differentiation engine, for CPU-based runtimes.

.. autosummary::
   :toctree: generated
   :recursive:
   
   pygrad


architectures
-------------

``architectures`` holds three example architectures defined solely using this library: A DNN, a CNN, and a Vanila Vaswani Transformer: https://github.com/baubels/pygrad/tree/master/architectures.

.. autosummary::
   :toctree: generated
   :recursive:
   
   architectures

examples
-------------

``examples`` shows training pipelines for each of the architectures defined in ``architectures``.
Running these will require installing the library with ``.[examples]``.

Please see: https://github.com/baubels/pygrad/tree/master/examples
