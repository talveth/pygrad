
"""
Module storing Module.
"""

import copy
from abc import ABC, abstractmethod

import numpy as np

from .tensor import Tensor


class Module(ABC):
    """
    Module Class.

    Allows for performing batched forward and backwards passes on a model without modifying the model directly.
    The subclassed models must perform any required **kwargs type checking.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        self.params, self.weights       = self.parameters(**kwargs)
        self.model_copy                 = None
        np.seterr(all='ignore')

    @abstractmethod
    def forward(self, **kwargs):
        """Ensure this method is defined in the subclass."""
        pass

    def __call__(self, **kwargs: Tensor)->Tensor:
        """
        Returns the forward pass output of the model on a batched input.

        Further:
            * Creates a batch-friendly version of the original model to do backprop with.
            * Creates topological and weight graphs of the batched model, storing them in self.model_copy.

        """
        if getattr(self, 'model_copy', None) is None:
            self.model_reset()
        output                  = self.model_copy.forward(**kwargs)
        self.model_copy.params, self.model_copy.weights  = output.create_graph()
        return output

    def model_reset(self):
        """
        Deletes the batched model.
        """
        self.model_copy     = None
        self.model_copy     = copy.deepcopy(self)

    def call_slow(self, **kwargs:Tensor)->Tensor:
        ## creates a batch-friendly version of the original model to do backprop with
        self.model_copy         = None
        self.model_copy         = copy.deepcopy(self)
        output                  = self.model_copy.forward(**kwargs)
        self.model_copy.params, self.model_copy.weights  = output.create_graph()
        return output

    def parameters(self, **kwargs):
        temp_x              = self.forward(**kwargs)
        params, weights     = temp_x.create_graph()
        return params, weights
