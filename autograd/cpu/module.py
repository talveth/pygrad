
import copy
from .tensor import Tensor
import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):
    """
    Module Class.
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

    def __call__(self, **kwargs: Tensor):
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

    def call_slow(self, **kwargs:Tensor):
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
