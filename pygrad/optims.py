
"""
Module storing (gradient descent) optimization methods.
"""

import copy

import numpy as np

from .tensor import Tensor


class SGD:
    """
    Vanilla Gradient Descent.
    
    """
    def __init__(self, model_parameters:list, lr:float=1e-5):
        """
        Initializes the SGD optimizer.

        :param model_parameters: This is a list of Tensors specifying the pre-order traversal of a Tensor's computational graph. This is given either from Tensor.create_graph[1] or for models subclassing Module as model.params
        :type model_parameters: list
        :param lr: the learning rate for SGD
        :type lr: float. Defaults to 1e-5.

        """
        assert isinstance(model_parameters, list), "model_parameters must be a list of Tensors"
        assert len(model_parameters) > 0, "The provided model_parameters contains no Tensors"
        assert isinstance(lr, float), "the lr must be a float"
        self.model_parameters   = model_parameters
        self.lr                 = lr

    def zero_grad(self):
        """
        Sets the gradient of each Tensor in model_parameters to 0.

        """
        for param in self.model_parameters:
            param.reset_grad()

    def step(self, loss:Tensor)->None:
        """
        Performs a single step of gradient descent on model_parameters according to the loss function's gradients.
        
        Gradients are both averaged across a batch, with Tensor values modified accordingly.
        
        :param loss: A Tensor specifying a loss function. This loss needs to have taken the output from the same model which provided self.model_parameters
        :type loss: Tensor

        """
        assert isinstance(loss, Tensor), "the loss must be a Tensor"
        for i, param in enumerate(self.model_parameters):
            # if param.grad.shape != loss.weights[i].grad.shape:
                # print(param, param.grad.shape, loss.weights[i].grad.shape)
            param.grad      = np.mean(loss.weights[i].grad, axis=0, keepdims=True)
            param.value     = param.value - self.lr * param.grad

    def step_single(self, loss:Tensor, batch_size, modify:bool=False)->None:
        """
        This function performing gradient descent on arbitrary batch sizes by allowing for any number of gradient updates before values are updated.

        The single step of gradient descent is split into two components.
            1. Model parameter gradients are adjusted according to the average of the loss gradients. This is set when modify=False.
            2. Model parameter values are updated. This is set when modify=True

        :param loss: A Tensor specifying a loss function. This loss needs to have taken the output from the same model which provided self.model_parameters
        :type loss: Tensor
        :param batch_size: The final batch_size to average gradients over.
        :type batch_size: int
        :param modify: Whether or not to modify the model values.
        :type modify: bool, defaults to False.

        """
        assert isinstance(loss, Tensor), "loss must be a Tensor"
        assert isinstance(batch_size, int), "batch_size must be an int"
        assert isinstance(modify, bool), "modify must be a bool"
        if not modify:
            for i, param in enumerate(self.model_parameters):
                param.grad      += np.mean(loss.weights[i].grad, axis=0, keepdims=True)
        elif modify:
            for i, param in enumerate(self.model_parameters):
                param.value     -=  self.lr * param.grad/batch_size


class SGD_Momentum:
    """
    Gradient Descent with Momentum.
    """
    def __init__(self, model_parameters:list, beta:float=0.9, lr:float=1e-5):
        """
        Initializes the SGD with momentum optimizer.

        :param model_parameters: This is a list of Tensors specifying the pre-order traversal of a Tensor's computational graph. This is given either from Tensor.create_graph[1] or for models subclassing Module as model.params
        :type model_parameters: list
        :param beta: the beta momentum parameter to use
        :type beta: float. Defaults to 0.9.
        :param lr: the learning rate for SGD
        :type lr: float. Defaults to 1e-5.

        """
        assert isinstance(model_parameters, list), "model_parameters must be a list of Tensors"
        assert len(model_parameters) > 0, "The provided model_parameters contains no Tensors"
        assert isinstance(lr, float), "the lr must be a float"
        assert isinstance(beta, float), "beta must be a float"
        assert beta>=0 and beta<= 1, "beta must be in [0,1]"
        self.model_parameters   = model_parameters
        self.model_momentums    = copy.deepcopy(model_parameters)
        self.zero_momentums()
        self.beta               = beta
        self.lr                 = lr

    def zero_grad(self):
        for param in self.model_parameters:
            param.reset_grad()

    def zero_momentums(self):
        for param in self.model_momentums:
            param.reset_grad()
            param.value = None

    def step(self, loss:Tensor):
        """
        Performs a single step of gradient descent with momentum on model_parameters according to the loss function's gradients.
        
        Gradients are both averaged across a batch, with Tensor values modified accordingly.
        
        :param loss: A Tensor specifying a loss function. This loss needs to have taken the output from the same model which provided self.model_parameters
        :type loss: Tensor

        """
        assert isinstance(loss, Tensor), "loss must be a Tensor"
        for i, param in enumerate(self.model_parameters):
            # if param.grad.shape != loss.weights[i].grad.shape:
                # print(param, param.grad.shape, loss.weights[i].grad.shape)
            param.grad                      = np.mean(loss.weights[i].grad, axis=0, keepdims=True)
            self.model_momentums[i].grad    = self.beta * self.model_momentums[i].grad + (1-self.beta) * param.grad
            param.value                     = param.value - self.lr * self.model_momentums[i].grad

    def step_single(self, loss, batch_size, modify:bool=False):
        """
        This function performing gradient descent on arbitrary batch sizes by allowing for any number of gradient updates before values are updated.

        The single step of gradient descent is split into two components.
            1. Model parameter gradients are adjusted according to the average of the loss gradients. This is set when modify=False.
            2. Model parameter values are updated. This is set when modify=True

        :param loss: A Tensor specifying a loss function. This loss needs to have taken the output from the same model which provided self.model_parameters
        :type loss: Tensor
        :param batch_size: The final batch_size to average gradients over.
        :type batch_size: int
        :param modify: Whether or not to modify the model values.
        :type modify: bool, defaults to False.
        """
        assert isinstance(loss, Tensor), "loss must be a Tensor"
        assert isinstance(batch_size, int), "batch_size must be an int"
        assert isinstance(modify, bool), "modify must be a bool"
        if not modify:
            for i, param in enumerate(self.model_parameters):
                param.grad      += np.mean(loss.weights[i].grad, axis=0, keepdims=True)
        elif modify:
            for i, param in enumerate(self.model_parameters):
                self.model_momentums[i].grad= self.beta * self.model_momentums[i].grad + (1-self.beta) * param.grad/batch_size
                param.value                 =  param.value - self.lr * self.model_momentums[i].grad


class RMSProp:
    """RMS Prop."""
    def __init__(self, model_parameters:list, beta:float=0.9, lr:float=1e-5):
        """
        Initializes the RMS Prop.

        :param model_parameters: This is a list of Tensors specifying the pre-order traversal of a Tensor's computational graph. This is given either from Tensor.create_graph[1] or for models subclassing Module as model.params
        :type model_parameters: list
        :param beta: the beta parameter to use in RMSProp.
        :type beta: float. Defaults to 0.9.
        :param lr: the learning rate.
        :type lr: float. Defaults to 1e-5.

        """
        assert isinstance(model_parameters, list), "model_parameters must be a list of Tensors"
        assert len(model_parameters) > 0, "The provided model_parameters contains no Tensors"
        assert isinstance(lr, float), "the lr must be a float"
        assert isinstance(beta, float), "beta must be a float"
        assert beta>=0 and beta<= 1, "beta must be in [0,1]"
        self.model_parameters   = model_parameters
        self.model_vs           = copy.deepcopy(model_parameters)
        self.zero_vs()
        self.beta               = beta
        self.lr                 = lr
        self.eps                = 1e-8

    def zero_grad(self):
        for param in self.model_parameters:
            param.reset_grad()

    def zero_vs(self):
        for param in self.model_vs:
            param.reset_grad()
            param.value = None

    def step(self, loss:Tensor):
        """
        Performs a single step of RMSProp on model_parameters according to the loss function's gradients.
        
        Gradients are both averaged across a batch, with Tensor values modified accordingly.
        
        :param loss: A Tensor specifying a loss function. This loss needs to have taken the output from the same model which provided self.model_parameters
        :type loss: Tensor
        """
        assert isinstance(loss, Tensor), "loss must be a Tensor"
        for i, param in enumerate(self.model_parameters):
            param.grad                      = np.sum(loss.weights[i].grad, axis=0, keepdims=True)
            self.model_vs[i].grad           = self.beta * self.model_vs[i].grad + (1-self.beta) * (param.grad**2)
            param.value                     = param.value - (self.lr*param.grad)/(np.sqrt(self.model_vs[i].grad) + self.eps)

    def step_single(self, loss, batch_size, modify:bool=False):
        """
        This function performing gradient descent on arbitrary batch sizes by allowing for any number of gradient updates before values are updated.

        The single step of gradient descent is split into two components.
            1. Model parameter gradients are adjusted according to the average of the loss gradients. This is set when modify=False.
            2. Model parameter values are updated. This is set when modify=True

        :param loss: A Tensor specifying a loss function. This loss needs to have taken the output from the same model which provided self.model_parameters
        :type loss: Tensor
        :param batch_size: The final batch_size to average gradients over.
        :type batch_size: int
        :param modify: Whether or not to modify the model values.
        :type modify: bool, defaults to False.
        """
        assert isinstance(loss, Tensor), "loss must be a Tensor"
        assert isinstance(batch_size, int), "batch_size must be an int"
        assert isinstance(modify, bool), "modify must be a bool"
        if not modify:
            for i, param in enumerate(self.model_parameters):
                param.grad      += np.sum(loss.weights[i].grad, axis=0, keepdims=True)
        elif modify:
            for i, param in enumerate(self.model_parameters):
                self.model_vs[i].grad           = self.beta * self.model_vs[i].grad + (1-self.beta) * ((param.grad)**2)
                param.value                     = param.value - (self.lr/np.sqrt(self.model_vs[i].grad + self.eps)) * (param.grad)


class Adam:
    """Adam Optimizer."""
    def __init__(self, model_parameters:list, beta1:float=0.9, beta2:float=0.999, eps=1e-8, lr:float=1e-5):
        """
        Initializes Adam.

        :param model_parameters: This is a list of Tensors specifying the pre-order traversal of a Tensor's computational graph. This is given either from Tensor.create_graph[1] or for models subclassing Module as model.params
        :type model_parameters: list
        :param beta1: the beta1 parameter to use
        :type beta1: float. Defaults to 0.9.
        :param beta2: the beta2 parameter to use
        :type beta2: float. Defaults to 0.999.
        :param eps: the epsilon to use
        :type eps: float. Defaults to 1e-8.
        :param lr: the learning rate.
        :type lr: float. Defaults to 1e-5.

        """
        assert isinstance(model_parameters, list), "model_parameters must be a list of Tensors"
        assert len(model_parameters) > 0, "The provided model_parameters contains no Tensors"
        assert isinstance(lr, float), "the lr must be a float"
        assert isinstance(beta1, float), "beta must be a float"
        assert beta1>=0 and beta1<= 1, "beta must be in [0,1]"
        assert isinstance(beta2, float), "beta must be a float"
        assert beta2>=0 and beta2<= 1, "beta must be in [0,1]"
        assert isinstance(eps, float), "beta must be a float"
        self.model_parameters   = model_parameters
        self.model_momentums    = copy.deepcopy(model_parameters)
        self.model_vs           = copy.deepcopy(model_parameters)
        self.zero_adam()
        self.beta1              = beta1
        self.beta2              = beta2
        self.lr                 = lr
        self.eps                = eps
        self.iter_num           = 0

    def zero_grad(self):
        """
        Resets the model parameter gradients.
        """
        for param in self.model_parameters:
            param.reset_grad()

    def zero_adam(self):
        """
        Resets the momentums and variances stored by Adam for each model parameter.
        """
        for param in self.model_vs:
            param.reset_grad()
            param.value = None
        for param in self.model_momentums:
            param.reset_grad()
            param.value = None

    def step(self, loss:Tensor):
        """
        Performs a single step of Adam on model_parameters according to the loss function's gradients.
        
        Gradients are both averaged across a batch, with Tensor values modified accordingly.
        
        :param loss: A Tensor specifying a loss function. This loss needs to have taken the output from the same model which provided self.model_parameters
        :type loss: Tensor

        """
        assert isinstance(loss, Tensor), "loss must be a Tensor"
        self.iter_num += 1
        for i, param in enumerate(self.model_parameters):
            param.grad                          = np.sum(loss.weights[i].grad, axis=0, keepdims=True)
            self.model_momentums[i].grad        = (self.beta1 * self.model_momentums[i].grad) + (1-self.beta1) * param.grad
            self.model_vs[i].grad               = (self.beta2 * self.model_vs[i].grad) + (1-self.beta2) * (param.grad**2)
            corr_m                              = self.model_momentums[i].grad/(1-self.beta1**self.iter_num)
            corr_v                              = self.model_vs[i].grad/(1-self.beta2**self.iter_num)
            param.value                         = param.value - (self.lr*corr_m)/(np.sqrt(corr_v) + self.eps)

    def step_single(self, loss, batch_size, modify:bool=False):
        """
        Perform gradient descent on a loss, with control over value modification.

        This function performing gradient descent on arbitrary batch sizes by allowing for any number of gradient updates before values are updated.

        The single step of gradient descent is split into two components.
            1. Model parameter gradients are adjusted according to the average of the loss gradients. This is set when modify=False.
            2. Model parameter values are updated. This is set when modify=True

        :param loss: A Tensor specifying a loss function. This loss needs to have taken the output from the same model which provided self.model_parameters
        :type loss: Tensor
        :param batch_size: The final batch_size to average gradients over.
        :type batch_size: int
        :param modify: Whether or not to modify the model values.
        :type modify: bool, defaults to False.

        """
        assert isinstance(loss, Tensor), "loss must be a Tensor"
        assert isinstance(batch_size, int), "batch_size must be an int"
        assert isinstance(modify, bool), "modify must be a bool"
        if not modify:
            for i, param in enumerate(self.model_parameters):
                param.grad                     += np.sum(loss.weights[i].grad, axis=0, keepdims=True)

        else:
            for i, param in enumerate(self.model_parameters):
                self.iter_num                  += 1
                self.model_momentums[i].grad    = (self.beta1 * self.model_momentums[i].grad) + (1-self.beta1) * param.grad
                self.model_vs[i].grad           = (self.beta2 * self.model_vs[i].grad) + (1-self.beta2) * (param.grad**2)
                corr_m                          = self.model_momentums[i].grad/(1-self.beta1**self.iter_num)
                corr_v                          = self.model_vs[i].grad/(1-self.beta2**self.iter_num)
                param.value                     = param.value - (self.lr*corr_m)/(np.sqrt(corr_v) + self.eps)




