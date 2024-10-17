
import numpy as np
import copy

class SGD:
    def __init__(self, model_parameters:list, lr:float=1e-5):
        self.model_parameters   = model_parameters
        self.lr                 = lr

    def zero_grad(self):
        for param in self.model_parameters:
            param.reset_grad()

    def step(self, loss):
        for i, param in enumerate(self.model_parameters):
            # if param.grad.shape != loss.weights[i].grad.shape:
                # print(param, param.grad.shape, loss.weights[i].grad.shape)
            param.grad      = np.mean(loss.weights[i].grad, axis=0, keepdims=True)
            param.value     = param.value - self.lr * param.grad

    def step_single(self, loss, batch_size, modify:bool=False):
        if not modify:
            for i, param in enumerate(self.model_parameters):
                param.grad      += np.mean(loss.weights[i].grad, axis=0, keepdims=True)
        elif modify:
            for i, param in enumerate(self.model_parameters):
                param.value     -=  self.lr * param.grad/batch_size


class SGD_Momentum:
    def __init__(self, model_parameters:list, beta:float=0.9, lr:float=1e-5):
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

    def step(self, loss):
        for i, param in enumerate(self.model_parameters):
            # if param.grad.shape != loss.weights[i].grad.shape:
                # print(param, param.grad.shape, loss.weights[i].grad.shape)
            param.grad                      = np.mean(loss.weights[i].grad, axis=0, keepdims=True)
            self.model_momentums[i].grad    = self.beta * self.model_momentums[i].grad + (1-self.beta) * param.grad
            param.value                     = param.value - self.lr * self.model_momentums[i].grad

    def step_single(self, loss, batch_size, modify:bool=False):
        if not modify:
            for i, param in enumerate(self.model_parameters):
                param.grad      += np.mean(loss.weights[i].grad, axis=0, keepdims=True)
        elif modify:
            for i, param in enumerate(self.model_parameters):
                self.model_momentums[i].grad= self.beta * self.model_momentums[i].grad + (1-self.beta) * param.grad/batch_size
                param.value                 =  param.value - self.lr * self.model_momentums[i].grad


class RMSProp:
    def __init__(self, model_parameters:list, beta:float=0.9, lr:float=1e-5):
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

    def step(self, loss):
        for i, param in enumerate(self.model_parameters):
            param.grad                      = np.mean(loss.weights[i].grad, axis=0, keepdims=True)
            self.model_vs[i].grad           = self.beta * self.model_vs[i].grad + (1-self.beta) * (param.grad**2)
            param.value                     = param.value - (self.lr/np.sqrt(self.model_vs[i].grad + self.eps)) * param.grad

    def step_single(self, loss, batch_size, modify:bool=False):
        if not modify:
            for i, param in enumerate(self.model_parameters):
                param.grad      += np.mean(loss.weights[i].grad, axis=0, keepdims=True)
        elif modify:
            for i, param in enumerate(self.model_parameters):
                self.model_vs[i].grad           = self.beta * self.model_vs[i].grad + (1-self.beta) * ((param.grad)**2)
                param.value                     = param.value - (self.lr/np.sqrt(self.model_vs[i].grad + self.eps)) * (param.grad)


class Adam:
    def __init__(self, model_parameters:list, beta1:float=0.9, beta2:float=0.999, eps=1e-8, lr:float=1e-5):
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
        for param in self.model_parameters:
            param.reset_grad()

    def zero_adam(self):
        for param in self.model_vs:
            param.reset_grad()
            param.value = None
        for param in self.model_momentums:
            param.reset_grad()
            param.value = None

    def step(self, loss):
        self.iter_num += 1
        for i, param in enumerate(self.model_parameters):
            param.grad                          = np.sum(loss.weights[i].grad, axis=0, keepdims=True)
            self.model_momentums[i].grad        = (self.beta1 * self.model_momentums[i].grad) + (1-self.beta1) * param.grad
            self.model_vs[i].grad               = (self.beta2 * self.model_vs[i].grad) + (1-self.beta2) * (param.grad**2)
            corr_m                              = self.model_momentums[i].grad/(1-self.beta1**self.iter_num)
            corr_v                              = self.model_vs[i].grad/(1-self.beta2**self.iter_num)
            param.value                         = param.value - (self.lr*corr_m)/(np.sqrt(corr_v) + self.eps)

    def step_single(self, loss, batch_size, modify:bool=False):
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




