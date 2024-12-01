
"""
Module storing class-defined layers.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from .constants import PRECISION
from .tensor import Tensor


class Dropout:
    """
    Dropout Class with specified rate parameter.
    Randomly masks input values with a probability of rate.
    
    Rate defaults to 0.1.
    """
    def __init__(self, rate:float=0.1):
        assert isinstance(rate, float),     "rate must be of float type"
        assert rate >= 0 and rate <= 1,     "rate must be in [0,1]"
        self.rate = rate

    def __call__(self, x:Tensor, training:bool=True)->Tensor:
        assert isinstance(x, Tensor),       "Assert x is of type Tensor"
        assert isinstance(training, bool),  "Assert training is of type bool"

        if training:
            n_points        = int(np.prod(x.shape)*self.rate)
            arr_indices     = np.unravel_index(np.random.choice(np.arange(0, np.prod(x.shape)), size=n_points, replace=False), x.shape)
            dropouted_pts   = x.mask_idcs(arr_indices)
            return dropouted_pts
        else:
            return x


class AddNorm:
    """
    Performs AddNorm on an input x and skip connection value skip.
    The forward pass performs the following, outputting a Tensor:
    
    y = x + skip
    mu= mean of y
    sd= sd of y
    output = gain * (y-mu)/sd + bias

    gain defaults to 1.0.
    bias defaults to 0.0.
    """
    def __init__(self, gain:float=1.0, bias:float=0.0, epsilon:float=1e-9):
        assert isinstance(gain, float),     "gain must be of type float"
        assert gain <= 1 and gain >= 0,     "gain must be in [0,1]"
        assert isinstance(bias, float),     "bias must be of type float"
        assert bias <= 1 and bias >= 0,     "bias must be in [0,1]"
        assert isinstance(epsilon, float),  "epsilon must be a float"
        self.gain       = gain
        self.bias       = bias
        self.epsilon    = epsilon

    def __call__(self, x:Tensor, skip:Tensor)->Tensor:
        assert isinstance(x, Tensor),       "x must be a Tensor."
        assert isinstance(skip, Tensor),    "skip must be a Tensor"
        assert x.shape == skip.shape,       "x and skip must be of the same shape"

        outsum          = x + skip
        mu              = outsum.mean(axis=-1,keepdims=True)
        sd              = outsum.std(axis=-1)
        outsum_normed   = self.gain * (outsum - mu) * ((sd**2 + self.epsilon)**-0.5) + self.bias
        return outsum_normed


class Linear:
    """
    Linear 2D layer. Performs Wx + B on an input x.
    
    Inputs and outputs are in 3D: (bs, h, w)
    Weights are initialized using Kaiming Uniform initialization.
    """
    def __init__(self, i_dim:int, o_dim:int, bias:bool=True, label:Union[None,int,str]="Linear", dtype=PRECISION) -> None:
        """
        Initializes a Dense Linear Layer with Kaiming Uniform initialization.

        A Dense linear layer is Wx + B.

        W is initialized as a Tensor of shape (1, i_dim, o_dim); with the leading dimension indicating the batch dimension.
        if bias is True: B is initialized as a Tensor of shape (1, 1, o_dim); the leading dimension indicating the batch dimension.

        :param i_dim: The input data dimension to the layer.
        :type i_dim: int
        :param o_dim: The output data dimension of the layer.
        :type o_dim: int
        :param bias: Whether or not to include the bias term. Defaults to True.
        :type bias: bool.
        :param label: An optional label to give to the layer.
        :type label: None, int, str (defaults to "Linear")
        :param dtype: The data type of the weights and gradients. Defaults to np.float64.
        :type dtype: The data types allowable by the Tensor class.
        """
        assert isinstance(i_dim, int), "i_dim must be an integer"
        assert i_dim > 0,              "i_dim must be positive"
        assert isinstance(o_dim, int), "o_dim must be an integer"
        assert o_dim > 0,              "o_dim must be positive"
        assert isinstance(label, (int, str)) or label is None, "label must be None, int, or a str."
        assert isinstance(bias, bool), "bias must be a boolean"

        self.dtype  = dtype
        self.i_dim  = i_dim
        self.o_dim  = o_dim
        self.W      = Tensor(value=self.init_kaiming_uniform("W"), label=f"W_{label}", dtype=self.dtype)
        if bias:
            self.B  = Tensor(value=self.init_kaiming_uniform("B"), label=f"b_{label}", dtype=self.dtype)
        else:
            self.B  = None
        self.tensor = None
        self.label  = label
        self.bias   = bias

    def __repr__(self) -> str:
        return f"Linear=(name={self.label}, W shape={self.W.shape}, B shape={self.B.shape if self.B is not None else None}"
        # Output shape={self.tensor.shape})"

    def __call__(self, x:Tensor)->Tensor:
        # x: (b>=1,n,m), self: (b>=1, m,o)
        assert isinstance(x, Tensor), f"Make sure the input to this is of type Tensor, got {type(x)}"
        assert len(x.shape) == 3, "Make sure the shape of the input is (b>=1,h,w)"
        if self.W.shape[0] != x.shape[0]:
            self.W.new_value(np.repeat(self.W.value, x.shape[0], axis=0))
        if self.B is not None:
            if self.B.shape[0] != x.shape[0]:
                self.B.new_value(np.repeat(self.B.value, x.shape[0], axis=0))
            assert self.B.shape[0] == x.shape[0],               f"ERROR, shape of self.B should be (x.shape[0], ...), got {self.B.shape, x.shape}"
        assert self.W.shape[0] == x.shape[0],                    "ERROR, shape of self.W should be (x.shape[0], ...)"

        if self.bias:
            out         = x@self.W + (self.B).reshape((self.B.shape[0], 1, self.B.shape[-1]))
        else:
            out         = x@self.W
        self.tensor = out
        out.label   = self.label
        return self.tensor

    def init_kaiming_uniform(self, component:str)->np.ndarray:
        # initializes the weights of a Tensor according to Kaiming Uniform initialization
        assert isinstance(component, str), "component must be a string" 
        assert component in ["W", "B"], "components must be one of 'W', 'B'"
        k = 1/self.i_dim
        if component == "W":
            return np.random.uniform(low=-k**0.5, high=k**0.5, size=(1, self.i_dim, self.o_dim)).astype(self.dtype)
        elif component == "B":
            return np.random.uniform(low=-k**0.5, high=k**0.5, size=(1, 1, self.o_dim)).astype(self.dtype)


class Softmax:
    """
    Performs Softmax on an input.
    """
    def __init__(self, label:Union[None,int,str]="Softmax"):
        assert isinstance(label, (int, str)) or label is None, "Label has to be None, int, or str."
        self.label  = label
        self.tensor = None

    def __repr__(self) -> str:
        return f"Softmax=(name={self.label})"

    def __call__(self, x:Tensor)->Tensor:
        assert isinstance(x, (Tensor)), "Make sure the input is of type Tensor"
        out         = x.softmax()
        self.tensor = out
        out.label   = self.label
        return self.tensor
    
    def parameters(self):
        return [0]


class Flatten:
    """
    Flattens an input by reshaping it into a 1D Tensor.
    """
    def __init__(self, label:Union[None,int,str]="Flatten")->None:
        assert isinstance(label, (str, int)) or label is None, "Ensure label is None, str, or int."
        self.label  = label
        self.value  = None
        self.tensor = None

    def __repr__(self) -> str:
        return f"Flatten=(name={self.label})"

    def __call__(self, x:Tensor)->Tensor:
        assert isinstance(x, (Tensor)), "Make sure the input is of type Tensor"
        out         = x.reshape((x.shape[0], 1, -1))
        self.tensor = out
        out.label   = self.label
        return self.tensor

    def parameters(self):
        return [0]


class Conv2D:
    """
    Performs Conv2D from an input dimension i_dim to an output dimension o_dim using a kernel (kH, kW).

    Kernels are initialized using Kaiming Uniform initialization.
    Only single strides are performed. No output padding is performed.
    """
    def __init__(self, o_dim:int, i_dim:int, kH:int, kW:int, bias:bool=True, label:Union[None,int,str]="Conv2D", dtype=PRECISION)->None:
        """
        Initialization for the Conv2D class.

        Conv2D is a set of kernels, that calls on an input data x: Cx + B.
        
        C is the convolution, B is the bias.
        
        C is of shape (1, o_dim, i_dim, kH, kW), the leading dimension being the batch dimension.
        
        If bias is True:
            B is of shape (1, o_dim, 1, 1)
        
        Weights are initialized via Kaiming Uniform Initialization.

        :param o_dim: The output channel dimension of the convolution. This indicates the number of kernels to apply to the input.
        :type o_dim: int
        :param i_dim: The number of channels of the input.
        :type i_dim: int
        :param kH: the height of the kernel
        :type kH: int
        :param kW: the width of the kernel
        :type kW: int
        :param bias: Whether or not to include the bias term after performing the initial convolution. Defaults to true.
        :type bias: bool
        :param label: A label for the layer.
        :type label: None, int, or str. Defaults to "Conv2D"
        :param dtype: The data type of the weights and gradients. Defaults to np.float64.
        :type dtype: The data types allowable by the Tensor class.

        """
        assert isinstance(o_dim, int), "o_dim must be an integer"
        assert o_dim>0, "o_dim must be positive"
        assert isinstance(i_dim, int), "i_dim must be an integer"
        assert i_dim>0, "i_dim must be positive"
        assert isinstance(kH, int), "kH must be an integer"
        assert kH>0, "kH must be positive"
        assert isinstance(kW, int), "kW must be an integer"
        assert kW>0, "kW must be positive"
        assert isinstance(bias, bool), "bias must be a boolean"
        assert isinstance(label, (int, str)) or label is None, "label must be None, int, or str"

        self.o_dim  = o_dim
        self.i_dim  = i_dim
        self.kH     = kH
        self.kW     = kW
        self.bias   = bias
        self.label  = label
        self.shape  = ()
        self.tensor = None
        self.dtype  = dtype

        self.kw     = Tensor(value=self.init_kaiming_uniform("kW"), label=f"ConvW_{label}", dtype=self.dtype)
        if bias:
            self.kb = Tensor(value=self.init_kaiming_uniform("kB"), label=f"ConvB_{label}", dtype=self.dtype)
        self.grad = None

    def __repr__(self) -> str:
        return f"Conv2D=(name={self.label}, kernel={self.kw.shape}, bias={self.kb.shape}, output={self.shape})"

    def __call__(self, x:Tensor)->Tensor:
        assert isinstance(x, Tensor), "Make sure the input to this is of type Tensor"
        assert len(x.shape) == 4, "Make sure the shape of the input is (b>=1,c,h,w)"
        if self.kw.shape[0] != x.shape[0]:
            self.kw.new_value(np.repeat(self.kw.value, x.shape[0], axis=0))
            self.kb.new_value(np.repeat(self.kb.value, x.shape[0], axis=0))
        assert self.kw.shape[0] == x.shape[0] and self.kb.shape[0] == x.shape[0], "ERROR, shape of self.kw and self.kb should be (x.shape[0], ...)"
        if self.bias:
            out         = self.kw.conv2D(x) + self.kb
        else:
            out         = self.kw.conv2D(x)
        self.shape      = out.shape
        self.tensor     = out
        return self.tensor

    def init_kaiming_uniform(self, component:str):
        # initializes the weights of a Tensor according to Kaiming Uniform initialization
        assert isinstance(component, str), "component must be a string" 
        assert component in ["kW", "kB"], "components must be one of 'W', 'B'"
        n = self.i_dim
        for i in [self.kH, self.kW]:
            n *= i
        k = 1./np.sqrt(n)
        if component == "kW":
            return np.random.uniform(low=-k**0.5, high=k**0.5, size=(1, self.o_dim, self.i_dim, self.kH, self.kW))
        elif component == "kB":
            return np.random.uniform(low=-k**0.5, high=k**0.5, size=(1, self.o_dim, 1, 1))
        else: raise "ERROR"
