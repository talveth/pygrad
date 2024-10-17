
import numpy as np
from .tensor import Tensor
from typing import Union


PRECISION = np.float64


class ReLU:
    def __init__(self, label:Union[None,int,str]="ReLU"):
        self.label  = label
        self.value  = None
        self.tensor = None

    def __repr__(self) -> str:
        if self.tensor is None:
            raise "Unable to __repr__, do a forward pass first."
        return f"ReLU=(name={self.label})"

    def __call__(self, x):
        out = x.relu()
        out.label = self.label
        self.tensor = out
        return self.tensor


class Dropout:
    def __init__(self, rate:float=0.1):
        assert isinstance(rate, float),     ""
        assert rate >= 0 and rate <= 1,     ""
        self.rate = rate

    def __call__(self, x, training:bool=True):
        assert isinstance(x, Tensor),       ""
        assert isinstance(training, bool),  ""

        if training:
            n_points        = int(np.prod(x.shape)*self.rate)
            arr_indices     = np.unravel_index(np.random.choice(np.arange(0, np.prod(x.shape)), size=n_points, replace=False), x.shape)
            dropouted_pts   = x.mask_idcs(arr_indices)
            return dropouted_pts
        else:
            return x


class AddNorm:
    def __init__(self, gain:float=0.1, epsilon:float=1e-9, dtype=PRECISION):
        self.dtype      = dtype
        self.gain       = np.array([gain]).astype(self.dtype)
        self.epsilon    = epsilon

    def __call__(self, x:Tensor, skip:Tensor):
        assert isinstance(x, Tensor),       ""
        assert isinstance(skip, Tensor),    ""
        assert x.shape == skip.shape,       ""

        outsum          = x + skip
        mu              = outsum.mean(axis=-1,keepdims=True)
        sd              = outsum.std(axis=-1)
        outsum_normed   = (outsum - mu) * ((sd**2 + self.epsilon)**-0.5)
        return outsum_normed


class Linear: # Ax + b
    def __init__(self, i_dim:int, o_dim:int, label:Union[None,int,str]="Linear", bias:bool=True, dtype=PRECISION) -> None:
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

    def __call__(self, x):
        # x: (b>=1,n,m), self: (b>=1, m,o)
        # assert isinstance(x, Tensor), f"Make sure the input to this is of type Tensor, got {type(x)}"
        assert len(x.shape) == 3, "Make sure the shape of the input is (b>=1,h,w)"
        if self.W.shape[0] != x.shape[0]:
            self.W.new_value(np.repeat(self.W.value, x.shape[0], axis=0))
        if self.B is not None:
            if self.B.shape[0] != x.shape[0]:
                self.B.new_value(np.repeat(self.B.value, x.shape[0], axis=0))
            assert self.B.shape[0] == x.shape[0],               "ERROR, shape of self.B should be (x.shape[0], ...)"
        assert self.W.shape[0] == x.shape[0],                   "ERROR, shape of self.W should be (x.shape[0], ...)"

        if self.bias:
            out         = x@self.W + (self.B).reshape((self.B.shape[0], 1, self.B.shape[-1]))
        else:
            out         = x@self.W
        self.tensor = out
        out.label   = self.label
        return self.tensor

    def init_kaiming_uniform(self, component:str):
        # initializes the weights of a Tensor according to Kaiming Uniform initialization
        assert isinstance(component, str), "component must be a string" 
        assert component in ["W", "B"], "components must be one of 'W', 'B'"
        k = 1/self.i_dim
        if component == "W":
            return np.random.uniform(low=-k**0.5, high=k**0.5, size=(1, self.i_dim, self.o_dim)).astype(self.dtype)
        elif component == "B":
            return np.random.uniform(low=-k**0.5, high=k**0.5, size=(1, 1, self.o_dim)).astype(self.dtype)


class Softmax:
    def __init__(self, label:Union[None,int,str]="Softmax"):
        self.label  = label
        self.tensor = None

    def __repr__(self) -> str:
        return f"Softmax=(name={self.label})"

    def __call__(self, x):
        assert isinstance(x, (Tensor)), "Make sure the input is of type Tensor"
        out         = x.softmax()
        self.tensor = out
        out.label   = self.label
        return self.tensor
    
    def parameters(self):
        return [0]


class Flatten:
    def __init__(self, label:Union[None,int,str]="Flatten"):
        self.label  = label
        self.value  = None
        self.tensor = None

    def __repr__(self) -> str:
        return f"Flatten=(name={self.label})"

    def __call__(self, x):
        assert isinstance(x, (Tensor)), "Make sure the input is of type Tensor"
        out         = x.flatten()
        self.tensor = out
        out.label   = self.label
        return self.tensor
    
    def parameters(self):
        return [0]


class Conv2D:
    def __init__(self, o_dim:int, i_dim:int, kH:int, kW:int, label:Union[None,int,str]="Conv2D", bias:bool=True):
        self.o_dim  = o_dim
        self.i_dim  = i_dim
        self.kH     = kH
        self.kW     = kW
        self.bias   = bias
        self.label  = label
        self.shape  = ()
        self.tensor = None

        self.kw     = Tensor(value=self.init_kaiming_uniform("kW"), label=f"ConvW_{label}")
        if bias:
            self.kb = Tensor(value=self.init_kaiming_uniform("kB"), label=f"ConvB_{label}")
        self.grad = None

    def __repr__(self) -> str:
        return f"Conv2D=(name={self.label}, kernel={self.kw.shape}, bias={self.kb.shape}, output={self.shape})"

    def __call__(self, x):
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
