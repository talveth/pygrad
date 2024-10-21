
from typing import Union

import numpy as np
from .numba_ops import softmax_grad
import copy


def array(*args, **kwargs):
    return Tensor(*args, **kwargs)


class Tensor:
    def __init__(self, value:Union[list, np.ndarray], label:str="",  dtype=np.float64, learnable:bool=True, leaf:bool=False, _prev:tuple=()) -> None:
        """
        Initializes a Tensor. 
        
        A Tensor at all times holds:
            - A value assigned to it indicating its value.
            - A gradient of the same shape as its value indicating its gradient.
            - A function indicating how to pass a gradient to its children Tensors.
            - A computational graph for all Tensors that eventually resulted in the Tensor.
            
        Tensors store operations performed, allowing them to calculate the gradients of all Tensors in their 
        computational graph via .backward().
        
        Initialization:
            value: The input Tensor value. Is either a numeric value or a list/np.ndarray. 
                Complex or boolean values are not supported. Value is automatically recast to dtype.
            label: A string giving the Tensor an identifiable name. Defaults to "".
            dtype: The dtype to cast the input value. Must be one of np.float16, np.float32, np.float64, np.float128.
            
            (optional)
            learnable: A boolean indicating whether or not to compute gradients for _prev Tensors this Tensor has.
                    Setting this to False means the computational graph will stop at this node.
                    This node will still have gradients computed.
            leaf: A boolean indicating if the Tensor is to be considered a leaf node in the computational graph.      
                Leaf nodes will have gradients tracked, but won't appear as a weight in self.weights.
            
            _prev: An empty tuple or a tuple of Tensor objects, referencing objects to pass gradients 
                too when doing a backwards pass. _prev is automatically filled when performing a 
                tensor method, manual specification is not necessary.        
        """
        
        self._is_valid_value(value)
        if not isinstance(_prev, tuple):
            raise TypeError(f"Expected '_prev' to be of type 'tuple', but got {type(_prev).__name__}")
        for val in _prev:
            assert isinstance(val, Tensor), f"Expected _prev values to be Tensor objects, for {type(val).__name__}"
        if not isinstance(label, str):
            raise TypeError(f"Expected 'label' to be of type 'str', but got {type(label).__name__}")
        if not isinstance(learnable, bool):
            raise TypeError(f"Expected 'learnable' to be of type 'bool', but got {type(learnable).__name__}")
        if not isinstance(leaf, bool):
            raise TypeError(f"Expected 'leaf' to be of type 'bool', but got {type(leaf).__name__}")      
        if not isinstance(dtype(), (np.float16, np.float32, np.float64, np.float128)):
            raise TypeError(f"Expected 'dtype' to be of type (np.float16,np.float32,np.float64,np.float128), but got {dtype}")

        self.dtype          = dtype
        self.value          = np.array(value, dtype=self.dtype)
        self.learnable      = learnable
        self.leaf           = leaf
        self.shape          = self.value.shape
        self.label          = label
        self._prev:tuple    = _prev
        self.topo           = None
        self.weights        = None
        self.grad           = np.zeros_like(self.value).view(self.dtype)
        self.bpass          = lambda : None
        # self.n_conv_calls   = 0
        np.seterr(all='ignore')

    def _is_valid_value(self, value, with_tensor:bool=False)->tuple:
        if not isinstance(value, self._valid_values(with_tensor)):
            raise TypeError(f"Expected 'value' to be of type 'int,float,np.integer,np.floating,np.ndarray,list', but got {type(value).__name__}")

    def _valid_values(self, with_tensor:bool=False):
        if not with_tensor:
            return (int, float, np.integer, np.floating, np.ndarray, list)
        else:
            return (int, float, np.integer, np.floating, np.ndarray, list, Tensor)

    def __repr__(self) -> str:
        if self.shape == ():
            return f"Tensor=(value={self.value}, name={self.label}, shape={self.shape}, dtype={self.dtype})"
        else:
            return f"Tensor=(name={self.label}, shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, idcs):
        return self.value[idcs]

    def reset_grad(self):
        """Resets the gradient of the Tensor."""
        self.grad  = np.zeros_like(self.value).view(self.dtype)

    def new_value(self, x):
        """Assigns a new value to the Tensor and resets gradients without changing computational graphs."""
        self._is_valid_value(x)
        self.value = x
        self.shape = self.value.shape
        self.grad  = np.zeros_like(self.value).view(self.dtype)

    def _determine_broadcasting(self, val1, val2)->tuple[np.ndarray, np.ndarray]:
        # determines the broadcasted shapes of inputs val1 and val2
        broadcast   = np.broadcast_shapes(val1.shape, val2.shape)
        sh1         = (1,) * (len(broadcast) - len(val1.shape)) + val1.shape
        sh2         = (1,) * (len(broadcast) - len(val2.shape)) + val2.shape
        shape1_broadcasts = np.array(broadcast)//np.array(sh1)
        shape2_broadcasts = np.array(broadcast)//np.array(sh2)
        return shape1_broadcasts, shape2_broadcasts

    def _broadcast_addgrad(self, grad_to_change, new_grad):
        sh1b, _     = self._determine_broadcasting(grad_to_change, new_grad)
        # across all broadcast dimensions, sum new_grad gradient
        for i in range(len(sh1b)):
            if sh1b[i] != 1:
                new_grad = np.sum(new_grad, axis=i, keepdims=True)
        # reshape new_grad to be the shape of grad_to_change
        s1, s2 = grad_to_change.shape, new_grad.shape
        if s1 == ():
            new_grad = np.sum(new_grad)
        elif len(s2)>len(s1):
            diff = len(s2)-len(s1)
            new_grad = new_grad.reshape(s2[diff:])
        return new_grad

    def __add__(self, other):
        self._is_valid_value(other, with_tensor=True)
        if isinstance(other, (float, int, np.integer, np.floating)):
            other   = np.array(other)
            other   = Tensor(other, learnable=False, leaf=True)
        new = Tensor(value=self.value + other.value, _prev=(self, other), leaf=True)

        def bpass():
            other.grad += self._broadcast_addgrad(other.grad, new.grad.copy())
            self.grad  += self._broadcast_addgrad(self.grad, new.grad.copy())
        new.bpass                   = bpass
        return new

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        new = Tensor(value=-1.0*self.value, _prev=(self,), leaf=True)
        def bpass():
            self.grad               = self.grad + -1.0 * new.grad
        new.bpass = bpass
        return new

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def sum(self, axis:Union[None,int,tuple]=None, keepdims:bool=False):
        """
        Returns a Tensor with value equal to the sum of all the Tensor values.

        If over_batch == True, the first dimension is assumed to be the batch 
        dimension, and instead the sum is taken over the other dimensions.
        
        over_batch: boolean
        axis: None, int, or tuple of ints, suitable for self.shape. 
                    If None, sum is performed over all axes.
        keepdims:bool, False
        """
        assert isinstance(keepdims, bool), "keepdims must be a boolean."
        assert isinstance(axis, (int, tuple, type(None))), "axis must be None, an int, or a tuple."

        if axis is None:
            axis = tuple([i for i in range(len(self.shape))])
        new_val             = np.sum(self.value, axis=axis, keepdims=keepdims)
        new                 = Tensor(value=new_val, _prev=(self,), learnable=True, leaf=True)
        def bpass():
            if keepdims:
                self.grad  += np.einsum('...,...->...', np.ones(self.shape), new.grad, optimize='optimal')
            else:
                new_grad    = np.expand_dims(new.grad, axis=axis)
                self.grad  += np.einsum('...,...->...', np.ones(self.shape), new_grad, optimize='optimal')
            # print(new.grad.shape, self.shape, self.grad.shape)
            # self_grad   = 
            # self.grad  += 
            # self.grad  += np.einsum('...,...->...', np.ones(self.shape), new.grad, optimize='optimal')
        new.bpass       = bpass
        return new

    def __pow__(self, n:int):
        new = Tensor(value=self.value**n, _prev=(self,), leaf=True)
        def bpass():
            self.grad                       += np.einsum('...,...->...', (n*self.value**(n-1)), new.grad, optimize='optimal')
        new.bpass = bpass
        return new

    def __matmul__(self, other):
        if isinstance(other, (np.ndarray)):
            other = Tensor(value=np.array(other, dtype=self.dtype), label="other", learnable=True, leaf=False)
        assert isinstance(other, Tensor), "other must be a Tensor"
        new_val         = self.value @ other.value
        new             = Tensor(value=new_val, _prev=(self, other), leaf=True)

        def bpass():
            # self.grad   = self.grad + new.grad @ np.transpose(other.value, (0,2,1))
            self.grad   += np.einsum('...bj,...kj->...bk', new.grad, other.value, optimize='optimal')
            # other.grad  = other.grad + np.transpose(self.value, (0,2,1)) @ new.grad
            other.grad  += np.einsum('...jb,...jk->...bk', self.value, new.grad, optimize='optimal')
        new.bpass       = bpass
        return new

    def reshape(self, shape:tuple):
        """
        Returns a a copy of the Tensor with the same data but reshaped shape.
        """
        assert isinstance(shape, tuple), "reshape input must be a tuple"
        new_val = np.reshape(self.value, shape)
        new     = Tensor(value=new_val, _prev=(self,), learnable=True, leaf=True)
        def bpass():
            self.grad   += np.reshape(new.grad, self.value.shape)
        new.bpass       = bpass
        return new

    def _invert_axes(self, axes:tuple):
        inverted = [0]*len(axes)
        for i, val in enumerate(axes):
            inverted[val] = i
        return tuple(inverted)

    def transpose(self, axes:tuple):
        """
        Returns a copy of the Tensor with the same data but transposed axes.
        """
        assert isinstance(axes, tuple), "axes must be a tuple"
        new_val     = np.transpose(self.value, axes)
        new         = Tensor(value=new_val, _prev=(self,), leaf=True)
        def bpass():
            self.grad   += np.transpose(new.grad, self._invert_axes(axes))
        new.bpass       = bpass
        return new

    def mean(self, axis:int=-1, keepdims:bool=True):
        """
        Returns the average of the Tensor's value along a given axis.
        
        axis: int, axis to mean along
        keepdims: bool, whether or not to keep the existing dimensions
        """
        assert isinstance(axis, int),       "axis must be an integer"
        assert isinstance(keepdims, bool),  "keepdims must be a boolean value"

        new_val = np.mean(self.value, axis=axis, keepdims=keepdims)
        new     = Tensor(value=new_val, _prev=(self,), learnable=True, leaf=True)

        def bpass():
            if keepdims:
                self.grad += (1/self.shape[axis]) * np.einsum('...,...->...', np.ones(self.shape), new.grad, optimize='optimal')
            else:
                new_grad = np.expand_dims(new.grad, axis=axis)
                self.grad += (1/self.shape[axis]) * np.einsum('...,...->...', np.ones(self.shape), new_grad, optimize='optimal')
        new.bpass = bpass
        return new

    def std(self, axis:int, keepdim=True):
        """
        Returns the standard deviation of the Tensor along the specified axis.
        Bias correction is not performed here.

        axis: int
        keepdim: bool
        """
        assert isinstance(axis, int),       "The specified axis must be an integer."
        assert isinstance(keepdim, bool),   ""
        N   = self.value.shape[axis]
        if N == 1: return 0 * self
        d2  = (self - self.mean(axis=axis,keepdims=True))**2  # abs is for complex `a`
        print(d2.value)
        var = d2.mean(axis=axis, keepdims=keepdim)            # no bias correction done here
        print(var.value)
        new = var**0.5
        return new

    def _broadcast_mulgrad(self, grad_to_change, new_grad):
        # """broadcasts"""
        sh1b, _     = self._determine_broadcasting(grad_to_change, new_grad)
        # across all broadcast dimensions, sum new_grad gradient
        for i in range(len(sh1b)):
            if sh1b[i] != 1:
                new_grad = np.sum(new_grad, axis=i, keepdims=True)
        # reshape new_grad to be the shape of grad_to_change
        s1, s2 = grad_to_change.shape, new_grad.shape
        if s1 == ():
            new_grad = np.sum(new_grad)
        elif len(s2)>len(s1):
            diff = len(s2)-len(s1)
            new_grad = new_grad.reshape(s2[diff:])
        return new_grad

    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            other = Tensor(other, learnable=True, leaf=True)
        assert isinstance(other, Tensor), "Ensure that the multiplier is a Tensor."
        
        if np.broadcast_shapes(self.shape, other.shape) != ():
            assert np.broadcast_shapes(self.shape, other.shape), "Shapes not broadcastable. Ensure shapes are broadcastable"
        
        new_val = np.einsum('...,...->...', self.value, other.value, optimize='optimal')
        new = Tensor(value=new_val, _prev=(self, other), leaf=True)

        def bpass():
            new_grad    = np.einsum('...,...->...', other.value, new.grad, optimize='optimal')
            self.grad  += self._broadcast_mulgrad(self.grad, new_grad)
            new_grad    = np.einsum('...,...->...', self.value, new.grad, optimize='optimal')
            other.grad  += self._broadcast_mulgrad(other.grad, new_grad)
          
        new.bpass = bpass
        return new

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return self**-1 * other

    @property
    def T(self):
        """
        Transposes the last two dimensions of self.
        """
        if len(self.shape) <= 1:
            return self
        new                          = Tensor(value=np.einsum('...ij->...ji', self.value), _prev=(self,), label=f"{self.label}.T", leaf=True)
        def bpass():
            self.grad               += np.einsum('...ij->...ji', new.grad, optimize='optimal')
        new.bpass                    = bpass
        return new

    def mask_idcs(self, mask_idcs:tuple, value:float=0.0):
        """
        Applies a mask to the Tensor via an array. Creates a copy of the Tensor.

        mask_idcs: tuple of indices from which to mask values of self.
        value: The mask value. Defaults to 0.0.
        """
        assert isinstance(mask_idcs, (tuple, np.ndarray)), "Ensure that mask_idcs is of type tuple."
        new_val             = copy.deepcopy(self.value)
        new_val[mask_idcs]  = value
        new                 = Tensor(value=new_val, _prev=(self,), leaf=True)

        def bpass():
            ones_arr        = np.ones_like(self.value)
            ones_arr[mask_idcs] = value
            self.grad      += ones_arr * new.grad
        new.bpass = bpass
        return new

    def relu(self):
        """
        Applies a point-wise ReLU to the Tensor. Outputs a new Tensor copy.
        """
        new = Tensor(value=np.maximum(0, self.value), _prev=(self,), leaf=True)
        def bpass():
            grad_mat                = np.copy(self.value)
            grad_mat[grad_mat<=0]   = 0.0
            grad_mat[grad_mat>0]    = 1.0
            self.grad               += np.einsum('...,...->...', grad_mat, new.grad, optimize='optimal')
        new.bpass                   = bpass
        return new

    def tanh(self):
        """
        Applies a point-wise tanh to the Tensor. Outputs a new Tensor copy.
        """
        ex, emx = np.exp(self.value), np.exp(-self.value)
        tanh_val= (ex-emx)/(ex+emx)
        new     = Tensor(value=tanh_val, _prev=(self,), leaf=True)
        def bpass():
            self.grad   += np.einsum('...,...->...', (1-tanh_val**2), new.grad, optimize='optimal')
        new.bpass = bpass
        return new

    def softmax(self):
        """
        Applies softmax to self. Softmax is performed on the last axis.
        
        self.shape has to be either 3 or 4 dimensional.
            - (B, H, W)
            - (B, O, H, W)

        Returns a copy of the Tensor, with the softmax'd value.
        """
        assert len(self.shape) in [3,4], f"Ensure len(self.shape) in [3, 4], got {len(self.shape)}"
        max_normalized_self = self.value - np.max(self.value, axis=-1, keepdims=True)
        numerator           = np.exp(max_normalized_self)
        denominator         = np.sum(numerator, axis=-1, keepdims=True)
        new_val             = np.divide(numerator, denominator) # (..., n, k)
        new_val[np.all(np.isclose(self.value, -1e9), axis=-1)] = 1/self.value.shape[-1]
        new                 = Tensor(value=new_val, _prev=(self,), leaf=True)    # 
        w                   = new_val.shape[-1]
        
        def bpass():
            if len(self.shape) == 4:
                def add_newaxis(arr, num_axes):
                    return arr[tuple([np.newaxis] * num_axes + [slice(None)] * arr.ndim)]
                
                np_eye          = add_newaxis(np.eye(w), len(new_val.shape)-2)
                diag_elem       = np.einsum('...hn,...inm->...hnm', new_val, np_eye, optimize='optimal')
                new_val_reshape = np.expand_dims(new_val, axis=-2)
                nondiags        = np.einsum('...ji,...jk->...ik', new_val_reshape, new_val_reshape, optimize='optimal')
                sum_grad        = diag_elem-nondiags
                new_grad        = np.transpose(new.grad[...,np.newaxis,:,:], (0,1,2,4,3))
                pre_self_grad       = sum_grad@new_grad
                idx                 = np.arange(pre_self_grad.shape[2])
                result              = np.transpose(pre_self_grad[:,:,idx,:,idx], (1,2,0,3))
                self.grad           += result

            elif len(self.shape) == 3:
                self.grad        += softmax_grad(new_val, new.grad)
            else:
                raise "ERROR, self.shape should be 3D or 4D"
        new.bpass           = bpass
        return new

    def sigmoid(self):
        """
        Applies sigmoid to self.

        Self has to be of shape (..., 1).
        """
        if self.shape != ():
            assert self.shape[-1] == 1,         "Ensure the input has shape (..., 1)"
        else:
            pass
        new_val             = 1/(1 + np.exp(-self.value))
        new                 = Tensor(value=new_val, _prev=(self,), leaf=True)
        def bpass():
            self.grad      += new_val * (1-new_val) * new.grad
        new.bpass           = bpass
        return new

    def log(self):
        """
        Applies the natural logarithm to self.

        Negative values will raise an error.
        """
        assert np.min(self.value) >= 0,      "Ensure the value of self is non-negative before logging"
        log_val             = np.log(self.value)
        new                 = Tensor(value=log_val, _prev=(self,), leaf=True)
        def bpass():
            multiplier      = 1/self.value
            self.grad      += multiplier * new.grad
        new.bpass           = bpass
        return new

    def conv2D(self, other):
        """
        Applies a 2D convolution on other using self as the kernel.
        Strides are set to 1 by default, with no padding.

        If self.shape = (1, out_channels, in_channels, kH, kW)
            other.shape = (bs, in_channels, H, W)
            output.shape = (bs, out_channels, H-kH+1, W-kW+1)

        other: 4D Tensor.
        """
        assert len(other.shape)                 == 4, "Ensure other.shape is 4D: (1, in_channels, H, W)"
        assert len(self.shape)                  == 5, "Ensure self.shape is  5D: (1, out_channels, in_channels, kH, kW)"
        assert other.shape[1] == self.shape[2],       "Ensure in_channels of self match in_channels of other."
        assert other.shape[-2] >= self.shape[-2],      f"Ensure the input {other.shape} is at least as large as the kernel {self.shape}."
        assert other.shape[-1] >= self.shape[-1],      f"Ensure the input {other.shape} is at least as large as the kernel {self.shape}."
        
        if self.shape[0] != other.shape[0]:
            self.new_value(np.repeat(self.value, other.shape[0], axis=0))

        # self.n_conv_calls                       += 1
        b, inC, H, W                            = other.shape
        b, outC, inC, kH, kW                    = self.shape
        output_np                               = np.zeros(shape=(b, outC, H-kH+1, W-kW+1))
        for i in range(H-kH+1):
            for j in range(W-kW+1):
                for c in range(outC):
                    output_np[:,c,i,j]  = np.einsum('i...->i', self.value[:,c,:,:,:] * other.value[:,:,i:i+kH,j:j+kW], optimize='optimal')
        new                             = Tensor(value=output_np, _prev=(self,other), label="Conv2D", leaf=True)

        def bpass():
            # using Pytorch convention here:
            # gradients are averaged across the out_channels arguments
            # if the same conv is called >1 times per forward pass, average the gradient over those number of times
            # average the other gradient with the number of out_channels
            
            # self.reset_grad()
            # other.reset_grad()
            for i in range(H-kH+1):
                for j in range(W-kW+1):
                    for c in range(outC):
                        self.grad[:,c,:,:,:]   += np.einsum('...,...->...', other.value[:,:,i:i+kH, j:j+kW], new.grad[:,c,i,j].reshape(-1,1,1,1), optimize='optimal')  # (b, in_channels, kH, kW)
                        temp_other_grad_a       = np.einsum('...,...->...', self.value, new.grad[:,c,i,j].reshape(-1,1,1,1,1), optimize='optimal')                     # (b,out_c, in_c, kH, kW)
                        other.grad[:,:,i:i+kH,j:j+kW] += np.sum(temp_other_grad_a, axis=(1))

            # self.grad                          *= self.n_conv_calls
            other.grad                         /= outC
        new.bpass                               = bpass
        return new

    def create_graph(self):
        """Creates an reverse-ordered topological graph."""
        topo = []
        weights = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build_topo(prev)
                if v.learnable:
                    topo.append(v)
                if not v.leaf:
                    weights.append(v)
        build_topo(self)
        return topo, weights

    def backward(self, reset_grad=True) -> list:
        """
        Computes the gradients of all Tensors in self's computation graph.
        Self is initialized with a gradient of 1, incrementing all children gradients by this multiplier.

        This method first creates two topological graphs of self.
            1. A backwards-pass graph including all Tensors contributing to self.
            2. The backwards-pass graph, now omitting all Tensors with leaf=True.
                This is useful for seeing the exact parameters contributing to the 
                current Tensor, ignoring any Tensors that were produced as intermediary
                values for producing the current Tensor.
        
        reset_grad: boolean, True means all gradients of the graph will be reset to 0.
                             False means they will remain untouched.
        """
        self.topo, self.weights = self.create_graph()
        if reset_grad:
            for node in self.topo:
                node.reset_grad()
        self.grad = np.ones_like(self.value, dtype=self.dtype)
        for node in reversed(self.topo):
            node.bpass()
