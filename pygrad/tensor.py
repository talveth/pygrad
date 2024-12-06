
"""
Module storing the main Tensor class and related methods.
"""

from __future__ import annotations
import warnings

import copy
from typing import Union

import numpy as np

from .constants import PRECISION
from .numba_ops import softmax_grad, conv2d_fwd, conv2d_bwd


warnings.simplefilter("once", category=UserWarning)


class Tensor:
    """
    The main Tensor object.
    """
    __slots__ = 'value', 'label', 'dtype', 'learnable', 'leaf', 'shape', '_prev', 'topo', 'weights', 'grad', 'bpass'

    def __init__(self, value:Union[list, np.ndarray], label:str="",  dtype=PRECISION, learnable:bool=True, leaf:bool=False, _prev:tuple=()) -> None:
        """
        Initializes a Tensor. 
        
        A Tensor at all times holds:
            - A value assigned to it indicating its value.
            - A gradient of the same shape as its value indicating its gradient.
            - A function indicating how to pass a gradient to its children Tensors.
            - A computational graph for all Tensors that eventually resulted in the Tensor.
            
        Tensors store operations performed, allowing them to calculate the gradients of all Tensors in their 
        computational graph via .backward().
        
        :param value: The input Tensor value. 
        :type value: Is either a numeric value or a list/np.ndarray. Complex or boolean values are not supported. Value is automatically recast to dtype.
        :param label: A string giving the Tensor an identifiable name. Defaults to "".
        :param dtype: The dtype to cast the input value. Must be one of np.bool, np.integer, np.floating.
        
        :param learnable: Optional. A boolean indicating whether or not to compute gradients for _prev Tensors this Tensor has.
                Setting this to False means the computational graph will stop at this node.
                This node will still have gradients computed.
        :param leaf: Optional. A boolean indicating if the Tensor is to be considered a leaf node in the computational graph.      
            Leaf nodes will have gradients tracked, but won't appear as a weight in self.weights.
        
        :param _prev: Optional. An empty tuple or a tuple of Tensor objects, referencing objects to pass gradients 
            too when doing a backwards pass. _prev is automatically filled when performing a 
            tensor method, manual specification is not necessary.        
        
        :return: A produced Tensor.
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
        if not isinstance(dtype(), (np.floating)):
            raise TypeError(f"Expected 'dtype' to be of type (np.floating), but got {dtype}")

        self.dtype          = dtype
        self.value          = np.array(value, dtype=self.dtype)
        self.learnable      = learnable
        self.leaf           = leaf
        self.shape          = self.value.shape
        self.label          = label
        self._prev:tuple    = _prev
        self.topo           = None
        self.weights        = None
        self.grad           = np.zeros_like(self.value, dtype=self.dtype)
        self.bpass          = lambda : None
        np.seterr(all='ignore')

    def _is_valid_value(self, value, with_tensor:bool=False)->tuple:
        if not isinstance(value, self._valid_values(with_tensor)):
            raise TypeError(f"Expected 'value' to be of type 'int,float,np.integer,np.floating,np.ndarray,list', but got {type(value).__name__}")

    def _valid_values(self, with_tensor:bool=False):
        if not with_tensor:
            return (int, float, np.integer, np.floating, np.ndarray, np.bool_, list)
        else:
            return (int, float, np.integer, np.floating, np.ndarray, np.bool_, list, Tensor)

    def __repr__(self) -> str:
        if self.shape == ():
            if self.label != "":
                return f"Tensor(name={self.label}, value={self.value}, shape={self.shape}, dtype={self.dtype})"
            else:
                return f"Tensor(value={self.value}, shape={self.shape}, dtype={self.dtype})"
        else:
            if self.label != "":
                return f"Tensor(name={self.label}, shape={self.shape}, dtype={self.dtype})"
            else:
                return f"Tensor(shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, idcs):
        """
        Fetches self.value[idcs]. Identical to NumPy syntax for fetching indices.
        """
        return self.value[idcs]

    def reset_grad(self)->None:
        """Resets the gradient of the Tensor to 0, maintaining all other attributes."""
        self.grad  = np.zeros_like(self.value, dtype=self.dtype)

    def new_value(self, x)->None:
        """
        Assigns a new value to the Tensor, Tensor.value = x, and resets gradients to 0 without changing computational graph topology.
        """
        self._is_valid_value(x)
        self.value = np.array(x, self.dtype)
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

    def _broadcast_addgrad(self, grad_to_change, new_grad)->np.ndarray:
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

    def __add__(self, other:Union[int, float, np.integer, np.floating, np.ndarray, Tensor])->Tensor:
        """
        Performs self.Tensor + other, returning a new Tensor object.

        :param other: the object to add with shape broadcastable to self.shape.
        :type other: (int, float, np.integer, np.floating, np.ndarray, list, Tensor)
        """
        self._is_valid_value(other, with_tensor=True)
        if isinstance(other, (float, int, np.integer, np.floating, np.ndarray)):
            other   = np.array(other, dtype=self.dtype)
            other   = Tensor(other, learnable=False, leaf=True, dtype=self.dtype)
        new = Tensor(value=self.value + other.value, _prev=(self, other), leaf=True, dtype=self.dtype)

        def bpass():
            other.grad += self._broadcast_addgrad(other.grad, new.grad.copy())
            self.grad  += self._broadcast_addgrad(self.grad, new.grad.copy())
        new.bpass                   = bpass
        return new

    def __radd__(self, other:Union[int, float, np.integer, np.floating, np.ndarray, Tensor])->Tensor:
        return self + other

    def __neg__(self)->Tensor:
        """
        Performs -1*self.Tensor, returning a new Tensor object.
        """
        new = Tensor(value=-1.0*self.value, _prev=(self,), leaf=True, dtype=self.dtype)
        def bpass():
            self.grad               = self.grad + -1.0 * new.grad
        new.bpass = bpass
        return new

    def __sub__(self, other:Union[int, float, np.integer, np.floating, np.ndarray, Tensor])->Tensor:
        return self + (-other)

    def __rsub__(self, other:Union[int, float, np.integer, np.floating, np.ndarray, Tensor])->Tensor:
        return (-self) + other

    def sum(self, axis:Union[None,int,tuple]=None, keepdims:bool=False)->Tensor:
        """
        Performs a summation on self.value according to axis chosen. Returns a new Tensor object.
        
        :param axis:     Determines which axis to sum self.value over. Defaults to None: summing over all axes.
        :type axis:      None (default), int, or tuple of ints.
        :param keepdims: Indicates whether to keep the current shape of self after summation.
        :type param:     bool, False (default).
        """
        assert isinstance(keepdims, bool), "keepdims must be a boolean."
        assert isinstance(axis, (int, tuple, type(None))), "axis must be None, an int, or a tuple."

        if axis is None:
            axis = tuple([i for i in range(len(self.shape))])
        new_val             = np.sum(self.value, axis=axis, keepdims=keepdims, dtype=self.dtype)
        new                 = Tensor(value=new_val, _prev=(self,), learnable=True, leaf=True, dtype=self.dtype)
        def bpass():
            if keepdims:
                self.grad  += np.einsum('...,...->...', np.ones(self.shape, dtype=self.dtype), new.grad, optimize='optimal', dtype=self.dtype)
            else:
                new_grad    = np.expand_dims(new.grad, axis=axis)
                self.grad  += np.einsum('...,...->...', np.ones(self.shape, dtype=self.dtype), new_grad, optimize='optimal', dtype=self.dtype)
        new.bpass       = bpass
        return new

    def __pow__(self, n:Union[int,float,np.integer,np.floating])->Tensor:
        """
        Raises the Tensor to a power of n.

        :param n: The power value to raise the current Tensor
        :type param: One of int,float,np.integer,np.floating.
        """
        assert isinstance(n, (int, float, np.integer, np.floating)), "The value `n` has to be of type int, float, np.integer, or np.floating."
        new = Tensor(value=self.value**n, _prev=(self,), leaf=True, dtype=self.dtype)
        def bpass():
            self.grad                       += np.einsum('...,...->...', (n*self.value**(n-1)), new.grad, optimize='optimal', dtype=self.dtype)
        new.bpass = bpass
        return new

    def __matmul__(self, other:Union[np.ndarray, Tensor])->Tensor:
        """
        Performs matrix multiplication with self and other: self@other.

        Matrix multiplication is performed between the last two dimensions of self and other, broadcasting all those remaining.

        :param other: The matrix to perform matrix multiplication against.
        :type param: Either a np.ndarray or Tensor of suitable shape.
        """
        if isinstance(other, (np.ndarray)):
            other = Tensor(value=np.array(other, dtype=self.dtype), label="other", learnable=True, leaf=False, dtype=self.dtype)
        assert isinstance(other, Tensor), "other must be a Tensor"
        new_val         = self.value @ other.value
        new             = Tensor(value=new_val, _prev=(self, other), leaf=True, dtype=self.dtype)

        def bpass():
            self.grad   += np.einsum('...bj,...kj->...bk', new.grad, other.value, optimize='optimal', dtype=self.dtype)
            other.grad  += np.einsum('...jb,...jk->...bk', self.value, new.grad, optimize='optimal', dtype=self.dtype)
        new.bpass       = bpass
        return new

    def reshape(self, shape:tuple)->Tensor:
        """
        Returns a new Tensor with Tensor.value.shape == shape.

        :param shape: A tuple indicating the new shape self.value has to take.
        :type param: tuple
        """
        assert isinstance(shape, tuple), "reshape input must be a tuple"
        new_val = np.reshape(self.value, shape)
        new     = Tensor(value=new_val, _prev=(self,), learnable=True, leaf=True, dtype=self.dtype)
        def bpass():
            self.grad   += np.reshape(new.grad, self.value.shape)
        new.bpass       = bpass
        return new

    def _invert_axes(self, axes:tuple)->tuple:
        inverted = [0]*len(axes)
        for i, val in enumerate(axes):
            inverted[val] = i
        return tuple(inverted)

    def transpose(self, axes:Union[None, tuple, list])->Tensor:
        """
        Returns a new Tensor with the same data but transposed axes.

        :param axes: If specified, it must be a tuple or list which contains a 
                     permutation of [0,1,…,N-1] where N is the number of axes of self. 
                     The ith axis of the returned array will correspond to the axis 
                     numbered axes[i] of the input. If not specified, defaults to 
                     the reverse of the order of the axes.
        """
        assert isinstance(axes, (tuple, list)) or axes is None, "axes must be None or a tuple or list"
        new_val     = np.transpose(self.value, axes)
        new         = Tensor(value=new_val, _prev=(self,), leaf=True, dtype=self.dtype)
        def bpass():
            self.grad   += np.transpose(new.grad, self._invert_axes(axes))
        new.bpass       = bpass
        return new

    def mean(self, axis:Union[int, tuple, None]=-1, keepdims:bool=True)->Tensor:
        """
        Returns a new Tensor with value being the average of self's value along a given axis.
        
        :param axis: The axis to perform a mean over.
        :type axis: None, int, tuple of ints
        :param keepdims: Whether or not to keep the existing dimensions. True is yes.
        :type keepdims: bool
        """
        assert isinstance(axis, (int, tuple)) or axis is None, "axis must be an integer, tuple of ints, or None"
        assert isinstance(keepdims, bool),  "keepdims must be a boolean value"

        new_val = np.mean(self.value, axis=axis, keepdims=keepdims, dtype=self.dtype)
        new     = Tensor(value=new_val, _prev=(self,), learnable=True, leaf=True, dtype=self.dtype)

        def bpass():
            if axis is None:
                c = 1/np.prod(self.shape)
            elif isinstance(axis, int):
                c = 1/self.shape[axis]
            else:
                c = 1/np.prod([self.shape[ax] for ax in axis])

            if keepdims:
                self.grad += c * np.einsum('...,...->...', np.ones(self.shape, dtype=self.dtype), new.grad, optimize='optimal', dtype=self.dtype)
            else:
                new_grad = np.expand_dims(new.grad, axis=axis)
                self.grad += c * np.einsum('...,...->...', np.ones(self.shape, dtype=self.dtype), new_grad, optimize='optimal', dtype=self.dtype)
        new.bpass = bpass
        return new

    def std(self, axis:int, keepdim=True)->Tensor:
        """
        Returns a new Tensor with value being the standard deviation of self along the specified axis.
        No bias correction performed.

        :param axis: axis over which to perform a standard deviation over
        :type axis: integer
        :param keepdim: whether or not to keep the axis dimension as self
        :type keepdim: bool (defaults to True)
        """
        assert isinstance(axis, int),       "The specified axis must be an integer."
        assert isinstance(keepdim, bool),   "Ensure keepdim is a boolean"
        N   = self.value.shape[axis]
        if N == 1: return 0 * self
        d2  = (self - self.mean(axis=axis,keepdims=True))**2  # abs is for complex `a`
        var = d2.mean(axis=axis, keepdims=keepdim)            # no bias correction done here
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

    def __mul__(self, other:Union[int, float, np.integer, np.floating, np.ndarray, Tensor])->Tensor:
        """
        Performs multiplication between the values of self and other. 
        If self and other are matrices, this is equivalent to the hadamard product.
        
        :param other: The value to multiply against. Must be broadcastable in shape to self.
        :type other:  One of int, float, np.integer, np.floating, np.ndarray, or Tensor.
        """
        if isinstance(other, (int, float, np.integer, np.floating, np.ndarray)):
            other = Tensor(other, learnable=True, leaf=True, dtype=self.dtype)
        assert isinstance(other, Tensor), "Ensure that the multiplier is a Tensor."
        
        if np.broadcast_shapes(self.shape, other.shape) != ():
            assert np.broadcast_shapes(self.shape, other.shape), "Shapes not broadcastable. Ensure shapes are broadcastable"
        
        new_val = np.einsum('...,...->...', self.value, other.value, optimize='optimal', dtype=self.dtype)
        new = Tensor(value=new_val, _prev=(self, other), leaf=True, dtype=self.dtype)

        def bpass():
            new_grad    = np.einsum('...,...->...', other.value, new.grad, optimize='optimal', dtype=self.dtype)
            self.grad  += self._broadcast_mulgrad(self.grad, new_grad)
            new_grad    = np.einsum('...,...->...', self.value, new.grad, optimize='optimal', dtype=self.dtype)
            other.grad  += self._broadcast_mulgrad(other.grad, new_grad)
          
        new.bpass = bpass
        return new

    def __rmul__(self, other:Union[int, float, np.integer, np.floating, np.ndarray, Tensor]):
        return self * other

    def __truediv__(self, other:Union[int, float, np.integer, np.floating, np.ndarray, Tensor]):
        return self * other**-1

    def __rtruediv__(self, other:Union[int, float, np.integer, np.floating, np.ndarray, Tensor]):
        return self**-1 * other

    @property
    def T(self) -> Tensor:
        """
        Transposes self.value.

        If self is 0 or 1 dimensional, no self is returned without modification.
        Otherwise, the last two dimensions of self are flipped.
        """
        if len(self.shape) <= 1:
            return self
        new                          = Tensor(value=np.einsum('...ij->...ji', self.value, dtype=self.dtype), _prev=(self,), 
                                              label=f"{self.label}.T", leaf=True, dtype=self.dtype)
        def bpass():
            self.grad               += np.einsum('...ij->...ji', new.grad, optimize='optimal', dtype=self.dtype)
        new.bpass                    = bpass
        return new

    def mask_idcs(self, mask_idcs:tuple, value:float=0.0)->Tensor:
        """
        Applies a mask to the Tensor via an array indicating indices of self.value. 
        Outputs a new Tensor.

        :param mask_idcs: tuple of indices from which to mask values of self.
        :type mask_idcs: tuple
        :param value: The mask value. Defaults to 0.0 indicating that chosen indices are now set to 0.
        :type value: float
        """
        assert isinstance(mask_idcs, (tuple, np.ndarray)), "Ensure that mask_idcs is of type tuple."
        assert isinstance(value, (float)), "Ensure value is of type float."
        new_val             = copy.deepcopy(self.value)
        new_val[mask_idcs]  = value
        new                 = Tensor(value=new_val, _prev=(self,), leaf=True, dtype=self.dtype)

        def bpass():
            ones_arr        = np.ones_like(self.value, dtype=self.dtype)
            ones_arr[mask_idcs] = value
            self.grad      += ones_arr * new.grad
        new.bpass = bpass
        return new

    def relu(self)->Tensor:
        """
        Applies a point-wise ReLU to the Tensor values. Outputs a new Tensor.
        """
        new = Tensor(value=np.maximum(0, self.value), _prev=(self,), leaf=True, dtype=self.dtype)
        def bpass():
            grad_mat                = np.copy(self.value)
            grad_mat[grad_mat<=0]   = 0.0
            grad_mat[grad_mat>0]    = 1.0
            self.grad               += np.einsum('...,...->...', grad_mat, new.grad, optimize='optimal', dtype=self.dtype)
        new.bpass                   = bpass
        return new

    def tanh(self)->Tensor:
        """
        Applies a point-wise tanh activation to the Tensor. Outputs a new Tensor.
        """
        ex, emx = np.exp(self.value), np.exp(-self.value)
        tanh_val= (ex-emx)/(ex+emx)
        new     = Tensor(value=tanh_val, _prev=(self,), leaf=True, dtype=self.dtype)
        def bpass():
            self.grad   += np.einsum('...,...->...', (1-tanh_val**2), new.grad, optimize='optimal', dtype=self.dtype)
        new.bpass = bpass
        return new

    def log(self)->Tensor:
        """
        Applies the natural logarithm to self.

        Negative values will raise an error.
        """
        # self.value[self.value<1e-6] = 1e-6
        assert np.min(self.value) >= 0,      f"Ensure the value of self is non-negative before logging, got {np.min(self.value)}"
        log_val             = np.log(self.value)
        new                 = Tensor(value=log_val, _prev=(self,), leaf=True, dtype=self.dtype)
        def bpass():
            self.grad      += new.grad/self.value
        new.bpass           = bpass
        return new

    def softmax(self)->Tensor:
        """
        Applies softmax to self. Softmax is performed on the last axis.
        
        self.shape has to be either 3 or 4 dimensional.
            - (B, H, W)
            - (B, O, H, W)

        Returns a copy of the Tensor, with the softmax'd value.
        """
        assert len(self.shape) in [3,4], f"Ensure len(self.shape) in [3, 4], got {len(self.shape)}"
        max_vals            = np.max(self.value, axis=-1, keepdims=True)
        new_val             = np.exp(self.value - (max_vals+np.log(np.sum(np.exp(self.value-max_vals), axis=-1, keepdims=True))))
        new                 = Tensor(value=new_val, _prev=(self,), leaf=True, dtype=self.dtype)    # 
        w                   = new_val.shape[-1]
        
        def bpass():
            if len(self.shape) == 4:
                def add_newaxis(arr, num_axes):
                    return arr[tuple([np.newaxis] * num_axes + [slice(None)] * arr.ndim)]
                
                np_eye          = add_newaxis(np.eye(w, dtype=self.dtype), len(new_val.shape)-2)
                diag_elem       = np.einsum('...hn,...inm->...hnm', new_val, np_eye, optimize='optimal', dtype=self.dtype); del np_eye
                nondiags        = np.einsum('...ji,...jk->...ik', np.expand_dims(new_val, axis=-2), np.expand_dims(new_val, axis=-2), optimize='optimal', dtype=self.dtype)
                sum_grad        = diag_elem-nondiags; del diag_elem
                new_grad        = np.transpose(new.grad[...,np.newaxis,:,:], (0,1,2,4,3))
                pre_self_grad       = sum_grad@new_grad
                idx                 = np.arange(pre_self_grad.shape[2])
                result              = np.transpose(pre_self_grad[:,:,idx,:,idx], (1,2,0,3))
                self.grad           += result

            elif len(self.shape) == 3:
                if self.dtype in [np.float16, np.float128]:
                    warnings.warn(f"Softmax grad calculations don't support {self.dtype}, casting to np.float64 for calculations.", category=UserWarning)
                    self.grad        += softmax_grad(new_val.astype(np.float64), new.grad.astype(np.float64)).astype(self.dtype)
                else:
                    self.grad        += softmax_grad(new_val, new.grad).astype(self.dtype)
            else:
                raise "ERROR, self.shape should be 3D or 4D"
        new.bpass           = bpass
        return new

    def softmax_log(self)->Tensor:
        """
        Computes .softmax().log() in one go. Use this if the former has numerical issues.
        
        self.shape has to be either 3 or 4 dimensional.
            - (B, H, W)
            - (B, O, H, W)

        Returns a copy of the Tensor, with the softmax'd value.
        """
        assert len(self.shape) in [3,4], f"Ensure len(self.shape) in [3, 4], got {len(self.shape)}"
        max_vals            = np.max(self.value, axis=-1, keepdims=True)
        new_val             = self.value - (max_vals+np.log(np.sum(np.exp(self.value-max_vals), axis=-1, keepdims=True)))
        new_val_exp         = np.exp(new_val)
        new                 = Tensor(value=new_val, _prev=(self,), leaf=True, dtype=self.dtype)

        def bpass():
            if len(self.shape) == 4:
                def add_newaxis(arr, num_axes):
                    return arr[tuple([np.newaxis] * num_axes + [slice(None)] * arr.ndim)]
                new_val         = new_val_exp
                np_eye          = add_newaxis(np.eye(new_val.shape[-1], dtype=self.dtype), len(new_val.shape)-2)
                diag_elem       = np.einsum('...hn,...inm->...hnm', new_val, np_eye, optimize='optimal', dtype=self.dtype); del np_eye
                nondiags            = np.einsum('...ji,...jk->...ik', np.expand_dims(new_val, axis=-2), np.expand_dims(new_val, axis=-2), optimize='optimal', dtype=self.dtype)
                sum_grad            = diag_elem-nondiags; del diag_elem
                new_grad            = np.transpose((new.grad/new_val)[...,np.newaxis,:,:], (0,1,2,4,3))
                pre_self_grad       = sum_grad@new_grad
                idx                 = np.arange(pre_self_grad.shape[2])
                result              = np.transpose(pre_self_grad[:,:,idx,:,idx], (1,2,0,3))
                self.grad           += result

            elif len(self.shape) == 3:
                if self.dtype in [np.float16, np.float128]:
                    warnings.warn(f"Softmax grad calculations don't support {self.dtype}, casting to np.float64 for calculations.", category=UserWarning)
                    new_val              = new_val_exp
                    self.grad        += softmax_grad(new_val.astype(np.float64), new.grad.astype(np.float64)).astype(self.dtype)
                else:
                    new_val              = new_val_exp
                    self.grad        += softmax_grad(new_val, new.grad/new_val).astype(self.dtype)
            else:
                raise "ERROR, self.shape should be 3D or 4D"
        new.bpass           = bpass
        return new


    def sigmoid(self)->Tensor:
        """
        Applies sigmoid activation to self, returning a new Tensor.

        self.value has to be of shape (..., 1).
        """
        if self.shape != ():
            assert self.shape[-1] == 1,         "Ensure the input has shape (..., 1)"

        new_val             = 1/(1 + np.exp(-self.value))
        new                 = Tensor(value=new_val, _prev=(self,), leaf=True, dtype=self.dtype)
        def bpass():
            self.grad      += new_val * (1-new_val) * new.grad
        new.bpass           = bpass
        return new

    def conv2D(self, other:Tensor)->Tensor:
        """
        Applies a 2D convolution on other using self as the kernel.
        Strides are set to 1 by default, with no padding.
        Output a new Tensor.

        If self.shape = (1, out_channels, in_channels, kH, kW)
            other.shape = (bs, in_channels, H, W)
            output.shape = (bs, out_channels, H-kH+1, W-kW+1)

        :param other: A 4D Tensor.
        :type other: Tensor.
        """
        assert isinstance(other, Tensor),   "Other must be a Tensor."
        assert len(other.shape)                 == 4, "Ensure other.shape is 4D: (1, in_channels, H, W)"
        assert len(self.shape)                  == 5, "Ensure self.shape is  5D: (1, out_channels, in_channels, kH, kW)"
        assert other.shape[1] == self.shape[2],       "Ensure in_channels of self match in_channels of other."
        assert other.shape[-2] >= self.shape[-2],      f"Ensure the input {other.shape} is at least as large as the kernel {self.shape}."
        assert other.shape[-1] >= self.shape[-1],      f"Ensure the input {other.shape} is at least as large as the kernel {self.shape}."
        
        if self.shape[0] != other.shape[0]:
            self.new_value(np.repeat(self.value, other.shape[0], axis=0))

        b, inC, H, W                    = other.shape
        b, outC, inC, kH, kW            = self.shape
        if self.dtype in [np.float16, np.float128]:
            output_np                       = conv2d_fwd(self.value.astype(np.float32), other.value.astype(np.float32))
        else:
            output_np                       = conv2d_fwd(self.value, other.value)
        new                                 = Tensor(value=output_np, _prev=(self,other), label="Conv2D", leaf=True, dtype=self.dtype)

        def bpass():
            # using Pytorch convention here:
            # gradients are averaged across the out_channels arguments
            if self.dtype in [np.float16, np.float128]:
                self_grad, other_grad            = conv2d_bwd(self.value.astype(np.float32), 
                                                              other.value.astype(np.float32), 
                                                              new.grad.astype(np.float32))
            else:
                self_grad, other_grad            = conv2d_bwd(self.value, other.value, new.grad)
            self.grad                           += self_grad
            other.grad                          += other_grad/self.shape[1]
        new.bpass                                = bpass
        return new

    def create_graph(self) -> tuple[list,list]:
        """
        Creates two reverse-ordered topological graphs: topo and weights.
        
        _topo_ is the full backwards pass computational graph, which includes all 
        intermediary Tensors. _weights_ is a subgraph, containing only the Tensors
        containing learnable weights.
        
        For example, performing y = x**2 + 1 will create the following graphs:
        
        - topo, containing: x**2 + 1, x**2, 1, and x.
        - weights, containing: x
        
        although all nodes in topo were responsible for producing a gradient for x, 
        only the x node contains weights which would need to be updated by this gradient.

        Both graphs are lists that perform a pre-order traversal starting at self as the root node.
        
        :return: topo[list], weights[list]
        """
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

    def backward(self, reset_grad=True) -> None:
        """
        Computes the gradients of all Tensors in self's computation graph, storing results in self.topo and self.weights.
        
        self is initialized with gradient 1, incrementing all children gradients by this multiplier.

        This method first creates two topological graphs of self.
            1. A backwards-pass graph including all Tensors contributing to self.
            2. The backwards-pass graph, now omitting all Tensors with leaf=True.
                This is useful for seeing the exact parameters contributing to the 
                current Tensor, ignoring any Tensors that were produced as intermediary
                values for producing the current Tensor.
        
        :param reset_grad: Whether or not to reset the current backwards pass gradients.
        :type boolean:     True means all gradients of the graph will be reset to 0.
                           False means they will remain untouched.
        """
        self.topo, self.weights = self.create_graph()
        if reset_grad:
            for node in self.topo:
                node.reset_grad()
        self.grad               = np.ones_like(self.value, dtype=self.dtype)
        for node in reversed(self.topo):
            node.bpass()


def array(*args, **kwargs)->Tensor:
    """
    Helper function designed for initializing a `Tensor` object in the same way as a NumPy array.
    Ensure inputs match those of `Tensor`.
    
    :return: A `Tensor` object with fields (*args, **kwargs)
    """
    return Tensor(*args, **kwargs)
