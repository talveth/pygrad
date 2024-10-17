
import numpy as np
import numba as nb
from opt_einsum import contract
from .numba_ops import softmax_grad
import copy

# tensor class
class Tensor:
    def __init__(self, value, _prev:tuple=(), label:str="", learnable:bool=True, leaf:bool=False, dtype=np.float64) -> None:
        self.dtype          = dtype
        self.value          = np.array(value).astype(self.dtype)
        self.learnable      = learnable
        self.leaf           = leaf
        self.shape          = self.value.shape
        self.label          = label
        self._prev:tuple    = _prev
        self.topo           = None
        self.grad           = np.zeros_like(self.value).astype(self.dtype)
        self.bpass          = lambda : None
        self.params         = [self]
        np.seterr(all='ignore')

    def reset_grad(self):
        self.grad  = np.zeros_like(self.value).astype(self.dtype)

    def new_value(self, x):
        self.value = x
        self.shape = self.value.shape
        self.grad  = np.zeros_like(self.value).astype(self.dtype)

    def __repr__(self) -> str:
        return f"Tensor=(name={self.label}, shape={self.shape}, dtype={self.dtype})"

    def __matmul__(self, other):
        # only support dense layer operations or multiply by float broadcasting ops
        # (1,n,m)*(1,m,0) or (1,1)*(1,m,o)
        # assert isinstance(other, (Tensor, int, float, np.ndarray)), f"Ensure that argument is of type Tensor or a float/numpy array, got {type(other)}"
        if isinstance(other, (int, float)):
            self_shape_length = len(self.value.shape)
            if self_shape_length == 3:
                other = np.array(other).reshape(1, 1, 1)
            elif self_shape_length == 4:
                other = np.array(other).reshape(1, 1, 1, 1)
            other = Tensor(value=other.astype(self.dtype), label="other", leaf=False)
        elif isinstance(other, (np.ndarray)):
            assert len(other.shape) == 3, "other, if np.ndarray, must be 3D (1, H, W)"
            other = Tensor(value=np.array(other).astype(self.dtype), label="other", leaf=False)

        assert len(self.value.shape) in [3,4],        "Ensure that self is 3D or 4D (1, H, W), (1, _, H, W)"
        assert len(other.shape) in [1,2,3,4],   "Ensure that other is 2D (1, _, W, I)"
        # new_val         = contract('...bj,...jk->...bk', self.value, other.value, optimize='optimal')
        new_val         = self.value @ other.value
        new             = Tensor(value=new_val, _prev=(self, other), leaf=True)

        def bpass():
            # broadcasting?!?!?
            # self.grad   = self.grad + new.grad @ np.transpose(other.value, (0,2,1))
            self.grad   += contract('...bj,...kj->...bk', new.grad, other.value, optimize='optimal')
            # other.grad  = other.grad + np.transpose(self.value, (0,2,1)) @ new.grad
            other.grad  += contract('...jb,...jk->...bk', self.value, new.grad, optimize='optimal')
        new.bpass       = bpass
        return new

    def reshape(self, shape:tuple):
        assert isinstance(shape, tuple), "shape input must be a tuple"
        new_val = np.reshape(self.value, shape)
        new     = Tensor(value=new_val, _prev=(self,), leaf=True)
        def bpass():
            self.grad   += np.reshape(new.grad, self.value.shape)
        new.bpass       = bpass
        return new

    def transpose(self, axes:tuple):
        assert isinstance(axes, tuple), "axes must be a tuple"
        assert len(axes) == len(self.value.shape), f"axes tuple must be the same length as self.shape. Got {len(axes)} should be {len(self.shape)}"
        new_val     = np.transpose(self.value, axes)
        new         = Tensor(value=new_val, _prev=(self,), leaf=True)
        def bpass():
            def invert_axes():
                inverted = [0]*len(axes)
                for i, val in enumerate(axes):
                    inverted[val] = i
                return tuple(inverted)
            self.grad   += np.transpose(new.grad, invert_axes())
        new.bpass       = bpass
        return new

    def __rmul__(self, other):
        return self * other

    def __getitem__(self, idcs):
        return self.value[idcs]

    def mean(self, axis:int=-1, keepdims:bool=True):
        new_val = np.mean(self.value, axis=axis, keepdims=keepdims)
        new     = Tensor(value=new_val, _prev=(self,), leaf=True)

        def bpass():
            # needs checking
            if keepdims:
                self.grad += (1/self.shape[axis]) * contract('...,...->...', np.ones(self.shape), new.grad, optimize='optimal')
            else:
                new_grad = np.expand_dims(new.grad, axis=axis)
                self.grad += (1/self.shape[axis]) * contract('...,...->...', np.ones(self.shape), new_grad, optimize='optimal')
        new.bpass = bpass
        return new

    def std(self, axis):
        N   = self.value.shape[axis]
        d2  = (self - self.mean(axis=axis,keepdims=True))**2  # abs is for complex `a`
        var = d2.mean(axis=axis, keepdims=True)     # no bias correction done here
        new = var**0.5
        return new

    def __mul__(self, other):
        # same sized terms only
        if not isinstance(other, Tensor):
            if isinstance(other, float):
                other_np = np.array([other])
                other_np = np.reshape(other_np, (1,)*len(self.shape))
            elif isinstance(other, np.ndarray):
                other_np = other
            else:
                raise ValueError
            other = Tensor(other_np)
        assert np.broadcast_shapes(self.shape, other.shape), "Shapes not broadcastable. Ensure shapes are broadcastable"
        new_val = contract('...,...->...', self.value, other.value, optimize='optimal')
        new = Tensor(value=new_val, _prev=(self, other), leaf=True)

        def determine_broadcasting(val1, val2):
            broadcast   = np.broadcast_shapes(val1.shape, val2.shape)
            sh1         = (1,) * (len(broadcast) - len(val1.shape)) + val1.shape
            sh2         = (1,) * (len(broadcast) - len(val2.shape)) + val2.shape
            shape1_broadcasts = np.array(broadcast)//np.array(sh1)
            shape2_broadcasts = np.array(broadcast)//np.array(sh2)
            return shape1_broadcasts, shape2_broadcasts

        def bpass():
            sh1b, _     = determine_broadcasting(self.grad, new.grad)
            new_grad    = contract('...,...->...', other.value, new.grad, optimize='optimal')
            for i in range(len(sh1b)):
                if sh1b[i] != 1:
                    new_grad = np.sum(new_grad, axis=i, keepdims=True)
            self.grad += new_grad
            
            sh1b, _     = determine_broadcasting(other.grad, new.grad)
            new_grad    = contract('...,...->...', self.value, new.grad, optimize='optimal')
            for i in range(len(sh1b)):
                if sh1b[i] != 1:
                    new_grad = np.sum(new_grad, axis=i, keepdims=True)
            other.grad += new_grad
            # self.grad   += contract('...,...->...', other.value, new.grad, optimize='optimal')
            # other.grad  += contract('...,...->...', self.value, new.grad, optimize='optimal')
        new.bpass = bpass
        return new

    def sum(self, over_batch:bool=False):
        assert isinstance(over_batch, bool), "Make sure over_batch is a boolean"
        if over_batch:
            new_val         = contract('...->', self.value, optimize='optimal')
        else:
            new_val         = contract('i...->i', self.value, optimize='optimal')
        
        if over_batch:
            new_val = new_val.reshape(1, 1, -1)
        else:
            reshape_dims = (self.shape[0], 1, -1) if len(self.shape) <= 3 else (self.shape[0], 1, 1, -1)
            new_val = new_val.reshape(reshape_dims)

        new             = Tensor(value=new_val, _prev=(self,), leaf=True)
        def bpass():
            self.grad  += contract('...,...->...', np.ones(self.shape), new.grad, optimize='optimal')
        new.bpass       = bpass
        return new

    def __add__(self, other):
        # (...,m) + (...,m) -> (..., m)
        assert isinstance(other, (Tensor, float, int)), "Other must be a float or Tensor"
        if isinstance(other, (float, int)):
            other   = np.array([other])[None,None,:]
            other   = Tensor(other, learnable=False, leaf=True)              # turns into a Tensor of shape (1,1,1)
        new = Tensor(value=self.value + other.value, _prev=(self, other), leaf=True)

        def determine_broadcasting(val1, val2):
            broadcast   = np.broadcast_shapes(val1.shape, val2.shape)
            sh1         = (1,) * (len(broadcast) - len(val1.shape)) + val1.shape
            sh2         = (1,) * (len(broadcast) - len(val2.shape)) + val2.shape
            shape1_broadcasts = np.array(broadcast)//np.array(sh1)
            shape2_broadcasts = np.array(broadcast)//np.array(sh2)
            return shape1_broadcasts, shape2_broadcasts

        def bpass():
            sh1b, _     = determine_broadcasting(other.grad, new.grad)
            new_grad    = new.grad
            for i in range(len(sh1b)):
                if sh1b[i] != 1:
                    new_grad = np.sum(new_grad, axis=i, keepdims=True)
            s1, s2 = other.grad.shape, new_grad.shape
            if len(s2)>len(s1):
                diff = len(s2)-len(s1)
                new_grad = new_grad.reshape(s2[diff:])
            other.grad += new_grad

            sh1b, _     = determine_broadcasting(self.grad, new.grad)
            new_grad    = new.grad
            for i in range(len(sh1b)):
                if sh1b[i] != 1:
                    new_grad = np.sum(new_grad, axis=i, keepdims=True)
            s1, s2 = other.grad.shape, new_grad.shape
            if len(s2)>len(s1):
                diff = len(s2)-len(s1)
                new_grad = new_grad.reshape(s2[diff:])
            self.grad += new_grad
        new.bpass                   = bpass
        return new

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

    def __pow__(self, n:int):
        new = Tensor(value=self.value**n, _prev=(self,), leaf=True)
        def bpass():
            self.grad                       += contract('...,...->...', (n*self.value**(n-1)), new.grad, optimize='optimal')
        new.bpass = bpass
        return new

    @property
    def T(self):
        # transposes the last two dimensions
        new                          = Tensor(value=contract('...ij->...ji', self.value), _prev=(self,), label=f"{self.label}.T", leaf=True)
        def bpass():
            self.grad               += contract('...ij->...ji', new.grad, optimize='optimal')
        new.bpass                    = bpass
        return new

    def relu(self):
        new = Tensor(value=np.maximum(0, self.value), _prev=(self,), leaf=True)
        def bpass():
            grad_mat                = np.copy(self.value)
            grad_mat[grad_mat<=0]   = 0.0
            grad_mat[grad_mat>0]    = 1.0
            self.grad               += contract('...,...->...', grad_mat, new.grad, optimize='optimal')
        new.bpass                   = bpass
        return new

    def tanh(self):
        # (1,W)
        ex, emx = np.exp(self.value), np.exp(-self.value)
        tanh_val= (ex-emx)/(ex+emx)
        new     = Tensor(value=tanh_val, _prev=(self,), leaf=True)
        def bpass():
            self.grad   += contract('...,...->...', (1-tanh_val**2), new.grad, optimize='optimal')
        new.bpass = bpass
        return new

    def mask_idcs(self, mask_idcs:np.ndarray, value:float=0):
        # assert len(self.shape) == len(mask_idcs),    ""       #TODO:fix this
        new_val             = copy.deepcopy(self.value)
        new_val[mask_idcs]  = value
        new                 = Tensor(value=new_val, _prev=(self,), leaf=True)

        def bpass():
            ones_arr        = np.ones_like(self.value)
            ones_arr[mask_idcs] = 0
            self.grad      += ones_arr * new.grad
        new.bpass = bpass
        return new

    def softmax(self):
        # (B>=1,H>=1,W)
        # outputs in 3D
        assert isinstance(self, Tensor), "Ensure that self is of type Tensor"
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
                diag_elem       = contract('...hn,...inm->...hnm', new_val, np_eye, optimize='optimal')
                new_val_reshape = np.expand_dims(new_val, axis=-2)
                nondiags        = contract('...ji,...jk->...ik', new_val_reshape, new_val_reshape, optimize='optimal')
                sum_grad        = diag_elem-nondiags
                new_grad        = np.transpose(new.grad[...,np.newaxis,:,:], (0,1,2,4,3))
                # idx             = np.arange(X.shape[2])
                pre_self_grad       = sum_grad@new_grad
                idx                 = np.arange(pre_self_grad.shape[2])
                result              = np.transpose(pre_self_grad[:,:,idx,:,idx], (1,2,0,3))
                self.grad           += result

            elif len(self.shape) == 3:
                self.grad        += softmax_grad(new_val, new.grad)
            else:
                raise "ERROR, self.shape should be 2 or 3D"
        new.bpass           = bpass
        return new

    def sigmoid(self):
        # self.shape (b>=1,1,1) only
        assert len(self.shape) in [2,3],    "Ensure len(self.shape) == 2"
        assert self.shape[-1] == 1,         "Ensure the input has shape (..., 1)"
        new_val             = 1/(1 + np.exp(-self.value))
        new                 = Tensor(value=new_val, _prev=(self,), leaf=True)
        def bpass():
            self.grad      += new_val * (1-new_val) * new.grad
        new.bpass           = bpass
        return new

    def log(self):
        # (1, W), logs in base E
        assert len(self.shape) in [3],      f"Ensure the self shape is 2D (b>=1, 1, w), got {self.shape}"
        assert np.min(self.value) >= 0,      "Ensure the value of self is non-negative before logging"
        
        # result              = np.where(self.value > 1e-10, self.value, -10)
        # new_val             = np.log(result, out=result, where=result > 0)
        log_val             = np.log(self.value)
        new                 = Tensor(value=log_val, _prev=(self,), leaf=True)
        def bpass():
            multiplier      = 1/self.value
            self.grad      += multiplier * new.grad
        new.bpass           = bpass
        return new

    def conv2D(self, other):
        # conv shape: (b>=1, out_channels, in_channels, kH, kW)
        # produces an output of shape (b>=1, out_channels, H-kH+1, W-kW+1)
        assert len(other.shape)                 == 4, "Ensure other.shape is 4D: (1, in_channels, H, W)"
        assert len(self.shape)                  == 5, "Ensure self.shape is  5D: (1, out_channels, in_channels, kH, kW)"
        
        n_conv_calls                            = 0
        n_conv_calls                           += 1
        b, inC, H, W                            = other.shape
        b, outC, inC, kH, kW                    = self.shape
        output_np                               = np.zeros(shape=(b, outC, H-kH+1, W-kW+1))
        for i in range(H-kH+1):
            for j in range(W-kW+1):
                for c in range(outC):
                    temp_self                   = Tensor(value=self.value[:,c,:,:,:], leaf=True)                                               # (b, in_channels, H, W)
                    output_np[:, c, i, j]       = (temp_self * Tensor(other.value[:,:,i:i+kH, j:j+kW], leaf=True)).sum().value[:,0,0,0]  # (b,)
        new                                     = Tensor(value=output_np, _prev=(self,other), label="ConvOut", leaf=True)

        def bpass():
            # using Pytorch convention here:
            # gradients are averaged across the out_channels arguments
            self.reset_grad()
            other.reset_grad()
            for i in range(H-kH+1):
                for j in range(W-kW+1):
                    for c in range(outC):
                        self.grad[:,c,:,:,:]   += contract('...,...->...', other.value[:,:,i:i+kH, j:j+kW], new.grad[:,c,i,j].reshape(-1,1,1,1), optimize='optimal')  # (b, in_channels, kH, kW)
                        temp_other_grad_a       = contract('...,...->...', self.value, new.grad[:,c,i,j].reshape(-1,1,1,1,1), optimize='optimal')                     # (b,out_c, in_c, kH, kW)
                        other.grad[:,:,i:i+kH, j:j+kW] += np.sum(temp_other_grad_a, axis=(1))                                                      # (b, in_c, kH, kW)
            # if the same conv is called >1 times per forward pass, average the gradient over those number of times
            self.grad                          *= n_conv_calls
            # average the other gradient with the number of out_channels
            other.grad                         /= outC
        new.bpass                               = bpass
        return new

    def maxpool2D(self, kH:int, kW:int):
        # ignores out-of-boundary pooling
        # self.shape == (b>=1, c, h, w)
        assert isinstance(kH, int) and isinstance(kW, int),  "Ensure kH and kW are integers"
        assert kH >=1 and kW >= 1,                           "Ensure kH and kW are >=1"
        assert kH <= self.shape[-2] and kW <= self.shape[-1],"Ensure the pooling kernel is not bigger than its input"
        assert len(self.shape) == 4,                         "Ensure self.shape is 4D"

        def pool2D(inp:np.ndarray, kH:int, kW:int, idx0:int, idx1:int) -> float:
            return np.max(inp[:,:,idx0:idx0+kH, idx1:idx1+kW], axis=(-2,-1))

        b, c, h, w = self.shape
        nh, nw     = np.floor((h-kH)/kH + 1).astype(int), np.floor((w-kW)/kW + 1).astype(int)
        new_val    = np.zeros((b, c, nh, nw))
        for i in range(nh):
            for j in range(nw):
                out_i = i*kH
                out_j = j*kW
                new_val[:,:,i,j] = pool2D(self.value, kH, kW, out_i, out_j)
        new        = Tensor(value=new_val, _prev=(self,), label="MaxPool'd")
    
        def bpass():
            new_val_repeated = np.repeat(np.repeat(new_val, kH, axis=-2), kW, axis=-1)
            new_grad_repeated=np.repeat(np.repeat(new.grad, kH, axis=-2), kW, axis=-1)
            self.grad       += (new_val_repeated == self.value).astype(self.dtype) * new_grad_repeated
        new.bpass  = bpass
        return new

    def flatten(self):
        # self.shape == (H, W) or (1, H, W) or (b, H, W) or (b, 1, H, W)
        # output: 3D
        assert len(self.shape) in [2,3,4],      "self.shape has to be 2,3 or 4D."
        if len(self.shape)                      == 2:       # (H, W)
            new_val                             = self.value.reshape(1, 1, -1)
        elif len(self.shape)                    == 3:       # (1, H, W)
            assert self.shape[0]                == 1, "Ensure the channel dimension is of size 1"
            new_val                             = self.value.reshape(1, 1, -1)
        elif len(self.shape)                    == 4:
            assert self.shape[1]                == 1, "Ensure the 1st dimension is of size 1"
            new_val                             = self.value.reshape(self.value.shape[0], 1, -1)
        else:
            raise "Error with input to flatten"
        new                                     = Tensor(value=new_val, _prev=(self,))
        def bpass():
            self.grad                           += new.grad.reshape(self.value.shape)
        new.bpass                               = bpass
        return new

    def create_graph(self):
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
        """Creates an reverse-ordered topological graph."""
        self.topo, self.weights = self.create_graph()
        if reset_grad:
            for node in self.topo:
                node.reset_grad()
        self.grad = np.ones_like(self.value).astype(self.dtype)
        for node in reversed(self.topo):
            node.bpass()

