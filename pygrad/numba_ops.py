
"""
Module storing numba-defined optimizations.
"""

import numba as nb
import numpy as np
from numba import prange


@nb.njit(fastmath=True,parallel=False,cache=True)
def combined_softmax_backprop(new_val,new_grad):
    s1,s2,s3 = new_val.shape
    res = np.empty((s1,s2,s3), dtype=new_val.dtype)
    for b in range(s1):
        for c in range(s2):
            for h in range(s3):
                acc=0
                for w in range(s3):
                    if w==h:
                        acc+=(new_val[b,c,h]*(1-new_val[b,c,h]))*new_grad[b,0,w]
                    else:
                        acc+=(-1*new_val[b,c,h]*new_val[b,c,w])*new_grad[b,0,w]
                res[b,c,h]=acc
    return res


@nb.njit(fastmath=True,parallel=False,cache=True)
def find_sum_grad_3D(new_val):
    s1,s2,s3 = new_val.shape
    res = np.empty((s1, s2, s3, s3), dtype=new_val.dtype)
    for b in range(s1):
        for c in range(s2):
            for w in range(s3):
                for i in range(s3):
                    if i==w:
                        res[b,c,w,i]=new_val[b,c,w]-new_val[b,c,w]**2
                    else:
                        res[b,c,w,i]=-1*new_val[b,c,w]*new_val[b,c,i]
    return res

@nb.njit(fastmath=True,parallel=False,cache=True)
def softmax_matmul_3D(A,B):
    a1, a2, a3, a4  = A.shape
    b1, b3, b4      = B.shape
    res             = np.empty((a1,b3,a3), dtype=A.dtype)
    for g in range(a1):
        for h in range(a2):
            for i in range(a3):
                acc=0
                for j in range(a4):
                    acc+=A[g,h,i,j]*B[g,h,j]
                res[g,h,i]=acc
    return res


@nb.njit(fastmath=True,parallel=True,cache=False)
def softmax_grad(new_val,grad):
    s1,s2,s3 = new_val.shape
    _,b3,_ = grad.shape
    res2     = np.empty((s1,b3,s3), dtype=new_val.dtype)
    for b in prange(s1):
        for c in prange(s2):
            for w in prange(s3):
                acc=0
                for i in prange(s3):
                    acc+=new_val[b,c,w]*((i==w)-new_val[b,c,i])*grad[b,c,i]
                res2[b,c,w]=acc
    return res2


@nb.njit(fastmath=True,parallel=False,cache=False)
def mat_sum(mat1,mat2):
    inC, kH, kW = mat1.shape
    res         = 0
    for c in range(inC):
        for h in range(kH):
            for w in range(kW):
                res += mat1[c,h,w]*mat2[c,h,w]
    return res

@nb.njit(fastmath=True,parallel=True,cache=False)
def conv2d_fwd(kernel,signal):
    b, outC, inC, kH, kW = kernel.shape
    b, inC, H, W         = signal.shape
    res                  = np.empty((b, outC, H-kH+1, W-kW+1), dtype=kernel.dtype)
    
    for i in nb.prange(H-kH+1):
        for j in nb.prange(W-kW+1):
            for c in nb.prange(outC):
                for bd in nb.prange(b):
                    res[bd,c,i,j] = mat_sum(kernel[bd,c,:,:,:],signal[bd,:,i:i+kH,j:j+kW])    
    return res


@nb.njit(fastmath=False,parallel=True,cache=False)
def conv2d_bwd(kernel, signal, grad):
    b, outC, inC, kH, kW = kernel.shape
    b, inC, H, W         = signal.shape
    self_grad            = np.zeros_like(kernel, dtype=kernel.dtype)
    other_grad           = np.zeros_like(signal, dtype=kernel.dtype)
    
    for i in range(H-kH+1):
        for j in range(W-kW+1):
            for c in range(outC):
                res_self_grad = np.empty((b,inC,kH,kW), dtype=kernel.dtype)
                res_othe_grad = np.empty((b,inC,kH,kW), dtype=kernel.dtype)
                for bd in range(b):
                    for ic in range(inC):
                        for h in range(kH):
                            for w in range(kW):
                                res_self_grad[bd,ic,h,w] = signal[bd,ic,i+h,j+w]*grad[bd,c,i,j]
                                res2 = 0
                                for oc in range(outC):
                                    res2 += kernel[bd,oc,ic,h,w]*grad[bd,c,i,j]
                                res_othe_grad[bd,ic,h,w] = res2
                self_grad[:,c,:,:,:]          += res_self_grad
                other_grad[:,:,i:i+kH,j:j+kW] += res_othe_grad
    return self_grad, other_grad
