
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
