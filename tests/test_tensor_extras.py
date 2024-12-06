

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numba as nb
import numpy as np
import torch

import pygrad
from pygrad.constants import PRECISION
from pygrad.tensor import Tensor

if PRECISION in [np.float16, np.float32]:
    tol = 1e-5
else:
    tol = 1e-8

class TestTensorExtras(unittest.TestCase):

    def test_std(self):
        shapes = [(10,20,20), (10,20), (1,1,1), (1,10,1), (1,1,1,1)]
        for shape in shapes:
            for axis in range(len(shape)):
                for keepdims in [True, False]:

                    a_val               = np.random.uniform(0, 1, shape)

                    # custom
                    a                   = Tensor(a_val)
                    a_fnc               = a.std(axis=axis, keepdim=keepdims)
                    tensor_loss         = (a_fnc).sum()
                    tensor_loss.backward()

                    # pytorch
                    at                  = torch.tensor(a_val, requires_grad=True)
                    at_fnc              = torch.std(at, dim=axis, keepdim=keepdims, correction=0)
                    at_fnc.retain_grad()
                    torch_loss          = (at_fnc).sum()
                    torch_loss.backward()

                    self.assertTrue(np.all(np.isclose(tensor_loss.value.item(), torch_loss.detach().numpy().item())))
                    self.assertTrue(a.grad.shape == tuple(at.grad.shape))
                    self.assertTrue(np.all(np.max(np.abs(a.grad-at.grad.detach().numpy()))<tol))
                    self.assertTrue(np.all(np.isclose(a_fnc.grad, at_fnc.grad, rtol=tol)))

    def test_mean(self):
        shapes = [(10,20,20), (10,20), (1,1,1), (1,10,1), (1,1,1,1)]
        for shape in shapes:
            for axis in range(len(shape)):
                for keepdims in [True, False]:
                    a_val               = np.random.uniform(0, 1, shape)

                    # custom
                    a                   = Tensor(a_val)
                    a_fnc               = a.mean(axis=axis, keepdims=keepdims)
                    tensor_loss         = (a_fnc).sum()
                    tensor_loss.backward()

                    # pytorch
                    at                  = torch.tensor(a_val, requires_grad=True)
                    at_fnc              = at.mean(axis=axis, keepdims=keepdims)
                    at_fnc.retain_grad()
                    torch_loss          = (at_fnc).sum()
                    torch_loss.backward()

                    self.assertTrue(tensor_loss.value, torch_loss.detach().numpy())
                    self.assertTrue(a.grad.shape == tuple(at.grad.shape))
                    self.assertTrue(np.all(np.isclose(a.grad, at.grad)))
                    self.assertTrue(np.all(np.isclose(a_fnc.grad, at_fnc.grad)))

    def test_sum(self):
        shapes = [(10,20,20), (10,20), (1,1,1), (1,10,1), (1,1,1,1)]
        for shape in shapes:
            for axis in range(len(shape)):
                for keepdims in [True, False]:
                    a_val               = np.random.uniform(0, 1, shape)

                    # custom
                    a                   = Tensor(a_val)
                    a_fnc               = a.sum(axis=axis, keepdims=keepdims)
                    tensor_loss         = (a_fnc).sum()
                    tensor_loss.backward()

                    # pytorch
                    at                  = torch.tensor(a_val, requires_grad=True)
                    at_fnc              = at.sum(axis=axis, keepdims=keepdims)
                    at_fnc.retain_grad()
                    torch_loss          = (at_fnc).sum()
                    torch_loss.backward()

                    self.assertTrue(tensor_loss.value, torch_loss.detach().numpy())
                    self.assertTrue(a.grad.shape == tuple(at.grad.shape))
                    self.assertTrue(np.all(np.isclose(a.grad, at.grad)))
                    self.assertTrue(np.all(np.isclose(a_fnc.grad, at_fnc.grad)))

    def test_relu(self):
        shapes = [(10,20,20), (10,20), (1,1,1), (), (1,10,1)]
        for shape in shapes:
            a_val               = np.random.uniform(0, 1, shape)

            # custom
            a                   = Tensor(a_val)
            a_relu              = a.relu()
            tensor_loss         = (a_relu).sum()
            tensor_loss.backward()

            # pytorch
            at                  = torch.tensor(a_val, requires_grad=True)
            at_relu             = torch.nn.functional.relu(at)
            at_relu.retain_grad()
            torch_loss          = (at_relu).sum()
            torch_loss.backward()

            self.assertTrue(tensor_loss.value, torch_loss.detach().numpy())
            self.assertTrue(a.grad.shape == tuple(at.grad.shape))
            self.assertTrue(np.all(np.isclose(a.grad, at.grad)))
            self.assertTrue(np.all(np.isclose(a_relu.grad, at_relu.grad)))    


    def test_softmax(self):
        test_shapes         = [(100,50,50), (1,1,5), (3,5,5), (1,2,2,5), (5,1,10,15)]
        
        for shape in test_shapes:
            nb.njit(fastmath=True,parallel=False,cache=True)(pygrad.numba_ops.softmax_grad.py_func)
            a_val               = np.random.uniform(0, 1, shape)

            # custom
            a                   = Tensor(a_val)
            a_sm                = a.softmax()
            tensor_loss         = (a_sm).sum()
            tensor_loss.backward()

            # pytorch
            at                  = torch.tensor(a_val, requires_grad=True)
            at_sm               = torch.nn.functional.softmax(at, dim=-1)
            at_sm.retain_grad()
            torch_loss          = (at_sm).sum()
            torch_loss.backward()

            self.assertTrue(np.all(np.isclose(a_sm.value, at_sm.detach().numpy())))
            self.assertTrue(tensor_loss.value, torch_loss.detach().numpy())
            self.assertTrue(a.grad.shape == tuple(at.grad.shape))
            self.assertTrue(np.all(np.max(np.abs(a.grad-at.grad.detach().numpy()))<tol))
            self.assertTrue(np.all(np.isclose(a_sm.grad, at_sm.grad, rtol=tol)))

    def test_log(self):
        test_shapes         = [(1,1,5), (3,5,5), (1,2,5), (), (1), (1,1,1), (1,1), (10,1), (1,10)]
        for shape in test_shapes:
            a_val               = np.random.uniform(0, 1, shape)

            # custom
            a                   = Tensor(a_val)
            a_sm                = a.log()
            tensor_loss         = (a_sm).sum()
            tensor_loss.backward()

            # pytorch
            at                  = torch.from_numpy(a_val)
            at.requires_grad    = True
            at_sm               = torch.log(at)
            at_sm.retain_grad()
            torch_loss          = (at_sm).sum()
            torch_loss.backward()

            self.assertTrue(np.all(np.isclose(a_sm.value, at_sm.detach().numpy())))
            self.assertTrue(a.grad.shape == tuple(at.grad.shape))
            self.assertTrue(np.all(np.isclose(a.grad, at.grad, rtol=tol)))
            self.assertTrue(np.all(np.isclose(a_sm.grad, at_sm.grad, rtol=tol)))


    def test_sigmoid(self):
        test_shapes         = [(1,1,1), (3,5,1), (2,1), (), (1), (10,1)]
        
        for shape in test_shapes:
            a_val               = np.random.uniform(0, 1, shape)

            # custom
            a                   = Tensor(a_val)
            a_sm                = a.sigmoid()
            tensor_loss         = (a_sm).sum()
            tensor_loss.backward()

            # pytorch
            at                  = torch.from_numpy(a_val)
            at.requires_grad    = True
            at_sm               = torch.nn.functional.sigmoid(at)
            at_sm.retain_grad()
            torch_loss          = (at_sm).sum()
            torch_loss.backward()

            self.assertTrue(a.grad.shape == tuple(at.grad.shape))
            self.assertTrue(np.all(np.isclose(a.grad, at.grad, rtol=tol)))
            self.assertTrue(np.all(np.isclose(a_sm.grad, at_sm.grad, rtol=tol)))

    def test_tanh(self):
        test_shapes         = [(1,1,1), (3,5,1), (2,1), (), (1), (10,1)]
        
        for shape in test_shapes:
            a_val               = np.random.uniform(0, 1, shape)

            # custom
            a                   = Tensor(a_val)
            a_sm                = a.tanh()
            tensor_loss         = (a_sm).sum()
            tensor_loss.backward()

            # pytorch
            at                  = torch.from_numpy(a_val)
            at.requires_grad    = True
            at_sm               = torch.nn.functional.tanh(at)
            at_sm.retain_grad()
            torch_loss          = (at_sm).sum()
            torch_loss.backward()

            self.assertTrue(np.all(np.isclose(a_sm.value, at_sm.detach().numpy())))
            self.assertTrue(a.grad.shape == tuple(at.grad.shape))
            self.assertTrue(np.all(np.isclose(a.grad, at.grad, rtol=tol)))
            self.assertTrue(np.all(np.isclose(a_sm.grad, at_sm.grad, rtol=tol)))

    def test_conv2D(self):

        # try out the newly created convs; more complex example
        # If self.shape = (1, out_channels, in_channels, kH, kW)
        #     other.shape = (bs, in_channels, H, W)
        #     output.shape = (bs, out_channels, H-kH+1, W-kW+1)
        input_shapes    = [
                            (1,1,1,1), (1,1,4,4), (1,1,10,10), (5,1,1,1), (5,1,4,4), (5,1,10,10),
                            (1,3,1,1), (1,3,4,4), (1,3,10,10),
                            (10,3,1,1), 
                            (10,3,4,4), 
                            (10,3,10,10),
                            (10,3,3,3), 
                            (10,3,4,4), 
                            (10,3,10,10),
                            (10,3,3,1), (10,3,1,3), (10,3,10,10),
                          ]
        kernel_shapes   = [
                            (1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1),
                            (1,1,3,1,1), (1,1,3,1,1), (1,1,3,1,1), 
                            (1,10,3,1,1), 
                            (1,10,3,1,1), 
                            (1,10,3,1,1),
                            (1,10,3,3,1), 
                            (1,10,3,1,3), (1,10,3,3,3),
                            (1,10,3,3,1), (1,10,3,1,3), (1,10,3,3,3),
                          ]
        
        for i in range(len(input_shapes)):
            nb.njit(fastmath=True,parallel=False,cache=True)(pygrad.numba_ops.conv2d_fwd.py_func)
            nb.njit(fastmath=True,parallel=False,cache=True)(pygrad.numba_ops.conv2d_bwd.py_func)
            inp_val             = np.random.uniform(0,1,input_shapes[i])
            ker_val             = np.random.uniform(0,1,kernel_shapes[i])

            X                   = Tensor(value=inp_val)
            k1                  = Tensor(value=ker_val)
            out1                = k1.conv2D(X)
            loss                = out1.sum()
            loss.backward()

            # pytorch equivalent
            xpy                 = torch.tensor(inp_val, requires_grad=True)
            k1_torch            = torch.tensor(ker_val[0], requires_grad=True)
            out1py              = torch.nn.functional.conv2d(xpy, k1_torch, bias=None, stride=1, padding=0, dilation=1)
            
            losspy              = torch.sum(out1py)
            losspy.backward(retain_graph=True)

            # print(loss.value/losspy.detach().numpy())
            # print(out1.value/out1py.detach().numpy())
            # print(k1.grad/k1_torch.grad.detach().numpy())
            # print(X.grad/xpy.grad.detach().numpy(), 'x.grad.shape: ', X.grad.shape, 'xpy.grad.shape: ', xpy.grad.shape, 'X: ', X.shape, 'k1: ', k1.shape)
            self.assertTrue(np.all(np.isclose(loss.value, losspy.detach().numpy(), rtol=tol)))
            self.assertTrue(np.all(np.isclose(out1.value, out1py.detach().numpy(), rtol=tol)))
            self.assertTrue(np.all(np.isclose(np.sum(k1.grad,axis=0), k1_torch.grad.detach().numpy(), rtol=tol))) # summing over the batch dimension
            self.assertTrue(np.all(np.isclose(X.grad, xpy.grad.detach().numpy(), rtol=tol)))

    # def test_conv2D_double(self):

    #     # try out the newly created convs; more complex example
    #     # If self.shape = (1, out_channels, in_channels, kH, kW)
    #     #     other.shape = (bs, in_channels, H, W)
    #     #     output.shape = (bs, out_channels, H-kH+1, W-kW+1)
    #     input_shapes    = [
    #                         # (1,1,1,1), (1,1,4,4), (1,1,10,10), (5,1,1,1), (5,1,4,4), (5,1,10,10),
    #                         (10,3,1,1), (10,3,4,4), (10,3,10,10),
    #                         (10,3,6,3), (10,3,2,6), (10,3,10,10),
    #                         (10,3,6,1), (10,3,1,6), (10,3,10,10),
    #                       ]
    #     kernel_shapes   = [
    #                         # (1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1), (1,1,1,1,1),
    #                         (1,3,3,1,1), (1,3,3,1,1), (1,3,3,1,1),
    #                         (1,3,3,3,1), (1,3,3,1,3), (1,3,3,3,3),
    #                         (1,3,3,3,1), (1,3,3,1,3), (1,3,3,3,3),
    #                       ]
        
    #     for i in range(len(input_shapes)):

    #         inp_val             = np.random.uniform(0,1,input_shapes[i])
    #         # inp_val             = np.ones(input_shapes[i])
    #         ker_val             = np.random.uniform(0,1,kernel_shapes[i])
    #         # ker_val             = np.ones(kernel_shapes[i])

    #         X                   = Tensor(value=inp_val, label="X")
    #         k1                  = Tensor(value=ker_val, label="ker")
    #         out1                = k1.conv2D(X)
    #         out1                = k1.conv2D(out1)
    #         loss                = out1.sum()
    #         loss.backward()

    #         # pytorch equivalent
    #         xpy                 = torch.tensor(inp_val, requires_grad=True)
    #         k1_torch            = torch.tensor(ker_val[0], requires_grad=True)
    #         out1py              = torch.nn.functional.conv2d(xpy, k1_torch, bias=None, stride=1, padding=0, dilation=1)
    #         out1py              = torch.nn.functional.conv2d(out1py, k1_torch, bias=None, stride=1, padding=0, dilation=1)
    #         losspy              = out1py.sum()
    #         losspy.backward()

    #         # self.assertTrue(np.all(np.isclose(loss.value, losspy.detach().numpy())))
    #         # self.assertTrue(np.all(np.isclose(out1.value, out1py.detach().numpy())))
    #         # print('k1 shapes', k1.grad.shape, k1_torch.grad.shape)
    #         # print('k1 grads', np.sum(k1.grad,axis=0)/k1_torch.grad.detach().numpy())
    #         # self.assertTrue(np.all(np.isclose(np.sum(k1.grad,axis=0), k1_torch.grad.detach().numpy()))) # summing over the batch dimension
    #         print('x shapes', X.grad.shape, X.shape, xpy.grad.shape, xpy.shape)
    #         print('xgrads', X.label, X.grad/xpy.grad.detach().numpy())
    #         self.assertTrue(np.all(np.isclose(X.grad, xpy.grad.detach().numpy())))



if __name__ == "__main__":
    unittest.main()
