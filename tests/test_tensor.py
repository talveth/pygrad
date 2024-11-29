
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numba as nb
import numpy as np
import torch

from pygrad.constants import PRECISION
from pygrad.tensor import Tensor

if PRECISION in [np.float16, np.float32]:
    tol = 1e-5
else:
    tol = 1e-8


class TestTensor(unittest.TestCase):
    """
    Class to test Tensor ops against Pytorch.
    """
    def test_matmul(self):
        # tests __mul__ between Tensors
        shapes_1 = [
                    (100,200,10),
                    (15,10,10,10,20,10),
                    (10,10)
                    ]
        shapes_2 = [
                    (100,10,200), 
                    (15,10,10,10,10,20),
                    (10,10)
                    ]

        for i in range(len(shapes_1)):
            a_val               = np.random.uniform(0, 1, shapes_1[i])
            b_val               = np.random.uniform(0, 1, shapes_2[i])

            # custom
            a                   = Tensor(a_val)
            b                   = Tensor(b_val)
            tensor_loss         = (a@b).sum()
            tensor_loss.backward()

            # pytorch
            at                  = torch.tensor(a_val, requires_grad=True)
            bt                  = torch.tensor(b_val, requires_grad=True)
            torch_loss          = (at@bt).sum()
            torch_loss.backward()

            # value
            self.assertTrue(np.all(np.isclose(tensor_loss.value, torch_loss.detach().numpy())))

            # gradients
            self.assertTrue(a.grad.shape == at.grad.shape),       ""
            self.assertTrue(b.grad.shape == bt.grad.shape),       ""
            self.assertTrue(np.all(np.isclose(a.grad, at.grad))), ""
            self.assertTrue(np.all(np.isclose(b.grad, bt.grad))), ""

    def test_add_sub(self):
        # tests __add__ and __sub__ between Tensors
        shapes_1 = [
                    (1,2,2), 
                    (10,20,20), (10,20,20), (10,20,20), 
                    (10,20,20),
                    (1), (1), (1), (1), (1),
                    (), (), (), (), 
                    (10,1), (10,1), (10,1), (10,1)
                    ]
        shapes_2 = [
                    (1,2,2), 
                    (1,20,20), (1,1,20), (10,1,20), 
                    (),
                    (1), (1,1), (10,1), (1,10,1), (10,1), 
                    (), (1), (1,10,1), (1,10),
                    (10,1), (1,1), (1), ()
                    ]
        for i in range(len(shapes_1)):
            for j in range(2):
                for k in range(2):
                    if j==0:
                        a_val = np.random.uniform(0, 1, shapes_1[i])
                        b_val = np.random.uniform(0, 1, shapes_2[i])
                    elif j==1:
                        b_val = np.random.uniform(0, 1, shapes_1[i])
                        a_val = np.random.uniform(0, 1, shapes_2[i])
                    
                    # custom
                    a = Tensor(a_val)
                    b = Tensor(b_val)

                    # pytorch
                    at = torch.tensor(a_val, requires_grad=True)
                    bt = torch.tensor(b_val, requires_grad=True)
                    
                    if k == 0:
                        tensor_loss = (a+b).sum()
                        print((a+b).shape, shapes_1[i], shapes_2[i])
                        tensor_loss.backward()
                        torch_loss = (at+bt).sum()
                        torch_loss.backward()
                    elif k == 1:
                        tensor_loss = (a-b).sum()
                        tensor_loss.backward()
                        torch_loss = (at-bt).sum()
                        torch_loss.backward()

                    # test value
                    self.assertTrue(tensor_loss.value, torch_loss.detach().numpy())

                    # test grads
                    self.assertTrue(a.grad.shape == at.grad.shape),       ""
                    self.assertTrue(b.grad.shape == bt.grad.shape),       ""
                    self.assertTrue(np.all(np.isclose(a.grad, at.grad))), ""
                    self.assertTrue(np.all(np.isclose(b.grad, bt.grad))), ""

    def test_tensor_reshape_grad(self):
        # tests .reshape
        a_val = np.random.uniform(0, 1, (10,20,20))

        # custom
        a = Tensor(a_val)
        tensor_loss = (a.reshape((1,10,20,20,1))).sum()
        tensor_loss.backward()

        # pytorch
        at = torch.tensor(a_val, requires_grad=True)
        torch_loss = (at.reshape(1,10,20,20,1)).sum()
        torch_loss.backward()

        self.assertTrue(a.grad.shape == tuple(at.grad.shape))
        self.assertTrue(np.all(np.isclose(a.grad, at.grad)))


    def test_tensor_transpose_grad_v1(self):
        # tests .transpose()

        a_val           = np.random.uniform(0, 1, (10,20,20))
        transposes      = [(1,0,2), (0,2,1)]
        pt_transposes   = [(1,0), (-1,-2)]

        for i in range(len(transposes)):

            # custom
            a               = Tensor(a_val)
            tensor_loss     = (a.transpose(transposes[i])).sum()
            tensor_loss.backward()

            # pytorch
            at              = torch.tensor(a_val, requires_grad=True)
            torch_loss      = (at.transpose(*pt_transposes[i])).sum()
            torch_loss.backward()

            self.assertTrue(a.grad.shape == tuple(at.grad.shape))
            self.assertTrue(np.all(np.isclose(a.grad, at.grad)))


    def test_mul(self):
        # tests __mul__ and between Tensors
        shapes_1 = [
                    (1,2,2), 
                    (10,20,20), (10,20,20), (10,20,20), 
                    (10,20,20),
                    (1), (1), (1), (1), (1),
                    (), (), (), (), 
                    (10,1), (10,1), (10,1), (10,1)
                    ]
        shapes_2 = [
                    (1,2,2), 
                    (1,20,20), (1,1,20), (10,1,20), 
                    (),
                    (1), (1,1), (10,1), (1,10,1), (10,1), 
                    (), (1), (1,10,1), (1,10),
                    (10,1), (1,1), (1), ()
                    ]
        for i in range(len(shapes_1)):
            for j in range(2):
                for k in range(2):
                    if j==0:
                        a_val = np.random.uniform(0, 1, shapes_1[i])
                        b_val = np.random.uniform(0, 1, shapes_2[i])
                    elif j==1:
                        b_val = np.random.uniform(0, 1, shapes_1[i])
                        a_val = np.random.uniform(0, 1, shapes_2[i])
                    
                    # custom
                    a = Tensor(a_val)
                    b = Tensor(b_val)

                    # pytorch
                    at = torch.tensor(a_val, requires_grad=True)
                    bt = torch.tensor(b_val, requires_grad=True)
                    
                    if k == 0:
                        tensor_loss = (a*b).sum()
                        print((a+b).shape, shapes_1[i], shapes_2[i])
                        tensor_loss.backward()
                        torch_loss = (at*bt).sum()
                        torch_loss.backward()
                    elif k == 1:
                        tensor_loss = (a/b).sum()
                        tensor_loss.backward()
                        torch_loss = (at/bt).sum()
                        torch_loss.backward()

                    # test value
                    self.assertTrue(tensor_loss.value, torch_loss.detach().numpy())

                    # test grads
                    self.assertTrue(a.grad.shape == at.grad.shape),       ""
                    self.assertTrue(b.grad.shape == bt.grad.shape),       ""
                    self.assertTrue(np.all(np.isclose(a.grad, at.grad))), ""
                    self.assertTrue(np.all(np.isclose(b.grad, bt.grad))), ""


    def test_tensor_pow_grad_v1(self):
        # checks __pow__
        pow_vals            = [-1,2,10]
        shapes              = [(10,20,20), (), (1)]
        for pow_val in pow_vals:
            for shape in shapes:
                a_val               = np.random.uniform(0, 1, shape)

                # custom
                a                   = Tensor(a_val)
                pow_val1            = a**pow_val
                tensor_loss         = pow_val1.sum()
                tensor_loss.backward()

                # pytorch
                at                  = torch.from_numpy(a_val)
                at.requires_grad    = True
                pow_val1t           = at**pow_val
                pow_val1t.retain_grad()
                torch_loss = (pow_val1t).sum()
                torch_loss.backward()

                self.assertTrue(pow_val1.grad.shape == tuple(pow_val1t.grad.shape))
                self.assertTrue(np.all(np.isclose(pow_val1.grad, pow_val1t.grad)))



    


# class TestNN(unittest.TestCase):

#     def test_scaled_attention_class(self):
        
#         batch_sizes = [1,10]
#         d_contexts  = [1,16,64]
#         d_models    = [1,16,64]

#         for batch_size in batch_sizes:
#             for d_context in d_contexts:
#                 for d_model in d_models:
#                     d_k        = d_model
                    
#                     Q = np.random.uniform(0,1,(batch_size,d_context,d_model))
#                     K = np.random.uniform(0,1,(batch_size,d_context,d_model))
#                     V = np.random.uniform(0,1,(batch_size,d_context,d_model))

#                     mask               = np.random.randint(0,2,(batch_size,d_context,d_context-1)).astype(bool)
#                     extra_trues        = np.ones((batch_size,d_context,1)).astype(bool)
#                     mask               = np.concatenate((mask, extra_trues), axis=-1)

#                     Qt, Kt, Vt = Tensor(Q, label='Q'), Tensor(K, label='K'), Tensor(V, label='V')
#                     QKT        = Qt * Kt.T
#                     multiplier = np.ones_like(QKT.value)*(1/np.sqrt(d_k))
#                     presoftmax = QKT.hadamard(multiplier)
#                     presoftmax.label = "attention presoftmax"
#                     maskt = np.where(mask==0, -1e9, 0)
#                     presoftmax.value += maskt
#                     Zt = (presoftmax.softmax()) * Vt

#                     tensor_loss = Zt.sum()
#                     tensor_loss.backward()

#                     # ...
#                     Qtt                 = torch.from_numpy(Q)
#                     Ktt                 = torch.from_numpy(K)
#                     Vtt                 = torch.from_numpy(V)
#                     masktt              = torch.from_numpy(mask)
#                     Qtt.requires_grad   = True
#                     Ktt.requires_grad   = True
#                     Vtt.requires_grad   = True

#                     Ztt                 = torch.nn.functional.scaled_dot_product_attention(Qtt, Ktt, Vtt, attn_mask=masktt)
#                     Ztt.retain_grad()
#                     torch_loss          = Ztt.sum()
#                     torch_loss.backward()

#                     self.assertTrue(Qt.grad.shape == tuple(Qtt.grad.shape))
#                     self.assertTrue(np.all(np.isclose(Qt.grad, Qtt.grad, rtol=1e-3)))
#                     self.assertTrue(np.all(np.isclose(Kt.grad, Ktt.grad, rtol=1e-5)))
#                     self.assertTrue(np.all(np.isclose(Vt.grad, Vtt.grad, rtol=1e-5)))
#                     self.assertTrue(np.all(np.isclose(Zt.grad, Ztt.grad, rtol=1e-5)))
#                     self.assertTrue(np.all(np.isclose(Zt.value, Ztt.detach().numpy(), rtol=1e-5)))

#     def test_layernorm(self):
#         addnorm             = cpu_transformer.AddNorm()
#         test_shapes         = [(1,1,5), (3,5,5), (1,2,2,5)]
#         for shape in test_shapes:
#             x_val               = np.random.uniform(0, 1, shape)
#             skip_val            = np.random.uniform(0, 1, shape)
            
#             # custom
#             x                   = Tensor(x_val)
#             sk                  = Tensor(skip_val)

#             outp                = addnorm(x, sk)
#             tensor_loss         = (outp).sum()
#             tensor_loss.backward()

#             # pytorch
#             x_valt                          = torch.from_numpy(x_val)
#             x_valt.requires_grad            = True
#             skip_valt                       = torch.from_numpy(skip_val)
#             skip_valt.requires_grad         = True

#             outp_torch                      = torch.nn.functional.layer_norm(x_valt+skip_valt, list([shape[-1]]))
#             outp_torch.retain_grad()
#             torch_loss                      = (outp_torch).sum()
#             torch_loss.backward()

#             self.assertTrue(outp.grad.shape == tuple(outp_torch.grad.shape))
#             self.assertTrue(np.all(np.isclose(outp.grad, outp_torch.grad, rtol=1e-9)))
#             self.assertTrue(np.all(np.isclose(outp.value, outp_torch.detach().numpy(), rtol=1e-9)))


#     def test_cce(self):
#         test_shapes         = [(2,1,5), (5,1,5), (4,1,2)]
#         for shape in test_shapes:
#             x_val           = np.random.uniform(0, 1, shape)
            
#             # generate labels
#             labels          = np.zeros_like(x_val)
#             for i in range(shape[0]):
#                 random_indices                                  = np.random.choice(shape[-1], size=shape[-2], replace=False)
#                 labels[i, np.arange(shape[-2]), random_indices] = 1

#             cce_loss        = cpu_transformer.CCELoss()
#             x_val_sm_c      = Tensor(x_val)
#             labels_c        = Tensor(labels)
#             loss_val        = cce_loss(x_val_sm_c, labels_c)
#             loss_val.backward()

#             cce_loss_torch  = torch.nn.CrossEntropyLoss()
#             x_val_sm_t      = torch.from_numpy(x_val[:,0,:])
#             x_val_sm_t.requires_grad = True
#             labels_t        = torch.from_numpy(labels[:,0,:])
#             labels_t.requires_grad = True
#             loss_val_t      = cce_loss_torch(x_val_sm_t, labels_t)
#             loss_val_t.retain_grad()
#             loss_val_t.backward()
#             self.assertTrue(np.all(np.isclose(x_val_sm_c.grad[:,0,:], x_val_sm_t.grad.detach().numpy(), rtol=1e-5)))
#             # self.assertTrue(np.all(np.isclose(loss_val.value[:,0,:], loss_val_t.detach().numpy(), rtol=1e-9)))


if __name__ == "__main__":
    unittest.main()
