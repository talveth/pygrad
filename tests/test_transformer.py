
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy

import numba as nb
import numpy as np
import torch
from transformer.transformers import Embedding, Transformer_torch

import pygrad
from architectures.transformer import Transformer
from pygrad.constants import PRECISION
from pygrad.optims import SGD, Adam, RMSProp
from pygrad.tensor import Tensor
from examples.transformer.utils import PrepareDataset

if PRECISION in [np.float32]:
    tol = 1e-5
else:
    tol = 1e-7

dtype_mapping = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.bool_: torch.bool
}

class TestTransformer(unittest.TestCase):
    """
    Tests a library-made Transformer compared to a working Pytorch equivalent.

    Tests:
    - Linear Layer
    - Multiheaded Attention
    - AddNorm
    - ReLU
    - Masking
    - Softmax
    - SGD/RMSProp/Adam Optimizers

    Note:
    - Custom Transformer's Encoder is replaced by the Pytorch's
    - Dropout == 0.0 during tests
    """

    def copy_weights(self, model_custom, model_torch, n_layers:int):
        # copy weights
        model_custom.encoder.embedding.torch_embs = model_torch.encoder.emb.torch_embs
        for i in range(n_layers):
            # mha copy
            model_custom.encoder.encoder_layers[i].Att_E1.WQ.W.value = (copy.copy(model_torch.encoder.layers[i].attention.W_q.weight.detach().numpy()).T)[None,...]
            model_custom.encoder.encoder_layers[i].Att_E1.WQ.B.value = copy.copy(model_torch.encoder.layers[i].attention.W_q.bias.detach().numpy())[None,None,...]
            model_custom.encoder.encoder_layers[i].Att_E1.WK.W.value = (copy.copy(model_torch.encoder.layers[i].attention.W_k.weight.detach().numpy()).T)[None,...]
            model_custom.encoder.encoder_layers[i].Att_E1.WK.B.value = copy.copy(model_torch.encoder.layers[i].attention.W_k.bias.detach().numpy())[None,None,...]
            model_custom.encoder.encoder_layers[i].Att_E1.WV.W.value = (copy.copy(model_torch.encoder.layers[i].attention.W_v.weight.detach().numpy()).T)[None,...]
            model_custom.encoder.encoder_layers[i].Att_E1.WV.B.value = copy.copy(model_torch.encoder.layers[i].attention.W_v.bias.detach().numpy())[None,None,...]
            model_custom.encoder.encoder_layers[i].Att_E1.WO.W.value = (copy.copy(model_torch.encoder.layers[i].attention.W_o.weight.detach().numpy()).T)[None,...]
            model_custom.encoder.encoder_layers[i].Att_E1.WO.B.value = copy.copy(model_torch.encoder.layers[i].attention.W_o.bias.detach().numpy())[None,None,...]

            # FFN copy
            model_custom.encoder.encoder_layers[i].FFN_L1.W.value = (copy.copy(model_torch.encoder.layers[i].ffn.linear1.weight.detach().numpy()).T)[None,...]
            model_custom.encoder.encoder_layers[i].FFN_L1.B.value = (copy.copy(model_torch.encoder.layers[i].ffn.linear1.bias.detach().numpy()).T)[None,...]
            model_custom.encoder.encoder_layers[i].FFN_L2.W.value = (copy.copy(model_torch.encoder.layers[i].ffn.linear2.weight.detach().numpy()).T)[None,...]
            model_custom.encoder.encoder_layers[i].FFN_L2.B.value = (copy.copy(model_torch.encoder.layers[i].ffn.linear2.bias.detach().numpy()).T)[None,...]

        model_custom.decoder.embedder.torch_embs = model_torch.decoder.emb.torch_embs
        for i in range(n_layers):
            # mha copy
            model_custom.decoder.decoder_layers[i].multihead_attention1.WQ.W.value = (copy.copy(model_torch.decoder.layers[i].self_attention.W_q.weight.detach().numpy().T))[None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention1.WQ.B.value = copy.copy(model_torch.decoder.layers[i].self_attention.W_q.bias.detach().numpy())[None,None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention1.WK.W.value = (copy.copy(model_torch.decoder.layers[i].self_attention.W_k.weight.detach().numpy().T))[None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention1.WK.B.value = copy.copy(model_torch.decoder.layers[i].self_attention.W_k.bias.detach().numpy())[None,None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention1.WV.W.value = (copy.copy(model_torch.decoder.layers[i].self_attention.W_v.weight.detach().numpy().T))[None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention1.WV.B.value = copy.copy(model_torch.decoder.layers[i].self_attention.W_v.bias.detach().numpy())[None,None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention1.WO.W.value = (copy.copy(model_torch.decoder.layers[i].self_attention.W_o.weight.detach().numpy().T))[None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention1.WO.B.value = copy.copy(model_torch.decoder.layers[i].self_attention.W_o.bias.detach().numpy())[None,None,...]

            model_custom.decoder.decoder_layers[i].multihead_attention2.WQ.W.value = (copy.copy(model_torch.decoder.layers[i].enc_dec_attention.W_q.weight.detach().numpy().T))[None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention2.WQ.B.value = copy.copy(model_torch.decoder.layers[i].enc_dec_attention.W_q.bias.detach().numpy())[None,None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention2.WK.W.value = (copy.copy(model_torch.decoder.layers[i].enc_dec_attention.W_k.weight.detach().numpy().T))[None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention2.WK.B.value = copy.copy(model_torch.decoder.layers[i].enc_dec_attention.W_k.bias.detach().numpy())[None,None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention2.WV.W.value = (copy.copy(model_torch.decoder.layers[i].enc_dec_attention.W_v.weight.detach().numpy().T))[None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention2.WV.B.value = copy.copy(model_torch.decoder.layers[i].enc_dec_attention.W_v.bias.detach().numpy())[None,None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention2.WO.W.value = (copy.copy(model_torch.decoder.layers[i].enc_dec_attention.W_o.weight.detach().numpy().T))[None,...]
            model_custom.decoder.decoder_layers[i].multihead_attention2.WO.B.value = copy.copy(model_torch.decoder.layers[i].enc_dec_attention.W_o.bias.detach().numpy())[None,None,...]

            # FFN copy
            model_custom.decoder.decoder_layers[i].FFN_L1.W.value = (copy.copy(model_torch.decoder.layers[i].ffn.linear1.weight.detach().numpy()).T)[None,...]
            model_custom.decoder.decoder_layers[i].FFN_L1.B.value = (copy.copy(model_torch.decoder.layers[i].ffn.linear1.bias.detach().numpy()).T)[None,...]
            model_custom.decoder.decoder_layers[i].FFN_L2.W.value = (copy.copy(model_torch.decoder.layers[i].ffn.linear2.weight.detach().numpy()).T)[None,...]
            model_custom.decoder.decoder_layers[i].FFN_L2.B.value = (copy.copy(model_torch.decoder.layers[i].ffn.linear2.bias.detach().numpy()).T)[None,...]

        # final FFN copy
        model_custom.decoder.linear.W.value = (copy.copy(model_torch.decoder.linear.weight.detach().numpy()).T)[None,...]
        model_custom.decoder.linear.B.value = (copy.copy(model_torch.decoder.linear.bias.detach().numpy()).T)[None,...]

        return model_custom, model_torch

    def test_transformer_adam(self):

        # load data
        prepare_dataset                 = PrepareDataset()
        max_dataset_length              = 100
        dataset, enc_vocab, dec_vocab   = prepare_dataset.load_dataset('tests/english-german-both.pkl', max_len=max_dataset_length, store=False)
        train, test                     = prepare_dataset.train_test_split(dataset, split=0.8, shuffle=True, store=False)
        trainX, trainY                  = prepare_dataset.create_enc_dec_inputs(train, store=False)

        for n_heads in [1,4,8]:
            for d_model in [16,32]:
                for n_layers in [0,2]:
                    d_k, d_v                = d_model, d_model
                    d_ff                    = 32
                    d_k, d_v                = 16, 16
                    d_ff                    = 32
                    dropout                 = 0.00
                    inp_enc_seq_length      = prepare_dataset.find_seq_length(train[:,0]) - 1
                    inp_dec_seq_length      = prepare_dataset.find_seq_length(train[:,1]) - 1
                    max_len                 = max(inp_dec_seq_length, inp_enc_seq_length) + 1

                    src_pad_idx = 0
                    trg_pad_idx = 0
                    trg_sos_idx = 1

                    # make transformers
                    model_custom            = Transformer(enc_vocab, dec_vocab, n_heads, d_model, d_k, d_v, d_ff, n_layers, 
                                                            inp_enc_seq_length, inp_dec_seq_length, label="Transformer", dropout=dropout)

                    model_torch = Transformer_torch(
                                        src_pad_idx, trg_pad_idx, trg_sos_idx,
                                        d_model=d_model,
                                        d_k=d_k,
                                        enc_voc_size=len(enc_vocab),
                                        dec_voc_size=len(dec_vocab),
                                        max_len=max_len,
                                        ffn_hidden=d_ff,
                                        n_head=n_heads,
                                        n_layers=n_layers,
                                        drop_prob=dropout,
                                        device='cpu')

                    d_vocab_enc    = len(enc_vocab)
                    d_vocab_dec    = len(dec_vocab)
                    model_custom.encoder.embedding = Embedding(d_vocab_enc, d_model)
                    model_custom.decoder.embedder  = Embedding(d_vocab_dec, d_model)
                    model_custom, model_torch = self.copy_weights(model_custom, model_torch, n_layers)
                    
                    # test forward pass:
                    beta1=0.9
                    beta2=0.999
                    eps=1e-8
                    lr=1e-3

                    optim_c             = Adam(model_custom.weights, beta1, beta2, eps, lr)
                    optim_torch         = torch.optim.Adam(model_torch.parameters(), lr=lr, betas=(beta1,beta2), eps=eps, weight_decay=0, amsgrad=False)
                    loss_fn_c           = CCELoss()
                    loss_fn_pt          = torch.nn.CrossEntropyLoss(ignore_index=src_pad_idx)
                    batch_size          = 1
                    n_epochs            = 5

                    for i in range(n_epochs):
                        model_custom.model_reset()
                        xe_inp       = trainX[:batch_size,1:]
                        xe_inp_c     = Tensor(copy.copy(xe_inp), dtype=PRECISION, leaf=True)
                        xe_inp_pt    = torch.tensor(copy.copy(xe_inp), dtype=dtype_mapping[PRECISION], requires_grad=True)
                        
                        xd_inp       = trainY[:batch_size,:-1]
                        xd_inp_c     = Tensor(copy.copy(xd_inp), dtype=PRECISION, leaf=True)
                        xd_inp_pt    = torch.tensor(copy.copy(xd_inp), dtype=dtype_mapping[PRECISION], requires_grad=True)

                        outp_custom  = model_custom(enc_inp=xe_inp_c, dec_inp=xd_inp_c, training=False)
                        outp_torch   = model_torch(src=xe_inp_pt, trg=xd_inp_pt)
                        outp_torch.retain_grad()

                        y_out        = Tensor(np.array(trainY[:batch_size,1:]), dtype=PRECISION, leaf=True)
                        y_out_c      = Tensor(model_custom.onehot_tokens(y_out), learnable=False, leaf=True, dtype=PRECISION)    
                        y_out_pt     = torch.tensor(copy.copy(y_out_c.value).reshape(-1, y_out_c.shape[-1]), dtype=dtype_mapping[PRECISION], requires_grad=True).argmax(dim=-1)

                        loss_c              = loss_fn_c(outp_custom, y_out_c, mask=True)
                        loss_pt             = loss_fn_pt(outp_torch.reshape(-1, y_out_c.shape[-1]), y_out_pt)
                        optim_c.zero_grad()
                        optim_torch.zero_grad()

                        loss_c.backward(reset_grad=True)
                        loss_pt.backward()

                        # test fwd passes
                        self.assertTrue(np.max(np.abs(outp_custom.value - outp_torch.detach().numpy()))<tol)
                        self.assertTrue(np.all(np.isclose(loss_c.value, loss_pt.detach().numpy(), rtol=tol)))

                        #### use adam optimizer and backprop
                        optim_c.step(loss_c)
                        optim_torch.step()


    def test_transformer_sgd(self):

        # load data
        prepare_dataset                 = PrepareDataset()
        max_dataset_length              = 100
        dataset, enc_vocab, dec_vocab   = prepare_dataset.load_dataset('tests/english-german-both.pkl', max_len=max_dataset_length, store=False)
        train, test                     = prepare_dataset.train_test_split(dataset, split=0.8, shuffle=True, store=False)
        trainX, trainY                  = prepare_dataset.create_enc_dec_inputs(train, store=False)

        for n_heads in [1,4,8]:
            for d_model in [16,32]:
                for n_layers in [1,2,4]:
                    d_k, d_v                = d_model, d_model
                    d_ff                    = 32
                    d_k, d_v                = 16, 16
                    d_ff                    = 32
                    dropout                 = 0.00
                    inp_enc_seq_length      = prepare_dataset.find_seq_length(train[:,0]) - 1
                    inp_dec_seq_length      = prepare_dataset.find_seq_length(train[:,1]) - 1
                    max_len                 = max(inp_dec_seq_length, inp_enc_seq_length) + 1

                    src_pad_idx = 0
                    trg_pad_idx = 0
                    trg_sos_idx = 1

                    # make transformers
                    model_custom            = Transformer(enc_vocab, dec_vocab, n_heads, d_model, d_k, d_v, d_ff, n_layers, 
                                                            inp_enc_seq_length, inp_dec_seq_length, label="Transformer", dropout=dropout)

                    model_torch = Transformer_torch(
                                        src_pad_idx, trg_pad_idx, trg_sos_idx,
                                        d_model=d_model,
                                        d_k=d_k,
                                        enc_voc_size=len(enc_vocab),
                                        dec_voc_size=len(dec_vocab),
                                        max_len=max_len,
                                        ffn_hidden=d_ff,
                                        n_head=n_heads,
                                        n_layers=n_layers,
                                        drop_prob=dropout,
                                        device='cpu')

                    d_vocab_enc    = len(enc_vocab)
                    d_vocab_dec    = len(dec_vocab)
                    model_custom.encoder.embedding = Embedding(d_vocab_enc, d_model)
                    model_custom.decoder.embedder  = Embedding(d_vocab_dec, d_model)
                    model_custom, model_torch = self.copy_weights(model_custom, model_torch, n_layers)
                    
                    # test forward pass:
                    eps=1e-8
                    lr=1e-3

                    optim_c             = SGD(model_custom.weights, lr)
                    optim_torch         = torch.optim.SGD(model_torch.parameters(), lr=lr)
                    loss_fn_c           = CCELoss()
                    loss_fn_pt          = torch.nn.CrossEntropyLoss(ignore_index=src_pad_idx)
                    batch_size          = 1
                    n_epochs            = 3

                    for i in range(n_epochs):
                        model_custom.model_reset()
                        xe_inp       = trainX[:batch_size,1:]
                        xe_inp_c     = Tensor(copy.copy(xe_inp), dtype=PRECISION, leaf=True)
                        xe_inp_pt    = torch.tensor(copy.copy(xe_inp), dtype=dtype_mapping[PRECISION], requires_grad=True)
                        
                        xd_inp       = trainY[:batch_size,:-1]
                        xd_inp_c     = Tensor(copy.copy(xd_inp), dtype=PRECISION, leaf=True)
                        xd_inp_pt    = torch.tensor(copy.copy(xd_inp), dtype=dtype_mapping[PRECISION], requires_grad=True)

                        outp_custom  = model_custom(enc_inp=xe_inp_c, dec_inp=xd_inp_c, training=False)
                        outp_torch   = model_torch(src=xe_inp_pt, trg=xd_inp_pt)
                        outp_torch.retain_grad()

                        y_out        = Tensor(np.array(trainY[:batch_size,1:]), dtype=PRECISION, leaf=True)
                        y_out_c      = Tensor(model_custom.onehot_tokens(y_out), learnable=False, leaf=True, dtype=PRECISION)    
                        y_out_pt     = torch.tensor(copy.copy(y_out_c.value).reshape(-1, y_out_c.shape[-1]), dtype=dtype_mapping[PRECISION], requires_grad=True).argmax(dim=-1)

                        loss_c              = loss_fn_c(outp_custom, y_out_c, mask=True)
                        loss_pt             = loss_fn_pt(outp_torch.reshape(-1, y_out_c.shape[-1]), y_out_pt)
                        optim_c.zero_grad()
                        optim_torch.zero_grad()

                        loss_c.backward(reset_grad=True)
                        loss_pt.backward()

                        # test fwd passes
                        print(np.max(np.abs(outp_custom.value - outp_torch.detach().numpy())))
                        self.assertTrue(np.all(np.abs(outp_custom.value - outp_torch.detach().numpy())<tol))
                        self.assertTrue(np.all(np.isclose(loss_c.value, loss_pt.detach().numpy(), rtol=tol)))

                        #### use adam optimizer and backprop
                        optim_c.step(loss_c)
                        optim_torch.step()


    def test_transformer(self):

        # load data
        prepare_dataset                 = PrepareDataset()
        max_dataset_length              = 100
        dataset, enc_vocab, dec_vocab   = prepare_dataset.load_dataset('tests/english-german-both.pkl', max_len=max_dataset_length, store=False)
        train, test                     = prepare_dataset.train_test_split(dataset, split=0.8, shuffle=True, store=False)
        trainX, trainY                  = prepare_dataset.create_enc_dec_inputs(train, store=False)

        for n_heads in [1,4,8]:
            for d_model in [16,32]:
                for n_layers in [0,2]:
                    d_k, d_v                = d_model, d_model
                    d_ff                    = 32
                    d_k, d_v                = 16, 16
                    d_ff                    = 32
                    dropout                 = 0.00
                    inp_enc_seq_length      = prepare_dataset.find_seq_length(train[:,0]) - 1
                    inp_dec_seq_length      = prepare_dataset.find_seq_length(train[:,1]) - 1
                    max_len                 = max(inp_dec_seq_length, inp_enc_seq_length) + 1

                    src_pad_idx = 0
                    trg_pad_idx = 0
                    trg_sos_idx = 1

                    # make transformers
                    model_custom            = Transformer(enc_vocab, dec_vocab, n_heads, d_model, d_k, d_v, d_ff, n_layers, 
                                                            inp_enc_seq_length, inp_dec_seq_length, label="Transformer", dropout=dropout)

                    model_torch = Transformer_torch(
                                        src_pad_idx, trg_pad_idx, trg_sos_idx,
                                        d_model=d_model,
                                        d_k=d_k,
                                        enc_voc_size=len(enc_vocab),
                                        dec_voc_size=len(dec_vocab),
                                        max_len=max_len,
                                        ffn_hidden=d_ff,
                                        n_head=n_heads,
                                        n_layers=n_layers,
                                        drop_prob=dropout,
                                        device='cpu')

                    d_vocab_enc    = len(enc_vocab)
                    d_vocab_dec    = len(dec_vocab)
                    model_custom.encoder.embedding = Embedding(d_vocab_enc, d_model)
                    model_custom.decoder.embedder  = Embedding(d_vocab_dec, d_model)
                    model_custom, model_torch = self.copy_weights(model_custom, model_torch, n_layers)
                    
                    # test forward pass:
                    beta1=0.9
                    eps=1e-8
                    lr=1e-3

                    optim_c             = RMSProp(model_custom.weights, beta1, lr)
                    optim_torch         = torch.optim.RMSprop(model_torch.parameters(), lr, beta1)
                    loss_fn_c           = CCELoss()
                    loss_fn_pt          = torch.nn.CrossEntropyLoss(ignore_index=src_pad_idx)
                    batch_size          = 1
                    n_epochs            = 5

                    for i in range(n_epochs):
                        model_custom.model_reset()
                        xe_inp       = trainX[:batch_size,1:]
                        xe_inp_c     = Tensor(copy.copy(xe_inp), dtype=PRECISION, leaf=True)
                        xe_inp_pt    = torch.tensor(copy.copy(xe_inp), dtype=dtype_mapping[PRECISION], requires_grad=True)
                        
                        xd_inp       = trainY[:batch_size,:-1]
                        xd_inp_c     = Tensor(copy.copy(xd_inp), dtype=PRECISION, leaf=True)
                        xd_inp_pt    = torch.tensor(copy.copy(xd_inp), dtype=dtype_mapping[PRECISION], requires_grad=True)

                        outp_custom  = model_custom(enc_inp=xe_inp_c, dec_inp=xd_inp_c, training=False)
                        outp_torch   = model_torch(src=xe_inp_pt, trg=xd_inp_pt)
                        outp_torch.retain_grad()

                        y_out        = Tensor(np.array(trainY[:batch_size,1:]), dtype=PRECISION, leaf=True)
                        y_out_c      = Tensor(model_custom.onehot_tokens(y_out), learnable=False, leaf=True, dtype=PRECISION)    
                        y_out_pt     = torch.tensor(copy.copy(y_out_c.value).reshape(-1, y_out_c.shape[-1]), dtype=dtype_mapping[PRECISION], requires_grad=True).argmax(dim=-1)

                        loss_c              = loss_fn_c(outp_custom, y_out_c, mask=True)
                        loss_pt             = loss_fn_pt(outp_torch.reshape(-1, y_out_c.shape[-1]), y_out_pt)
                        optim_c.zero_grad()
                        optim_torch.zero_grad()

                        loss_c.backward(reset_grad=True)
                        loss_pt.backward()

                        # test fwd passes
                        self.assertTrue(np.max(np.abs(outp_custom.value - outp_torch.detach().numpy()))<tol)
                        self.assertTrue(np.all(np.isclose(loss_c.value, loss_pt.detach().numpy(), rtol=tol)))

                        #### use adam optimizer and backprop
                        optim_c.step(loss_c)
                        optim_torch.step()


class CCELoss:
    # using a modified CCELoss to match that of the Pytorch Implementation
    def __init__(self, label="CCELoss"):
        self.label = label
    
    def __repr__(self):
        return f"CCELoss=(name={self.label})"
    
    def __call__(self, pred, target, mask:bool=False):
        assert isinstance(pred, (Tensor)),          "Ensure the prediction is of type Tensor"
        assert isinstance(target, (Tensor)),        "Ensure the target is of type Tensor"
        assert len(pred.shape)      == 3,           f"Ensure the prediction shape is 3, (batch_size, 1, w), got: {pred.shape}"
        assert len(target.shape)    == 3,           f"Ensure the target shape is 3, (batch_size, 1, w), got: {target.shape}"
        assert pred.shape           == target.shape,"Ensure pred and target have the same shape"
        assert np.max(target.value) <= 1 and np.min(target.value) >= 0, "Ensure targets have values in {0,1}"
        if pred.shape[1] != 1:
            pred = pred.reshape((-1,1,pred.shape[-1]))
            target = target.reshape((-1,1,target.shape[-1]))
        
        if not mask:
            loss             = (target * (pred.softmax().log())).sum()
            loss_multiplier  = Tensor(value=np.ones_like(loss.value, dtype=pred.dtype)*(-1/pred.shape[0]), dtype=pred.dtype)
            loss             = loss * loss_multiplier
        else:
            mask_np          = (target[:,:,0] != 1)[...,None]
            mask             = Tensor(mask_np, learnable=False, leaf=True, dtype=pred.dtype)
            loss             = ((target*mask)*(pred.softmax().log())).sum()
            loss_multiplier  = Tensor(value=np.ones_like(loss.value, dtype=pred.dtype)*(-1/np.sum(mask[:])), dtype=pred.dtype)
            loss             = loss * loss_multiplier
        return loss

if __name__ == "__main__":
    unittest.main()
