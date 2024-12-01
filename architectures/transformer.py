

import numpy as np

from pygrad.activations import ReLU
from pygrad.basics import AddNorm, Dropout, Linear
from pygrad.constants import PRECISION
from pygrad.module import Module
from pygrad.tensor import Tensor

class Embedding:
    def __init__(self, d_vocab, d_model, dtype=PRECISION) -> None:
        self.d_vocab        = d_vocab
        self.d_model        = d_model
        self.dtype          = dtype
        self.cust_emb       = np.random.normal(0,0.8,size=(d_vocab, d_model))

    def generate_pos_embeddings(self, max_len: int) -> np.ndarray:
        encoding    = np.zeros((max_len, self.d_model))
        pos         = np.arange(0, max_len)
        pos         = pos[:,None]
        _2i = np.arange(0, self.d_model, step=2)

        encoding[:, 0::2] = np.sin(pos / (10000 ** (_2i / self.d_model)))
        encoding[:, 1::2] = np.cos(pos / (10000 ** (_2i / self.d_model)))
        return encoding
    
    def onehot_tokens(self, inp:np.ndarray, d_vocab) -> np.ndarray:
        inp         = np.asarray(inp, dtype=int)
        num_samples = inp.size  # Get the total number of elements
        one_hot     = np.zeros((num_samples, d_vocab), dtype=int)
        one_hot[np.arange(num_samples), inp.flatten()] = 1
        one_hot     = one_hot.reshape(inp.shape[0], -1, d_vocab)
        return one_hot
    
    def generate_normal_embeddings(self, input):
        inp_oh      = self.onehot_tokens(input.value, self.d_vocab)
        return inp_oh @ self.cust_emb

    def gen_embeddings(self, max_len, input:Tensor):
        pos_embs = self.generate_pos_embeddings(max_len)
        oth_embs = self.generate_normal_embeddings(input)
        return Tensor(pos_embs + oth_embs, label="embeddings", leaf=True, dtype=self.dtype)


class Attention:
    def __init__(self, d_k, n_heads=None, label="Attention", eps=1e9, dtype=PRECISION):
        # Q: (b>=1, n, d_k)
        self.dtype      = dtype
        self.eps        = eps
        self.label      = label
        self.n_heads    = 1 if n_heads is None else n_heads
        self.d_k        = d_k
    
    def __call__(self, Q:Tensor, K:Tensor, V:Tensor, mask=None):
        self.QKT            = Q @ K.T
        multiplier          = Tensor(np.ones_like(self.QKT.value)*(1/np.sqrt(self.d_k/self.n_heads)))
        presoftmax          = self.QKT * multiplier
        presoftmax.label    = "attention presoftmax"
        #
        if mask is not None:
            self.mask = mask
            masked_presoftmax = presoftmax.mask_idcs(mask.value.astype(bool), value=-1e9)
        softmax         = masked_presoftmax.softmax()
        Z               = softmax @ V
        return Z


class MultiHeadAttention:
    def __init__(self, n_heads, d_n:int, d_k:int, d_v:int, d_model:int, dtype=PRECISION, label:str="MHA"):
        self.attention  = Attention(d_v, n_heads=n_heads, label=label, dtype=PRECISION)
        self.dtype      = dtype
        self.label      = label
        self.n_heads    = n_heads
        self.d_n        = d_n
        self.d_k        = d_k
        self.d_v        = d_v
        self.d_model    = d_model
        self.WQ         = Linear(i_dim=self.d_model,   o_dim=self.d_model,   bias=True,     label="WQ", dtype=self.dtype) 
        self.WK         = Linear(i_dim=self.d_model,   o_dim=self.d_model,   bias=True,     label="WK", dtype=self.dtype)
        self.WV         = Linear(i_dim=self.d_model,   o_dim=self.d_model,   bias=True,     label="WV", dtype=self.dtype)
        self.WO         = Linear(i_dim=self.d_model,   o_dim=self.d_model,   bias=True,     label="WO", dtype=self.dtype)

    def reshape_Tensor(self, x, shape_into:bool=True)->Tensor:
        assert isinstance(shape_into, bool), ""
        assert isinstance(x, Tensor)
        if shape_into:
            new_x   = x.reshape((x.shape[0], x.shape[1], self.n_heads, -1))
            new_xt  = new_x.transpose((0,2,1,3))
        if not shape_into:
            new_x   = x.transpose((0,2,1,3))
            new_xt  = new_x.reshape((new_x.shape[0], new_x.shape[1], -1))
        return new_xt

    def __call__(self, Q:Tensor, K:Tensor, V:Tensor, mask=None):
        # Q/K/V: (bs, d_c, d_m)
        Q_reshaped = self.reshape_Tensor(self.WQ(Q))                                # (bs, n_h, d_ce, d_m/n_h)
        K_reshaped = self.reshape_Tensor(self.WK(K))                                # (bs, n_h, d_ce, d_m/n_h)
        V_reshaped = self.reshape_Tensor(self.WV(V))                                # (bs, n_h, d_ce, d_m/n_h)
        # mask should be of shape (bs, n_heads, d_context_enc, d_context_enc)
        preout              = self.attention(Q_reshaped, K_reshaped, V_reshaped, mask=mask)      # (bs, n_h, d_c, d_m/n_h)
        preout_backshape    = self.reshape_Tensor(preout, shape_into=False)            # (bs, d_c, d_m)
        out                 = self.WO(preout_backshape)
        out.label           = "MultiHeadAttention Output"                                     # (bs, d_c, d_model)
        return out


class EncoderLayer:
    # @profile
    def __init__(self, n_heads, d_model, d_k, d_v, d_ff, d_context, dropout, dtype=PRECISION):
        # ...
        self.dtype          = dtype
        self.d_context      = d_context
        self.dropout_rate   = dropout
        #
        self.Att_E1         = MultiHeadAttention(n_heads, self.d_context, d_k, d_v, d_model, label="MHA - Encoder - 1st")  # (b, n, d_model)
        self.dropout1       = Dropout(self.dropout_rate)
        self.AddNorm_1      = AddNorm(gain=1.0)
        self.FFN_L1         = Linear(i_dim=d_model, o_dim=d_ff, bias=True, label="FFN_L1")
        self.FFN_relu1      = ReLU()
        self.FFN_L2         = Linear(i_dim=d_ff, o_dim=d_model, bias=True, label="FFN_L2")
        self.dropout2       = Dropout(self.dropout_rate)
        self.AddNorm_2      = AddNorm(gain=1.0)                                      # (b, n, d_model)

    def __call__(self, x: Tensor, padding_mask, training):
        Att1                = self.Att_E1(Q=x, K=x, V=x, mask=padding_mask)
        Z                   = self.AddNorm_1(self.dropout1(Att1, training), x)          # (b, n, d_model)
        enc_ffn             = self.FFN_L2(self.FFN_relu1(self.FFN_L1(Z)))
        enc_outp            = self.AddNorm_2(self.dropout2(enc_ffn, training), Z)
        return enc_outp                                                                    # (b, n, d_model)


class Encoder:
    def __init__(self, d_vocab,
                 n_heads, d_model, d_k, d_v, d_ff, n_layers, d_context, dropout):

        self.d_vocab        = d_vocab
        self.d_context      = d_context
        self.dropout_rate   = dropout
        self.embedding      = Embedding(self.d_vocab, d_model)
        self.dropout        = Dropout(self.dropout_rate)        
        self.encoder_layers = [EncoderLayer(n_heads, d_model, d_k, d_v, d_ff, self.d_context, dropout) for _ in range(n_layers)]

    def __call__(self, x:Tensor, padding_mask:np.ndarray, training:bool):  
        x                   = self.embedding.gen_embeddings(self.d_context, x)
        x                   = self.dropout(x, training)
        for layer in self.encoder_layers:
            x = layer(x=x, padding_mask=padding_mask, training=training)
        return x


class DecoderLayer:
    def __init__(self, n_heads, d_model, d_k, d_v, d_ff, d_context, dropout):
        self.d_context                  = d_context
        self.dropout_rate               = dropout

        self.multihead_attention1       = MultiHeadAttention(n_heads, self.d_context, d_k, d_v, d_model, label="MHA - Decoder - 1st")
        self.dropout1                   = Dropout(self.dropout_rate)
        self.add_norm1                  = AddNorm()
        self.multihead_attention2       = MultiHeadAttention(n_heads, self.d_context, d_k, d_v, d_model, label="MHA - Decoder - 2nd")
        self.dropout2                   = Dropout(self.dropout_rate)
        self.add_norm2                  = AddNorm()
        self.FFN_L1                     = Linear(i_dim=d_model, o_dim=d_ff, bias=True, label="dec_FFN_L1")
        self.FFN_relu1                  = ReLU()
        self.FFN_L2                     = Linear(i_dim=d_ff, o_dim=d_model, bias=True, label="dec_FFN_L2")
        self.dropout3                   = Dropout(self.dropout_rate)
        self.add_norm3                  = AddNorm()

    def __call__(self, x, encoder_output, lookahead_mask, cross_mask, training:bool):
        self.input_to_self_attention = x, lookahead_mask
        att1 = self.multihead_attention1(Q=x, K=x, V=x, mask=lookahead_mask)
        self.att1 = att1
        x    = self.add_norm1(self.dropout1(att1, training), x)
        self.pre_crossattention = x, encoder_output, encoder_output
        att2 = self.multihead_attention2(Q=x, K=encoder_output, V=encoder_output, mask=cross_mask)
        x    = self.add_norm2(self.dropout2(att2, training), x)
        ffn  = self.FFN_L2(self.FFN_relu1(self.FFN_L1(x)))
        x    = self.add_norm3(self.dropout3(ffn, training), x)
        return x

class Decoder:
    def __init__(self, d_vocab,
                 n_heads, d_model, d_k, d_v, d_ff, n_layers, d_context, dropout):
        
        self.d_vocab        = d_vocab
        self.dropout_rate   = dropout
        self.d_context      = d_context
        self.dropout        = Dropout(self.dropout_rate)
        self.embedder       = Embedding(self.d_vocab, d_model)
        self.decoder_layers = [DecoderLayer(
            n_heads=n_heads,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            d_context=d_context,
            dropout=self.dropout_rate) for _ in range(n_layers)]
        self.linear         = Linear(d_model, self.d_vocab)

    def __call__(self, x, encoder_output, lookahead_mask, cross_mask, training:bool):
        x        = self.embedder.gen_embeddings(self.d_context, x)
        x        = self.dropout(x, training)
        for layer in self.decoder_layers:
            x    = layer(x=x, encoder_output=encoder_output, 
                         lookahead_mask=lookahead_mask, cross_mask=cross_mask, training=training)
        x = self.linear(x)
        return x

def onehot_tokens(inp:Tensor, d_vocab:int) -> Tensor:
    inp         = inp.value
    d_vocab     = 10000
    num_samples = inp.size  # Get the total number of elements
    one_hot     = np.zeros((num_samples, d_vocab), dtype=int)
    one_hot[np.arange(num_samples), inp.flatten()] = 1
    one_hot     = one_hot.reshape(inp.shape[0], -1, d_vocab)
    return one_hot

class Transformer(Module):
    print(f"Make sure PRECISION is np.float32 or higher, currently: {PRECISION}")
    def __init__(self,
                 enc_tokenizer, dec_tokenizer,
                 n_heads=8, d_model=512, d_k=512, d_v=512, d_ff=2048, 
                 n_layers=6, enc_seq_length=1, dec_seq_length=1, batch_size=1, label="Transformer", 
                 dropout=0.10, dtype=PRECISION):
        
        self.dtype          = dtype
        self.label          = label
        self.n_heads        = n_heads
        self.dropout_rate   = dropout
        self.d_vocab_enc    = len(enc_tokenizer)
        self.d_vocab_dec    = len(dec_tokenizer)

        ## ...
        self.d_model        = d_model

        # ...
        self.dropout                = Dropout(self.dropout_rate)
        self.encoder                = Encoder(self.d_vocab_enc,
                                      n_heads, d_model, d_k, d_v, d_ff, 
                                      n_layers=n_layers, d_context=enc_seq_length, dropout=self.dropout_rate)
        self.decoder                = Decoder(self.d_vocab_dec,
                                      n_heads, d_model, d_k, d_v, d_ff, 
                                      n_layers=n_layers, d_context=dec_seq_length, dropout=self.dropout_rate)
        super().__init__(enc_inp=Tensor(np.ones((batch_size, enc_seq_length), dtype=self.dtype), leaf=True), 
                         dec_inp=Tensor(np.ones((batch_size, dec_seq_length), dtype=self.dtype), leaf=True),
                         training=True)

    def forward(self, enc_inp:Tensor, dec_inp:Tensor, training:bool):
        # x should be the preprocessed inputs of shape (bs, d_context), values per row indicating 
        enc_padding_mask                       = self.make_enc_mask(enc_inp)
        dec_lookahead_mask                     = self.make_dec_lookahead_mask(dec_inp)
        cross_mask                             = self.make_cross_mask(enc_inp, dec_inp)
        
        enc_output                     = self.encoder(x=enc_inp, padding_mask=enc_padding_mask, training=training)
        self.enc_output                = enc_output
        dec_output                     = self.decoder(x=dec_inp, encoder_output=enc_output,
                                                      lookahead_mask=dec_lookahead_mask, cross_mask=cross_mask,
                                                      training=training)
        return dec_output

    def padding_mask(self, x):
        mask = np.equal(x, 0).astype(self.dtype)
        return mask
    
    def lookahead_mask(self, shape):
        mask = np.where(np.arange(shape)[None, :, None] < np.arange(shape), 1, np.zeros((shape,shape)))
        return mask

    def make_enc_mask(self, enc_inp):
        enc_padding_mask:np.ndarray    = self.padding_mask(enc_inp.value)[:,np.newaxis,:,np.newaxis]      # (bs,1,d_context_enc,1);
        enc_padding_mask               = np.tile(enc_padding_mask, (1,self.n_heads,1,enc_padding_mask.shape[-2]))       # (bs, n_heads, d_context_enc, d_context_enc)
        enc_padding_mask               = Tensor(enc_padding_mask, label="enc_padding_mask", learnable=False, leaf=True).T
        self.enc_padding_mask          = enc_padding_mask
        return enc_padding_mask

    def make_dec_lookahead_mask(self, dec_inp):
        dec_padding_mask:np.ndarray    = self.padding_mask(dec_inp.value)[:,np.newaxis,:,np.newaxis]# (bs,1,d_context,1);
        dec_padding_mask               = np.tile(dec_padding_mask, (1,self.n_heads,1,dec_padding_mask.shape[-2])) # (bs,n_heads,d_context_dec,d_context_dec);
        dec_padding_mask               = Tensor(dec_padding_mask, label="dec_padding_mask", learnable=False, leaf=True).T
        dec_lookahead_mask:np.ndarray  = self.lookahead_mask(dec_inp.shape[1])[np.newaxis,:,:,:]    # (1,1,d_context_dec,d_context_dec)
        dec_lookahead_mask             = np.tile(dec_lookahead_mask, (dec_inp.shape[0], self.n_heads, 1, 1))      # (bs,n_heads,d_context_dec,d_context_dec)
        dec_lookahead_mask             = np.maximum(np.maximum(dec_padding_mask.value, dec_padding_mask.T.value), dec_lookahead_mask)
        dec_lookahead_mask             = Tensor(dec_lookahead_mask, label="dec_lookahead_mask", learnable=False, leaf=True)        
        self.dec_lookahead_mask        = dec_lookahead_mask
        return dec_lookahead_mask

    def make_cross_mask(self, enc_inp, dec_inp):
        cross1      = np.where(np.arange(dec_inp.value.shape[-1])[None, :, None] < np.arange(enc_inp.value.shape[-1]), 
                               1, 
                               np.zeros((dec_inp.value.shape[-1],enc_inp.value.shape[-1])))
        cross12     = np.tile(cross1, (1,self.n_heads,1,1))
        cross21     = self.padding_mask(enc_inp.value)[:,None,:,None]
        cross22     = np.transpose(np.tile(cross21, (1,self.n_heads,1,dec_inp.shape[-1])), (0,1,3,2))
        # cross_mask  = Tensor(np.maximum(cross12, cross22))
        cross_mask  = Tensor(cross22)
        self.cross12    = cross12
        self.cross_mask = cross22
        return cross_mask

    def onehot_tokens(self, inp:Tensor) -> np.ndarray:
        d_vocab     = self.d_vocab_dec
        inp         = np.asarray(inp.value, dtype=int)
        num_samples = inp.size  # Get the total number of elements
        one_hot     = np.zeros((num_samples, d_vocab), dtype=int)
        one_hot[np.arange(num_samples), inp.flatten()] = 1
        one_hot     = one_hot.reshape(inp.shape[0], -1, d_vocab)
        return one_hot
