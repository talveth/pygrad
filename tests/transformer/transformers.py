
import torch
from torch import nn
import numpy as np

import os, sys
sys.path.append(os.path.abspath(os.path.dirname('./../../')))
from autograd.cpu.tensor import Tensor
from autograd.cpu.basics import Linear, ReLU, AddNorm, Dropout
from autograd.cpu.module import Module

PRECISION = np.float64

dtype_mapping = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.bool_: torch.bool
}

class TransformerEmbedding_torch(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_vocab, d_model, max_len, device="cpu", dtype=PRECISION):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(TransformerEmbedding_torch, self).__init__()

        self.dtype_mapping  = dtype_mapping
        self.torch_embs     = torch.nn.Embedding(num_embeddings=d_vocab, 
                                                 embedding_dim=d_model)

        # same size with input matrix (for adding with input matrix)
        self.dtype = dtype
        self.encoding = torch.zeros(max_len, d_model, device=device, dtype=dtype_mapping[self.dtype])
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device, dtype=dtype_mapping[self.dtype])
        pos = pos.unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device, dtype=dtype_mapping[self.dtype])
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        torch_emb = self.torch_embs(x.type(torch.int)).type(self.dtype_mapping[self.dtype])
        _, seq_len = x.size()
        return self.encoding[:seq_len,:] + torch_emb


class PositionwiseFeedForward_torch(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1, dtype=PRECISION):
        super(PositionwiseFeedForward_torch, self).__init__()
        self.dtype      = dtype
        self.linear1    = nn.Linear(d_model, hidden, dtype=self.dtype)
        self.linear2    = nn.Linear(hidden, d_model, dtype=self.dtype)
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LayerNorm_torch(nn.Module):
    def __init__(self, d_model, eps=1e-12, dtype=PRECISION):
        super(LayerNorm_torch, self).__init__()
        self.dtype      = dtype
        self.gamma  = nn.Parameter(torch.ones(d_model, dtype=self.dtype))
        self.beta   = nn.Parameter(torch.zeros(d_model, dtype=self.dtype))
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        # out = self.gamma * out + self.beta
        return out

class MultiHeadAttention_torch(torch.nn.Module):
    def __init__(self, d_model, d_k, n_heads, dtype=PRECISION):

        super(MultiHeadAttention_torch, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"      
        self.dtype      = dtype  # ...
        self.d_k        = d_k                                                   # Model's dimension
        self.num_heads  = n_heads                                               # Number of attention heads
        self.W_q        = torch.nn.Linear(d_model, d_model, dtype=self.dtype)   # Query transformation
        self.W_k        = torch.nn.Linear(d_model, d_model, dtype=self.dtype)   # Key transformation
        self.W_v        = torch.nn.Linear(d_model, d_model, dtype=self.dtype)   # Value transformation
        self.W_o        = torch.nn.Linear(d_model, d_model, dtype=self.dtype)   # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        self.QKT            = torch.matmul(Q, K.transpose(-2, -1))
        self.QKT.retain_grad()
        attn_scores         = self.QKT * (1/torch.sqrt(torch.tensor(self.d_k/self.num_heads, dtype=self.dtype)))
        if mask is not None:
            self.mask       = mask
            attn_scores     = attn_scores.masked_fill(mask == 0, -1e9)  
        self.attn_scores    = attn_scores
        self.attn_scores.retain_grad()
        attn_probs          = torch.softmax(attn_scores, dim=-1)
        self.softmax     = attn_probs
        self.softmax.retain_grad()
        output              = torch.matmul(attn_probs, V)
        self.Z          = output
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q_sh = self.split_heads(self.W_q(Q))
        K_sh = self.split_heads(self.W_k(K))
        V_sh = self.split_heads(self.W_v(V))
        
        attn_output = self.combine_heads(self.scaled_dot_product_attention(Q_sh, K_sh, V_sh, mask))
        output = self.W_o(attn_output)
        return output

class EncoderLayer_torch(nn.Module):

    def __init__(self, d_model, d_k, ffn_hidden, n_heads, drop_prob, dtype=PRECISION):
        super(EncoderLayer_torch, self).__init__()
        self.dtype      = dtype_mapping[dtype]
        self.attention  = MultiHeadAttention_torch(d_model, d_k=d_k, n_heads=n_heads, dtype=self.dtype)
        self.norm1      = LayerNorm_torch(d_model=d_model, dtype=self.dtype)
        self.dropout1   = nn.Dropout(p=drop_prob)

        self.ffn        = PositionwiseFeedForward_torch(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob, dtype=self.dtype)
        self.norm2      = LayerNorm_torch(d_model=d_model, dtype=self.dtype)
        self.dropout2   = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(Q=x, K=x, V=x, mask=src_mask)
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder_torch(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, d_k, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # d_vocab, d_model, max_len, device
        self.emb = TransformerEmbedding_torch(
                                        d_vocab=enc_voc_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        device="cpu")
        self.layers = nn.ModuleList([EncoderLayer_torch(d_model=d_model, d_k=d_k,
                                                  ffn_hidden=ffn_hidden,
                                                  n_heads=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class DecoderLayer_torch(nn.Module):
    def __init__(self, d_model, d_k, ffn_hidden, n_head, drop_prob, dtype=PRECISION):
        super(DecoderLayer_torch, self).__init__()
        self.self_attention = MultiHeadAttention_torch(d_model, d_k=d_k, n_heads=n_head, dtype=dtype)
        self.norm1 = LayerNorm_torch(d_model=d_model, dtype=dtype)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention_torch(d_model, d_k=d_k, n_heads=n_head, dtype=dtype)
        self.norm2 = LayerNorm_torch(d_model=d_model, dtype=dtype)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward_torch(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob, dtype=dtype)
        self.norm3 = LayerNorm_torch(d_model=d_model, dtype=dtype)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _x = dec
        self.input_to_self_attention = dec, trg_mask
        x = self.self_attention(Q=dec, K=dec, V=dec, mask=trg_mask)
        self.att1 = x
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            self.pre_crossattention = x, enc, enc
            x = self.enc_dec_attention(Q=x, K=enc, V=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x

class Decoder_torch(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, d_k, ffn_hidden, n_head, n_layers, drop_prob, device, dtype=PRECISION):
        super().__init__()
        self.dtype = dtype_mapping[dtype]
        self.emb = TransformerEmbedding_torch(
                                        d_vocab=dec_voc_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        device=device)
        self.layers = nn.ModuleList([DecoderLayer_torch(d_model=d_model, d_k=d_k,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob, dtype=self.dtype)
                                     for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size, dtype=self.dtype)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        output = self.linear(trg)
        return output

class Transformer_torch(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx,
                 enc_voc_size, 
                 dec_voc_size, 
                 d_model, d_k,
                 n_head, 
                 max_len,
                 ffn_hidden, 
                 n_layers, 
                 drop_prob, 
                 device='cpu'):
        super().__init__()
        self.src_pad_idx    = src_pad_idx
        self.trg_pad_idx    = trg_pad_idx
        self.trg_sos_idx    = trg_sos_idx
        self.device         = device
        self.encoder    = Encoder_torch(d_model=d_model, d_k=d_k,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder    = Decoder_torch(d_model=d_model, d_k=d_k,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask    = self.make_src_mask(src)
        trg_mask    = self.make_trg_mask(trg)
        enc_src     = self.encoder(src, src_mask)
        self.enc_src = enc_src
        output      = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):   # padding masks
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        self.src_mask = src_mask
        return src_mask

    def make_trg_mask(self, trg):   # decoder autoregressive mask
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        self.trg_mask = trg_mask
        return trg_mask



class Tokenizer:
    def __init__(self, dtype=PRECISION):
        self.dtype = dtype

    def padding_mask(self, x:np.ndarray):
        # x.shape == (b,w)
        mask = np.equal(x, 0).astype(self.dtype)
        return mask

    def lookahead_mask(self, shape:int): # (shape,shape)
        mask = np.where(np.arange(shape)[None, :, None] < np.arange(shape), 1, np.zeros((shape,shape)))
        return mask


class Embedding:
    def __init__(self, d_vocab, d_model, dtype=PRECISION) -> None:
        self.d_vocab        = d_vocab
        self.d_model        = d_model
        self.dtype          = dtype
        self.torch_embs     = torch.nn.Embedding(num_embeddings=d_vocab, 
                                                 embedding_dim=d_model)

    def generate_pos_embeddings(self, max_len: int) -> np.ndarray:
        encoding    = np.zeros((max_len, self.d_model), dtype=self.dtype)
        pos         = np.arange(0, max_len, dtype=self.dtype)
        pos         = pos[:,None]
        _2i         = np.arange(0, self.d_model, step=2, dtype=self.dtype)

        encoding[:, 0::2] = np.sin(pos / (10000 ** (_2i / self.d_model)))
        encoding[:, 1::2] = np.cos(pos / (10000 ** (_2i / self.d_model)))
        return encoding

    def gen_embeddings(self, max_len, input):
        # input shape: (bs, max_len)
        # output:      (bs, max_len, d_model)
        pos_embs    = self.generate_pos_embeddings(max_len)
        input_torch = torch.tensor(input.value, dtype=torch.int)
        oth_embs    = (self.torch_embs(input_torch)).detach().numpy().astype(self.dtype)
        embedded    = Tensor(value=pos_embs + oth_embs, label="embeddings", leaf=True, dtype=self.dtype)
        return embedded


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
    def __init__(self, 
                 enc_tokenizer, dec_tokenizer,
                 n_heads=8, d_model=512, d_k=512, d_v=512, d_ff=2048, 
                 n_layers=6, enc_seq_length=1, dec_seq_length=1, batch_size=1, label="Transformer", 
                 dropout=0.00, dtype=PRECISION):
        
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


