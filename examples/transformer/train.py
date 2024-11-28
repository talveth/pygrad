
# PYTHONPATH=. python ....py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from architectures.transformer import Transformer
from autograd.cpu.losses import CCELoss
from autograd.cpu.optims import Adam
from autograd.cpu.tensor import Tensor
from examples.transformer.utils import accuracy_fn, infer_one, save_model, PrepareDataset

import numpy as np
import tqdm
import gc


def main():

    # load dataset
    prepare_dataset                 = PrepareDataset()
    max_dataset_length              = 1000
    dataset, enc_vocab, dec_vocab   = prepare_dataset.load_dataset('./data/english-german-both.pkl', max_len=max_dataset_length, store=True)
    train, test                     = prepare_dataset.train_test_split(dataset, split=0.8, shuffle=True, store=True)
    trainX, trainY                  = prepare_dataset.create_enc_dec_inputs(train, store=True)
    
    # network params
    n_heads                 = 8
    d_model                 = 256
    d_k, d_v                = 256, 256
    d_ff                    = 1024
    n_layers                = 3
    dropout                 = 0.10
    inp_enc_seq_length      = prepare_dataset.find_seq_length(train[:,0]) - 1
    inp_dec_seq_length      = prepare_dataset.find_seq_length(train[:,1]) - 1

    model                   = Transformer(enc_vocab, dec_vocab, n_heads, d_model, d_k, d_v, d_ff, n_layers, 
                                            inp_enc_seq_length, inp_dec_seq_length, label="Transformer", dropout=dropout)


    print(f"training Transformer with: \n n_heads={n_heads}, d_model={d_model}, d_k={d_k}, d_v={d_v}, d_ff={d_ff}, n_layers={n_layers}")
    
    # training params
    loss_fn                 = CCELoss()
    n_epochs                = 50
    batch_size              = 64
    mini_batchsize          = 8
    optimizer               = Adam(model.weights, beta1=0.9, beta2=0.98, eps=1e-6, lr=1.0)
    warmup_steps            = 4000

    def train_one_minibatch(step_num, i, pbar, batch_size, mini_batchsize):
        i_copy          = i
        optimizer.zero_grad()
        model.model_reset()

        loss_avg        = []
        y_preds, targets= [], []
        
        for _ in range(0, batch_size, mini_batchsize):
            encoder_input   = Tensor(trainX[i_copy:i_copy+mini_batchsize,1:], leaf=True)
            decoder_input   = Tensor(trainY[i_copy:i_copy+mini_batchsize,:-1], leaf=True)
            decoder_output  = Tensor(trainY[i_copy:i_copy+mini_batchsize,1:], leaf=True)
            if encoder_input.shape[0] != mini_batchsize or decoder_input.shape[0]!= mini_batchsize:
                break
            y_pred          = model(enc_inp=encoder_input, dec_inp=decoder_input, training=True)  # calls either new or same model copy
            target_labels   = Tensor(model.onehot_tokens(decoder_output), learnable=False, leaf=True)
            loss            = loss_fn(y_pred, target_labels, mask=True)
            ##### determine the learning rate ######
            arg1            = step_num ** -0.5
            arg2            = step_num * (warmup_steps ** -1.5)
            optimizer.lr    = (model.d_model ** -0.5) * np.min([arg1, arg2]) * 5
            ########################################
            loss.backward(reset_grad=True)                          # resets model copy graph gradients first, then computes new gradient
            loss_avg.append((1/batch_size) * loss.value.item())
            y_preds.extend(y_pred.value)
            targets.extend(target_labels.value)
            optimizer.step_single(loss, batch_size, modify=False)
            i_copy      += mini_batchsize
            pbar.set_postfix({'lr': optimizer.lr, 
                              'step_num': step_num, 
                              'i_copy': i_copy%batch_size, 
                              'batch_size': batch_size,
                              'rolling batch loss': np.mean(loss_avg),
                              'rolling batch accuracy:': accuracy_fn(np.array(y_preds), np.array(targets)).item()
                              })
        
        accuracy_fn(y_pred.value, target_labels.value).item()
        optimizer.step_single(loss, batch_size, modify=True)
        gc.collect()
        return loss_avg

    print("starting training")
    step_num                = 1
    for e in range(n_epochs):
        # random_perms = np.random.permutation(trainX.shape[0])
        # trainX = np.array(trainX)[random_perms]
        # trainY = np.array(trainY)[random_perms]
        losses = []
        with tqdm.tqdm(range(0, len(trainX)-batch_size, batch_size)) as pbar:
            for i in pbar:
                loss = train_one_minibatch(step_num, i, pbar, batch_size, mini_batchsize)
                step_num       += 1
                losses.extend(loss)
        
        infer_one(model, prepare_dataset)
        save_model(f"examples/transformer/model_saves/model_epoch_{e}", model, prepare_dataset)
    print("Finished training model.")

if __name__ == "__main__":
    main()
