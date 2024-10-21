
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autograd.cpu.losses import CCELoss
from autograd.cpu.optims import RMSProp
from autograd.cpu.tensor import Tensor

from examples.cnn.utils import accuracy_fn, save_model
from architectures.cnn import CNN

import numpy as np
import tqdm
import gc

def main():

    # load data
    trainX = np.load("data/MNIST_trainX.npy")*255.
    trainY = np.load("data/MNIST_trainY.npy")

    # prepare model
    model       = CNN()
    loss_fn     = CCELoss()
    optim       = RMSProp(model.weights, beta=0.9, lr=0.01)
    n_epochs    = 2
    batch_size  = 16

    # train
    for e in range(n_epochs):
        model.model_reset()
        random_perms = np.random.permutation(trainX.shape[0])
        trainX = np.array(trainX)[random_perms]
        trainY = np.array(trainY)[random_perms]
        with tqdm.tqdm(range(0, len(trainX)//50-batch_size, batch_size)) as pbar:
            for batch_idx in pbar:
                optim.zero_grad()
                x_val = Tensor(trainX[batch_idx:batch_idx+batch_size], learnable=False, leaf=True)
                y_true= Tensor(trainY[batch_idx:batch_idx+batch_size], learnable=False, leaf=True)
                y_pred = model(x=x_val, training=True)

                loss = loss_fn(y_pred, y_true)
                loss.backward()

                optim.step(loss)
                model.model_reset()
                
                pbar.set_postfix({'epoch': e,
                                'lr': optim.lr,
                                'batch_idx': batch_idx,
                                'batch loss': loss.value.item(),
                                'batch pred accuracy:': accuracy_fn(y_pred.value, y_true.value).item()
                                })
                gc.collect()
        save_model(f"examples/cnn/model_saves/model_epoch_{e}", model)


if __name__ == "__main__":
    main()

