
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gc
import os
import sys

import numpy as np
import tqdm

from architectures.dnn import DNN
from pygrad.losses import CCELoss
from pygrad.optims import SGD
from pygrad.tensor import Tensor
from examples.dnn.utils import accuracy_fn, save_model


def main():

    # load data
    trainX = np.load("examples/cnn/MNIST_trainX.npy")*255.
    trainY = np.load("examples/cnn/MNIST_trainY.npy")

    # prepare model
    model = DNN()
    loss_fn     = CCELoss()
    optim       = SGD(model.weights, lr=0.01)
    n_epochs    = 2
    batch_size  = 64

    # train
    print("Training DNN")
    for e in range(n_epochs):
        random_perms = np.random.permutation(trainX.shape[0])
        trainX = np.array(trainX)[random_perms]
        trainY = np.array(trainY)[random_perms]
        model.model_reset()
        with tqdm.tqdm(range(0, len(trainX)-batch_size, batch_size)) as pbar:
            for batch_idx in pbar:
                optim.zero_grad()
                x_val = Tensor(trainX[batch_idx:batch_idx+batch_size], learnable=False, leaf=True)
                y_true= Tensor(trainY[batch_idx:batch_idx+batch_size], learnable=False, leaf=True)
                y_pred = model(x=x_val)

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
        optim.lr /= 10
        save_model(f"examples/dnn/model_saves/model_epoch_{e}", model)


if __name__ == "__main__":
    main()
