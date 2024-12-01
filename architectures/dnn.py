
import numpy as np

from pygrad.activations import ReLU
from pygrad.basics import Dropout, Flatten, Linear
from pygrad.module import Module
from pygrad.tensor import Tensor
from pygrad.constants import PRECISION


class DNN(Module):
    def __init__(self, batch_size=1, label="DNN", 
                 dropout=0.10, dtype=PRECISION):
        
        self.dtype          = dtype
        self.label          = label
        self.dropout_rate   = dropout

        self.flatten        = Flatten()
        self.dense1         = Linear(i_dim=28*28, o_dim=100)
        self.dropout1       = Dropout(rate=self.dropout_rate)
        self.relu1          = ReLU()
        self.dense2         = Linear(i_dim=100, o_dim=10)
        super().__init__(x=Tensor(np.ones((batch_size, 1, 28, 28), dtype=self.dtype), leaf=True))

    def forward(self, x:Tensor):
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        return x
