
import numpy as np

from pygrad.activations import ReLU
from pygrad.basics import Conv2D, Dropout, Flatten, Linear
from pygrad.module import Module
from pygrad.tensor import Tensor
from pygrad.constants import PRECISION


class CNN(Module):
    def __init__(self, batch_size=1, label="CNN", 
                 dropout=0.10, dtype=PRECISION):
        
        self.dtype          = dtype
        self.label          = label
        self.dropout_rate   = dropout
        self.conv1          = Conv2D(o_dim=4, i_dim=1, kH=5, kW=5)
        self.relu1          = ReLU()
        self.conv2          = Conv2D(o_dim=1, i_dim=4, kH=5, kW=5)
        self.relu2          = ReLU()
        self.dropout        = Dropout(self.dropout_rate)
        self.flatten        = Flatten()
        self.dense1         = Linear(i_dim=20*20, o_dim=100)
        self.relu3          = ReLU()
        self.dense2         = Linear(i_dim=100, o_dim=10)
        super().__init__(x=Tensor(np.ones((batch_size, 1, 28, 28), dtype=self.dtype), leaf=True), training=False)

    def forward(self, x:Tensor, training:bool=False):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.dropout(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu3(x)
        x = self.dense2(x)
        return x
