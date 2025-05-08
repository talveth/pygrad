
.. _examples:

Examples
=========

Attributes:
------------------------

.. code-block:: Python

    from pygrad.tensor import Tensor
    a = Tensor(1)
    a.value     # -> 1.0
    a.grad      # -> 0.0
    a.label     # -> ""
    a.dtype     # -> PRECISION (pygrad.constants)
    a.learnable # -> True
    a.leaf      # -> False
    a._prev     # -> ()

    # calling a.backward()
    a.grad      # -> 1.0
    a.topo      # -> [a]
    a.weights   # -> [a]


Attributes after basic ops:
---------------------------

.. code-block:: Python

    from pygrad.tensor import Tensor
    a = Tensor(1, label="a")
    b = Tensor(2)
    c = a*b
    c.value     # -> 2.0
    c.grad      # -> 0.0
    a.label     # -> "a"
    c.label     # -> ""
    c.dtype     # -> PRECISION (pygrad.constants)
    c.learnable # -> True
    c.leaf      # -> False
    c._prev     # -> (a,b)

    # calling c.backward()
    a.grad      # -> 2.0
    b.grad      # -> 1.0
    a.topo      # -> []
    b.topo      # -> []
    c.topo      # -> [a, b, c]
    c.weights   # -> [a,b]


Working with NumPy arrays:
---------------------------

.. code-block:: Python

    from pygrad.tensor import Tensor
    import numpy as np
    x   = np.random.uniform(0,1,(10,5))
    xt1 = Tensor(x)
    
    xt1.value       # -> x
    xt1.grad.shape  # -> x.shape

    x + 1           # -> broadcasted x + 1
    x*10            # -> broadcasted x * 10
    

Manual Gradient Descent:
------------------------

.. code-block:: Python

    from pygrad.tensor import Tensor
    n_iters = 1000
    stepsize= 0.01
    x       = Tensor(1)

    for _ in range(n_iters):
        loss_fn = (x-1.5)**2                    # define a loss function
        loss_fn.backward(reset_grad=True)       # run .backward() to compute gradients
        x.value = x.value - stepsize*x.grad

    print(x.value, loss_fn.value)
    # -> 1.4336... 0.0045...


Gradient Descent with ``optim``:
-------------------------------------------

.. code-block:: Python

    from pygrad.tensor import Tensor
    from pygrad.optims import SGD

    x     = Tensor([1])
    y     = x**2 + 1
    optim = SGD(y.create_graph()[1], lr=0.01)

    for _ in range(100):
        optim.zero_grad()
        y    = x**2 + 1
        loss = (y-1.5)**2
        loss.backward()
        optim.step(loss)

    print(x.value, y.value, loss.value) 
    # -> 0.7100436 1.50433688 1.88085134e-05


Full Deep Neural Network:
-------------------------

Note, the following requires the MNIST dataset.

.. code-block:: Python

    import numpy as np
    import tqdm
    from pygrad.tensor import Tensor
    from pygrad.module import Module
    from pygrad.losses import CCELoss
    from pygrad.optims import SGD
    from pygrad.basics import Linear, Dropout, Flatten
    from pygrad.activations import ReLU

    PRECISION = np.float64
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

    def accuracy_fn(y_pred:np.ndarray, y_true:np.ndarray):
        return np.mean(np.argmax(y_pred, axis=-1, keepdims=True) == np.argmax(y_true, axis=-1, keepdims=True))


    # load data
    trainX = np.load("data/MNIST_trainX.npy")*255.
    trainY = np.load("data/MNIST_trainY.npy")

    # prepare model
    model       = DNN()
    loss_fn     = CCELoss()
    optim       = SGD(model.weights, lr=0.1)
    n_epochs    = 2
    batch_size  = 16

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


