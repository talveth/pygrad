.. _modules:

Modules
============

.. _tensor:

tensor
------

The ``tensor`` module contains the main `Tensor` class responsible for providing automatic differentiation capability.

The `Tensor` object is a NumPy array holding gradients, supporting a variety of common operations, computing gradients when request.

Values and Gradients
*********************

In its simplest form, `Tensors` are initialized with a `value`. 
Values accepted are (non-complex) numeric types, lists, and NumPy arrays.:

.. code-block:: Python

   a = Tensor(value=1)
   b = Tensor(np.random.normal(0,1,(100,100)))
   c = Tensor([2])

Gradients of each `Tensor` are initialized ``0``, stored in ``.grad``, and are of the same shape as the passed-in ``value``. 
Gradients are ``0`` until backpropagation is manually called upon either the Tensor or a referee:

.. code-block:: Python

   print(a.value, a.grad)
   # -> 1.0 0.0


Operations and Topological Graphs
**********************************

`Tensors` support a variety of operations as methods:

    * dunder methods: __add__, __sub__, __neg__, __truediv__, __mul__, __matmul__, __pow__
    * other common ops: sum, reshape, transpose, T, mean, std, conv2D, mask_idcs
    * activations: relu, tanh, sigmoid, softmax
    * NumPy-level broadcasting and slicing

For example, the below are all valid:

.. code-block:: Python

    x = Tensor(value=np.ones((10,20,30)))

    # built-ins
    y = x + 2*x - x/2 + 3
    y = (x**2)@np.ones((10,30,20))
    y = x @ np.ones((1,20,30))      # broadcasting is performed
    y = (-1*(x+2*x-3*x)/4 @ np.ones((10,30,20))) **2

    # other
    y = x.sum(axis=0); x.sum(axis=(0,1,2)); x.sum()
    y = x.mean() + x.std()
    y = x.transpose(); x.T
    y = x.reshape((30,-1))
    x2     = np.ones((1, 3, 32, 32))
    kernel = np.ones((1, 1, 3, kH=5, kW=5))
    kernel.Conv2D(x2)

    # activations
    x.relu()
    x.tanh()
    x.reshape((-1,20,1)).sigmoid()
    x.softmax()

Applying an operation on a `Tensor` will always produce a new Tensor whose operands are those Tensors' children.
By applying successive operations on a `Tensor`, a computational graph is built and stored, thus, each `Tensor`
has memory of the `Tensors` used to create it. The direct children of a Tensor can be found in its ``._prev`` attribute.

A `Tensor`'s computational graph can be generated at any point in time with `.create_graph()`. 
This method returns two computational graphs: a complete topological graph, ``topo`` and a weights-only graph ``weights``.

Both graphs are reverse-ordered (pre-order traversed) lists of operands performed on `self`, with `self` as root.

    * ``topo`` stores all Tensors that interacted to produce the Tensor.
    * ``weights`` only Tensors with learnable weights that produced the Tensor.

The following illustrates their difference:

.. code-block:: Python

    x = Tensor(value=1, label='x')
    y = x**2 + x + 1
    topo, weights = y.create_graph()

In the above, 3 operations are performed on a created Tensor ``x``, thus, ``topo`` is a list with the following 
elements in this order:

    #. ``x``
    #. ``x**2``
    #. ``x**2 + x``
    #. ``y=x**2 + x + 1``

``weights`` however only contains the ``x`` Tensor, as in producing ``y``, 
only one Tensor had learnable weights, with all intermediary Tensors not contributing uniquely to ``y``.

Backwards propagation
************************

Having a set of ``weights`` for a Tensor allows performing backpropagation on that Tensor, 
updating only those Tensors whose values directly contribute, ignoring the rest. 
This is how ``backward()`` is implemented in the library.

Backpropagation can be directly performed on a Tensor at any time using the ``.backward()`` method.
This populates the gradients of all contibuting weights to that Tensor.

.. code-block:: Python

    x = Tensor(value=1, label='x')
    y = x**2 + x + 1
    y.backward()
    print(x.grad)           
    # -> 3.0
    y.backward()
    print(x.grad)
    # -> 3.0
    y.backward(reset_grad=False)
    print(x.grad)
    # -> 11.0

Backprop can be applied as many times as needed on a Tensor, however will default to resetting all previous backwards passes.
To perform backpropagation multiple times on the same computational graph, set ``reset_grad=False``.
Each new backprop adds the previous gradients to the new one, using these added gradients for gradient 
computations further down the computational graph.

Gradient Descent example
************************

The following shows a simple example of performing gradient descent on the Tensor ``x=1``.

.. code-block:: Python

    from pygrad.tensor import Tensor

    n_iters = 1000
    stepsize= 0.01
    x       = Tensor(1)

    for _ in range(n_iters):
        loss_fn = (x-1.5)**2
        loss_fn.backward(reset_grad=True)
        x.value = x.value - stepsize*x.grad

    print(x.value, loss_fn.value)
    # -> 1.4336... 0.0045...


For performing backprop automatically with more complex functions, use the :ref:`optims` module.
Vectorized backprop with batched data is also supported, however to not modify the underlying model 
whilst performing vectorized forward and backwards passes will require creating a subclass of ``Module``. 

For all Tensor methods see: :ref:`Tensor Methods <tensormethods>`

.. _basics:

basics, activations, and losses
-------------------------------

The ``basics, activations, and losses`` modules extend the functionality of the ``Tensor``, by providing Pytorch-like classes
that create a variety of higher-order Tensors commonly used in deep-learning.

These include:

    * Dropout, AddNorm, Linear, Softmax, Flatten, Conv2D layers,
    * ReLU activation,
    * BCELoss, CCELoss losses.

The above classes contain no dependencies other than the ``Tensor`` object and ``NumPy``.
Since ``Tensors``s use ``NumPy`` arrays under the hood, creating custom classes is thus very simple.
For example, defining Dropout is done as follows:

.. code-block:: Python

    from pygrad.tensor import Tensor
    import numpy as np

    class Dropout:
        def __init__(self, rate:float=0.1):
            self.rate = rate

        def __call__(self, x:Tensor, training:bool=True) -> Tensor:
            if training:
                n_points        = int(np.prod(x.shape)*self.rate)
                arr_indices     = np.unravel_index(np.random.choice(np.arange(0, np.prod(x.shape)), 
                                                    size=n_points, 
                                                    replace=False), x.shape)
                dropouted_pts   = x.mask_idcs(arr_indices)
                return dropouted_pts
            else:
                return x


Backpropagation can now be done using this Class, no different than with any other Tensor:

.. code-block:: Python

    x = Tensor(np.array([1,1,1,1]))
    d = Dropout(0.5)
    otp = d(x)
    otp.value
    # -> array([1., 0., 0., 1.])
    otp.backward()
    x.grad
    # -> array([1., 0., 0., 1.])

For all methods in basics, activations, and losses, see: :ref:`basics, activations, losses <basics_methods>`.

.. _optims:

optims
-------

Classes for gradient descent such as SGD, SGD with Momentum, RMSProp, and Adam are defined here.
Optimizers are designed to work with the ``weights`` of a Tensor (called a ``model``), each having a ``.zero_grad`` method 
for resetting Tensor gradients, a ``.step`` method for updating model weights given a loss function, and a ``.step_single`` method
for updating model weights progressively in a memory-sensitive manner when model weights are large. 
This method is further explained under :ref:`module`.

Basic usage is the same across all optimizers; 
initialize the optimizer with the model weights along with optimizer-specific parameters; 
reset the model gradient; 
do a forward pass and a backwards pass with a specified loss function;
and step with the optimizer, feeding in the loss function.

.. code-block:: Python

    from pygrad.tensor import Tensor
    from pygrad.optims import SGD

    x     = Tensor([1])
    y     = x**2 + 1
    model = y.create_graph()[1]                # fetching .weights from Tensor y
    optim = SGD(model, lr=0.01)

    for _ in range(100):
        optim.zero_grad()
        y    = x**2 + 1
        loss = (y-1.5)**2
        loss.backward()
        optim.step(loss)

    print(x.value, y.value, loss.value)
    # -> 0.7100436 1.50433688 1.88085134e-05

For all available options, see: :ref:`optims options <optimsmethods>`.

.. _module:

module
-------

The ``Module`` class gives the ability to perform batched forward and backward passes on the model without mutating the model. 
Functions defined as classes representing models can also easily use optimizers as defined in :ref:`optims`.

Below shows how to convert a class-defined function into one subclassing ``Module``.

.. code-block:: Python

    class DNN:
        """Dense Neural Network, (28,28) -> (10) """
        def __init__(self, dtype=np.float32):
            
            self.dtype          = dtype
            self.flatten        = Flatten()
            self.dense1         = Linear(i_dim=28*28, o_dim=100)
            self.relu1          = ReLU()
            self.dense2         = Linear(i_dim=100, o_dim=10)

        def forward(self, x:Tensor):
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.relu1(x)
            x = self.dense2(x)
            return x


The following have to now take place:
    #. Subclassing (``pygrad.module.Module``)
    #. A line ``super().__init__``, passing in the expected model forward-pass inputs, with:
        * each input that is of type ``Tensor`` has to have set ``leaf=True``
    #. Any calling of the model that has ``Tensor`` inputs requires the tensor to have ``leaf=True``
    #. Any calling of the model requires keyword inputs.

.. code-block:: Python

    class DNN2(Module):
        def __init__(self, dtype=PRECISION):
            
            self.dtype          = dtype
            batch_size          = 1
            self.flatten        = Flatten()
            self.dense1         = Linear(i_dim=28*28, o_dim=100)
            self.relu1          = ReLU()
            self.dense2         = Linear(i_dim=100, o_dim=10)
            super().__init__(x=Tensor(np.ones((batch_size, 1, 28, 28), dtype=self.dtype), leaf=True))

        def forward(self, x:Tensor):
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.relu1(x)
            x = self.dense2(x)
            return x

By subclassing, model forward passes can be performed by calling the model on the needed inputs, 
ensuring that all input keyword arguments are specified:

.. code-block:: Python

    model = DNN2()
    model(**kwargs)


The below now illustrates the difference:

.. code-block:: Python

    dnn1    = DNN()
    dnn2    = DNN2()

    batch_size = 16
    x          = Tensor(np.ones((batch_size,28,28), dtype=np.float32), leaf=True)

    fwd1    = dnn1.forward(x)   # shape=(batch_size, 1, 10)
    fwd2    = dnn2(x=x)         # shape=(batch_size, 1, 10)

    fwd1.backward()
    fwd2.backward()

    dnn1.dense2.W.value.shape, dnn2.dense2.W.value.shape
    # -> ((batch_size, 100, 10), (1, 100, 10))
    dnn1.dense2.W.grad.shape, dnn2.dense2.W.grad.shape
    # -> ((batch_size, 100, 10), (1, 100, 10))


Both versions are able to apply batched forward passes on the input. 
However, due to Tensor automatically rescaling due to broadcasting, only the model subclassing ``Module`` 
is able to maintain the originally instantiated shape of values and gradients.

Performing gradient descent on the original model would require resetting values and gradient shapes
of each model weight, and updating gradients according to the batched versions, undoing any broadcasting. 
This process is done automatically when subclassing ``Module``, with the batched copy of the ``model`` 
available under ``model.copy``.


Using ``Module`` makes it easy to perform gradient descent with :ref:`optims`:

    * Model weights are found in ``model.weights``. These weights are given to the optimizer for updating.
    * Batched model data which is stored in the model after calling on batched data is reset with ``model.model_reset()``. If this is not reset, the previous model gradients will accumulate. This will also stop the model from training if different batch sizes are given from one training epoch from the next.

.. code-block:: Python

    from pygrad.tensor import Tensor
    from pygrad.optims import SGD
    import numpy as np

    model       = DNN2()                        # defined previously, subclassing Module
    optim       = SGD(model.weights, lr=0.1)    # model.weights property is available

    n_epochs    = 25
    batch_size  = 4

    for _ in range(n_epochs):
        x           = Tensor(np.random.uniform(0,1,(batch_size,28,28)), leaf=True)
        y_true      = Tensor(np.ones((batch_size,1,10), dtype=np.float32), leaf=True)

        model.model_reset()
        optim.zero_grad()
        y_pred = model(x=x)
        loss   = ((y_pred - y_true)**2).sum(axis=-1).mean(axis=0, keepdims=False) # averages over the batch
        loss.backward()
        optim.step(loss)

The model's weights will be updated here according to losses averaged over the batch, but without any change to their originally defined shape.
For more training examples, see :ref:`examples`.

For class methods, see :ref:`Module <module_methods>`
