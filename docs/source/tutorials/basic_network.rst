.. _basic-network:

Building a Basic Network
=========================

This tutorial will guide you through creating your first neural network using the athena library.

Prerequisites
-------------

Make sure you have athena installed. See :ref:`Installation <install>` for details.

Creating a Simple Network
-------------------------

Let's build a simple feedforward neural network for a classification problem.

Step 1: Import the Library
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   program basic_network
     use athena
     implicit none

     ! Variable declarations will go here

   end program basic_network

Step 2: Define the Network Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now define a variable to hold the network:

.. code-block:: fortran

   type(network_type) :: net

Step 3: Build the Network Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

And build the network layer by layer:

.. code-block:: fortran

   integer :: num_inputs, num_hidden, num_outputs

   ! Define network dimensions
   num_inputs = 784    ! e.g., 28x28 flattened image
   num_hidden = 128
   num_outputs = 10    ! 10 classes

   ! Add first hidden layer with ReLU activation
   call net%add(full_layer_type( &
        num_inputs=num_inputs, &
        num_outputs=num_hidden, &
        activation="relu"))

   ! Add second hidden layer
   call net%add(full_layer_type( &
        num_outputs=num_hidden, &
        activation="relu"))

   ! Add output layer with softmax for classification
   call net%add(full_layer_type( &
        num_outputs=num_outputs, &
        activation="softmax"))

.. note::

   For simple networks (i.e. single input layer), the input layer is created automatically during the compile step.

Step 4: Compile the Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialise the network with an optimiser and loss function:

.. code-block:: fortran

   type(adam_optimiser_type) :: optimiser
   type(cce_loss_type) :: loss

   ! Set up optimiser
   optimiser = adam_optimiser_type(learning_rate=0.001)

   ! Set up loss function
   loss = cce_loss_type()

   ! Compile network
   call net%compile(optimiser=optimiser, loss_method=loss, accuracy_method="mse")

Currently, the accuracy method needs to be specified for ``train()`` and ``test()`` functions to work correctly.
This may be updated in future releases.
If it is not set, the following error will occur when calling ``train()`` or ``test()`` and the program will terminate:

.. code-block:: fortran

  ERROR: accuracy method not set


Complete Example
----------------

Here's the complete program:

.. code-block:: fortran

   program basic_network
     use athena
     implicit none

     type(network_type) :: net
     type(adam_optimiser_type) :: optimiser
     type(cce_loss_type) :: loss
     integer :: num_inputs, num_hidden, num_outputs

     ! Define dimensions
     num_inputs = 784
     num_hidden = 128
     num_outputs = 10

     ! Build network
     call net%add(full_layer_type(num_inputs=num_inputs, num_outputs=num_hidden, activation="relu"))
     call net%add(full_layer_type(num_outputs=num_hidden, activation="relu"))
     call net%add(full_layer_type(num_outputs=num_outputs, activation="softmax"))

     ! Compile
     optimiser = adam_optimiser_type(learning_rate=0.001)
     loss = cce_loss_type()
     call net%compile(optimiser=optimiser, loss=loss)

     ! Print network summary
     call net%print_summary()

   end program basic_network

Network Variations
------------------

Below are some variations you can try to modify the network architecture, optimiser, and loss function.

Different Architectures
~~~~~~~~~~~~~~~~~~~~~~~

Different architectures can be achieved through adding or modifying layers in the network.
For a full list of available layers, see :ref:`Layers <layers>` or refer to the :ref:`API <api>`.

**Deeper Network:**

.. code-block:: fortran

   call net%add(full_layer_type(num_inputs=num_inputs, num_outputs=256, activation="relu"))
   call net%add(full_layer_type(num_outputs=128, activation="relu"))
   call net%add(full_layer_type(num_outputs=64, activation="relu"))
   call net%add(full_layer_type(num_outputs=num_outputs, activation="softmax"))

**With Dropout for Regularisation:**

.. code-block:: fortran

   call net%add(input_layer_type(input_shape=[num_inputs]))
   call net%add(full_layer_type(num_outputs=128, activation="relu"))
   call net%add(dropout_layer_type(rate=0.5, num_masks=10))
   call net%add(full_layer_type(num_outputs=128, activation="relu"))
   call net%add(dropout_layer_type(rate=0.5, num_masks=10))
   call net%add(full_layer_type(num_outputs=num_outputs, activation="softmax"))

**With Batch Normalisation:**

.. code-block:: fortran

   call net%add(input_layer_type(input_shape=[num_inputs]))
   call net%add(full_layer_type(num_outputs=128, activation="relu"))
   call net%add(batchnorm1d_layer_type(num_channels=128))
   call net%add(full_layer_type(num_outputs=128, activation="relu"))
   call net%add(batchnorm1d_layer_type(num_channels=128))
   call net%add(full_layer_type(num_outputs=num_outputs, activation="softmax"))

For more complex architectures, such as convolution, residual, or physics-informed networks, refer to the respective tutorials in the :ref:`Layers <layers>` section.

Different Optimisers
~~~~~~~~~~~~~~~~~~~~

Different optimisation algorithms can be used to train the network.
The different optimisers are suited to different types of problems and datasets.
For a full list of available optimisers, see :ref:`Optimisers <optimisers>`.

**SGD with Momentum:**

.. code-block:: fortran

   optimiser = sgd_optimiser_type( &
        learning_rate=0.01, &
        momentum=0.9, &
        nesterov=.true.)

**RMSprop:**

.. code-block:: fortran

   optimiser = rmsprop_optimiser_type( &
        learning_rate=0.001, &
        beta=0.9)

Different Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~~

Choice of loss function depends on the task at hand.
Different loss functions guide the learning process in different ways and can be used, in some cases, to encode prior knowledge about the problem.
For a full list of available loss functions, see :ref:`Loss Functions <loss-functions>`.

**For Regression:**

.. code-block:: fortran

   type(mse_loss_type) :: loss
   loss = mse_loss_type()

**For Binary Classification:**

.. code-block:: fortran

   type(bce_loss_type) :: loss
   loss = bce_loss_type()

Next Steps
----------

Now that you've created a basic network, learn how to:

* :ref:`Train your model <training-model>`
* :ref:`Save and load models <saving-loading>`
* :ref:`Follow examples and advanced tutorials <tutorials>`

See Also
--------

* :ref:`Layers <layers>` - Complete list of available layers
* :ref:`Optimisers <optimisers>` - Available optimization algorithms
* :ref:`Loss Functions <loss-functions>` - Available loss functions
