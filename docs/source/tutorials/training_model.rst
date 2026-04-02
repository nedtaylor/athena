.. _training-model:

Training a Model
================

This tutorial covers how to train, evaluate, and monitor your neural network.

Prerequisites
-------------

You should have a compiled network. See :ref:`Building a Basic Network <basic-network>`.

Preparing Training Data
-----------------------

Currently, the ``train`` and ``test`` functions accept the following data types:

* 1D, 2D, 3D, 4D, or 5D ``real`` arrays
* scalar or 2D array of type ``array_type`` from the diffstruc library (also imported with athena)
* ``array_ptr_type`` a container containing pointers to multiple ``array_type`` instances (NOT FULLY SUPPORTED YET)
* scalar or 1D array of type ``array_ptr_type`` from the athena library (i.e. a container containg pointers to ``array_type`` instances)
* 1D or 2D array of type ``graph_type`` from the graphstruc library (also imported with athena)

Real arrays and ``array_type`` arrays are the most likely to be used for typical training scenarios.
``graph_type`` is useful for graph neural networks.

The expected format of the input and output data depends on the network architecture and task.
However, the consistency is to have samples (or batch size) as the last dimension.
For spatial data, the channels are often the second to last dimension.
For example, for a fully connected network with 100 samples, each with 784 features (e.g., flattened 28x28 images), the input data should be shaped as (784, 100).
For a convolutional network processing 28x28 RGB channel images with 100 samples, the input data should be shaped as (28, 28, 3, 100).

Data Format
~~~~~~~~~~~

Here, we illustrate preparing data for training a simple fully connected network for classification.

.. code-block:: fortran

   real(real32), allocatable :: train_data(:,:)
   real(real32), allocatable :: train_labels(:,:)
   real(real32), allocatable :: test_data(:,:)
   real(real32), allocatable :: test_labels(:,:)

   integer :: num_samples, num_features, num_classes

   ! Allocate arrays
   num_samples = 1000
   num_features = 784
   num_classes = 10

   allocate(train_data(num_features, num_samples))
   allocate(train_labels(num_classes, num_samples))

For classification, labels should be one-hot encoded:

.. code-block:: fortran

   ! Example: one-hot encode label for class 3 (out of 10 classes)
   train_labels(:, sample_idx) = 0.0
   train_labels(3, sample_idx) = 1.0

If using the ``array_type``, it would be prepared the following way:

.. code-block:: fortran

   use diffstruc__array_type

   type(array_type) :: train_data_array, train_labels_array
   integer :: i

   ! Initialise array_type instances
   call train_data_array%allocate(shape=[num_features, num_samples])
   call train_labels_array%allocate(shape=[num_classes, num_samples])

   ! Fill data
   do i = 1, num_samples
      train_data_array%val(:, i) = ... ! fill with sample data
      train_labels_array%val(:, i) = 0.0
      train_labels_array%val(3, i) = 1.0 ! one-hot encode class 3
   end do

In athena examples, it is common to see a 2D ``array_type`` array be used for both input data and labels, with both dimensions set to a length of 1.
This is done because the expected input argument for the forward pass procedure (``forward()``) of layers is a 2D ``array_type`` array, and the output component of each layer is also a 2D ``array_type`` array.
This is because it is most adaptable to any type of data, including data for simple networks to representing graph data and multi-input data,

Training the Network
--------------------

Basic Training
~~~~~~~~~~~~~~

For basic training, specify the input data, output labels, batch size, and number of epochs.
The ``train()`` function handles the training loop internally.

.. code-block:: fortran

   integer :: num_epochs, batch_size

   num_epochs = 50
   batch_size = 32

   ! Train the network
   call net%train( &
        input=train_data, &
        output=train_labels, &
        batch_size=batch_size, &
        num_epochs=num_epochs)

The above is the high-level training interface.

Internally, ``train()`` switches the network to training mode (and internally resets the mode at the end of the call). For low-level
manual loops that use ``forward()`` directly, call
``set_training_mode()`` yourself when you need training-time behaviour from
layers such as dropout or batch normalisation.

For more custom training logic, a low-level training loop can be implemented manually.
This will take the form:

.. code-block:: fortran

   integer :: n
   type(array_type), pointer :: output(:,:), loss(:,:)
   type(loss_type), pointer :: loss

   do n = 1, num_epochs
     ! 1. Forward pass
     call network%forward(x)

     ! 2. Set expected output
     network%expected_array = y_array

     ! 3. Backward pass (compute gradients)
     loss => network%loss_eval(1, 1)
     call loss%grad_reverse()

     ! 4. Update weights
     call network%update()
     call loss%nullify_graph()
     nullify(loss)
   end do

There are many ways that this can be modified beyond this basic structure, depending on your needs, but this works as a starting point.
When deciding between using the built-in ``train()`` function or implementing a custom training loop, consider the following:

**High-level ``train()``**

* Production code
* Standard training workflows
* Batch processing
* Built-in metrics and logging

**Low-level loop**

* Learning how training works
* Custom training logic
* Research and experimentation
* Fine-grained control needed


Evaluating the Model
--------------------

After training, the model performance can be evaluated.

Making Predictions
~~~~~~~~~~~~~~~~~~

Predictions can be made using the ``predict()`` function.

.. code-block:: fortran

   real(real32), allocatable :: predictions(:,:)

   ! Get predictions for test data
   predictions = net%predict(test_data)

The ``predict()`` method switches the network to inference mode (and internally resets the mode at the end of the call). If you later
return to a custom training loop based on ``forward()``, call
``set_training_mode()`` before continuing.

This can take in ``real``, ``array_type``, or ``graph_type`` input data.
Unless you are using advanced network architectures (such as multi-input networks or graph neural networks), the input data will either be a ``real`` array of rank 1-5, or a scalar ``array_type`` array.
The graph input will typically be of rank 1, representing a list of graphs.
For all input types using simple network architectures, the output will be a 2D array with shape:

* ``real``: :math:`(O, N)`, where :math:`O` is the number of output features (e.g., classes) and :math:`N` is the number of samples.
* ``array_type``: :math:`(1, 1)` (for compatibility reasons it is returned as a 2D array, even for single-output networks)
* ``graph_type``: :math:`(1, N)`, where :math:`N` is the number of graphs.

For more advanced architectures, please refer to :ref:`Network Outputs <network-outputs>`.


Computing Convergence Metrics (e.g., Accuracy and Loss)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loss is always computed during ``train()`` and ``test()``.

Accuracy is optional and is only computed when an accuracy method is configured
(for example with ``compile(..., accuracy_method=...)``).

When accuracy is enabled, the final value after training or testing is available
via ``accuracy_val``.

.. code-block:: fortran

   write(*,*), "Test Accuracy:", net%accuracy_val

The loss method must also be specified during compilation of the network.
The loss value after a training or testing run can be accessed via the ``loss_val`` attribute of the network.

.. code-block:: fortran

   call net%train(test_data, test_labels)
   write(*,*), "Test Loss:", net%loss_val

For detailed ``train()`` argument documentation, including ``print_precision``,
``scientific_print``, and when ``train_acc`` is printed, see
:ref:`train() Subroutine <train-subroutine>` and
:ref:`Network Modes <network-modes>`.

Complete Training Example
--------------------------

.. code-block:: fortran

   program train_network
     use athena
     implicit none

     type(network_type) :: net
     type(adam_optimiser_type) :: optimiser
     type(cce_loss_type) :: loss

     real(real32), allocatable :: train_data(:,:), train_labels(:,:)
     real(real32), allocatable :: test_data(:,:), test_labels(:,:)
     real(real32), allocatable :: predictions(:,:)
     real(real32) :: accuracy, test_loss

     integer :: num_epochs, batch_size
     integer :: correct, total, i, pred_class, true_class

     ! Load or generate data (not shown)
     ! allocate and fill train_data, train_labels, test_data, test_labels

     ! Build network
     call net%add(full_layer_type(num_inputs=784, num_outputs=128, activation="relu"))
     call net%add(full_layer_type(num_outputs=10, activation="softmax"))

     ! Compile
     optimiser = adam_optimiser_type(learning_rate=0.001)
     loss = cce_loss_type()
     call net%compile(optimiser=optimiser, loss_method=loss, accuracy_method="mse")

     ! Train
     num_epochs = 50
     batch_size = 32
     call net%train( &
          input=train_data, &
          output=train_labels, &
          batch_size=batch_size, &
          num_epochs=num_epochs)

     ! Evaluate
     predictions = net%predict(test_data)
     call net%test(test_data, test_labels)
     write(*,*) "Test loss: ", net%loss_val
     write(*,*) "Test accuracy: ", net%accuracy_val

   end program train_network

Advanced Training Techniques
-----------------------------

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~

The learning rate can be adjusted during training using learning rate schedules.

.. code-block:: fortran

   use athena__learning_rate_decay

   type(step_lr_decay_type) :: lr_schedule

   ! Reduce learning rate by 0.5 every 10 epochs
   lr_schedule = step_lr_decay_type(decay_steps=10, decay_rate=0.5)

   optimiser = adam_optimiser_type( &
        learning_rate=0.001, &
        lr_decay=lr_schedule)

Other learning rate schedules are available; see :ref:`Learning Rate Decay <lr-decay>`.

Plateau Detection
~~~~~~~~~~~~~~~~~

A plateau detection can be used to determine whether no additional improvement is being made during training.
This option can be specified during ``train()`` by setting the ``plateau_threshold`` argument to the value of the minimum change in loss to be considered an improvement.

.. code-block:: fortran

   call net%train( &
        input=train_data, &
        output=train_labels, &
        batch_size=batch_size, &
        num_epochs=num_epochs, &
        plateau_threshold=1.0e-4)

The default value is ``0.0``, which disables plateau detection.

Troubleshooting
---------------

Below are some common issues and potential solutions when training neural networks.
This is just a brief overview; much more detailed troubleshooting guides can be found in the literature.

Loss Not Decreasing
~~~~~~~~~~~~~~~~~~~

* **Check learning rate**: Too high or too low can prevent learning
* **Verify data**: Ensure proper normalisation and correct labels
* **Try different optimiser**: Adam often works better than SGD
* **Check network architecture**: May be too shallow or too deep

Overfitting
~~~~~~~~~~~

* **Add dropout layers**: Helps regularisation
* **Reduce network size**: Fewer parameters
* **Get more training data**: Or use data augmentation
* **Add L2 regularisation**: Penalise large weights

Underfitting
~~~~~~~~~~~~

* **Increase network capacity**: More layers or neurons
* **Train longer**: More epochs
* **Reduce regularisation**: If too aggressive
* **Check data quality**: Ensure sufficient information

Next Steps
----------

* :ref:`Save and load your trained model <saving-loading>`
* :ref:`Try the MNIST example <mnist-example>`

See Also
--------

* :ref:`Optimisers <optimisers>` - Optimisation algorithms
* :ref:`Loss Functions <loss-functions>` - Available loss functions
