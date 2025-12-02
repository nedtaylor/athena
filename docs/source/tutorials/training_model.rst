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

* 1D, 2D, 3D, 4D, or 5D real arrays
* scalar or 2D array of type ``array_type`` from the diffstruc library (alos imported with athena)
* ``array_container_type`` derived type from the diffstruc library
* scalar or 2D array of type ``array_container_type`` from the athena library (i.e. a container containg mutliple ``array_type`` instances)
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

This can take in ``real``, ``array_type``, ``array_container_type``, or ``graph_type`` input data.
The output will be the same type as the input data, but two-dimensional (except for ``array_container_type``, which will output ``array_type``).
For ``real`` data, therefore, the output will be a 2D real array, where the first dimension corresponds to the output features (e.g., classes), and the second dimension corresponds to the samples.
For more control over output format, the argument ``output_as_array`` can be set to ``.true.`` to enforce output as ``array_type`` output, regardless of input type.
Finally, a polymorphic version of ``predict()`` is available called ``predict_generic()`` that can handle any of the above input types and outputs either an ``array_type`` or ``graph_type`` output, depending on the value of the input argument ``output_as_graph``.

Computing Convergence Metrics (e.g., Accuracy and Loss)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The accuracy is computed during the ``train()`` and ``test()`` functions.
The accuracy method must be specified during compilation of the network using the ``compile()`` function.
The accuracy value after a training or testing run can be accessed via the ``accuracy_val`` attribute of the network.

.. code-block:: fortran

   write(*,*), "Test Accuracy:", net%accuracy_val

The loss method must also be specified during compilation of the network.
The loss value after a training or testing run can be accessed via the ``loss_val`` attribute of the network.

.. code-block:: fortran

   call net%train(test_data, test_labels)
   write(*,*), "Test Loss:", net%loss_val

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

Other learning rate schedules are available; see :ref:`Learning Rate Decay <learning-rate-decay>`.

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
