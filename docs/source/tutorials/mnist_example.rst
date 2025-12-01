.. _mnist-example:

MNIST Classification Example
=============================

Complete example of training a convolutional neural network on the MNIST handwritten digit dataset.

This tutorial walks through the actual ``example/mnist`` code from the athena repository.

Overview
--------

This example demonstrates:

* Loading MNIST data using custom reader
* Building a simple CNN architecture
* Training with the high-level ``train()`` method
* Testing model accuracy
* Saving trained models

Network Architecture
--------------------

The network uses a simple but effective architecture:

.. code-block:: fortran

   ! Pad input images
   call network%add(pad2d_layer_type( &
        input_shape=[28, 28, 1], &
        padding=[1, 1], &
        method="constant"))

   ! Convolutional layer with 32 filters
   call network%add(conv2d_layer_type( &
        num_filters=32, &
        kernel_size=3, &
        stride=1, &
        padding="none", &
        activation="relu"))

   ! Max pooling
   call network%add(maxpool2d_layer_type( &
        pool_size=2, &
        stride=2))

   ! First fully connected layer
   call network%add(full_layer_type( &
        num_outputs=100, &
        activation="relu", &
        kernel_initialiser="he_uniform", &
        bias_initialiser="he_uniform"))

   ! Output layer
   call network%add(full_layer_type( &
        num_outputs=10, &
        activation="softmax", &
        kernel_initialiser="glorot_uniform", &
        bias_initialiser="glorot_uniform"))

**Architecture breakdown:**

1. **Padding layer**: Adds 1-pixel border to maintain spatial dimensions
2. **Conv2D**: 32 filters × 3×3 kernels with ReLU activation
3. **MaxPool2D**: 2×2 pooling reduces spatial dimensions by half
4. **Flatten**: Converts 2D feature maps to 1D vector
5. **Dense**: 100 hidden units with ReLU and He initialization
6. **Output**: 10 classes with softmax and Glorot initialization

Complete Program Structure
---------------------------

.. code-block:: fortran

   program mnist_example
     use athena
     use read_mnist, only: read_mnist_db
     use inputs
     implicit none

     type(network_type) :: network

     ! Data arrays
     real(real32), allocatable :: input_images(:,:,:,:), test_images(:,:,:,:)
     integer, allocatable :: labels(:), test_labels(:)
     real(real32), allocatable :: input_labels(:,:)

     ! Parameters
     integer, parameter :: num_classes = 10
     integer :: num_samples, num_samples_test
     integer :: i

     ! Initialize from configuration file
     call set_global_vars(param_file="example/mnist/test_job.in")

     ! Load training data
     call read_mnist_db("data/MNIST_train.txt", input_images, labels, &
                        1, image_size, "none")
     num_samples = size(input_images, 4)

     ! Load test data
     call read_mnist_db("data/MNIST_test.txt", test_images, test_labels, &
                        1, image_size, "none")
     num_samples_test = size(test_images, 4)

     ! Shuffle training data
     call shuffle(input_images, labels, 4, seed)

     ! Build network (see architecture above)
     call build_network(network)

     ! Compile network
     call network%compile( &
          optimiser=adam_optimiser_type(learning_rate=0.001_real32), &
          loss_method="categorical_crossentropy", &
          accuracy_method="categorical_accuracy", &
          batch_size=128, &
          verbose=1)

     ! Convert labels to one-hot encoding
     allocate(input_labels(num_classes, num_samples), source=0.0_real32)
     do i = 1, num_samples
       input_labels(labels(i), i) = 1.0_real32
     end do

     ! Train network
     call network%train( &
          input_images, input_labels, &
          num_epochs=10, &
          batch_size=128, &
          shuffle_batches=.true., &
          verbose=1)

     ! Save trained model
     call network%print(file="mnist_trained.net")

     ! Test on test set
     allocate(input_labels(num_classes, num_samples_test))
     input_labels = 0.0
     do i = 1, num_samples_test
       input_labels(test_labels(i), i) = 1.0
     end do

     call network%test(test_images, input_labels)

     write(*, '(A,F0.5)') "Test accuracy: ", network%accuracy_val
     write(*, '(A,F0.5)') "Test loss: ", network%loss_val

   end program mnist_example


Key Methods Used
----------------

Training with ``train()``
~~~~~~~~~~~~~~~~~~~~~~~~~

The high-level ``train()`` method handles the training loop:

.. code-block:: fortran

   call network%train( &
        input_data, &          ! Input images [width, height, channels, n_samples]
        target_data, &         ! One-hot labels [n_classes, n_samples]
        num_epochs, &          ! Number of training epochs
        batch_size, &          ! Samples per batch
        shuffle_batches=.true., &  ! Shuffle each epoch
        verbose=1)             ! Print progress

This automatically:

* Splits data into batches
* Performs forward and backward passes
* Updates weights
* Tracks loss and accuracy
* Prints progress

Testing with ``test()``
~~~~~~~~~~~~~~~~~~~~~~~

Evaluate the trained model:

.. code-block:: fortran

   call network%test(test_images, test_labels)

   ! Access results
   print *, "Test accuracy:", network%accuracy_val
   print *, "Test loss:", network%loss_val

Saving and Loading
~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   ! Save trained model
   call network%print(file="mnist_model.net")

   ! Load saved model
   call network%read(file="mnist_model.net")

Data Preparation
----------------

One-Hot Encoding Labels
~~~~~~~~~~~~~~~~~~~~~~~~

Convert integer labels (0-9) to one-hot vectors:

.. code-block:: fortran

   integer :: labels(num_samples)  ! Integer labels
   real(real32) :: one_hot(10, num_samples)
   integer :: i

   ! Initialize to zeros
   one_hot = 0.0_real32

   ! Set corresponding class to 1.0
   do i = 1, num_samples
     one_hot(labels(i), i) = 1.0_real32
   end do

Shuffling Data
~~~~~~~~~~~~~~

Shuffle training data each epoch for better generalization:

.. code-block:: fortran

   use athena, only: shuffle

   ! Shuffle along dimension 4 (sample dimension)
   call shuffle(input_images, labels, dim=4, seed)

Configuration File
------------------

The example uses a configuration file (``test_job.in``) for hyperparameters:

.. code-block:: text

   # Network architecture
   cv_num_filters = 32
   padding_method = "constant"

   # Training parameters
   num_epochs = 10
   batch_size = 128
   learning_rate = 0.001

   # Data settings
   data_dir = "data/"
   shuffle_dataset = .true.

   # Output
   output_file = "mnist_trained.net"
   verbosity = 1

Load configuration in code:

.. code-block:: fortran

   use inputs

   call set_global_vars(param_file="example/mnist/test_job.in")

Why This Design?
----------------

Simple but Effective
~~~~~~~~~~~~~~~~~~~~

* **Single conv layer**: Sufficient for MNIST's simple patterns
* **100 hidden units**: Good balance between capacity and speed
* **Padding layer**: Explicit control over spatial dimensions
* **He/Glorot initialization**: Proper weight initialization for each activation

The architecture prioritizes:

1. **Clarity**: Easy to understand and modify
2. **Speed**: Trains quickly on CPU
3. **Effectiveness**: Achieves >98% accuracy

Expected Results
----------------

Running the example should produce:

.. code-block:: text

   ...
    NUMBER OF LAYERS 7
    Starting training...
   epoch=1, batch=1, learning_rate=.010, loss=2.305, accuracy=.094
   epoch=1, batch=20, learning_rate=.010, loss=1.313, accuracy=.545
   ...
   epoch=1, batch=1540, learning_rate=.010, loss=.217, accuracy=.934
    Convergence achieved, accuracy threshold reached
    Exiting training loop
    Writing network to file...
    Writing finished
    Starting testing...
    Testing finished
   Overall accuracy=.96820
   Overall loss=10.24504

See Also
--------

* :ref:`Regression Examples <regression-example>` - Function approximation with low-level API
* :ref:`ResNet Tutorial <resnet-example>` - Advanced CNN architectures
* :ref:`2D Convolutional Layer <conv2d-layer>` - Convolutional layer details
* :ref:`Fully-Connected Layer <full-layer>` - Fully-connected layer details
* :ref:`Saving and Loading <saving-loading>` - Model persistence
