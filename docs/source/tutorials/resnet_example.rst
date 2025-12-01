.. _resnet-example:

Residual Networks (ResNet)
===========================

This tutorial covers building Residual Networks (ResNets) for image processing and computer vision tasks using skip connections.

What are ResNets?
-----------------

Residual Networks solve the degradation problem in very deep neural networks by introducing skip connections (residual connections).

Key concepts:

* **Residual blocks**: Add input directly to output via skip connections [#f1]_
* **Skip connections**: Allow gradients to flow directly through the network
* **Identity mappings**: Enable training of very deep networks (100+ layers) [#f2]_
* **Convolutional layers**: Extract features at each level


Building a Basic ResNet
-----------------------

Residual Block Concept
~~~~~~~~~~~~~~~~~~~~~~

A residual block performs: ``output = F(x) + x``

Where:
* ``F(x)`` is the transformation (conv layers)
* ``x`` is the identity (skip connection)
* ``+`` is element-wise addition

Simple ResNet Example
~~~~~~~~~~~~~~~~~~~~~

.. note::
   The examples in this tutorial use simplified ResNet architectures without batch normalisation for clarity.
   The example found in :git:`example/resnet/src/main.f90` contains a more complete implementation.

For image classification:

.. code-block:: fortran

   program basic_resnet
     use athena
     implicit none

     type(network_type) :: net
     type(adam_optimiser_type) :: optimiser
     type(cce_loss_type) :: loss
     integer :: layer_id

     ! Image dimensions: 28x28 grayscale images
     ! Data format: [width, height, channels]
     integer, parameter :: width = 28, height = 28, channels = 1
     integer, parameter :: num_classes = 10

     ! Build ResNet architecture
     ! Initial conv layer
     call net%add(input_layer_type(input_shape=[width, height, channels]))
     call net%add(conv2d_layer_type( &
          num_filters=64, kernel_size=3, padding="same", activation="relu"))

     ! First residual block (64 filters)
     layer_id = net%num_layers  ! Save for skip connection
     call net%add(conv2d_layer_type( &
          num_filters=64, kernel_size=3, padding="same", activation="relu"))
     call net%add(conv2d_layer_type( &
          num_filters=64, kernel_size=3, padding="same"))
     ! Add skip connection
     call net%add(add_layer_type( &
          input_layer_ids=[layer_id, net%num_layers], input_rank=3), &
          input_list=[layer_id, net%num_layers], operator="+")
     call net%add(actv_layer_type(activation="relu"))

     ! Pooling and output
     call net%add(maxpool2d_layer_type(pool_size=2))
     call net%add(flatten_layer_type(input_rank=3))
     call net%add(full_layer_type(num_outputs=num_classes, activation="softmax"))

     ! Compile
     optimiser = adam_optimiser_type(learning_rate=0.001_real32)
     loss = cce_loss_type()
     call net%compile(optimiser=optimiser, loss_method=loss)

     call net%print_summary()

   end program basic_resnet

The skip connection can be introduced in one of two ways; either by saving the layer ID before the residual path and using it in the ``add_layer_type``, or by using the ``input_list`` parameter to specify which layers to combine.
The former method is shown above to be more explicit, while the latter is more concise.
The latter method is shown below and just automates the tracking of layer IDs and handling of building out the addition layer operation.
Direct use of ``add_layer_type`` requires specification of the ``input_rank`` parameter to indicate the rank of the input tensors (3 for 2D images due to the two spatial dimensions and channels).

Building Residual Blocks
------------------------

Helper Subroutine for Residual Blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a reusable subroutine for adding residual blocks:

.. code-block:: fortran

   subroutine add_residual_block(net, num_filters, stride)
     type(network_type), intent(inout) :: net
     integer, intent(in) :: num_filters
     integer, optional, intent(in) :: stride
     integer :: stride_, skip_id

     stride_ = 1
     if (present(stride)) stride_ = stride

     ! Save layer ID for skip connection
     skip_id = net%num_layers

     ! First conv layer in block
     call net%add(conv2d_layer_type( &
          num_filters=num_filters, kernel_size=3, &
          stride=stride_, padding="same", activation="relu"))

     ! Second conv layer in block
     call net%add(conv2d_layer_type( &
          num_filters=num_filters, kernel_size=3, padding="same"))

     ! Skip connection with addition
     call net%add(add_layer_type( &
          input_layer_ids=[skip_id, net%num_layers], input_rank=3), &
          input_list=[skip_id, net%num_layers], operator="+")

     ! Final activation
     call net%add(actv_layer_type(activation="relu"))

   end subroutine add_residual_block

Deeper ResNet Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~

ResNet-18 style network:

.. code-block:: fortran

   program resnet18_style
     use athena
     implicit none

     type(network_type) :: net
     type(adam_optimiser_type) :: optimiser
     type(cce_loss_type) :: loss

     ! Initial convolution
     call net%add(input_layer_type(input_shape=[224, 224, 3]))
     call net%add(conv2d_layer_type( &
          num_filters=64, kernel_size=7, stride=2, padding="same"))
     call net%add(batchnorm2d_layer_type(num_channels=64))
     call net%add(actv_layer_type(activation="relu"))
     call net%add(maxpool2d_layer_type(pool_size=3, stride=2))

     ! Residual blocks - Stage 1 (64 filters)
     call add_residual_block(net, 64)
     call add_residual_block(net, 64)

     ! Residual blocks - Stage 2 (128 filters)
     call add_residual_block(net, 128, stride=2)
     call add_residual_block(net, 128)

     ! Residual blocks - Stage 3 (256 filters)
     call add_residual_block(net, 256, stride=2)
     call add_residual_block(net, 256)

     ! Residual blocks - Stage 4 (512 filters)
     call add_residual_block(net, 512, stride=2)
     call add_residual_block(net, 512)

     ! Global average pooling and classifier
     call net%add(avgpool2d_layer_type(pool_size=7))
     call net%add(flatten_layer_type(input_rank=3))
     call net%add(full_layer_type(num_outputs=1000, activation="softmax"))

     optimiser = adam_optimiser_type(learning_rate=0.001_real32)
     loss = cce_loss_type()
     call net%compile(optimiser=optimiser, loss_method=loss)

   contains
     ! Include add_residual_block subroutine here
   end program resnet18_style

Projection Shortcuts
~~~~~~~~~~~~~~~~~~~~

When dimensions change, use 1x1 convolutions for the skip connection:

.. code-block:: fortran

   subroutine add_residual_block_projection(net, num_filters, stride)
     type(network_type), intent(inout) :: net
     integer, intent(in) :: num_filters, stride
     integer :: skip_id, main_path_id

     skip_id = net%num_layers

     ! Main path
     call net%add(conv2d_layer_type( &
          num_filters=num_filters, kernel_size=3, &
          stride=stride, padding="same", activation="relu"))
     call net%add(conv2d_layer_type( &
          num_filters=num_filters, kernel_size=3, padding="same"))
     main_path_id = net%num_layers

     ! Projection shortcut (1x1 conv to match dimensions)
     call net%add(conv2d_layer_type( &
          num_filters=num_filters, kernel_size=1, stride=stride), &
          input_list=[skip_id])

     ! Add skip connection
     call net%add(add_layer_type( &
          input_layer_ids=[net%num_layers, main_path_id], input_rank=3), &
          input_list=[net%num_layers, main_path_id], operator="+")
     call net%add(actv_layer_type(activation="relu"))

   end subroutine add_residual_block_projection

ResNet for Small Images (CIFAR-10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adapted architecture for 32x32 images:

.. code-block:: fortran

   program resnet_cifar10
     use athena
     implicit none

     type(network_type) :: net
     integer :: i

     ! Initial conv (no pooling for small images)
     call net%add(input_layer_type(input_shape=[32, 32, 3]))
     call net%add(conv2d_layer_type( &
          num_filters=16, kernel_size=3, padding="same", activation="relu"))

     ! Stage 1: 16 filters
     do i = 1, 3
        call add_residual_block(net, 16)
     end do

     ! Stage 2: 32 filters with stride
     call add_residual_block(net, 32, stride=2)
     do i = 1, 2
        call add_residual_block(net, 32)
     end do

     ! Stage 3: 64 filters with stride
     call add_residual_block(net, 64, stride=2)
     do i = 1, 2
        call add_residual_block(net, 64)
     end do

     ! Global average pooling
     call net%add(avgpool2d_layer_type(pool_size=8))
     call net%add(flatten_layer_type(input_rank=3))
     call net%add(full_layer_type(num_outputs=10, activation="softmax"))

   end program resnet_cifar10

Understanding Skip Connections
------------------------------

The following sections explain how to implement skip connections within athena.
These rely on the ``merge_layer_type`` derived type and the ``input_list`` argument of the ``add()`` method of the ``network_type`` derived type.
There are currently two options of ``merge_layer_type`` that can be used for skip connections: addition (``add_layer_type``) and concatenation (``concat_layer_type``).
Additionally, the output of any layer can be easily broadcast to multiple subsequent layers using the ``input_list`` argument.

How Skip Connections Work
~~~~~~~~~~~~~~~~~~~~~~~~~~

In athena, there are two ways we can implement the ResNet skip connections.

.. code-block:: fortran

   ! Save the layer ID before the residual path
   skip_id = net%num_layers

   ! Add transformation layers (F(x))
   call net%add(conv2d_layer_type(...))

   ! Add skip connection: output = F(x) + x
   call net%add(add_layer_type( &
        input_layer_ids=[skip_id, net%num_layers], input_rank=3), &
        input_list=[skip_id, net%num_layers], operator="+")

Or, we can use the ``input_list`` argument to specify which layers to merge directly:

.. code-block:: fortran

   ! Save the layer ID before the residual path
   skip_id = net%num_layers

   ! Add transformation layers (F(x))
   call net%add(conv2d_layer_type(...))

   ! Add skip connection: output = F(x) + x
   call net%add(add_layer_type( &
        input_layer_ids=[skip_id, net%num_layers], input_rank=3), &
        input_list=[skip_id, net%num_layers], operator="+")

The latter is the more concise and preferred method.
The ``input_list`` accepts values between ``-num_layers + 1`` and ``num_layers`` to specify which layers to merge.
Negative indices count backwards from the most recently added layer (``-1`` refers to the last added layer, ``-2`` the second last, etc.).
Positive indices refer to absolute layer IDs (i.e. the order in which layers have been added to the network via the ``add()`` method).
``0`` refers to the input layer of the network (i.e. the data input); if multiple input layers exist, this refers to the input to the first layer added.

Key Points
~~~~~~~~~~

* **Layer IDs**: Track ``net%num_layers`` to reference previous layers
* **Add layer**: Combines outputs element-wise from specified layers
* **Input list**: Specifies which layers to merge (typically start and end of block)
* **Operator**: Use ``"+"`` for addition (residual connections) and ``"||"`` for concatenation

Dimension Matching
~~~~~~~~~~~~~~~~~~

Skip connections require matching dimensions:

**Option 1: Same dimensions (identity shortcut)**

.. code-block:: fortran

   ! Both input and output have same shape
   ! Can directly add
   call net%add(add_layer_type(...), input_list=[skip_id, current_id], operator="+")

**Option 2: Different dimensions (projection shortcut)**

.. code-block:: fortran

   ! Use 1x1 convolution to match dimensions
   skip_id = net%num_layers

   ! Main path
   call net%add(conv2d_layer_type(num_filters=128, kernel_size=3, stride=2))
   main_id = net%num_layers

   ! Projection path for skip connection
   call net%add(conv2d_layer_type(num_filters=128, kernel_size=1, stride=2), &
                input_list=[skip_id])

   ! Combine
   call net%add(add_layer_type( &
          input_layer_ids=[net%num_layers, main_id], input_rank=3), &
          input_list=[net%num_layers, main_id], operator="+")

Next Steps
----------

* Implement your own ResNet for image classification
* Try the :ref:`MNIST example <mnist-example>` and adapt it for ResNet
* Experiment with different residual block designs
* Learn about :ref:`custom layers <custom-layers>` for advanced architectures

See Also
--------

* :ref:`conv2d_layer <conv2d-layer>` - Convolutional layers
* :ref:`batchnorm2d_layer <batchnorm2d-layer>` - Batch normalisation
* :ref:`Activation layers <activation-layer>` - Activation functions
* :ref:`Basic Network Tutorial <basic-network>` - Foundation concepts
* :ref:`Training Guide <training-model>` - Training deep networks

Further Reading
---------------

.. rubric:: Footnotes

.. [#f1] `He et al., "Deep Residual Learning for Image Recognition" (2015) <https://doi.org/10.48550/arXiv.1512.03385>`_
.. [#f2] `He et al., "Identity Mappings in Deep Residual Networks" (2016) <https://doi.org/10.48550/arXiv.1603.05027>`_
