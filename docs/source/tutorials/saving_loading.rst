.. _saving-loading:

Saving and Loading Models
==========================

This tutorial covers how to save trained models to disk and load them for later use, using both athena's native format and ONNX for interoperability.

Why Save Models?
----------------

* **Preserve training results**: Avoid retraining
* **Share models**: Distribute trained models to others
* **Deploy models**: Use in production applications
* **Resume training**: Continue from checkpoints
* **Interoperability**: Export to ONNX for use with other frameworks

Athena Native Format
--------------------

Saving a Model
~~~~~~~~~~~~~~

Save a network using athena's native text format:

.. code-block:: fortran

   type(network_type) :: net

   ! After building and training the network...

   ! Save to file (simple one-line call)
   call net%print(file="trained_model.txt")

The ``print`` method saves:

* Network architecture (all layer types and configurations)
* Trained weights and biases
* Optimiser settings
* Training metadata (epoch, batch size, loss, accuracy)

Loading a Model
~~~~~~~~~~~~~~~

Load a previously saved network:

.. code-block:: fortran

   type(network_type) :: net

   ! Load from file
   call net%read(file="trained_model.txt")

   ! Network is now ready to use
   call net%forward(test_data)
   call net%compile()

The loaded network includes the complete architecture and all trained parameters.
Currently, it is still necessary to call ``compile()`` after loading to ensure checks on network architecture;
for example, to verify layer compatibility, ensure input layers are present and layer connections are valid.

Complete Example
----------------

Save After Training
~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   program save_trained_model
     use athena
     implicit none

     type(network_type) :: net
     type(adam_optimiser_type) :: optimiser
     type(cce_loss_type) :: loss
     real(real32), allocatable :: train_data(:,:), train_labels(:,:)

     ! Load or generate train_data and train_labels here

     ! Build network
     call net%add(full_layer_type(num_inputs=784, num_outputs=128, activation="relu"))
     call net%add(full_layer_type(num_outputs=10, activation="softmax"))

     ! Compile
     optimiser = adam_optimiser_type(learning_rate=0.001_real32)
     loss = cce_loss_type()
     call net%compile(optimiser=optimiser, metrics=["loss"], loss_method=loss, accuracy_method="mse")

     ! Train network (assuming train_data and train_labels exist)
     call net%train(train_data, train_labels, num_epochs=50, &
                    batch_size=32)

     ! Save trained model
     call net%print(file="model.txt")

     write(*,*) "Model saved successfully!"

   end program save_trained_model

Load and Resume Training
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   program resume_training
     use athena
     implicit none

     type(network_type) :: net
     real(real32), allocatable :: train_data(:,:), train_labels(:,:)

     ! Load or generate train_data and train_labels here

     ! Load the saved model
     call net%read(file="model.txt")
     call net%compile()

     write(*,*) "Model loaded successfully!"
     write(*,*) "Previous epoch:", net%epoch
     write(*,*) "Previous loss:", net%loss_val
     write(*,*) "Previous accuracy:", net%accuracy_val

     ! Continue training from checkpoint
     call net%train(train_data, train_labels, num_epochs=50, &
                    batch_size=net%batch_size)

     ! Save updated model
     call net%print(file="model_continued.txt")

   end program resume_training

Load for Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   program use_saved_model
     use athena
     implicit none

     type(network_type) :: net
     real(real32), allocatable :: test_data(:,:,:,:)
     real(real32), allocatable :: prediction(:,:)

     ! Load or generate test_data here

     ! Load the saved model
     call net%read(file="model.txt")
     call net%compile()

     write(*,*) "Model loaded successfully!"
     write(*,*) "Number of layers:", net%num_layers

     ! Use for predictions (assuming test_data is loaded)
     prediction = net%predict(test_data)

   end program use_saved_model

Native Format Details
---------------------

File Structure
~~~~~~~~~~~~~~

Athena saves models in a human-readable text format with the following structure:

.. code-block:: text

   NETWORK_SETTINGS
      ATHENA_VERSION = 1.0.0
      NAME = my_network
      EPOCH = 50
      BATCH_SIZE = 32
      ACCURACY = 0.987
      LOSS = 0.045
      LOSS_METHOD = categorical_crossentropy
      OPTIMISER: adam
         LEARNING_RATE = 0.001
      END OPTIMISER
   END NETWORK_SETTINGS

   FULL || []
      NUM_INPUTS = 784
      NUM_OUTPUTS = 128
      USE_BIAS = T
      ACTIVATION: relu
      END ACTIVATION
   WEIGHTS
      (weight values...)
   END WEIGHTS
   END FULL

   ... (additional layers)

This format is:

* **Human-readable**: Easy to inspect and debug
* **Version-tracked**: Includes athena version for compatibility
* **Complete**: All architecture and parameters preserved
* **Editable**: Can be manually modified if needed (advanced use)

Model Checkpointing
-------------------

athena does not currently have built-in checkpointing during training, but you can implement this manually by saving the model at desired intervals.
It does, however, print out the current epoch, loss, and accuracy after ``batch_print_step`` batches during training, which can help monitor progress.
This argument can be set when calling the ``train()`` method.

ONNX Interoperability
----------------------

Athena supports exporting to and importing from ONNX (Open Neural Network Exchange) format for interoperability with other frameworks like PyTorch, TensorFlow, and more.
Currently, this uses the human-readable .json-like format, so does not provide the binary efficiency benefits of standard ONNX files.

An example of writing and reading ONNX models can be found :git:`here<example/onnx/src/main.f90>`.
The example can be run using fpm with the command:

.. code-block:: bash

   fpm run --example onnx

.. note::

    Whilst the framework for ONNX support in athena is implemented, some layers and features may not yet be fully supported.
    Please use the `issue tracker <https://github.com/nedtaylor/athena/issues>`_ to report any problems or request additional ONNX features.


Export to ONNX
~~~~~~~~~~~~~~

Export a trained athena network to ONNX format:

.. code-block:: fortran

   use athena
   implicit none

   type(network_type) :: net

   ! Build and train network...
   call net%add(conv2d_layer_type( &
        input_shape=[28, 28, 1], &
        num_filters=6, kernel_size=3, activation="relu"))
   call net%add(maxpool2d_layer_type(pool_size=2, stride=2))
   call net%add(full_layer_type(num_outputs=10, activation="softmax"))

   call net%compile( &
        optimiser=adam_optimiser_type(learning_rate=0.01_real32), &
        loss_method="categorical_crossentropy", batch_size=32)

   ! Export to ONNX
   call write_onnx("model.json", net)

The ONNX file can then be:

* Loaded in Python with ``onnx`` or ``onnxruntime``
* Converted to other formats (PyTorch, TensorFlow, etc.)
* Deployed in production environments
* Optimised with ONNX Runtime

Import from ONNX
~~~~~~~~~~~~~~~~

Load an ONNX model into athena:

.. code-block:: fortran

   use athena
   implicit none

   type(network_type) :: net

   ! Read ONNX file
   net = read_onnx("model.json")
   call net%compile( &
        optimiser=adam_optimiser_type(learning_rate=0.01_real32), &
        metrics=["loss"], &
        loss_method="cce", accuracy_method="mse" &
   )

   ! Network is ready to use
   write(*,*) "Loaded network with", net%num_layers, "layers"

   ! Use for inference
   call net%forward(test_data)

Complete ONNX Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

    program onnx_example
      use athena
      implicit none

      type(network_type) :: network_write, network_read
      character(256) :: onnx_file

      ! Create a simple network
      write(*,*) "Creating a simple test network..."

      call network_write%add(conv2d_layer_type( &
          input_shape=[28,28,1], &
          num_filters=6, &
          kernel_size=3, &
          activation="relu"))

      call network_write%add(maxpool2d_layer_type( &
          pool_size=2, &
          stride=2))

      call network_write%add(full_layer_type( &
          num_outputs=12, &
          activation="relu"))

      call network_write%add(full_layer_type( &
          num_outputs=10, &
          activation="softmax"))

      call network_write%compile( &
          optimiser=base_optimiser_type(learning_rate=0.01), &
          loss_method="categorical_crossentropy", &
          batch_size=1)

      ! Write network to ONNX
      onnx_file = "test_network.json"
      write(*,*) "Writing network to ONNX file: ", trim(onnx_file)
      call write_onnx(onnx_file, network_write)
      write(*,*) "ONNX file written successfully"

      ! Read network from ONNX
      write(*,*) ""
      write(*,*) "Reading network from ONNX file..."
      network_read = read_onnx(onnx_file)

      write(*,*) ""
      write(*,*) "Network read completed"
      write(*,*) "Number of layers in original network: ", network_write%num_layers
      write(*,*) "Number of layers in read network: ", network_read%num_layers

    end program onnx_example


ONNX Compatibility
~~~~~~~~~~~~~~~~~~

**Supported operations:**

* Fully connected (Dense/Gemm/MatMul)
* Convolutional (Conv1D, Conv2D, Conv3D)
* Pooling (MaxPool, AveragePool)
* Activation functions (ReLU, Sigmoid, Tanh, Softmax, etc.)
* Batch normalisation
* Dropout
* Flatten, Reshape
* Message passing layers
  * Duvenaud-style graph convolution
  * Kipf-style graph convolution
* Neural operators

**Limitations:**

* Some athena-specific features may not translate to ONNX
* Graph-based layers (message passing) have limited ONNX support
* Neural operators can be imported and exported by athena, but currently not supported by other frameworks

Safety and Reliability
----------------------

* **Save before long training runs**: Protect against crashes
* **Keep multiple checkpoints**: Don't overwrite the only copy
* **Test loading**: Verify models load correctly after saving
* **Version control**: Track model files using tools such as `Weights \& Biases <https://wandb.ai>`_

Common Issues
-------------

File Not Found
~~~~~~~~~~~~~~

Check file exists before loading:

.. code-block:: fortran

   logical :: file_exists

   inquire(file="trained_model.txt", exist=file_exists)
   if (.not. file_exists) then
      write(*,*) "Error: Model file not found!"
      stop
   end if

   call net%read(file="trained_model.txt")

Version Compatibility
~~~~~~~~~~~~~~~~~~~~~

The native format includes athena version information:

.. code-block:: fortran

   ! Saved files include:
   ! ATHENA_VERSION = 2.0.0

* Models saved with newer versions may not load in older athena
* Check release notes for breaking changes
* Test model loading after athena updates

It is always intended that major version updates (e.g., 1.x to 2.x) may introduce breaking changes,
so models saved with a newer major version may not be compatible with older versions of athena.
Meanwhile, minor version updates (e.g., 1.0 to 1.1) should maintain compatibility where possible.

ONNX Version Issues
~~~~~~~~~~~~~~~~~~~

* Athena exports to ONNX IR version 8
* Ensure target framework supports this version
* Some frameworks may need ONNX model conversion

Next Steps
----------

* :ref:`Try complete examples <mnist-example>`
* Learn about :ref:`custom components <custom-layers>`

See Also
--------

* :ref:`Building Networks <basic-network>`
* :ref:`Training Models <training-model>`
