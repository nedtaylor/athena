.. _regression-example:

Function Approximation (Regression)
====================================

Complete examples of training neural networks for regression tasks - predicting continuous values.

This tutorial walks through the actual ``example/sine`` and ``example/simple`` code from the athena repository.

Overview
--------

These examples demonstrate:

* Building networks for function approximation
* Training with MSE loss for regression
* Using low-level training loops
* Making predictions on continuous outputs

Sine Wave Approximation
------------------------

The ``example/sine`` demonstrates approximating :math:`y = \frac{\sin(x) + 1}{2}` over the range :math:`x \in [0, 2\pi]`.

.. rubric:: :h3style:`Network Architecture`

.. code-block:: fortran

   ! 1 input → 5 hidden units → 1 output
   call network%add(full_layer_type( &
        num_inputs=1, &
        num_outputs=5, &
        activation="tanh"))

   call network%add(full_layer_type( &
        num_outputs=1, &
        activation="sigmoid"))

**Why these choices:**

* **Tanh activation**: Bounded output [-1, 1], good for smooth functions
* **Sigmoid output**: Maps to [0, 1], matching our normalised sine wave
* **5 hidden units**: Sufficient capacity for smooth periodic function

.. rubric:: :h3style:`Training Loop`

The example uses a low-level training loop for fine control:

.. code-block:: fortran

   program sine
     use athena
     use coreutils, only: real32, pi
     implicit none

     type(network_type) :: network
     real(real32), dimension(1,1) :: x, y
     type(array_type) :: x_array(1), y_array(1,1)
     type(array_type), pointer :: loss

     integer, parameter :: num_iterations = 10000
     integer, parameter :: test_size = 30
     real(real32), dimension(1, test_size) :: x_test, y_test, y_pred
     integer :: n, i

     ! Build network
     call network%add(full_layer_type(num_inputs=1, num_outputs=5, &
          activation="tanh"))
     call network%add(full_layer_type(num_outputs=1, activation="sigmoid"))

     ! Compile with base optimiser (SGD) and MSE loss
     call network%compile( &
          optimiser=base_optimiser_type(learning_rate=1.0_real32), &
          loss_method="mse", &
          metrics=["loss"], &
          verbose=1)

     call network%set_batch_size(1)

     ! Generate test data
     do i = 1, test_size
       x_test(1,i) = ((i - 1) * 2.0 * pi) / test_size
       y_test(1,i) = (sin(x_test(1,i)) + 1.0) / 2.0
     end do

     ! Allocate arrays for training
     call x_array(1)%allocate(array_shape=[1,1])
     call y_array(1,1)%allocate(array_shape=[1,1])

     ! Training loop
     print *, "Training network"
     print *, "Iteration, Loss"

     do n = 0, num_iterations
       ! Generate random training point
       call random_number(x)
       x = x * 2.0 * pi
       y = (sin(x) + 1.0) / 2.0

       ! Store in array_type format
       x_array(1)%val = x
       y_array(1,1)%val = y

       ! Training step
       call network%set_batch_size(1)
       call network%forward(x)
       network%expected_array = y_array
       loss => network%loss_eval(1, 1)
       call loss%grad_reverse()
       call network%update()

       ! Print progress every 1000 iterations
       if (mod(n, 1000) == 0) then
         y_pred = network%predict(input=x_test)
         write(*, '(I7,1X,F9.6)') n, sum((y_pred - y_test)**2) / size(y_pred)
       end if
     end do

   end program sine

**Key points:**

* **Random sampling**: Each iteration uses a random x value
* **Online learning**: Batch size of 1, updates after each sample
* **Low-level control**: Direct access to forward, backward, and update steps
* **Learning rate = 1.0**: Higher learning rate works well for simple functions

Simple Function Example
------------------------

The ``example/simple`` demonstrates learning a fixed mapping from 3 inputs to 2 outputs.

.. rubric:: :h3style:`Network Architecture`

.. code-block:: fortran

   ! 3 inputs → 5 hidden units → 2 outputs
   call network%add(full_layer_type( &
        num_inputs=3, &
        num_outputs=5, &
        activation="tanh"))

   call network%add(full_layer_type( &
        num_outputs=2, &
        activation="sigmoid"))


.. rubric:: :h3style:`Training on Fixed Data`

.. code-block:: fortran

   program simple
     use athena
     use coreutils, only: real32
     implicit none

     type(network_type) :: network
     real(real32), allocatable :: x(:,:), y(:,:), prediction(:,:)
     type(array_type) :: x_array(1), y_array(1,1)
     type(array_type), pointer :: loss

     integer, parameter :: num_iterations = 500
     integer :: n

     ! Build network
     call network%add(full_layer_type(num_inputs=3, num_outputs=5, &
          activation="tanh"))
     call network%add(full_layer_type(num_outputs=2, activation="sigmoid"))

     ! Compile
     call network%compile( &
          optimiser=base_optimiser_type(learning_rate=1.0_real32), &
          loss_method="mse", &
          metrics=["loss"], &
          verbose=1)

     call network%set_batch_size(1)

     ! Define fixed training data
     x = reshape([0.2, 0.4, 0.6], [3, 1])
     y = reshape([0.123456, 0.246802], [2, 1])

     ! Allocate array_type containers
     call x_array(1)%allocate(source=x)
     call y_array(1,1)%allocate(source=y)

     ! Training loop
     print *, "Training network"
     print *, "Iteration, Predictions"

     do n = 0, num_iterations
       ! Training step
       call network%forward(x)
       network%expected_array = y_array
       loss => network%loss_eval(1, 1)
       call loss%grad_reverse()
       call network%update()

       ! Print progress every 50 iterations
       prediction = network%predict(input=x)
       if (mod(n, 50) == 0) then
         write(*, '(I7,2(1X,F9.6))') n, prediction
       end if
     end do

   end program simple

**Key points:**

* **Memorisation task**: Learning one specific input-output mapping
* **Convergence**: Network should converge to exact outputs
* **Demonstration**: Shows basic training mechanics without complexity

Understanding the Training Process
----------------------------------

.. rubric:: :h3style:`Low-Level Training Loop`

Both examples use a manual training loop for educational purposes:

.. code-block:: fortran

   ! 1. Forward pass
   call network%forward(x)

   ! 2. Set expected output
   network%expected_array = y_array

   ! 3. Backward pass (compute gradients)
   loss => network%loss_eval(1, 1)
   call loss%grad_reverse()

   ! 4. Update weights
   call network%update()

This gives you full control over each training step.

.. rubric:: :h3style:`Using array_type`

Athena uses ``array_type`` for internal computations.
For simple networks, this never needs to be exposed.
However, if you need to prepare data in ``array_type``, here's how:

.. code-block:: fortran

   type(array_type) :: x_array(1), y_array(1,1)

   ! Allocate
   call x_array(1)%allocate(array_shape=[num_inputs, batch_size])
   call y_array(1,1)%allocate(array_shape=[num_outputs, batch_size])

   ! Assign values
   x_array(1)%val = x  ! Copy from regular array

   ! Or allocate from source
   call x_array(1)%allocate(source=x)

.. rubric:: :h3style:`Making Predictions`

Use the ``predict()`` method for inference:

.. code-block:: fortran

   real(real32), dimension(num_inputs, num_samples) :: test_x
   real(real32), allocatable :: predictions(:,:)

   predictions = network%predict(input=test_x)
   ! Returns: [num_outputs, num_samples]

See Also
--------

* :ref:`MNIST Example <mnist-example>` - Classification with high-level API
* :ref:`ResNet Tutorial <resnet-example>` - Advanced architectures
* :ref:`Training Guide <training-model>` - Training best practices
