.. _inverse-design-example:

Inverse Design Example
======================

This tutorial walks through the actual
``example/inverse_design/src/main.f90`` program in the athena repository.

The example trains a small regression network, runs the built-in
``inverse_design()`` routine, and then repeats the same idea with a manual
inverse-design loop.

Overview
--------

The example uses the mapping:

.. math::

   y = \frac{2x + 0.5}{2.5}

The network is trained on :math:`x \in [0, 1]`, then asked to find an input
that produces the target output :math:`y_t = 0.6`.

Analytically, this corresponds to :math:`x = 0.5`, but the learned inverse is
based on the trained network approximation rather than the exact formula.

Running the Example
-------------------

.. code-block:: bash

   fpm run --example inverse_design

Structure of the Program
------------------------

The example has three main stages:

1. train a network on a simple 1D regression problem
2. use ``network%inverse_design()`` to optimise the input
3. repeat the process manually to show what the built-in routine is doing

Step 1: Build and Train the Network
-----------------------------------

The example creates a compact fully connected network:

.. code-block:: fortran

   call network%add(full_layer_type( &
        num_inputs=1, num_outputs=16, activation="tanh"))
   call network%add(full_layer_type(num_outputs=1, activation="sigmoid"))
   call network%compile( &
        optimiser = sgd_optimiser_type(learning_rate=0.1_real32), &
        loss_method = "mse", &
        metrics = ["loss"], &
        verbose = 0 &
   )
   call network%set_batch_size(1)

Training is done with a low-level loop rather than ``train()`` so the example
stays close to the mechanics used later during inverse design:

.. code-block:: fortran

   do i = 1, num_train
      call random_number(x)
      y(1,1) = (2._real32 * x(1,1) + 0.5_real32) / 2.5_real32

      x_array(1)%val = x
      y_array(1,1)%val = y

      call network%forward(x)
      network%expected_array = y_array
      loss => network%loss_eval(1, 1)
      call loss%grad_reverse()
      call network%update()
   end do

After training, the program checks the learned mapping at :math:`x = 0.5`.
This value is usually close to 0.6, but not exactly 0.6 because the network is
still only an approximation.

Step 2: Built-In Inverse Design
-------------------------------

The program then asks the network for an input that produces a target output of
0.6:

.. code-block:: fortran

   real(real32) :: target_y(1,1), x_init(1,1)
   real(real32), allocatable :: x_opt(:,:)

   target_y(1,1) = 0.6_real32
   x_init(1,1) = 0.1_real32

   x_opt = network%inverse_design( &
        target = target_y, &
        x_init = x_init, &
        optimiser = sgd_optimiser_type(learning_rate=0.1_real32), &
        steps = inverse_steps &
   )

Two details here matter:

* the example uses real 2D arrays, so ``x_opt`` is also a real 2D array
* the initial guess is intentionally far from the expected analytical value

The program then verifies the result with:

.. code-block:: fortran

   predicted = network%predict(input=x_opt)

This confirms whether the optimised input actually drives the network to the
requested output.

Step 3: Manual Inverse Design Loop
----------------------------------

The final section shows the same idea without using ``network%inverse_design()``.

It initialises an input variable, marks it for gradient tracking, and updates
that input directly with an optimiser:

.. code-block:: fortran

   opt = sgd_optimiser_type(learning_rate=0.1_real32)
   call opt%init(num_params=1)

   call cx(1,1)%allocate(source=reshape([0.1_real32], [1,1]))
   call cx(1,1)%set_requires_grad(.true.)
   call cy(1,1)%allocate(source=reshape([0.6_real32], [1,1]))

Inside the loop, the key operations are:

.. code-block:: fortran

   call network%forward(cx)
   call network%model(root_id)%layer%output(1,1)%set_requires_grad(.true.)

   network%expected_array = cy
   closs => network%loss_eval(1, 1)
   call closs%grad_reverse()

   if (associated(network%model(root_id)%layer%output(1,1)%grad)) then
      cx_grad = network%model(root_id)%layer%output(1,1)%grad%val(:,1)
   else
      cx_grad = 0._real32
   end if
   cx_flat = cx(1,1)%val(:,1)
   call opt%minimise(param=cx_flat, gradient=cx_grad)
   cx(1,1)%val(:,1) = cx_flat

   call closs%nullify_graph()
   deallocate(closs)
   nullify(closs)
   call network%reset_gradients()

This manual version is useful because it makes the inverse-design mechanics
explicit:

* the forward model stays the same
* the loss is still computed in output space
* the gradient is extracted at the input
* only the input variable is updated

Why the Returned Input Is Not Exactly 0.5
-----------------------------------------

When the example is run, the optimised input is typically close to 0.48 rather
than exactly 0.5, while still producing a network output of essentially 0.6.

That is expected. The inverse-design routine is solving the inverse problem for
the trained network, not for the analytical function. In the current example
run, the trained model predicts about 0.6178 at :math:`x = 0.5`, so the best
network-consistent inverse for output 0.6 is slightly smaller than 0.5.

Representative Output
---------------------

One run of the current example produced:

.. code-block:: text

   Sanity check: predict(0.5) =   0.617821
   Expected:     (2*0.5+0.5)/2.5 =   0.600000

   --- Built-in inverse_design results ---
   Optimised input:      0.480400
   Predicted output:     0.600000
   Target output:        0.600000
   Output error:        1.192E-07
   Input error:         1.960E-02

   --- Custom inverse design results ---
   Optimised input:      0.480400
   Predicted output:     0.600000
   Target output:        0.600000
   Output error:        1.192E-07
   Input error:         1.960E-02

The exact numbers will vary from run to run, but the built-in and manual
approaches should converge to the same solution when configured the same way.

What the Example Demonstrates
-----------------------------

This example shows that:

* athena can optimise inputs with the same autodiff machinery used for training
* the built-in routine is a convenience wrapper around a standard manual loop
* the network remains usable after inverse design
* the inverse solution depends on the learned model, not only on the underlying
  analytical function

See Also
--------

* :ref:`Inverse Design <inverse-design>`
* :ref:`Training a Model <training-model>`
* :ref:`Function Approximation <regression-example>`
