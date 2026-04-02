.. _inverse-design:

Inverse Design
==============

This tutorial introduces inverse design in athena: given a trained model
:math:`y = f(x)`, find an input :math:`x^*` whose predicted output matches a
desired target :math:`y_t`.

Unlike standard training, inverse design keeps the network parameters fixed and
optimises the input instead.

Overview
--------

Inverse design solves an optimisation problem of the form:

.. math::

   x^* = \arg\min_x \mathcal{L}(f(x), y_t)

where :math:`f` is the trained network, :math:`y_t` is the target output, and
:math:`\mathcal{L}` is the loss used to compare the prediction with the target.

In athena, inverse design is useful when you want to:

* find an input that produces a desired response from a surrogate model
* solve inverse problems using a differentiable neural network
* optimise controllable inputs while leaving the trained model unchanged

Workflow
--------

The inverse-design workflow is:

1. Build and compile a network.
2. Train the network in the normal forward direction.
3. Choose a target output and an initial input guess.
4. Run ``network%inverse_design(...)`` to optimise the input.
5. Verify the result with ``predict()``.

This is conceptually similar to training, but the variable being updated is the
input rather than the weights.

Requirements
------------

Before calling ``inverse_design()``, the network should already be compiled and
trained.

The implementation relies on the network loss function to measure how close the
current prediction is to the target. If no loss has been configured,
``inverse_design()`` falls back to MSE internally.

Because inverse design uses backpropagation through the network, inference mode
is temporarily disabled during the optimisation loop and restored afterward.

What Stays Fixed
----------------

Inverse design does not train the model weights.

Internally, athena:

* runs forward and backward passes through the existing network
* computes gradients with respect to the input layer output
* updates only the input variables
* resets parameter gradients after each step
* restores the saved model parameters before returning

The intent is that the trained network remains unchanged after the inverse
design call.

Built-In API
------------

The public interface is the generic procedure:

.. code-block:: fortran

   x_opt = network%inverse_design(target, x_init, optimiser, steps)

Supported argument families are:

* ``real(real32), dimension(:,:)``
* scalar ``array_type``
* ``array_type, dimension(:,:)``

The result type matches the interface branch that is called:

* real input returns a real 2D array
* scalar ``array_type`` input returns a scalar ``array_type``
* 2D ``array_type`` input returns a 2D ``array_type`` array

Parameters
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Argument
     - Type
     - Description
   * - ``target``
     - real 2D array or ``array_type``
     - Desired network output.
   * - ``x_init``
     - real 2D array or ``array_type``
     - Initial guess for the input to optimise.
   * - ``optimiser``
     - ``class(base_optimiser_type)``
     - Optional optimiser used to update the input.
   * - ``steps``
     - ``integer``
     - Number of optimisation iterations.

Optimiser Behaviour
~~~~~~~~~~~~~~~~~~~

If ``optimiser`` is provided, athena clones that optimiser and uses it for the
input updates.

If ``optimiser`` is omitted, athena does not reuse the network optimiser
object directly. Instead, it creates a plain ``base_optimiser_type`` using the
network optimiser's learning rate.

Typical Usage
-------------

For real-array inputs, a typical call looks like this:

.. code-block:: fortran

   real(real32) :: target_y(1,1), x_init(1,1)
   real(real32), allocatable :: x_opt(:,:)

   target_y(1,1) = 0.6_real32
   x_init(1,1) = 0.1_real32

   x_opt = network%inverse_design( &
        target=target_y, &
        x_init=x_init, &
        optimiser=sgd_optimiser_type(learning_rate=0.1_real32), &
        steps=2000)

After optimisation, verify the result with a forward prediction:

.. code-block:: fortran

   real(real32) :: predicted(1,1)

   predicted = network%predict(input=x_opt)
   write(*,'(A,F10.6)') "Predicted output: ", predicted(1,1)

Manual Workflow
---------------

The built-in routine is convenient, but the same idea can be implemented
manually:

1. Run ``forward()`` with the current input.
2. Set the target output on the network.
3. Evaluate the loss with ``loss_eval()``.
4. Call ``grad_reverse()`` to backpropagate.
5. Extract the gradient with respect to the input.
6. Update the input with an optimiser.
7. Reset gradients and clear graph state.

This is useful when you need custom logging, constraints, early stopping, or a
specialised update rule.

.. important::

   In a manual inverse-design loop, do not call ``network%update()`` unless
   you intentionally want to update the model parameters. Standard inverse
   design updates the input, not the network weights.

Practical Notes
---------------

* The inverse solution is the solution of the trained network, not necessarily
  the analytical inverse of the original function.
* Multiple inputs can produce similar outputs, so the result can depend on the
  initial guess and optimiser settings.
* If the network is only an approximation of the target mapping, an apparently
  correct inverse input may differ from the exact mathematical value.

Worked Example
--------------

For a complete walkthrough of the repository example, see
:ref:`Inverse Design Example <inverse-design-example>`.

See Also
--------

* :ref:`Training a Model <training-model>`
* :ref:`Building a Basic Network <basic-network>`
* :ref:`Function Approximation <regression-example>`
* :ref:`Custom Loss Functions <custom-loss>`
