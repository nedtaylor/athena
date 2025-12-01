.. _custom-loss:

Creating Custom Loss Functions
===============================

You can implement custom loss functions by extending the ``base_loss_type``.

Base Loss Type
--------------

All loss functions inherit from ``base_loss_type`` and must implement the ``compute`` method.

Required Procedures
~~~~~~~~~~~~~~~~~~~

* **compute**: Calculate the loss given predicted and expected values

Essential Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   type, extends(base_loss_type) :: custom_loss_type
      ! Add custom parameters here
      real(real32) :: custom_param = 1.0_real32
    contains
      procedure, pass(this) :: compute => compute_custom
   end type custom_loss_type

Example: Focal Loss
-------------------

Here's a complete example of a custom loss function (Focal Loss for handling class imbalance):

.. code-block:: fortran

   module my_focal_loss
     use coreutils, only: real32
     use athena__loss, only: base_loss_type
     use diffstruc, only: array_type, operator(-), operator(*), operator(**), &
                          log, sum, mean
     implicit none

     type, extends(base_loss_type) :: focal_loss_type
        real(real32) :: alpha = 0.25_real32   ! Balance factor
        real(real32) :: gamma = 2.0_real32     ! Focusing parameter
      contains
        procedure, pass(this) :: compute => compute_focal
     end type focal_loss_type

     interface focal_loss_type
        module function setup(alpha, gamma) result(loss)
          real(real32), optional, intent(in) :: alpha, gamma
          type(focal_loss_type) :: loss
        end function setup
     end interface

   contains

     function setup(alpha, gamma) result(loss)
       real(real32), optional, intent(in) :: alpha, gamma
       type(focal_loss_type) :: loss

       loss%name = "focal"
       if(present(alpha)) loss%alpha = alpha
       if(present(gamma)) loss%gamma = gamma
       loss%epsilon = 1.E-10_real32
     end function setup

     function compute_focal(this, predicted, expected) result(output)
       class(focal_loss_type), intent(in), target :: this
       type(array_type), dimension(:,:), intent(inout), target :: predicted
       type(array_type), dimension(size(predicted,1),size(predicted,2)), &
            intent(in) :: expected
       type(array_type), pointer :: output

       type(array_type) :: pt, focal_weight, loss_val

       ! Clip predictions to prevent log(0)
       pt = predicted
       ! Note: In practice, add clipping: max(epsilon, min(1-epsilon, predicted))

       ! Compute focal weight: (1 - pt)^gamma
       focal_weight = (1.0_real32 - pt)**this%gamma

       ! Compute focal loss: -alpha * (1-pt)^gamma * log(pt)
       loss_val = -this%alpha * focal_weight * log(pt + this%epsilon) * expected

       ! Average over samples
       allocate(output)
       output = mean(sum(loss_val))

     end function compute_focal

   end module my_focal_loss

Example: Custom Regression Loss
--------------------------------

A loss function that combines MSE with a penalty term:

.. code-block:: fortran

   module my_regularized_mse
     use coreutils, only: real32
     use athena__loss, only: base_loss_type
     use diffstruc, only: array_type, operator(-), operator(*), operator(**), &
                          mean, sum, abs

     type, extends(base_loss_type) :: regularized_mse_type
        real(real32) :: lambda = 0.01_real32  ! Regularization strength
      contains
        procedure, pass(this) :: compute => compute_regularized_mse
     end type regularized_mse_type

   contains

     function compute_regularized_mse(this, predicted, expected) result(output)
       class(regularized_mse_type), intent(in), target :: this
       type(array_type), dimension(:,:), intent(inout), target :: predicted
       type(array_type), dimension(size(predicted,1),size(predicted,2)), &
            intent(in) :: expected
       type(array_type), pointer :: output

       type(array_type) :: mse_term, reg_term, diff

       ! Compute MSE
       diff = predicted - expected
       mse_term = mean((diff)**2)

       ! Add regularization term (e.g., sum of absolute predictions)
       reg_term = this%lambda * mean(abs(predicted))

       allocate(output)
       output = mse_term + reg_term

     end function compute_regularized_mse

   end module my_regularized_mse

Working with array_type
------------------------

The ``array_type`` supports automatic differentiation. Use these operations:

Basic Operations
~~~~~~~~~~~~~~~~

.. code-block:: fortran

   ! Arithmetic
   result = a + b      ! Addition
   result = a - b      ! Subtraction
   result = a * b      ! Multiplication
   result = a / b      ! Division
   result = a ** 2     ! Power

   ! Functions
   result = log(a)     ! Natural logarithm
   result = abs(a)     ! Absolute value
   result = exp(a)     ! Exponential
   result = sqrt(a)    ! Square root

   ! Reductions
   result = sum(a)     ! Sum all elements
   result = mean(a)    ! Mean of elements

   ! Conditionals (preserve gradients)
   result = merge(a, b, condition)

Best Practices
--------------

1. **Numerical Stability**

   - Add epsilon to prevent log(0) or division by zero
   - Clip extreme values when necessary
   - Use log-space computations when dealing with very small probabilities

2. **Gradient Flow**

   - Ensure all operations support backpropagation
   - Use ``merge`` instead of ``if-else`` for conditionals
   - Avoid operations that break gradient flow

3. **Batch Processing**

   - Loss should average over the batch dimension
   - Use ``mean`` for proper averaging
   - Consider sample weights if needed

4. **Testing**

   - Verify loss decreases during training
   - Check gradients are computed correctly
   - Test with known inputs and expected outputs
   - Compare against reference implementations

5. **Documentation**

   - Cite original papers for novel losses
   - Document the mathematical formulation
   - Explain when to use the loss function
   - Provide guidance on hyperparameter tuning

Common Patterns
---------------

Classification Loss Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   function compute_classification(this, predicted, expected) result(output)
     ! ...

     ! Clip predictions to valid probability range
     p = max(this%epsilon, min(1.0 - this%epsilon, predicted))

     ! Compute cross-entropy or similar
     loss = -expected * log(p)

     ! Average over samples
     output = mean(sum(loss))
   end function

Regression Loss Template
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   function compute_regression(this, predicted, expected) result(output)
     ! ...

     ! Compute error
     error = predicted - expected

     ! Apply loss function to error
     loss = error**2  ! Or abs(error), or custom function

     ! Average over samples
     output = mean(loss)
   end function

See Also
--------

* :ref:`Built-in Loss Functions <loss-functions>`
* :ref:`Custom Layers <custom-layers>`
* ``diffstruc`` module documentation for array operations
