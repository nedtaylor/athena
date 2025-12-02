.. _custom-optimisers:

Creating Custom Optimisers
===========================

You can implement custom optimisation algorithms by extending the ``base_optimiser_type``.

Base Optimiser Type
--------------------

All optimisers inherit from ``base_optimiser_type`` and must implement the ``minimise`` procedure.

Required Procedures
~~~~~~~~~~~~~~~~~~~

* **minimise**: Apply gradients to parameters to minimise the loss
* **init_gradients**: Initialise optimiser-specific gradient accumulators

Essential Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   type, extends(base_optimiser_type) :: custom_optimiser_type
      ! Add optimiser state variables here
      real(real32), allocatable, dimension(:) :: state_variables
    contains
      procedure, pass(this) :: init_gradients => init_gradients_custom
      procedure, pass(this) :: minimise => minimise_custom
   end type custom_optimiser_type

Example: Custom Momentum-Based Optimiser
-----------------------------------------

Here's a complete example of a custom optimiser with adaptive momentum:

.. code-block:: fortran

   module my_adaptive_optimiser
     use coreutils, only: real32
     use athena__optimiser, only: base_optimiser_type
     implicit none

     type, extends(base_optimiser_type) :: adaptive_momentum_optimiser_type
        real(real32) :: beta = 0.9_real32
        real(real32) :: epsilon = 1.E-8_real32
        real(real32), allocatable, dimension(:) :: velocity
        real(real32), allocatable, dimension(:) :: grad_variance
      contains
        procedure, pass(this) :: init_gradients => init_gradients_adaptive
        procedure, pass(this) :: minimise => minimise_adaptive
     end type adaptive_momentum_optimiser_type

     interface adaptive_momentum_optimiser_type
        module function setup(learning_rate, beta, epsilon, num_params, &
                             regulariser, clip_dict, lr_decay) result(optimiser)
          use athena__regulariser, only: base_regulariser_type
          use athena__clipper, only: clip_type
          use athena__learning_rate_decay, only: base_lr_decay_type

          real(real32), optional, intent(in) :: learning_rate, beta, epsilon
          integer, optional, intent(in) :: num_params
          class(base_regulariser_type), optional, intent(in) :: regulariser
          type(clip_type), optional, intent(in) :: clip_dict
          class(base_lr_decay_type), optional, intent(in) :: lr_decay
          type(adaptive_momentum_optimiser_type) :: optimiser
        end function setup
     end interface

   contains

     function setup(learning_rate, beta, epsilon, num_params, &
                   regulariser, clip_dict, lr_decay) result(optimiser)
       use athena__regulariser, only: base_regulariser_type
       use athena__clipper, only: clip_type
       use athena__learning_rate_decay, only: base_lr_decay_type

       real(real32), optional, intent(in) :: learning_rate, beta, epsilon
       integer, optional, intent(in) :: num_params
       class(base_regulariser_type), optional, intent(in) :: regulariser
       type(clip_type), optional, intent(in) :: clip_dict
       class(base_lr_decay_type), optional, intent(in) :: lr_decay
       type(adaptive_momentum_optimiser_type) :: optimiser

       ! Initialise base optimiser
       optimiser%base_optimiser_type = base_optimiser_type( &
            learning_rate, num_params, regulariser, clip_dict, lr_decay)

       optimiser%name = "adaptive_momentum"
       if(present(beta)) optimiser%beta = beta
       if(present(epsilon)) optimiser%epsilon = epsilon

       if(present(num_params)) call optimiser%init_gradients(num_params)
     end function setup

     subroutine init_gradients_adaptive(this, num_params)
       class(adaptive_momentum_optimiser_type), intent(inout) :: this
       integer, intent(in) :: num_params

       if(allocated(this%velocity)) deallocate(this%velocity)
       if(allocated(this%grad_variance)) deallocate(this%grad_variance)

       allocate(this%velocity(num_params), source=0.0_real32)
       allocate(this%grad_variance(num_params), source=0.0_real32)
     end subroutine init_gradients_adaptive

     subroutine minimise_adaptive(this, params, gradients)
       class(adaptive_momentum_optimiser_type), intent(inout) :: this
       real(real32), dimension(:), intent(inout) :: params
       real(real32), dimension(size(params)), intent(in) :: gradients

       real(real32) :: effective_lr, adaptive_momentum
       integer :: i

       ! Update iteration counter
       this%iter = this%iter + 1

       ! Get current learning rate (with decay if applicable)
       effective_lr = this%learning_rate
       if(allocated(this%lr_decay)) then
         effective_lr = effective_lr * this%lr_decay%get_factor(this%epoch, this%iter)
       end if

       ! Update gradient variance estimate
       this%grad_variance = this%beta * this%grad_variance + &
                           (1.0_real32 - this%beta) * gradients**2

       ! Compute adaptive momentum coefficient
       do i = 1, size(params)
         adaptive_momentum = this%beta / (1.0_real32 + sqrt(this%grad_variance(i)))

         ! Update velocity with adaptive momentum
         this%velocity(i) = adaptive_momentum * this%velocity(i) + &
                           (1.0_real32 - adaptive_momentum) * gradients(i)

         ! Apply regularisation if present
         if(this%regularisation) then
           this%velocity(i) = this%velocity(i) + &
                             this%regulariser%apply(params(i))
         end if

         ! Apply gradient clipping if configured
         if(this%clip_dict%active) then
           this%velocity(i) = this%clip_dict%clip(this%velocity(i))
         end if

         ! Update parameters
         params(i) = params(i) - effective_lr * this%velocity(i)
       end do
     end subroutine minimise_adaptive

   end module my_adaptive_optimiser

Optimiser Components
--------------------

Learning Rate Decay
~~~~~~~~~~~~~~~~~~~

Use the built-in learning rate decay mechanisms:

.. code-block:: fortran

   effective_lr = this%learning_rate
   if(allocated(this%lr_decay)) then
     effective_lr = effective_lr * this%lr_decay%get_factor(this%epoch, this%iter)
   end if

Regularisation
~~~~~~~~~~~~~~

Apply regularisation penalties during parameter updates:

.. code-block:: fortran

   if(this%regularisation) then
     gradient = gradient + this%regulariser%apply(params)
   end if

Gradient Clipping
~~~~~~~~~~~~~~~~~

Prevent exploding gradients with clipping:

.. code-block:: fortran

   if(this%clip_dict%active) then
     gradient = this%clip_dict%clip(gradient)
   end if

Common Optimiser Patterns
--------------------------

Momentum Methods
~~~~~~~~~~~~~~~~

Accumulate gradients over time to smooth updates:

.. math::

   v_t = \beta v_{t-1} + (1-\beta) g_t \\
   \theta_{t+1} = \theta_t - \eta v_t

Adaptive Learning Rates
~~~~~~~~~~~~~~~~~~~~~~~~

Scale learning rate per parameter based on gradient history:

.. math::

   \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t

where :math:`v_t` accumulates squared gradients.

Bias Correction
~~~~~~~~~~~~~~~

Correct for initialisation bias in moment estimates:

.. math::

   \hat{m}_t = \frac{m_t}{1 - \beta^t}

Best Practices
--------------

1. **Hyperparameter Defaults**: Provide sensible defaults based on literature
2. **Numerical Stability**: Add epsilon terms to prevent division by zero
3. **State Management**: Properly initialise and manage optimiser state variables
4. **Memory Efficiency**: Only allocate what's necessary
5. **Testing**: Test on various problem types and network architectures
6. **Documentation**: Cite original papers and explain the algorithm

Performance Considerations
--------------------------

* **In-Place Operations**: Modify parameters in-place when possible
* **Vectorisation**: Use array operations instead of loops where practical
* **Memory Allocation**: Allocate state variables once in ``init_gradients``
* **Conditional Checks**: Minimise checks inside the optimisation loop

See Also
--------

* :ref:`Built-in Optimisers <optimisers>`
* :ref:`Learning Rate Decay <api>`
* :ref:`Regularisation <api>`
* Example implementations in ``src/lib/mod_optimiser.f90``
