.. _custom-initialisers:

Creating Custom Initialisers
=============================

You can create custom weight initialisation strategies by extending the ``base_init_type``.

Base Initialiser Type
----------------------

All initialisers inherit from ``base_init_type`` and must implement the ``initialise`` procedure.

Required Procedures
~~~~~~~~~~~~~~~~~~~

* **initialise**: Initialise weights/biases with the desired distribution or strategy

Essential Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   type, extends(base_init_type) :: custom_init_type
      ! Add custom parameters here
      real(real32) :: custom_param = 1.0_real32
    contains
      procedure, pass(this) :: initialise => initialise_custom
   end type custom_init_type

Example: Truncated Normal Initialiser
--------------------------------------

Here's a complete example of a custom initialiser:

.. code-block:: fortran

   module my_truncated_normal_init
     use coreutils, only: real32
     use athena__misc_types, only: base_init_type
     use athena__random, only: normal_dist
     implicit none

     type, extends(base_init_type) :: truncated_normal_init_type
        real(real32) :: lower_bound = -2.0_real32
        real(real32) :: upper_bound = 2.0_real32
      contains
        procedure, pass(this) :: initialise => initialise_truncated_normal
     end type truncated_normal_init_type

     interface truncated_normal_init_type
        module function setup(mean, std, lower, upper) result(initialiser)
          real(real32), optional, intent(in) :: mean, std, lower, upper
          type(truncated_normal_init_type) :: initialiser
        end function setup
     end interface

   contains

     function setup(mean, std, lower, upper) result(initialiser)
       real(real32), optional, intent(in) :: mean, std, lower, upper
       type(truncated_normal_init_type) :: initialiser

       initialiser%name = "truncated_normal"
       if(present(mean)) initialiser%mean = mean
       if(present(std)) initialiser%std = std
       if(present(lower)) initialiser%lower_bound = lower
       if(present(upper)) initialiser%upper_bound = upper
     end function setup

     subroutine initialise_truncated_normal(this, input, fan_in, fan_out, spacing)
       class(truncated_normal_init_type), intent(inout) :: this
       real(real32), dimension(..), intent(out) :: input
       integer, optional, intent(in) :: fan_in, fan_out
       integer, dimension(:), optional, intent(in) :: spacing

       real(real32) :: value
       integer :: i, n

       ! Get total number of elements
       select rank(input)
       rank(1)
         n = size(input, 1)
       rank(2)
         n = size(input, 1) * size(input, 2)
       rank(3)
         n = size(input, 1) * size(input, 2) * size(input, 3)
       rank(4)
         n = size(input, 1) * size(input, 2) * size(input, 3) * size(input, 4)
       end select

       ! Initialise with truncated normal distribution
       do i = 1, n
         do
           call normal_dist(value, this%mean, this%std)
           if(value >= this%lower_bound .and. value <= this%upper_bound) exit
         end do

         ! Assign to input array based on rank
         select rank(input)
         rank(1)
           input(i) = value
         rank(2)
           input(mod(i-1, size(input,1))+1, (i-1)/size(input,1)+1) = value
         ! Add more ranks as needed
         end select
       end do
     end subroutine initialise_truncated_normal

   end module my_truncated_normal_init

Fan-In and Fan-Out
------------------

Many initialisation schemes depend on the layer dimensions:

.. code-block:: fortran

   subroutine initialise_custom(this, input, fan_in, fan_out, spacing)
     class(custom_init_type), intent(inout) :: this
     real(real32), dimension(..), intent(out) :: input
     integer, optional, intent(in) :: fan_in, fan_out
     integer, dimension(:), optional, intent(in) :: spacing

     real(real32) :: scale_factor

     ! Calculate scale based on fan-in and fan-out
     if(present(fan_in) .and. present(fan_out)) then
       scale_factor = sqrt(2.0 / real(fan_in + fan_out))
     else if(present(fan_in)) then
       scale_factor = sqrt(1.0 / real(fan_in))
     else
       scale_factor = 1.0
     end if

     ! Use scale_factor in initialisation
     ! ...
   end subroutine initialise_custom

Registering Your Initialiser
-----------------------------

To use your custom initialiser, add it to the initialiser setup:

.. code-block:: fortran

   ! In mod_initialiser.f90, add to initialiser_setup function:
   case("truncated_normal")
     initialiser = truncated_normal_init_type()

Common Initialisation Patterns
-------------------------------

Variance Scaling
~~~~~~~~~~~~~~~~

Scale initialisation based on layer size to maintain activation variance:

.. math::

   \text{scale} = \sqrt{\frac{2}{n_{in} + n_{out}}}

This is the basis for Glorot (Xavier) initialisation.

Uniform vs Normal
~~~~~~~~~~~~~~~~~

Choose between uniform and normal distributions:

* **Uniform**: ``U[-a, a]`` - bounded, good for shallow networks
* **Normal**: ``N(0, σ²)`` - unbounded, common in modern architectures

Layer-Specific Initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different layer types may need different strategies:

* **Convolutional**: Consider kernel size in fan-in/fan-out calculation
* **Recurrent**: Often use orthogonal initialisation for hidden states
* **Batch Norm**: Typically initialise γ=1, β=0

Best Practices
--------------

1. **Activation Awareness**: Match initialisation to the activation function
2. **Scale Appropriately**: Prevent vanishing/exploding gradients
3. **Reproducibility**: Use seeded random number generation for reproducible results
4. **Documentation**: Clearly document the mathematical basis and use cases
5. **Testing**: Verify that weights have the expected statistical properties

See Also
--------

* :ref:`Built-in Initialisers <initialisers>`
* :ref:`Custom Layers <custom-layers>`
* Example implementations in ``src/lib/mod_initialiser_*.f90``
