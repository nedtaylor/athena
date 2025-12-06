.. _custom-activations:

Creating Custom Activation Functions
=====================================

You can extend athena with custom activation functions by implementing the ``base_actv_type`` interface.

Base Activation Type
--------------------

All activation functions inherit from ``base_actv_type`` and must implement four key procedures:

Required Procedures
~~~~~~~~~~~~~~~~~~~

* **apply**: Apply the activation function to input values
* **reset**: Reset any internal state or attributes
* **apply_attributes**: Load parameters from ONNX attributes
* **export_attributes**: Export parameters as ONNX attributes

Essential Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   type, extends(base_actv_type) :: custom_actv_type
      ! Add custom parameters here
      real(real32) :: custom_param = 1.0_real32
    contains
      procedure, pass(this) :: apply => apply_custom
      procedure, pass(this) :: reset => reset_custom
      procedure, pass(this) :: apply_attributes => apply_attributes_custom
      procedure, pass(this) :: export_attributes => export_attributes_custom
   end type custom_actv_type

Example: Custom Parametric Activation
--------------------------------------

Here's a complete example of a custom activation function:

.. code-block:: fortran

   module my_custom_activation
     use coreutils, only: real32
     use athena__misc_types, only: base_actv_type, onnx_attribute_type
     use diffstruc, only: array_type
     implicit none

     type, extends(base_actv_type) :: parametric_actv_type
        real(real32) :: alpha = 0.1_real32  ! Custom parameter
      contains
        procedure, pass(this) :: apply => apply_parametric
        procedure, pass(this) :: reset => reset_parametric
        procedure, pass(this) :: apply_attributes => apply_attributes_parametric
        procedure, pass(this) :: export_attributes => export_attributes_parametric
     end type parametric_actv_type

     interface parametric_actv_type
        module function setup(alpha) result(activation)
          real(real32), optional, intent(in) :: alpha
          type(parametric_actv_type) :: activation
        end function setup
     end interface

   contains

     function setup(alpha) result(activation)
       real(real32), optional, intent(in) :: alpha
       type(parametric_actv_type) :: activation

       activation%name = "parametric"
       if(present(alpha)) activation%alpha = alpha
     end function setup

     function apply_parametric(this, val) result(output)
       class(parametric_actv_type), intent(in) :: this
       type(array_type), intent(in) :: val
       type(array_type), pointer :: output

       ! Example: f(x) = x if x > 0, alpha * x otherwise
       allocate(output)
       output = merge(val, val * this%alpha, val > 0.0_real32)
     end function apply_parametric

     subroutine reset_parametric(this)
       class(parametric_actv_type), intent(inout) :: this

       ! Reset to default values if needed
       this%alpha = 0.1_real32
       this%scale = 1.0_real32
     end subroutine reset_parametric

     subroutine apply_attributes_parametric(this, attributes)
       class(parametric_actv_type), intent(inout) :: this
       type(onnx_attribute_type), dimension(:), intent(in) :: attributes
       integer :: i

       do i = 1, size(attributes)
         select case(trim(attributes(i)%name))
         case("alpha")
           read(attributes(i)%val, *) this%alpha
         end select
       end do
     end subroutine apply_attributes_parametric

     pure function export_attributes_parametric(this) result(attributes)
       class(parametric_actv_type), intent(in) :: this
       type(onnx_attribute_type), allocatable, dimension(:) :: attributes
       character(20) :: alpha_str

       allocate(attributes(1))
       write(alpha_str, '(F20.10)') this%alpha
       attributes(1) = onnx_attribute_type("alpha", "float", trim(alpha_str))
     end function export_attributes_parametric

   end module my_custom_activation

Registering Your Activation
----------------------------

To use your custom activation in athena networks, add it to the activation setup:

.. code-block:: fortran

   ! In mod_activation.f90, add to activation_setup function:
   case("parametric")
     activation = parametric_actv_type()

Mathematical Considerations
---------------------------

Gradient Computation
~~~~~~~~~~~~~~~~~~~~

When implementing activation functions for training, consider the gradient:

* The ``apply`` function works with ``array_type`` which tracks gradients automatically
* Ensure your function is differentiable where needed
* Use ``merge`` for conditional operations to maintain gradient flow

Numerical Stability
~~~~~~~~~~~~~~~~~~~

* Avoid operations that can cause overflow (e.g., large exponentials)
* Add small epsilon values where division by zero might occur
* Consider the range of expected input values

Best Practices
--------------

1. **Naming**: Use descriptive names that indicate the activation's behavior
2. **Parameters**: Provide sensible defaults for any parameters
3. **Documentation**: Include the mathematical formula in comments
4. **Testing**: Test with various input ranges including edge cases
5. **ONNX Support**: Implement attribute import/export for model portability

See Also
--------

* :ref:`Built-in Activation Functions <activation-functions>`
* :ref:`ONNX Export <api>`
* Example implementations in ``src/lib/mod_activation_*.f90``
