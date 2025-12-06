.. _custom-layers:

Creating Custom Layers
======================

The athena library is designed with extensibility in mind, allowing you to create custom layers by extending the ``base_layer_type``.

Base Layer Type
---------------

All layers in athena inherit from ``base_layer_type``, which provides the core functionality and interface that all layers must implement.

Key Components
~~~~~~~~~~~~~~

A custom layer must implement the following deferred procedures:

* **init**: Initialise the layer with proper shapes and allocations
* **read**: Read layer configuration from file
* **forward**: Implement the forward pass computation

Essential Structure
~~~~~~~~~~~~~~~~~~~

The essential structure of a custom layer is as follows:

.. code-block:: fortran

   type, extends(base_layer_type) :: custom_layer_type
      ! Add custom attributes here
    contains
      procedure, pass(this) :: init => init_custom
      procedure, pass(this) :: read => read_custom
      procedure, pass(this) :: forward => forward_custom
   end type custom_layer_type

When implementing your custom layer, it is recommended to look at existing implemented layers in the athena source code for guidance.
For a basic layer (non-learnable), refer to the (:git:`activation layer<src/lib/mod_activation_layer.f90>`.
For a learnable layer, see the (:git:`fully connected layer<src/lib/mod_full_layer.f90>`.

Example: Simple Scaling Layer
------------------------------

Here's a complete example of a custom layer that scales inputs:

.. code-block:: fortran

   module my_scaling_layer
     use coreutils, only: real32
     use athena__base_layer, only: base_layer_type
     use diffstruc, only: array_type
     implicit none

     type, extends(base_layer_type) :: scaling_layer_type
        real(real32) :: scale_factor = 1.0_real32
      contains
        procedure, pass(this) :: init => init_scaling
        procedure, pass(this) :: read => read_scaling
        procedure, pass(this) :: forward => forward_scaling
     end type scaling_layer_type

   contains

     subroutine init_scaling(this)
       class(scaling_layer_type), intent(inout) :: this

       ! Set layer type
       this%name = "scaling"

       ! Allocate output
       if(allocated(this%output)) deallocate(this%output)
       allocate(this%output(1,1))

       ! Initialise output arrays
       call this%output(1,1)%allocate(this%input_shape)
     end subroutine init_scaling

     subroutine forward_scaling(this, input)
       class(scaling_layer_type), intent(inout) :: this
       type(array_type), dimension(:,:), intent(in) :: input
       integer :: i

       this%output(1,1) => input(1,1) * this%scale_factor
       this%output%is_temporary = .false.

     end subroutine forward_scaling

     subroutine read_scaling(this, unit)
       class(scaling_layer_type), intent(inout) :: this
       integer, intent(in) :: unit

       ! Implement reading scale_factor from file
     end subroutine read_scaling

   end module my_scaling_layer

All layers store their output in the ``output`` attribute inherited from ``base_layer_type``.
This ensures compatibility with the rest of the athena framework.
The output is a rank 2 array of ``array_type`` objects.
Typically, the shape is set to ``(1,1)``, as the data and samples are all stored in the ``array_type`` objects.
The shape of output only differs for layers that produce multiple outputs or handle graph data.

Advanced Features
-----------------

Learnable Parameters
~~~~~~~~~~~~~~~~~~~~

For layers with learnable parameters, extend ``learnable_layer_type`` instead:

.. code-block:: fortran

   type, extends(learnable_layer_type) :: custom_learnable_layer_type
    contains
      ! ... implement required procedures
   end type custom_learnable_layer_type

The learnable parameters wihin the layer are stored in the ``params`` attribute inherited from ``learnable_layer_type``.
This is a rank 1 array of ``array_type`` objects, each representing a learnable parameter tensor.

The backpropagation is handled automatically by the athena framework through its use of pointers and the automatic differentiation library `diffstruc <https://github.com/nedtaylor/diffstruc>`_.

Best Practices
--------------

1. **Shape Handling**: Always properly set ``input_shape`` and ``output_shape``
2. **Memory Management**: Allocate/deallocate arrays in ``init`` and cleanup procedures
3. **Batch Processing**: Ensure your layer handles variable batch sizes correctly
4. **Documentation**: Add clear comments explaining the layer's purpose and parameters
5. **Testing**: Create comprehensive tests for your custom layer

See Also
--------

* :ref:`Base Layer API <api>`
* :ref:`Learnable Layers <layers>`
* Example implementations in ``src/lib/mod_*_layer.f90``
