.. _reshape-layer:

Reshape Layer
=============

``reshape_layer_type``

.. code-block:: fortran

  reshape_layer_type(
    output_shape,
    input_shape=...,
  )


The ``reshape_layer_type`` derived type provides a layer that reshapes the input tensor to a specified output shape (per batch sample).
This is commonly used when transitioning between layers that require different input shapes.

Arguments
---------

* **output_shape_shape** (`integer, dimension(:)`): Shape of the output data (excluding batch dimension).
* **input_shape** (`integer, dimension(:)`): Shape of the input data (excluding batch dimension).

Shape:
------

* Input: ``(input_shape, batch_size)``.
* Output: ``(output_shape, batch_size)``.
