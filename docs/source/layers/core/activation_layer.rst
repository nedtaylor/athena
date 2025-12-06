.. _activation-layer:

Activation Layer
================

``activation_layer_type``

.. code-block:: fortran

  activation_layer_type(
    activation,
    input_shape=...
  )


The ``activation_layer_type`` derived type provides a layer that applies an activation function element-wise to the input data.

Arguments
---------

* **activation** (`class(*)`): Activation function for the layer.

  * Accepts `character(*)` or `class(base_actv_type)`.
  * See :ref:`Activation Functions <activation-functions>` for available options.
  * Required argument.

* **input_shape** (`integer, dimension(:)`): Shape of the input data (excluding batch dimension).

Shape:
------

* Input: ``(input_shape, batch_size)``.
* Output: ``(input_shape, batch_size)`` - same shape as input.
