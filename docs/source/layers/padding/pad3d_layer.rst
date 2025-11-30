.. _pad3d-layer:

3D Padding Layer
================

``pad3d_layer_type``

.. code-block:: fortran

  pad3d_layer_type(
    padding,
    method,
    input_shape=...,
    batch_size=...
  )


The ``pad3d_layer_type`` derived type provides a 3D padding layer that adds padding to the input tensor along the spatial dimensions.

Arguments
---------

* **padding** (`integer, dimension(:)`): Padding sizes ``[pad_width, pad_height, pad_depth]``. Required argument.
* **method** (`character(*)`): Padding method. Required argument.

  * ``"none"``/``"valid"``: No padding is applied.
  * ``"same"``/``"constant"``/``"zero"``: Pad with a constant value (zeros).
  * ``"full"``: Same as "same" but assumes padding such that each element has equal contributions and the output size is increased accordingly.
  * ``"circular"``: Wrap around the input values.
  * ``"reflection"``: Reflect the values at the boundaries.
  * ``"replication"``: Replicate the edge values.

* **input_shape** (`integer, dimension(:)`): Shape of the input data (width, height, depth, channels).
* **batch_size** (`integer`): **SOON TO BE DEPRECATED**. Batch size for the layer. Handled automatically during training and inference.

Shape:
------

* Input: ``(width, height, depth, channels, batch_size)``.
* Output: ``(width + 2*padding(1), height + 2*padding(2), depth + 2*padding(3), channels, batch_size)``.
