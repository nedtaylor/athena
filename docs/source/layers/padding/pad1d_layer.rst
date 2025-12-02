.. _pad1d-layer:

1D Padding Layer
================

``pad1d_layer_type``

.. code-block:: fortran

  pad1d_layer_type(
    padding,
    method,
    input_shape=...
  )


The ``pad1d_layer_type`` derived type provides a 1D padding layer that adds padding to the input tensor along the width dimension.

Arguments
---------

* **padding** (`integer` or `integer, dimension(1)`): Padding size. Required argument.
* **method** (`character(*)`): Padding method. Required argument.

  * ``"none"``/``"valid"``: No padding is applied.
  * ``"same"``/``"constant"``/``"zero"``: Pad with a constant value (zeros).
  * ``"full"``: Same as "same" but assumes padding such that each element has equal contributions and the output size is increased accordingly.
  * ``"circular"``: Wrap around the input values.
  * ``"reflection"``: Reflect the values at the boundaries.
  * ``"replication"``: Replicate the edge values.

* **input_shape** (`integer, dimension(:)`): Shape of the input data (width, channels).

Shape:
------

* Input: ``(width, channels, batch_size)``.
* Output: ``(width + 2 * padding(1), channels, batch_size)``.
