.. _maxpool1d-layer:

1D Max Pooling Layer
====================

``maxpool1d_layer_type``

.. code-block:: fortran

  maxpool1d_layer_type(
    pool_size=2,
    stride=...,
    input_shape=...
  )


The ``maxpool1d_layer_type`` derived type provides a 1D max pooling layer.
This layer performs downsampling by dividing the input into pooling regions and taking the maximum value in each region.

Arguments
---------

* **pool_size** (`integer` or `integer, dimension(1)`): Size of the pooling window. Default: ``2``.
* **stride** (`integer` or `integer, dimension(1)`): Stride of the pooling operation. Default: ``pool_size``.
* **input_shape** (`integer, dimension(:)`): Shape of the input data (width, channels).

Shape:
------

* Input: ``(width, channels, batch_size)``.
* Output: ``(width_out, channels, batch_size)``.

where:

.. math::

   \text{width_out} = \left\lfloor \frac{\text{width} - \text{pool_size}}{\text{stride}} + 1 \right\rfloor
