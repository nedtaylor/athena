.. _maxpool2d-layer:

2D Max Pooling Layer
====================

``maxpool2d_layer_type``

.. code-block:: fortran

  maxpool2d_layer_type(
    pool_size=2,
    stride=...,
    input_shape=...,
    batch_size=...
  )


The ``maxpool2d_layer_type`` derived type provides a 2D max pooling layer.
This layer performs downsampling by dividing the input into pooling regions and taking the maximum value in each region.

Arguments
---------

* **pool_size** (`integer` or `integer, dimension(2)`): Size of the pooling window. If a single integer is provided, the same value is used for height and width. Default: ``2``.
* **stride** (`integer` or `integer, dimension(2)`): Stride of the pooling operation. Default: ``pool_size``.
* **input_shape** (`integer, dimension(:)`): Shape of the input data (width, height, channels).
* **batch_size** (`integer`): **SOON TO BE DEPRECATED**. Batch size for the layer. Handled automatically during training and inference.

Shape:
------

* Input: ``(width, height, channels, batch_size)``.
* Output: ``(width_out, height_out, channels, batch_size)``.

where:

.. math::

   \text{height_out} &= \left\lfloor \frac{\text{height} - \text{pool_size}[0]}{\text{stride}[0]} + 1 \right\rfloor \\
   \text{width_out} &= \left\lfloor \frac{\text{width} - \text{pool_size}[1]}{\text{stride}[1]} + 1 \right\rfloor
