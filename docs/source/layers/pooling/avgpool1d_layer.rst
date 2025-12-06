.. _avgpool1d-layer:

1D Average Pooling Layer
=========================

``avgpool1d_layer_type``

.. code-block:: fortran

  avgpool1d_layer_type(
    pool_size=2,
    stride=...,
    input_shape=...,
    padding="valid"
  )


The ``avgpool1d_layer_type`` derived type provides a 1D average pooling layer.
This layer performs downsampling by dividing the input into pooling regions and computing the average of each region.

Arguments
---------

* **pool_size** (`integer` or `integer, dimension(1)`): Size of the pooling window. Default: ``2``.
* **stride** (`integer` or `integer, dimension(1)`): Stride of the pooling operation. Default: ``pool_size``.
* **input_shape** (`integer, dimension(:)`): Shape of the input data (width, channels).
* **padding** (`character(*)`): Padding method, if any, to be applied to the input data prior to pooling. Refer to :ref:`1D padding layer <pad1d-layer>` for options. Default: ``"valid"``, i.e. no padding.

Shape:
------

* Input: ``(width, channels, batch_size)``.
* Output: ``(width_out, channels, batch_size)``.

where:

.. math::

   \text{width_out} = \left\lfloor \frac{\text{width} - \text{pool_size}}{\text{stride}} + 1 \right\rfloor
