.. _dropblock2d-layer:

2D DropBlock Layer
==================

``dropblock2d_layer_type``

.. code-block:: fortran

  dropblock2d_layer_type(
    rate,
    block_size,
    input_shape=...
  )


The ``dropblock2d_layer_type`` derived type provides a 2D DropBlock layer for regularisation.
Unlike standard dropout which drops individual elements randomly, DropBlock drops contiguous regions (blocks) of feature maps.
This is particularly effective for convolutional networks as it forces the network to learn more robust spatial features.

Arguments
---------

* **rate** (`real(real32)`): Fraction of the units to drop. Must be between 0 and 1. Required argument.
* **block_size** (`integer`): Size of the square blocks to drop. Required argument.
* **input_shape** (`integer, dimension(:)`): Shape of the input data (width, height, channels).

Shape:
------

* Input: ``(width, height, channels, batch_size)``.
* Output: ``(width, height, channels, batch_size)``.

Notes:
------

DropBlock is designed for convolutional layers and drops spatially contiguous regions rather than individual activations.
The layer is inactive during inference.
