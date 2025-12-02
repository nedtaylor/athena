.. _dropblock3d-layer:

3D DropBlock Layer
==================

``dropblock3d_layer_type``

.. code-block:: fortran

  dropblock3d_layer_type(
    rate,
    block_size,
    input_shape=...,
    batch_size=...
  )


The ``dropblock3d_layer_type`` derived type provides a 3D DropBlock layer for regularisation.
Unlike standard dropout which drops individual elements randomly, DropBlock drops contiguous regions (blocks) of feature maps.
This is particularly effective for 3D convolutional networks as it forces the network to learn more robust spatial features.

Arguments
---------

* **rate** (`real(real32)`): Fraction of the units to drop. Must be between 0 and 1. Required argument.
* **block_size** (`integer`): Size of the cubic blocks to drop. Required argument.
* **input_shape** (`integer, dimension(:)`): Shape of the input data (width, height, depth, channels).
* **batch_size** (`integer`): **SOON TO BE DEPRECATED**. Batch size for the layer. Handled automatically during training and inference.

Shape:
------

* Input: ``(width, height, depth, channels, batch_size)``.
* Output: ``(width, height, depth, channels, batch_size)``.

Notes:
------

DropBlock is designed for 3D convolutional layers and drops spatially contiguous cubic regions rather than individual activations.
The layer is inactive during inference.
