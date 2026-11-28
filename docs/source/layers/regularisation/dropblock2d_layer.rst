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

* **rate** (`real(real32)`): Fraction of the units to drop. Must be in the range ``[0, 1)``. Required argument.
* **block_size** (`integer`): Size of the square blocks to drop. Must be between 1 and the smallest spatial dimension. Required argument.
* **input_shape** (`integer, dimension(:)`): Shape of the input data (width, height, channels). The input rank must be 3.

Shape:
------

* Input: ``(width, height, channels, batch_size)``.
* Output: ``(width, height, channels, batch_size)``.

Notes:
------

DropBlock is designed for convolutional layers and drops spatially contiguous regions rather than individual activations.
A fresh mask is sampled on every forward pass during training, so the dropped regions change between iterations.
Training uses count-based normalisation, scaling the retained activations by ``count_elements / count_ones``.
The layer is the identity during inference, with no scaling applied.
