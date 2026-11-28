.. _dropblock3d-layer:

3D DropBlock Layer
==================

``dropblock3d_layer_type``

.. code-block:: fortran

  dropblock3d_layer_type(
    rate,
    block_size,
    input_shape=...
  )


The ``dropblock3d_layer_type`` derived type provides a 3D DropBlock layer for regularisation.
Unlike standard dropout which drops individual elements randomly, DropBlock drops contiguous regions (blocks) of feature maps.
This is particularly effective for 3D convolutional networks as it forces the network to learn more robust spatial features.

Arguments
---------

* **rate** (`real(real32)`): Fraction of the units to drop. Must be in the range ``[0, 1)``. Required argument.
* **block_size** (`integer`): Size of the cubic blocks to drop. Must be between 1 and the smallest spatial dimension. Required argument.
* **input_shape** (`integer, dimension(:)`): Shape of the input data (width, height, depth, channels). The input rank must be 4.

Shape:
------

* Input: ``(width, height, depth, channels, batch_size)``.
* Output: ``(width, height, depth, channels, batch_size)``.

Notes:
------

DropBlock is designed for 3D convolutional layers and drops spatially contiguous cubic regions rather than individual activations.
A fresh mask is sampled on every forward pass during training, so the dropped regions change between iterations.
Training uses count-based normalisation, scaling the retained activations by ``count_elements / count_ones``.
The layer is the identity during inference, with no scaling applied.
