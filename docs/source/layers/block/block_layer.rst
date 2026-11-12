.. _block-layer:

Block Layer
======================

``block_layer_type``

.. code-block:: fortran

  conv1d_layer_type(
    layers,
    input_shape=...
  )


The ``block_layer_type`` derived type provides a block layer that allows for the composition of multiple layers into a single unit.
This is useful for creating complex architectures by combining different types of layers together.

Arguments
---------

* **layers** (`type(container_layer_type)`): 1D array of layer containers that make up the block. Each layer container contains a single layer (needed for polymorphic operations).
* **input_shape** (`integer, dimension(:)`): Shape of the input data (channels, width).

Shape:
------

* Input: ``(..., batch_size)``.
* Output: ``(..., batch_size)``.
