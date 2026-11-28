.. _block-layer:

Block Layer
======================

``block_layer_type``

.. code-block:: fortran

  block_layer_type( &
    layers=layers, &
    input_shape=... &
  )


The ``block_layer_type`` derived type provides a block layer that allows for the composition of multiple layers into a single unit.
This is useful for creating complex architectures by combining different types of layers together.

The base block is strictly sequential: each layer receives the output of the previous layer.
Architectures with branching or merged paths require a custom extension, such as the resnet example.

Arguments
---------

* **layers** (`type(container_layer_type)`): Optional 1D array of layer containers that make up the block. Each layer container contains a single layer (needed for polymorphic operations).
* **input_shape** (`integer, dimension(:)`): Shape of one input sample (excluding the batch dimension).

Parameters
----------

The parameters and gradients of all learnable layers in the block are aggregated, so ``get_params``, ``set_params``, ``get_gradients``, and ``set_gradients`` operate on the block as a whole.

Shape:
------

* Input: ``(..., batch_size)``.
* Output: ``(..., batch_size)``.

File I/O
--------

Printing a block writes its layers as nested layer cards.
Reading a block reconstructs the layers in order and round-trips nested sequential blocks, preserving their weights.
