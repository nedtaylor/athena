.. _flatten-layer:

Flatten Layer
=============

``flatten_layer_type``

.. code-block:: fortran

  flatten_layer_type(
    input_shape=...,
    input_rank=...,
    batch_size=...
  )


The ``flatten_layer_type`` derived type provides a layer that flattens the input tensor into a 1D vector (per batch sample).
This is commonly used when transitioning from convolutional layers to fully connected layers.

Arguments
---------

* **input_shape** (`integer, dimension(:)`): Shape of the input data (excluding batch dimension).
* **input_rank** (`integer`): Rank/number of dimensions of the input (excluding batch dimension).
* **batch_size** (`integer`): **SOON TO BE DEPRECATED**. Batch size for the layer. Handled automatically during training and inference.

Shape:
------

* Input: ``(input_shape, batch_size)``.
* Output: ``(product(input_shape), batch_size)``.
