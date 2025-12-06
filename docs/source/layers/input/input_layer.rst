.. _input-layer:

Input Layer
===========

``input_layer_type``

.. code-block:: fortran

  input_layer_type(
    input_shape=...,
    index=...,
    use_graph_input=.false.
  )


The ``input_layer_type`` derived type provides an input layer that defines the shape of data entering the network.
This layer doesn't perform any computation and is typically used as the first layer in a network.

Arguments
---------

* **input_shape** (`integer, dimension(:)`): Shape of the input data (excluding batch dimension).
* **index** (`integer`): Index of the layer in the network.
* **use_graph_input** (`logical`): Whether to use graph-structured input data. Default: ``.false.``.

Shape:
------

* Input: ``(input_shape, batch_size)``.
* Output: ``(input_shape, batch_size)``.
