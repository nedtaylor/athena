.. _dropout-layer:

Dropout Layer
=============

``dropout_layer_type``

.. code-block:: fortran

  dropout_layer_type(
    rate,
    num_masks,
    input_shape=...,
    batch_size=...
  )


The ``dropout_layer_type`` derived type provides a dropout layer for regularisation.
During training, randomly sets a fraction of input units to 0 at each update, which helps prevent overfitting.
The layer is inactive during inference.

Arguments
---------

* **rate** (`real(real32)`): Fraction of the input units to drop. Must be between 0 and 1. Required argument.
* **num_masks** (`integer`): Number of unique dropout masks to generate and cycle through. Required argument.
* **input_shape** (`integer, dimension(:)`): Shape of the input data (excluding batch dimension).
* **batch_size** (`integer`): **SOON TO BE DEPRECATED**. Batch size for the layer. Handled automatically during training and inference.

Shape:
------

* Input: ``(input_shape, batch_size)``.
* Output: ``(input_shape, batch_size)``.

Notes:
------

During training, outputs are scaled by :math:`\frac{1}{1-\text{rate}}` to maintain the expected sum of activations.
