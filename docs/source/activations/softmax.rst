.. _softmax-activation:

Softmax Activation
==================

``softmax_actv_type``

.. code-block:: fortran

  softmax_actv_type(
    scale=1.0,
    attributes=...
  )


The softmax activation function converts a vector of values into a probability distribution.

.. math::

   f(x_i) = s \frac{e^{x_i}}{\sum_j e^{x_j}}

where :math:`s` is a scaling factor (default ``1.0``).
The output values are in the range (0, 1) and sum to 1, making this activation ideal for multi-class classification problems.

Arguments
---------

* **scale** (`real`): Scaling factor for the output. Default: ``1.0``.
* **attributes** (`array`): Optional ONNX attributes.

Shape:
------

* Input: Any shape.
* Output: Same shape as input, with values in range (0, 1) that sum to 1.

Notes:
------

Typically used in the output layer for multi-class classification tasks.
