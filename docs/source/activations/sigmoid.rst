.. _sigmoid-activation:

Sigmoid Activation
==================

``sigmoid_actv_type``

.. code-block:: fortran

  sigmoid_actv_type(
    scale=1.0,
    attributes=...
  )


The sigmoid activation function squashes values to a range between 0 and 1.

.. math::

   f(x) = s \frac{1}{1 + e^{-x}}

where :math:`s` is a scaling factor (default ``1.0``).
This activation is commonly used in binary classification problems and in the output layer of networks predicting probabilities.

Arguments
---------

* **scale** (`real`): Scaling factor for the output. Default: ``1.0``.
* **attributes** (`array`): Optional ONNX attributes.

Shape:
------

* Input: Any shape.
* Output: Same shape as input, with values in range (0, 1).
