.. _swish-activation:

Swish Activation
================

``swish_actv_type``

.. code-block:: fortran

  swish_actv_type(
    scale=1.0,
    beta=1.0,
    attributes=...
  )


The Swish activation function is a self-gated activation function.

.. math::

   f(x) = s x \cdot \sigma(\beta x) = s \frac{x}{1 + e^{-\beta x}}

where :math:`s` is a scaling factor (default ``1.0``), :math:`\sigma` is the sigmoid function and :math:`\beta` is a parameter that controls the "steepness" of the activation.
By default, :math:`\beta = 1.0`.
Swish has been shown to work better than ReLU on deeper models across a variety of challenging datasets. It is smooth and non-monotonic.

Arguments
---------

* **scale** (`real`): Scaling factor for the output. Default: ``1.0``.
* **beta** (`real`): Parameter that controls the "steepness" of the activation. Default: ``1.0``.
* **attributes** (`array`): Optional ONNX attributes.

Shape:
------

* Input: Any shape.
* Output: Same shape as input.
