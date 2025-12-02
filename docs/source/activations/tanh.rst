.. _tanh-activation:

Tanh Activation
===============

``tanh_actv_type``

.. code-block:: fortran

  tanh_actv_type(
    scale=1.0,
    attributes=...
  )


The hyperbolic tangent (tanh) activation function squashes values to a range between -1 and 1.

.. math::

   f(x) = s \tanh(x) = s \frac{e^x - e^{-x}}{e^x + e^{-x}}

where :math:`s` is a scaling factor (default ``1.0``).
This activation is zero-centered, which can make optimisation easier compared to sigmoid. It's commonly used in recurrent neural networks.

Arguments
---------

* **scale** (`real`): Scaling factor for the output. Default: ``1.0``.
* **attributes** (`array`): Optional ONNX attributes.

Shape:
------

* Input: Any shape.
* Output: Same shape as input, with values in range (-1, 1).
