.. _linear-activation:

Linear Activation
=================

``linear_actv_type``

.. code-block:: fortran

  linear_actv_type(
    scale=1.0,
    attributes=...
  )


The linear activation function returns the input unchanged (identity function).

.. math::

   f(x) = s x

where :math:`s` is a scaling factor (default is 1.0).

Arguments
---------

* **scale** (`real`): Scaling factor for the output. Default: ``1.0``.
* **attributes** (`array`): Optional ONNX attributes.

Shape:
------

* Input: Any shape.
* Output: Same shape as input.
