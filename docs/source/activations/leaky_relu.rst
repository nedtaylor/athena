.. _leaky-relu-activation:

Leaky ReLU Activation
=====================

``leaky_relu_actv_type``

.. code-block:: fortran

  leaky_relu_actv_type(
    scale=1.0,
    alpha=0.01,
    attributes=...
  )


The Leaky ReLU activation function allows a small gradient when the unit is not active.

.. math::

   f(x) = \begin{cases}
   s x & \text{if } x > 0 \\
   s \alpha x & \text{otherwise}
   \end{cases}

where :math:`\alpha` is a small constant (typically 0.01) and :math:`s` is a scaling factor.

Arguments
---------

* **scale** (`real`): Scaling factor for the output. Default: ``1.0``.
* **alpha** (`real(real32)`): Slope for negative values. Default: ``0.01``.
* **attributes** (`array`): Optional ONNX attributes.

Shape:
------

* Input: Any shape.
* Output: Same shape as input.
