.. _selu-activation:

SELU Activation
===============

``selu_actv_type``

.. code-block:: fortran

  selu_actv_type(
    scale=1.0,
    alpha=1.67326,
    lambda=1.0507,
    attributes=...
  )


The Scaled Exponential Linear Unit (SELU) activation function is a self-normalizing activation.

.. math::

   f(x) = \lambda \begin{cases}
   s x & \text{if } x > 0 \\
   s \alpha (e^x - 1) & \text{if } x \leq 0
   \end{cases}

where :math:`s` is a scaling factor, and the default values :math:`\alpha \approx 1.67326` and :math:`\lambda \approx 1.0507` are derived to enable self-normalization.
The values of :math:`\alpha` and :math:`\lambda` have been taken directly from the PyTorch implementation to ensure consistency.

Arguments
---------

* **scale** (`real(real32)`): Scaling factor for the output. Default: ``1.0``.
* **alpha** (`real(real32)`): Scale for the exponential function. Default: ``1.67326``.
* **lambda** (`real(real32)`): Scale for the output. Default: ``1.0507``.
* **attributes** (`array`): Optional ONNX attributes.

Shape:
------

* Input: Any shape.
* Output: Same shape as input.

Notes:
------

SELU enables self-normalizing properties in neural networks. For best results, use with ``lecun_normal`` weight initialization and ensure inputs are normalized.
