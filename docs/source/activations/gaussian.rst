.. _gaussian-activation:

Gaussian Activation
===================

``gaussian_actv_type``

.. code-block:: fortran

  gaussian_actv_type(
    scale=1.0,
    sigma=1.5,
    mu=0.0,
    attributes=...
  )


The Gaussian activation function applies a Gaussian (bell curve) transformation to the input.

.. math::

   f(x) = s \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

where :math:`\sigma` is the standard deviation, :math:`\mu` is the mean, and :math:`s` is a scaling factor.
This activation is useful for radial basis function networks and certain specialized architectures.

Arguments
---------

* **scale** (`real`): Scaling factor for the output. Default: ``1.0``.
* **sigma** (`real`): Standard deviation of the Gaussian function. Default: ``1.5``.
* **mu** (`real`): Mean of the Gaussian function. Default: ``0.0``.
* **attributes** (`array`): Optional ONNX attributes.

Shape:
------

* Input: Any shape.
* Output: Same shape as input.
