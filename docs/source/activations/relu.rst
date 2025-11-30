.. _relu-activation:

ReLU Activation
===============

``relu_actv_type``

.. code-block:: fortran

  relu_actv_type()


The Rectified Linear Unit (ReLU) activation function outputs the input directly if positive, otherwise zero.

.. math::

   f(x) = s \max(0, x)

ReLU is one of the most commonly used activation functions in deep learning due to its simplicity and effectiveness in addressing the vanishing gradient problem.

Arguments
---------

* **scale** (`real`): Scaling factor for the output. Default: ``1.0``.
* **attributes** (`array`): Optional ONNX attributes.

Shape:
------

* Input: Any shape.
* Output: Same shape as input.
