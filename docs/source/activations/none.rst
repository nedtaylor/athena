.. _none-activation:

No Activation
=============

``none_actv_type``

.. code-block:: fortran

  none_actv_type()


A placeholder activation that performs no operation on the input. This is equivalent to the linear activation but explicitly indicates no activation function is used.

.. math::

   f(x) = x

Shape:
------

* Input: Any shape.
* Output: Same shape as input.
