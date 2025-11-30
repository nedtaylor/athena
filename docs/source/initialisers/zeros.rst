.. _zeros-initialiser:

Zeros Initialiser
=================

``zeros_init_type``

.. code-block:: fortran

  zeros_init_type()


Initializes all weights to zero.

.. math::

   W = 0

This is commonly used for bias initialization but should not be used for weight initialization as it would prevent neurons from learning different features.

Shape:
------

Initializes weights based on the shape provided during layer setup.
