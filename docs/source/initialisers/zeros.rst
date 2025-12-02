.. _zeros-initialiser:

Zeros Initialiser
=================

``zeros_init_type``

.. code-block:: fortran

  zeros_init_type()


Initialises all weights to zero.

.. math::

   W = 0

This is commonly used for bias initialisation but should not be used for weight initialisation as it would prevent neurons from learning different features.

Shape:
------

Initialises weights based on the shape provided during layer setup.
