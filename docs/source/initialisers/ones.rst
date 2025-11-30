.. _ones-initialiser:

Ones Initialiser
================

``ones_init_type``

.. code-block:: fortran

  ones_init_type()


Initializes all weights to one.

.. math::

   W = 1

This is rarely used for weight initialization. It may be used for specific layer parameters like batch normalization gamma values.

Shape:
------

Initializes weights based on the shape provided during layer setup.
