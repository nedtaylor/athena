.. _ones-initialiser:

Ones Initialiser
================

``ones_init_type``

.. code-block:: fortran

  ones_init_type()


Initialises all weights to one.

.. math::

   W = 1

This is rarely used for weight initialisation. It may be used for specific layer parameters like batch normalisation gamma values.

Shape:
------

Initialises weights based on the shape provided during layer setup.
