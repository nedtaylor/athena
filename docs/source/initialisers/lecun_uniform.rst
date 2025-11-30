.. _lecun-uniform-initialiser:

LeCun Uniform Initialiser
==========================

``lecun_uniform_init_type``

.. code-block:: fortran

  lecun_uniform_init_type()


Draws samples from a uniform distribution designed for SELU activation.

.. math::

   W \sim U\left[-\sqrt{\frac{3}{n_{in}}}, \sqrt{\frac{3}{n_{in}}}\right]

where :math:`n_{in}` is the number of input units.

This initialisation is specifically designed for networks using SELU activations to maintain self-normalizing properties.

Shape:
------

Initializes weights based on the shape provided during layer setup.
