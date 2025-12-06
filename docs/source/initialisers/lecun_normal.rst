.. _lecun-normal-initialiser:

LeCun Normal Initialiser
=========================

``lecun_normal_init_type``

.. code-block:: fortran

  lecun_normal_init_type()


Draws samples from a normal distribution designed for SELU activation.

.. math::

   W \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n_{in}}}\right)

where :math:`n_{in}` is the number of input units.

This initialisation is specifically designed for networks using SELU activations to maintain self-normalizing properties.

Shape:
------

Initialises weights based on the shape provided during layer setup.
