.. _he-normal-initialiser:

He Normal Initialiser
======================

``he_normal_init_type``

.. code-block:: fortran

  he_normal_init_type()


Draws samples from a normal distribution designed for layers with ReLU activation.

.. math::

   W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)

where :math:`n_{in}` is the number of input units.

This initialisation is specifically designed for networks using ReLU activations and helps prevent vanishing/exploding gradients.

Shape:
------

Initialises weights based on the shape provided during layer setup.
