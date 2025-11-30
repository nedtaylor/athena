.. _he-uniform-initialiser:

He Uniform Initialiser
=======================

``he_uniform_init_type``

.. code-block:: fortran

  he_uniform_init_type()


Draws samples from a uniform distribution designed for layers with ReLU activation.

.. math::

   W \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]

where :math:`n_{in}` is the number of input units.

This initialisation is specifically designed for networks using ReLU activations and helps prevent vanishing/exploding gradients.

Shape:
------

Initialises weights based on the shape provided during layer setup.
