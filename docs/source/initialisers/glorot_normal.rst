.. _glorot-normal-initialiser:

Glorot Normal Initialiser
==========================

``glorot_normal_init_type``

.. code-block:: fortran

  glorot_normal_init_type()


Also known as Xavier normal initialization. Draws samples from a normal distribution centered at zero.

.. math::

   W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)

where :math:`n_{in}` is the number of input units and :math:`n_{out}` is the number of output units.

This initialisation helps maintain the variance of activations across layers and is well-suited for networks using sigmoid or tanh activations.

Shape:
------

Initialises weights based on the shape provided during layer setup.
