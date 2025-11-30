.. _glorot-uniform-initialiser:

Glorot Uniform Initialiser
===========================

``glorot_uniform_init_type``

.. code-block:: fortran

  glorot_uniform_init_type()


Also known as Xavier uniform initialization. Draws samples from a uniform distribution within bounds that depend on the number of input and output units.

.. math::

   W \sim U\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]

where :math:`n_{in}` is the number of input units and :math:`n_{out}` is the number of output units.

This initialisation helps maintain the variance of activations across layers and is well-suited for networks using sigmoid or tanh activations.

Shape:
------

Initialises weights based on the shape provided during layer setup.
