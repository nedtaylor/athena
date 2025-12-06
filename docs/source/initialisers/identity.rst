.. _identity-initialiser:

Identity Initialiser
====================

``ident_init_type``

.. code-block:: fortran

  ident_init_type()


Initialises weights as the identity matrix.

.. math::

   W_{ij} = \begin{cases}
   1 & \text{if } i = j \\
   0 & \text{otherwise}
   \end{cases}

This is useful for certain architectures like residual connections where you want to preserve the input initially.

Shape:
------

Initialises square weight matrices based on the shape provided during layer setup.
