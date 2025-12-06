.. _gaussian-initialiser:

Gaussian Initialiser
====================

``gaussian_init_type``

.. code-block:: fortran

  gaussian_init_type(mean=0.0, std=1.0)


Draws samples from a normal (Gaussian) distribution with specified mean and standard deviation.

.. math::

   W \sim \mathcal{N}(\mu, \sigma^2)

Arguments
---------

* **mean** (`real`): Mean of the distribution. Default: ``0.0``.
* **std** (`real`): Standard deviation of the distribution. Default: ``1.0``.

Shape:
------

Initialises weights based on the shape provided during layer setup.

Notes:
------

Also accessible as ``normal_init_type`` for convenience.
