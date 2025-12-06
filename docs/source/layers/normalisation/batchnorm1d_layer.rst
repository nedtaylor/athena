.. _batchnorm1d-layer:

1D Batch Normalisation Layer
=============================

``batchnorm1d_layer_type``

.. code-block:: fortran

  batchnorm1d_layer_type(
    num_channels=...,
    num_inputs=...,
    momentum=0.99,
    epsilon=1.0e-5,
    gamma_init_mean=1.0,
    gamma_init_std=0.02,
    beta_init_mean=0.0,
    beta_init_std=0.02,
    kernel_initialiser=...,
    bias_initialiser=...,
    moving_mean_initialiser=...,
    moving_variance_initialiser=...,
    input_shape=...
  )


The ``batchnorm1d_layer_type`` derived type provides a 1D batch normalisation layer.
This layer applies batch normalisation, which normalises the inputs to have mean of 0 and variance of 1, then applies a learned affine transformation.

.. math::

   y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

where :math:`\mu` and :math:`\sigma^2` are the batch mean and variance, and :math:`\gamma` and :math:`\beta` are learned parameters.

Arguments
---------

* **num_channels** (`integer`): Number of channels in the input.
* **num_inputs** (`integer`): Number of input features. Alternative to ``num_channels``.
* **momentum** (`real(real32)`): Momentum for running mean and variance. Default: ``0.99``.
* **epsilon** (`real(real32)`): Small value added to variance for numerical stability. Default: ``1.0e-5``.
* **gamma_init_mean** (`real(real32)`): Mean for gamma (scale) initialisation. Default: ``1.0``.
* **gamma_init_std** (`real(real32)`): Standard deviation for gamma initialisation. Default: ``0.02``.
* **beta_init_mean** (`real(real32)`): Mean for beta (shift) initialisation. Default: ``0.0``.
* **beta_init_std** (`real(real32)`): Standard deviation for beta initialisation. Default: ``0.02``.
* **kernel_initialiser** (`character(*)`): Initialiser for gamma parameters (see :ref:`Initialisers <initialisers>`).
* **bias_initialiser** (`character(*)`): Initialiser for beta parameters (see :ref:`Initialisers <initialisers>`).
* **moving_mean_initialiser** (`character(*)`): Initialiser for running mean (see :ref:`Initialisers <initialisers>`).
* **moving_variance_initialiser** (`character(*)`): Initialiser for running variance (see :ref:`Initialisers <initialisers>`).
* **input_shape** (`integer, dimension(:)`): Shape of the input data.

Shape:
------

* Input: ``(1, num_channels, batch_size)``.
* Output: ``(1, num_channels, batch_size)``.
