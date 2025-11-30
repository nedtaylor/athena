.. _batchnorm3d-layer:

3D Batch Normalization Layer
=============================

``batchnorm3d_layer_type``

.. code-block:: fortran

  batchnorm3d_layer_type(
    num_channels=...,
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
    input_shape=...,
    batch_size=...
  )


The ``batchnorm3d_layer_type`` derived type provides a 3D batch normalization layer.
This layer applies batch normalization over 5D inputs (channels, depth, height, width, batch), normalizing over the spatial dimensions and batch.

.. math::

   y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

where :math:`\mu` and :math:`\sigma^2` are the batch mean and variance computed per channel, and :math:`\gamma` and :math:`\beta` are learned parameters.

Arguments
---------

* **num_channels** (`integer`): Number of channels in the input.
* **momentum** (`real(real32)`): Momentum for running mean and variance. Default: ``0.99``.
* **epsilon** (`real(real32)`): Small value added to variance for numerical stability. Default: ``1.0e-5``.
* **gamma_init_mean** (`real(real32)`): Mean for gamma (scale) initialization. Default: ``1.0``.
* **gamma_init_std** (`real(real32)`): Standard deviation for gamma initialization. Default: ``0.02``.
* **beta_init_mean** (`real(real32)`): Mean for beta (shift) initialization. Default: ``0.0``.
* **beta_init_std** (`real(real32)`): Standard deviation for beta initialization. Default: ``0.02``.
* **kernel_initialiser** (`character(*)`): Initialiser for gamma parameters (see :ref:`Initialisers <initialisers>`).
* **bias_initialiser** (`character(*)`): Initialiser for beta parameters (see :ref:`Initialisers <initialisers>`).
* **moving_mean_initialiser** (`character(*)`): Initialiser for running mean (see :ref:`Initialisers <initialisers>`).
* **moving_variance_initialiser** (`character(*)`): Initialiser for running variance (see :ref:`Initialisers <initialisers>`).
* **input_shape** (`integer, dimension(:)`): Shape of the input data (channels, depth, height, width).
* **batch_size** (`integer`): **SOON TO BE DEPRECATED**. Batch size for the layer. Handled automatically during training and inference.

Shape:
------

* Input: ``(width, height, depth, num_channels, batch_size)``.
* Output: ``(width, height, depth, num_channels, batch_size)``.
