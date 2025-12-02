.. _rmsprop-optimiser:

RMSprop Optimiser
=================

``rmsprop_optimiser_type``

.. code-block:: fortran

  rmsprop_optimiser_type(
    learning_rate=0.01,
    beta=0.9,
    epsilon=1.0e-8,
    num_params=...,
    regulariser=...,
    clip_dict=...,
    lr_decay=...
  )


Root Mean Square Propagation (RMSprop) optimiser adapts the learning rate for each parameter.

The update rule:

.. math::

   v_t &= \beta v_{t-1} + (1 - \beta) [\nabla L(\theta_t)]^2 \\
   \theta_{t+1} &= \theta_t - \eta \frac{\nabla L(\theta_t)}{\sqrt{v_t} + \epsilon}

where :math:`\eta` is the learning rate, :math:`\beta` is the decay rate, and :math:`\epsilon` prevents division by zero.

Arguments
---------

* **learning_rate** (`real(real32)`): Step size for parameter updates. Default: ``0.01``.
* **beta** (`real(real32)`): Exponential decay rate for moving average. Default: ``0.9``.
* **epsilon** (`real(real32)`): Small constant for numerical stability. Default: ``1.0e-8``.
* **num_params** (`integer`): Number of parameters to optimise.
* **regulariser** (`class(base_regulariser_type)`): Regularisation method (e.g., L2 regularisation).
* **clip_dict** (`type(clip_type)`): Gradient clipping configuration.
* **lr_decay** (`class(base_lr_decay_type)`): Learning rate decay schedule.

Notes:
------

RMSprop is particularly effective for recurrent neural networks and non-stationary problems. It divides the learning rate by a running average of recent gradient magnitudes.
