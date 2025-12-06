.. _adagrad-optimiser:

Adagrad Optimiser
=================

``adagrad_optimiser_type``

.. code-block:: fortran

  adagrad_optimiser_type(
    learning_rate=0.01,
    epsilon=1.0e-8,
    num_params=...,
    regulariser=...,
    clip_dict=...,
    lr_decay=...
  )


Adaptive Gradient (Adagrad) optimiser adapts the learning rate for each parameter based on historical gradients.

The update rule:

.. math::

   G_t &= G_{t-1} + [\nabla L(\theta_t)]^2 \\
   \theta_{t+1} &= \theta_t - \eta \frac{\nabla L(\theta_t)}{\sqrt{G_t} + \epsilon}

where :math:`\eta` is the learning rate, :math:`G_t` accumulates squared gradients, and :math:`\epsilon` prevents division by zero.

Arguments
---------

* **learning_rate** (`real`): Step size for parameter updates. Default: ``0.01``.
* **epsilon** (`real`): Small constant for numerical stability. Default: ``1.0e-8``.
* **num_params** (`integer`): Number of parameters to optimise.
* **regulariser** (`class(base_regulariser_type)`): Regularisation method (e.g., L2 regularisation).
* **clip_dict** (`type(clip_type)`): Gradient clipping configuration.
* **lr_decay** (`class(base_lr_decay_type)`): Learning rate decay schedule.

Notes:
------

Adagrad performs larger updates for infrequent parameters and smaller updates for frequent parameters. It works well for sparse data but can cause premature convergence due to aggressive learning rate decay.
