.. _plp-optimiser:

PLP Optimiser
=============

``plp_optimiser_type``

.. code-block:: fortran

  plp_optimiser_type(
    learning_rate=0.01,
    momentum=0.0,
    nesterov=.false.,
    num_params=...,
    regulariser=...,
    clip_dict=...,
    lr_decay=...
  )


Parameters Linear Prediction (PLP) applies the paper's three-point linear
predictor after every third SGD-style update.

The update rule for one PLP cycle is:

.. math::

   \theta_1 &= \operatorname{SGD}(\theta_0) \\
   \theta_2 &= \operatorname{SGD}(\theta_1) \\
   \theta_3 &= \operatorname{SGD}(\theta_2) \\
   m_{12} &= \frac{\theta_1 + \theta_2}{2} \\
   m_{23} &= \frac{\theta_2 + \theta_3}{2} \\
   s &= m_{23} - m_{12} \\
   \theta_{\mathrm{pred}} &= m_{23} + s

where :math:`\theta_1`, :math:`\theta_2`, and :math:`\theta_3` are the
parameter values after three consecutive SGD-style updates.

Arguments
---------

* **learning_rate** (`real(real32)`): Step size for the underlying SGD updates. Default: ``0.01``.
* **momentum** (`real(real32)`): Momentum factor used by the underlying SGD step. Default: ``0.0``.
* **nesterov** (`logical`): Whether to use Nesterov momentum in the underlying SGD step. Default: ``.false.``.
* **num_params** (`integer`): Number of parameters to optimise.
* **regulariser** (`class(base_regulariser_type)`): Regularisation method.
* **clip_dict** (`type(clip_type)`): Gradient clipping configuration.
* **lr_decay** (`class(base_lr_decay_type)`): Learning rate decay schedule.

Notes:
------

This implementation follows Ying et al., *Enhancing deep neural network training
efficiency and performance through linear prediction* (Scientific Reports,
2024), with the prediction step fixed to ``1`` as described in the paper.
