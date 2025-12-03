.. _adam-optimiser:

Adam Optimiser
==============

``adam_optimiser_type``

.. code-block:: fortran

  adam_optimiser_type(
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1.0e-8,
    num_params=...,
    regulariser=...,
    clip_dict=...,
    lr_decay=...
  )


Adaptive Moment Estimation (Adam) optimiser combines ideas from RMSprop and momentum.

The update rule:

.. math::

   m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t) \\
   v_t &= \beta_2 v_{t-1} + (1 - \beta_2) [\nabla L(\theta_t)]^2 \\
   \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
   \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
   \theta_{t+1} &= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

where :math:`\eta` is the learning rate, :math:`\beta_1` and :math:`\beta_2` are exponential decay rates, and :math:`\epsilon` prevents division by zero.

Arguments
---------

* **learning_rate** (`real`): Step size for parameter updates. Default: ``0.01``.
* **beta1** (`real`): Exponential decay rate for first moment estimates. Default: ``0.9``.
* **beta2** (`real`): Exponential decay rate for second moment estimates. Default: ``0.999``.
* **epsilon** (`real`): Small constant for numerical stability. Default: ``1.0e-8``.
* **num_params** (`integer`): Number of parameters to optimise.
* **regulariser** (`class(base_regulariser_type)`): Regularisation method (e.g., L2 regularisation).
* **clip_dict** (`type(clip_type)`): Gradient clipping configuration.
* **lr_decay** (`class(base_lr_decay_type)`): Learning rate decay schedule.

Notes:
------

Adam is one of the most popular optimisation algorithms for deep learning due to its adaptive learning rates and robustness to hyperparameter choices.
