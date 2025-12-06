.. _sgd-optimiser:

SGD Optimiser
=============

``sgd_optimiser_type``

.. code-block:: fortran

  sgd_optimiser_type(
    learning_rate=0.01,
    momentum=0.0,
    nesterov=.false.,
    num_params=...,
    regulariser=...,
    clip_dict=...,
    lr_decay=...
  )


Stochastic Gradient Descent (SGD) optimiser with optional momentum and Nesterov acceleration.

The update rule without momentum:

.. math::

   \theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)

With momentum:

.. math::

   v_{t+1} &= \mu v_t + \nabla L(\theta_t) \\
   \theta_{t+1} &= \theta_t - \eta v_{t+1}

With Nesterov momentum:

.. math::

   v_{t+1} &= \mu v_t + \nabla L(\theta_t - \eta \mu v_t) \\
   \theta_{t+1} &= \theta_t - \eta v_{t+1}

where :math:`\eta` is the learning rate and :math:`\mu` is the momentum coefficient.

Arguments
---------

* **learning_rate** (`real(real32)`): Step size for parameter updates. Default: ``0.01``.
* **momentum** (`real(real32)`): Momentum factor. Default: ``0.0`` (no momentum).
* **nesterov** (`logical`): Whether to use Nesterov momentum. Default: ``.false.``.
* **num_params** (`integer`): Number of parameters to optimise.
* **regulariser** (`class(base_regulariser_type)`): Regularisation method (e.g., L2 regularisation).
* **clip_dict** (`type(clip_type)`): Gradient clipping configuration.
* **lr_decay** (`class(base_lr_decay_type)`): Learning rate decay schedule.

Notes:
------

SGD is the fundamental optimisation algorithm for neural networks. Adding momentum helps accelerate convergence and reduces oscillation.
