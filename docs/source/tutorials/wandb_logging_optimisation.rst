.. _wandb-logging-optimisation:

wandb Logging and Hyperparameter Optimisation
=============================================

This tutorial shows two ways to use Weights & Biases (wandb) with athena:

1. Built-in logging through ``wandb_network_type``
2. Custom logging and sweeps via the direct ``wandb-fortran`` API in ``athena_wandb``

Prerequisites
-------------

You need ``fpm >= 0.13.0`` and a Python environment with ``wandb`` installed.

.. code-block:: bash

   python -m pip install wandb
   wandb login

In the athena repository, prepare build flags for Python embedding. If using a python virtual environment, activate it first:

.. code-block:: bash

   conda activate myenv  # or source myenv/bin/activate
   source tools/setup_wf_env.sh

All commands in this tutorial assume the ``wandb`` feature is enabled:

.. code-block:: bash

   fpm build --features wandb

Built-in Training Logging (wandb_network_type)
----------------------------------------------

Use ``wandb_network_type`` when you want logging integrated into the standard
``train`` workflow with minimal changes.

The key example is:

* ``example/wandb_network_sine``

Run it with:

.. code-block:: bash

   source tools/setup_wf_env.sh
   fpm run --example wandb_network_sine --features wandb

How it works:

* ``wandb_setup`` initialises the run
* training uses the normal ``train`` path
* loss and accuracy are logged each epoch via an internal post-epoch hook

This is the best option for quick experiment tracking in regular athena workflows.

Custom Logging in Your Own Training Loop
----------------------------------------

Use the direct API from ``athena_wandb`` when you need full control over
logging frequency, metric naming, and what gets tracked.

A minimal custom loop pattern is:

.. code-block:: fortran

   use athena_wandb

   call wandb_init(project="my-project", name="my-run")
   call wandb_config_set("learning_rate", learning_rate)

   do epoch = 1, num_epochs
      ! custom forward/backward/update steps
      call wandb_log("loss", loss_value, step=epoch)
      call wandb_log("accuracy", accuracy_value, step=epoch)
   end do

   call wandb_finish()

Reference examples:

* ``example/wandb_sine`` for manual metric/config logging
* ``example/wandb_pinn_burgers`` for custom training and physics-based metrics

Hyperparameter Optimisation with Sweeps
---------------------------------------

Use wandb sweeps when searching over training hyperparameters.

The key example is:

* ``example/wandb_sweep``

Run it with:

.. code-block:: bash

   source tools/setup_wf_env.sh
   fpm run --example wandb_sweep --features wandb

This example demonstrates:

* creating a sweep config (search space + optimisation target)
* creating the sweep on wandb with ``wandb_sweep``
* launching an agent and iterating runs with sampled parameters
* logging final metrics so wandb can rank trials

Mapping to athena Examples
--------------------------

Pick the example based on your goal:

* ``wandb_network_sine``: easiest built-in logging with ``wandb_network_type``
* ``wandb_sine``: manual logging in a custom loop
* ``wandb_sweep``: hyperparameter optimisation
* ``wandb_pinn_burgers``: advanced physics-informed training + wandb tracking

Further Reading
---------------

For full API coverage (including offline mode and sweep helpers), see
``wandb-fortran`` documentation:

* https://wandb-fortran.readthedocs.io/en/latest/
* https://wandb-fortran.readthedocs.io/en/latest/usage.html
