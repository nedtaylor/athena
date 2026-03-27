.. _training-config:

Training Configuration
======================

The athena library provides gradient clipping, learning rate decay, and regularisation to improve training stability and generalisation.
All components are configured through the optimiser.

.. toctree::
   :maxdepth: 1
   :caption: Training Components

   train_subroutine
   gradient_clipping
   lr_decay
   regularisation

Basic Usage
-----------

Combine training components when compiling the network:

.. code-block:: fortran

   use athena

   type(network_type) :: network
   type(clip_type) :: clipper
   type(exp_lr_decay_type) :: lr_schedule
   type(l2_regulariser_type) :: regulariser

   ! Configure components
   clipper = clip_type(clip_norm=1.0)
   lr_schedule = exp_lr_decay_type(decay_rate=0.01)
   regulariser = l2_regulariser_type()
   regulariser%l2 = 0.001

   ! Compile with all components
   call network%compile( &
        optimiser_type=adam_optimiser_type( &
             learning_rate=0.001, &
             clip_dict=clipper, &
             lr_decay=lr_schedule, &
             regulariser=regulariser), &
        loss_method="categorical_crossentropy")

See the individual component pages for available options and detailed usage.

See Also
--------

* :ref:`Optimisers <optimisers>`: Available optimisation algorithms
* :ref:`Network Outputs <network-outputs>`: Advanced training techniques

Examples
--------

* ``example/rnn_timeseries``: Gradient clipping with recurrent networks
* ``example/msgpass_euler``: Gradient clipping and learning rate decay with GNNs
* ``example/pinn_burgers``: Clipping for physics-informed networks
