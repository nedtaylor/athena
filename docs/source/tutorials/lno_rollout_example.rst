.. _lno-rollout-example:

LNO Rollout Example
===================

Complete example of autoregressive rollout training with a Laplace Neural Operator (LNO) on a 1D heat-equation surrogate task.

This tutorial walks through the ``example/lno_rollout`` code from the athena repository.

Overview
--------

This example demonstrates:

* Training a two-layer ``dynamic_lno_layer_type`` network
* Autoregressive multi-step rollout loss (not single-step teacher forcing only)
* Shared deterministic dataset generation for reproducible comparisons
* Cross-language parity checks between Fortran and Python implementations

Problem Setup
-------------

The example learns to step forward synthetic heat-equation trajectories on a fixed 1D grid.

.. rubric:: :h3style:`State and Boundary Conditions`

* Grid points: ``n_grid = 48``
* Left boundary: ``bc_left = -1``
* Right boundary: ``bc_right = 1``
* PDE parameters: ``alpha = 1e-2``, ``dt = 8e-4``

The initial profile is built from shared sine-mode coefficients:

.. math::

   u_0(x) = b_l + (b_r-b_l)x + c_1\sin(\pi x) + c_2\sin(2\pi x) + c_3\sin(3\pi x)

Both Fortran and Python read the same coefficient table from ``example/lno_rollout/shared/rollout_coeffs.csv`` and generate trajectories with an implicit finite-difference heat solver.

.. rubric:: :h3style:`Dataset Split`

* Total samples: ``24``
* Train trajectories: ``16``
* Validation trajectories: ``4``
* Held-out benchmark trajectory: ``index 21``

Network Architecture
--------------------

The model is a two-layer dynamic LNO network:

.. code-block:: fortran

   call network%add(dynamic_lno_layer_type( &
        num_inputs=n_grid, num_outputs=n_hidden, num_modes=n_modes, &
        activation='relu'))
   call network%add(dynamic_lno_layer_type( &
        num_outputs=n_grid, num_modes=n_modes, activation='none'))

with defaults:

* ``n_hidden = 32``
* ``n_modes = 16``

.. rubric:: :h3style:`What dynamic_lno_layer_type learns`

Each layer combines:

* Learnable Laplace poles :math:`\mu_k`
* Learnable residues :math:`\beta_k`
* A local bypass linear map :math:`W`
* Optional bias :math:`b`

Conceptually, the layer computes a spectral operator path plus local bypass, then applies activation.

Training Loop
-------------

The training objective is rollout-consistent: each sample is unrolled for multiple steps, and prediction is fed back as the next input.

.. code-block:: fortran

   do step_idx = 1, rollout_train_steps
      tgt(1,1)%val(:,1) = trajectories(:, step_idx, sample_idx)
      call network%forward(inp)
      network%expected_array = tgt
      ptr1 => network%loss_eval(1, 1)
      ptr2 => ptr1%duplicate_graph()
      if(step_idx .eq. 1) then
         loss => ptr2
      else
         loss => loss + ptr2
      end if

      inp(1,1)%val = network%predict(input=inp(1,1)%val(:,1:1))
      call clip_state(inp(1,1)%val(:,1), -4.0_real32, 4.0_real32)
      inp(1,1)%val(1,1) = bc_left
      inp(1,1)%val(n_grid,1) = bc_right
   end do

   loss => loss / real(rollout_train_steps, real32)
   call loss%grad_reverse()
   call network%update()

Key points:

* The loss is averaged across rollout steps before backpropagation.
* State clipping and boundary re-enforcement stabilise long-horizon rollouts.
* Validation is fully autoregressive, matching inference behaviour.

Shared Initialisation and Parity
--------------------------------

To compare Fortran and Python fairly, the example uses a shared deterministic initialiser.

.. rubric:: :h3style:`Initialisation Scheme`

For each dynamic LNO layer:

* Poles are set to :math:`\mu_k = k\pi`
* Residues, bypass weights, and bias are filled from the same LCG sequence
* Bases are rebuilt after pole assignment:

.. code-block:: fortran

   call layer%rebuild_bases()

This keeps parameter ordering and initial values aligned across implementations.

Running the Example
-------------------

From the repository root:

.. code-block:: bash

   fpm run lno_rollout --example

This writes benchmark outputs to:

* ``example/lno_rollout/shared/fortran_benchmark.txt``

Optional Python comparison (from ``example/lno_rollout/python``):

.. code-block:: bash

   python main.py --model lno

Python writes:

* ``example/lno_rollout/shared/python_benchmark.json``
* ``example/lno_rollout/shared/python_final_state.csv``

The Fortran program can optionally load ``python_final_state.csv`` for exact final-state parity checks.

Understanding the Benchmark Metrics
-----------------------------------

After training, the Fortran run reports:

* Relative final-state L2 error (%):

.. math::

   100 \cdot \frac{\|\hat{u}_T-u_T\|_2}{\|u_T\|_2 + 10^{-12}}

* Maximum absolute final-state error:

.. math::

   \max_i |\hat{u}_{T,i}-u_{T,i}|

where :math:`\hat{u}_T` is the predicted final state and :math:`u_T` is the reference PDE final state.

Practical Notes
---------------

* The example uses a low learning rate (``1e-4``) with Adam for stable rollout optimisation.
* Rollout training is more sensitive than one-step prediction; clipping and boundary projection are important.
* If you modify ``n_grid`` or rollout horizon, keep Fortran and Python configs in sync for valid parity comparisons.

See Also
--------

* :ref:`PINN Example <pinn-example>` - Physics-informed losses with custom derivatives
* :ref:`Regression Examples <regression-example>` - Basic low-level training loops
* :ref:`Training Guide <training-model>` - Training workflow and optimisation details
