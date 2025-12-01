.. _pinn-example:

Physics-Informed Neural Networks
=================================

Complete examples demonstrating physics-informed neural networks (PINNs) that incorporate physical laws and constraints into the learning process.

This tutorial covers the ``example/pinn_burgers`` and ``example/pinn_chemical`` examples from the athena repository.

Overview
--------

Physics-informed neural networks combine:

* **Data-driven learning** from observations
* **Physics-based constraints** from governing equations
* **Automatic differentiation** for computing derivatives
* **Custom loss functions** that encode physical laws

These examples demonstrate two approaches:

1. **Burgers equation** (``pinn_burgers``): Solving PDE with custom loss function
2. **Chemical forces** (``pinn_chemical``): Predicting forces via automatic differentiation

Burgers Equation Example
-------------------------

This example is taken from `this online article <https://www.marktechpost.com/2025/03/28/a-step-by-step-guide-to-solve-1d-burgers-equation-with-physics-informed-neural-networks-pinns-a-pytorch-approach-using-automatic-differentiation-and-collocation-methods/>`_
and ported to Fortran using the athena library.
The Python version is available :git:`here <example/pinn_burgers/pytorch_comparison.py>`, which can optionally read initial parameter values and data from athena output files for comparison.
If loading in athena initialised parameters, the outputs should match closely.

Solving PDEs with Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``example/pinn_burgers`` solves the 1D Burgers equation:

.. math::

   \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}

where :math:`u(x,t)` is velocity, :math:`\nu` is viscosity.

Network Architecture
~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   ! Input: [x, t] coordinates
   ! Output: u(x,t) velocity

   call network%add(input_layer_type(input_shape=[2]))

   call network%add(full_layer_type( &
        num_inputs=2, &
        num_outputs=50, &
        activation="tanh", &
        kernel_initialiser="glorot_uniform"))

   call network%add(full_layer_type( &
        num_outputs=50, &
        activation="tanh"))

   call network%add(full_layer_type( &
        num_outputs=50, &
        activation="tanh"))

   call network%add(full_layer_type( &
        num_outputs=50, &
        activation="tanh"))

   call network%add(full_layer_type( &
        num_outputs=1, &
        activation="none"))  ! No activation on output

**Architecture notes:**

* **Tanh activations**: Smooth, differentiable, bounded
* **Multiple hidden layers**: Capacity to approximate complex solutions
* **Linear output**: No activation on final layer for physical quantities

Custom Loss Function
~~~~~~~~~~~~~~~~~~~~

The key to PINNs is encoding the PDE in the loss:

.. code-block:: fortran

    function loss_func(this) result(loss)
      class(network_type), intent(inout), target :: this
      !! Instance of the burgers network type

      type(array_type), pointer :: loss

      type(array_type), pointer :: u_i, u_xx, u_t, u_x

      type(array_type), pointer :: input, u0_pred, f_pred, u_left_pred, u_right_pred, u
      type(array_type), pointer :: loss_f, loss_0, loss_b

      call this%forward(X_f)
      this%model(this%root_vertices(1))%layer%output(1,1)%id = 1
      ! find what the new input loc is now
      u => this%model(this%leaf_vertices(1))%layer%output(1,1)%duplicate_graph()
      input => u%get_ptr_from_id(1)

      ! set direction (t,x) to compute u_t, u_x, u_xx
      call input%set_direction([0._real32, 1._real32])
      call input%set_requires_grad(.true.)
      u_t => u%grad_forward(input)
      call input%set_direction([1._real32, 0._real32])
      u_x => u%grad_forward(input)
      u_xx => u_x%grad_forward(input)

      f_pred => u_t + u * u_x - nu * u_xx
      loss_f => mean( f_pred ** 2._real32, 2 )

      ! boundary conditions
      call this%forward(X_b_left)
      u_left_pred => &
          this%model(this%leaf_vertices(1))%layer%output(1,1)%duplicate_graph()

      call this%forward(X_b_right)
      u_right_pred => &
          this%model(this%leaf_vertices(1))%layer%output(1,1)%duplicate_graph()
      loss_b => &
          mean( u_left_pred ** 2._real32, 2 ) + &
          mean( u_right_pred ** 2._real32, 2 )

      ! zero time condition
      call this%forward(X_0)
      u0_pred => this%model(this%leaf_vertices(1))%layer%output(1,1)
      loss_0 => mean( ( u0_pred - u0 ) ** 2._real32, 2)

      loss => loss_f + loss_0 + loss_b
      loss%is_temporary = .false.

    end function loss_func

**Loss components:**

1. **Physics loss** (``loss_f``): PDE residual should be zero
2. **Data loss** (``loss_u``): Match observed values
3. **Initial conditions** (``loss_ic``): Enforce :math:`u(x,0) = u_0(x)`
4. **Boundary conditions** (``loss_bc``): Enforce boundary values


The loss function computes the PDE residual using automatic differentiation to obtain necessary derivatives.
This relies heavily on the `diffstruc <https://github.com/nedtaylor/diffstruc>`_ library integrated into athena, which requires use of pointers for correct and memory-efficient operation.

Chemical Forces Example
------------------------

Predicting Forces from Energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``example/pinn_chemical`` predicts molecular forces using physics: forces are negative gradients of energy.

.. math::

   \mathbf{F}_i = -\nabla_{R_i} E(\mathbf{R})

where :math:`E` is energy, :math:`\mathbf{R}` are atomic positions, :math:`\mathbf{F}_i` is force on atom :math:`i`.

Network Architecture
~~~~~~~~~~~~~~~~~~~~

Similar to msgpass_chemical.

.. code-block:: fortran

    call network%add(duvenaud_msgpass_layer_type( &
        num_time_steps = num_time_steps, &
        num_vertex_features = [ graphs_in(1,1)%num_vertex_features ], &
        num_edge_features =   [ graphs_in(1,1)%num_edge_features ], &
        num_outputs = num_dense_inputs, &
        kernel_initialiser = 'glorot_normal', &
        readout_activation = 'softmax', &
        min_vertex_degree = 1, &
        max_vertex_degree = 10, &
        batch_size = batch_size &
    ))
    call network%add(full_layer_type( &
        num_inputs  = num_dense_inputs, &
        num_outputs = 128, &
        batch_size  = batch_size, &
        activation = 'leaky_relu', &
        kernel_initialiser = 'he_normal', &
        bias_initialiser = 'ones' &
    ))
    call network%add(full_layer_type( &
        num_outputs = 64, &
        batch_size  = batch_size, &
        activation = 'leaky_relu', &
        kernel_initialiser = 'he_normal', &
        bias_initialiser = 'ones' &
    ))
    call network%add(full_layer_type( &
        num_outputs = num_outputs, &
        batch_size  = batch_size, &
        activation = 'leaky_relu', &
        kernel_initialiser = 'he_normal', &
        bias_initialiser = 'ones' &
    ))

Custom Forces Loss
~~~~~~~~~~~~~~~~~~

This example uses a custom loss function combining energy and force losses.
Whilst the example still performs its own training loop, this is an example of how the ``base_loss_type`` can be extended to create custom loss functions and integrate them into the athena training framework.

This force loss function is merely provided as an example of how to create custom loss functions for physics informed learning;
it is not the optimal way to implement force matching in a neural network.

.. code-block:: fortran

    type, extends(base_loss_type) :: forces_loss_type
      real(real32) :: alpha, beta
      type(network_type), pointer :: network
      type(array_type), dimension(:), allocatable :: expected_forces
    contains
      procedure :: compute => compute_forces
    end type forces_loss_type

Implementation:

.. code-block:: fortran

    function compute_forces( this, predicted, expected ) result(output)
      implicit none
      class(forces_loss_type), intent(in), target :: this
      !! Instance of the loss function type
      type(array_type), dimension(:,:), intent(inout), target :: predicted
      type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
          expected
      !! Predicted and expected values
      type(array_type), pointer :: output

      integer :: s
      integer :: num_atoms
      type(array_type), pointer :: input, forces, forces_loss

      do s = 1, size(predicted, 2)
        input => this%network%model(this%network%root_vertices(1))%layer%output(1,s)
        call input%set_requires_grad(.true.)
        num_atoms = size(input%val, dim=2)
        forces => predicted(1,1)%grad_forward(input)
        if(s.eq.1)then
            forces_loss => sum( forces - this%expected_forces(s), dim=2 ) ** 2 / &
                real(num_atoms, real32)
        else
            forces_loss => forces_loss + &
                sum( forces - this%expected_forces(s), dim=2 ) ** 2 / &
                real(num_atoms, real32)
        end if
      end do
      output => this%alpha * ( predicted(1,1) - expected(1,1) ) ** 2._real32 + &
          this%beta * forces_loss

    end function compute_forces

Key Takeaways
-------------

1. **Physics improves generalisation**: Networks extrapolate better outside training data
2. **Custom losses enable flexibility**: Encode any physical constraint
3. **Automatic differentiation is powerful**: Compute exact derivatives efficiently
4. **Balance is crucial**: Tune weights between data and physics losses

When to Use PINNs
~~~~~~~~~~~~~~~~~

**Good for:**

* Solving PDEs with sparse/noisy data
* Incorporating known physics into learned models
* Inverse problems (inferring parameters from data)
* Multi-physics problems (coupling multiple equations)
* Enforcing physical constraints (conservation laws, symmetries)

**Not ideal for:**

* Purely data-driven problems without known physics
* When physics is unknown or complex to encode
* Very high-dimensional PDEs (>10 dimensions)
* When black-box models are sufficient

**Advantages over traditional methods:**

* Handle irregular domains easily
* Naturally incorporate data
* Mesh-free (no discretisation required)
* Infer parameters from data simultaneously

See Also
--------

* :ref:`Message Passing Example <msgpass-example>` - Graph neural networks
* :ref:`Regression Examples <regression-example>` - Basic training concepts
* :ref:`MNIST Example <mnist-example>` - Standard supervised learning
