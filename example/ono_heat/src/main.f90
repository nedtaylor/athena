program orthogonal_neural_operator_heat
  !! Orthogonal Neural Operator for the 1D heat equation
  !!
  !! Demonstrates the Orthogonal Neural Operator (ONO) by learning
  !! the solution operator for the 1D heat equation:
  !!
  !! $$\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2}$$
  !!
  !! The ONO learns the mapping from an initial condition \(u_0(x)\)
  !! to the solution at a later time: \( u_0 \mapsto u(t^*, x) \).
  !!
  !! ## Architecture
  !!
  !! - ONO layer: N_grid -> 32 (relu), orthogonal basis k=8
  !! - ONO layer: 32 -> N_grid (none), orthogonal basis k=8
  !!
  !! ## Data generation
  !!
  !! Initial conditions are random superpositions of low-frequency sines:
  !! $$u_0(x) = \sum_{m=1}^{M} a_m \sin(m \pi x)$$
  !!
  !! The exact solution at time \(t^*\) is:
  !! $$u(t^*, x) = \sum_{m=1}^{M} a_m e^{-\nu m^2 \pi^2 t^*} \sin(m\pi x)$$
  !!
  !! ## Reference
  !!
  !! Luo et al., "Improved Operator Learning by Orthogonal Attention",
  !! arXiv:2310.12487v4, 2024.
  use athena
  use coreutils, only: real32
  use constants_mnist, only: pi
  implicit none

  !-----------------------------------------------------------------------------
  ! Problem parameters
  !-----------------------------------------------------------------------------
  integer, parameter :: N_grid = 32       !! spatial grid points
  integer, parameter :: N_modes = 5       !! Fourier modes for IC generation
  integer, parameter :: N_train = 200     !! number of training samples
  integer, parameter :: N_test  = 50      !! number of test samples
  integer, parameter :: N_hidden = 32     !! hidden dimension
  integer, parameter :: k_basis = 8       !! orthogonal basis size
  integer, parameter :: num_epochs = 100  !! training epochs
  integer, parameter :: batch_size = 20   !! batch size
  real(real32), parameter :: nu = 0.01_real32    !! diffusivity
  real(real32), parameter :: t_star = 1.0_real32 !! target time
  real(real32), parameter :: lr = 0.001_real32   !! learning rate

  !-----------------------------------------------------------------------------
  ! Data arrays
  !-----------------------------------------------------------------------------
  real(real32), allocatable :: x_train(:,:), y_train(:,:)
  real(real32), allocatable :: x_test(:,:),  y_test(:,:)
  real(real32), allocatable :: y_pred(:,:)
  real(real32) :: x_grid(N_grid)
  real(real32) :: coeffs(N_modes)

  !-----------------------------------------------------------------------------
  ! Network
  !-----------------------------------------------------------------------------
  type(network_type) :: network
  type(array_type), dimension(1,1) :: inp, tgt
  type(array_type), pointer :: loss

  integer :: i, m, n, epoch
  real(real32) :: mse, dx


  !-----------------------------------------------------------------------------
  ! Set random seed for reproducibility
  !-----------------------------------------------------------------------------
  call random_setup(42, restart=.false.)


  !-----------------------------------------------------------------------------
  ! Generate spatial grid  x in [0, 1]
  !-----------------------------------------------------------------------------
  dx = 1.0_real32 / real(N_grid - 1, real32)
  do i = 1, N_grid
     x_grid(i) = real(i - 1, real32) * dx
  end do


  !-----------------------------------------------------------------------------
  ! Generate training data: u0 -> u(t*)
  !   u0(x)    = sum_m a_m sin(m*pi*x)
  !   u(t*,x)  = sum_m a_m exp(-nu*m^2*pi^2*t*) sin(m*pi*x)
  !-----------------------------------------------------------------------------
  allocate(x_train(N_grid, N_train), y_train(N_grid, N_train))
  allocate(x_test(N_grid, N_test),   y_test(N_grid, N_test))

  ! Training samples
  do n = 1, N_train
     call random_number(coeffs)
     coeffs = 2.0_real32 * coeffs - 1.0_real32   ! in [-1, 1]

     do i = 1, N_grid
        x_train(i, n) = 0.0_real32
        y_train(i, n) = 0.0_real32
        do m = 1, N_modes
           x_train(i, n) = x_train(i, n) + &
                coeffs(m) * sin(real(m, real32) * pi * x_grid(i))
           y_train(i, n) = y_train(i, n) + &
                coeffs(m) * exp(-nu * real(m**2, real32) * pi**2 * t_star) * &
                sin(real(m, real32) * pi * x_grid(i))
        end do
     end do
  end do

  ! Test samples
  do n = 1, N_test
     call random_number(coeffs)
     coeffs = 2.0_real32 * coeffs - 1.0_real32

     do i = 1, N_grid
        x_test(i, n) = 0.0_real32
        y_test(i, n) = 0.0_real32
        do m = 1, N_modes
           x_test(i, n) = x_test(i, n) + &
                coeffs(m) * sin(real(m, real32) * pi * x_grid(i))
           y_test(i, n) = y_test(i, n) + &
                coeffs(m) * exp(-nu * real(m**2, real32) * pi**2 * t_star) * &
                sin(real(m, real32) * pi * x_grid(i))
        end do
     end do
  end do

  write(*,'(A)') "Orthogonal Neural Operator — 1D Heat Equation"
  write(*,'(A)') "=============================================="
  write(*,'(A,I0,A,I0)') "Grid: ", N_grid, ", Basis: ", k_basis
  write(*,'(A,I0,A,I0)') "Train samples: ", N_train, ", Test samples: ", N_test
  write(*,'(A,F6.4,A,F6.4)') "Diffusivity: ", nu, ", Target time: ", t_star
  write(*,*)


  !-----------------------------------------------------------------------------
  ! Build network:  ONO(N_grid -> N_hidden, relu)  ->  ONO(N_hidden -> N_grid)
  !-----------------------------------------------------------------------------
  call network%add(orthogonal_nop_block_type( &
       num_inputs = N_grid, num_outputs = N_hidden, &
       num_basis = k_basis, activation = "relu"))
  call network%add(orthogonal_nop_block_type( &
       num_outputs = N_grid, &
       num_basis = k_basis))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate = lr), &
       loss_method = "mse", &
       metrics = ["loss"], &
       verbose = 1)
  call network%set_batch_size(1)

  write(*,'(A,I0)') "Number of parameters: ", network%get_num_params()
  write(*,*)


  !-----------------------------------------------------------------------------
  ! Print initial orthogonality metric
  !-----------------------------------------------------------------------------
  select type(layer => network%model(1)%layer)
  type is(orthogonal_nop_block_type)
     write(*,'(A,ES10.3)') "Layer 1 orthogonality metric: ", &
          layer%get_orthogonality_metric()
  end select
  select type(layer => network%model(2)%layer)
  type is(orthogonal_nop_block_type)
     write(*,'(A,ES10.3)') "Layer 2 orthogonality metric: ", &
          layer%get_orthogonality_metric()
  end select
  write(*,*)


  !-----------------------------------------------------------------------------
  ! Training loop (manual mini-batch)
  !-----------------------------------------------------------------------------
  call inp(1,1)%allocate([N_grid, 1])
  call tgt(1,1)%allocate([N_grid, 1])

  write(*,'(A)') "Epoch   Train MSE   Test MSE   Orth1       Orth2"
  write(*,'(A)') "-----   ---------   --------   -----       -----"

  do epoch = 1, num_epochs
     ! -- Train on all samples one by one --
     do n = 1, N_train
        inp(1,1)%val(:,1) = x_train(:, n)
        tgt(1,1)%val(:,1) = y_train(:, n)

        call network%set_batch_size(1)
        call network%forward(inp)
        network%expected_array = tgt
        call network%reset_gradients()
        loss => network%loss_eval(1, 1)
        call loss%grad_reverse()
        call network%update()
     end do

     ! -- Evaluate on test set --
     if(mod(epoch, 10) .eq. 0 .or. epoch .eq. 1)then
        mse = 0.0_real32
        do n = 1, N_test
           y_pred = network%predict(input=x_test(:, n:n))
           mse = mse + sum((y_pred(:,1) - y_test(:,n))**2) / real(N_grid, real32)
        end do
        mse = mse / real(N_test, real32)

        ! Print with orthogonality metrics
        select type(l1 => network%model(1)%layer)
        type is(orthogonal_nop_block_type)
           select type(l2 => network%model(2)%layer)
           type is(orthogonal_nop_block_type)
              write(*,'(I5,3X,ES10.3,3X,ES10.3,3X,ES10.3,3X,ES10.3)') &
                   epoch, 0.0_real32, mse, &
                   l1%get_orthogonality_metric(), &
                   l2%get_orthogonality_metric()
           end select
        end select
     end if
  end do


  !-----------------------------------------------------------------------------
  ! Final evaluation
  !-----------------------------------------------------------------------------
  write(*,*)
  write(*,'(A)') "Final test predictions (first sample):"
  y_pred = network%predict(input=x_test(:, 1:1))
  write(*,'(A)') "  x         u_true      u_pred      error"
  write(*,'(A)') "  -------   ---------   ---------   ---------"
  do i = 1, N_grid
     write(*,'(2X,F7.4,3X,F10.6,3X,F10.6,3X,ES10.3)') &
          x_grid(i), y_test(i,1), y_pred(i,1), &
          abs(y_test(i,1) - y_pred(i,1))
  end do

  ! Final orthogonality
  write(*,*)
  select type(layer => network%model(1)%layer)
  type is(orthogonal_nop_block_type)
     write(*,'(A,ES10.3)') "Final layer 1 max(|Q^T Q - I|) = ", &
          layer%get_orthogonality_metric()
  end select
  select type(layer => network%model(2)%layer)
  type is(orthogonal_nop_block_type)
     write(*,'(A,ES10.3)') "Final layer 2 max(|Q^T Q - I|) = ", &
          layer%get_orthogonality_metric()
  end select

  call inp(1,1)%deallocate()
  call tgt(1,1)%deallocate()
  deallocate(x_train, y_train, x_test, y_test)

end program orthogonal_neural_operator_heat
