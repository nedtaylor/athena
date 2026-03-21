program test_kan_layer
  use coreutils, only: real32
  use athena, only: &
       kan_layer_type, &
       base_layer_type, &
       network_type, &
       sgd_optimiser_type, &
       adam_optimiser_type
  use athena__kan_layer, only: read_kan_layer
  use diffstruc, only: array_type
  implicit none

  class(base_layer_type), allocatable :: kan_layer1, kan_layer2
  class(base_layer_type), allocatable :: read_layer
  integer :: unit
  logical :: success = .true.


!-------------------------------------------------------------------------------
! Test 1: Basic construction with num_inputs specified
!-------------------------------------------------------------------------------
  write(*,*) "Test 1: Basic construction"
  kan_layer1 = kan_layer_type( &
       num_inputs=3, num_outputs=5, n_basis=4, spline_degree=3)

  if(.not. kan_layer1%name .eq. 'kan')then
     success = .false.
     write(0,*) 'KAN layer has wrong name: '//kan_layer1%name
  end if

  if(any(kan_layer1%input_shape .ne. [3]))then
     success = .false.
     write(0,*) 'KAN layer has wrong input_shape'
  end if

  if(any(kan_layer1%output_shape .ne. [5]))then
     success = .false.
     write(0,*) 'KAN layer has wrong output_shape'
  end if

  select type(kan_layer1)
  type is(kan_layer_type)
     if(kan_layer1%num_inputs .ne. 3)then
        success = .false.
        write(0,*) 'KAN layer has wrong num_inputs'
     end if
     if(kan_layer1%num_outputs .ne. 5)then
        success = .false.
        write(0,*) 'KAN layer has wrong num_outputs'
     end if
     if(kan_layer1%n_basis .ne. 4)then
        success = .false.
        write(0,*) 'KAN layer has wrong n_basis'
     end if
     if(kan_layer1%spline_degree .ne. 3)then
        success = .false.
        write(0,*) 'KAN layer has wrong spline_degree'
     end if
  class default
     success = .false.
     write(0,*) 'KAN layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 2: Deferred initialisation
!-------------------------------------------------------------------------------
  write(*,*) "Test 2: Deferred initialisation"
  kan_layer2 = kan_layer_type(num_outputs=10, n_basis=6)
  call kan_layer2%init([4])

  if(any(kan_layer2%input_shape .ne. [4]))then
     success = .false.
     write(0,*) 'B-spline KAN layer (deferred) has wrong input_shape'
  end if

  if(any(kan_layer2%output_shape .ne. [10]))then
     success = .false.
     write(0,*) 'B-spline KAN layer (deferred) has wrong output_shape'
  end if


!-------------------------------------------------------------------------------
! Test 3: Forward pass produces correct output shape
!-------------------------------------------------------------------------------
  write(*,*) "Test 3: Forward pass output shape"
  forward_shape: block
    type(network_type) :: network
    real(real32), dimension(2,1) :: x_fwd
    real(real32), allocatable, dimension(:,:) :: y_out

    call network%add(kan_layer_type( &
         num_inputs=2, num_outputs=3, n_basis=5, spline_degree=3))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', &
         metrics=['loss'], &
         batch_size=1, &
         verbose=0 &
    )

    x_fwd(:,1) = [0.5_real32, -0.3_real32]

    ! Check output shape: should be [3, 1]
    y_out = network%predict(input=x_fwd)
    if(size(y_out, 1) .ne. 3)then
       success = .false.
       write(0,*) 'B-spline KAN forward output has wrong first dimension:', &
            size(y_out, 1)
    end if
    if(size(y_out, 2) .ne. 1)then
       success = .false.
       write(0,*) 'B-spline KAN forward output has wrong second dimension:', &
            size(y_out, 2)
    end if

  end block forward_shape


!-------------------------------------------------------------------------------
! Test 4: Gradient propagation through parameters
!-------------------------------------------------------------------------------
  write(*,*) "Test 4: Gradient propagation"
  gradient_check: block
    type(network_type) :: network
    type(array_type), pointer :: loss
    real(real32), dimension(2,1) :: x_grad
    real(real32), dimension(3,1) :: y_grad

    call network%add(kan_layer_type( &
         num_inputs=2, num_outputs=3, n_basis=4, spline_degree=3))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', &
         metrics=['loss'], &
         batch_size=1, &
         verbose=0 &
    )

    x_grad(:,1) = [0.5_real32, -0.3_real32]
    y_grad(:,1) = [0.1_real32, 0.2_real32, 0.3_real32]

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(source=y_grad)

    call network%forward(x_grad)
    loss => network%loss_eval(1, 1)
    call loss%grad_reverse()

    ! Check that gradients exist on weight and bias parameters
    select type(layer => network%model(1)%layer)
    type is(kan_layer_type)
       ! Check weights gradients (params(1))
       if(.not.associated(layer%params(1)%grad))then
          success = .false.
          write(0,*) 'B-spline KAN weight gradients not computed'
       end if

       ! Check bias gradients (params(2))
       if(.not.associated(layer%params(2)%grad))then
          success = .false.
          write(0,*) 'B-spline KAN bias gradients not computed'
       end if
    class default
       success = .false.
       write(0,*) 'model(1) is not kan_layer_type'
    end select

    call loss%nullify_graph()
    loss => null()

  end block gradient_check


!-------------------------------------------------------------------------------
! Test 5: Parameters update during optimisation
!-------------------------------------------------------------------------------
  write(*,*) "Test 5: Parameter update in network"
  training: block
    type(network_type) :: network
    type(array_type), pointer :: loss
    real(real32), dimension(1,1) :: x, y
    real(real32), allocatable :: params_before(:), params_after(:)

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=5, spline_degree=3))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', &
         metrics=['loss'], &
         batch_size=1, &
         verbose=0 &
    )

    x(1,1) = 0.5_real32
    y(1,1) = 0.3_real32

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(source=y)

    ! Get parameters before training
    select type(layer => network%model(1)%layer)
    type is(kan_layer_type)
       allocate(params_before(size(layer%params(1)%val(:,1))))
       params_before = layer%params(1)%val(:,1)
    class default
       success = .false.
       write(0,*) 'model(1) is not kan_layer_type in training test'
    end select

    ! Run one training step
    call network%forward(x)
    loss => network%loss_eval(1, 1)
    call loss%grad_reverse()
    call network%update()

    ! Get parameters after training
    select type(layer => network%model(1)%layer)
    type is(kan_layer_type)
       allocate(params_after(size(layer%params(1)%val(:,1))))
       params_after = layer%params(1)%val(:,1)
    class default
       success = .false.
       write(0,*) 'model(1) is not kan_layer_type after training'
    end select

    ! Check that weights changed
    if(all(abs(params_before - params_after) .lt. 1.E-12))then
       success = .false.
       write(0,*) 'B-spline KAN weights did not update during training'
    end if

    call loss%nullify_graph()
    loss => null()
    if(allocated(params_before)) deallocate(params_before)
    if(allocated(params_after)) deallocate(params_after)

  end block training


!-------------------------------------------------------------------------------
! Test 6: B-spline basis partition of unity
!   B-spline basis functions should sum to 1 within the interior knot range
!-------------------------------------------------------------------------------
  write(*,*) "Test 6: B-spline basis partition of unity"
  partition_unity: block
    type(kan_layer_type) :: layer
    real(real32), allocatable :: x_exp(:,:), bvals(:,:)
    real(real32) :: bsum
    integer :: i, K, p
    integer, parameter :: n_test = 20
    real(real32) :: x_val

    K = 8
    p = 3
    layer = kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=K, spline_degree=p)

    allocate(x_exp(K, 1))
    allocate(bvals(K, 1))

    do i = 1, n_test
       ! Sample points inside [-1, 1]
       x_val = -1.0_real32 + 2.0_real32 * real(i - 1, real32) / &
            real(n_test - 1, real32)
       x_exp(:, 1) = x_val

       call layer%evaluate_bspline_basis(x_exp, 1, bvals)
       bsum = sum(bvals(:, 1))

       if(abs(bsum - 1.0_real32) .gt. 1.0E-5_real32)then
          success = .false.
          write(0,'(A,F8.4,A,F10.6)') &
               '  B-spline basis sum /= 1 at x=', x_val, &
               ', sum=', bsum
       end if
    end do

    deallocate(x_exp, bvals)

  end block partition_unity


!-------------------------------------------------------------------------------
! Test 7: Learn sin(x) regression
!-------------------------------------------------------------------------------
  write(*,*) "Test 7: Learning sin(x)"
  learn_sin: block
    type(network_type) :: network
    type(array_type), pointer :: loss
    real(real32), dimension(1,1) :: x, y
    real(real32) :: loss_start, loss_end, loss_val
    integer :: n
    integer, parameter :: num_iterations = 2000
    integer :: seed_size
    integer, allocatable :: seed(:)

    seed_size = 8
    allocate(seed(seed_size))
    seed = 42
    call random_seed(put=seed)

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=10, spline_degree=3))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', &
         metrics=['loss'], &
         batch_size=1, &
         verbose=0 &
    )

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(array_shape=[1,1])

    loss_start = 0.0_real32
    loss_end = 0.0_real32

    do n = 1, num_iterations
       call random_number(x)
       x = x * 6.2832_real32   ! [0, 2*pi]
       y(1,1) = (sin(x(1,1)) + 1.0_real32) / 2.0_real32  ! scale to [0, 1]

       network%expected_array(1,1)%val = y

       call network%set_batch_size(1)
       call network%forward(x)
       loss => network%loss_eval(1, 1)
       loss_val = loss%val(1,1)

       ! Record loss at start and end
       if(n .le. 10) loss_start = loss_start + loss_val / 10.0_real32
       if(n .gt. num_iterations - 10) &
            loss_end = loss_end + loss_val / 10.0_real32

       call loss%grad_reverse()
       call network%update()
       call loss%nullify_graph()
       loss => null()
    end do

    write(*,'(A,F10.6)') '  Start loss (avg first 10):  ', loss_start
    write(*,'(A,F10.6)') '  End loss   (avg last 10):   ', loss_end

    if(loss_end .ge. loss_start)then
       success = .false.
       write(0,*) 'B-spline KAN layer failed to decrease loss on sin(x) task'
    end if

    deallocate(seed)

  end block learn_sin


!-------------------------------------------------------------------------------
! Test 8: File I/O
!-------------------------------------------------------------------------------
  write(*,*) "Test 8: File I/O"
  kan_layer1 = kan_layer_type( &
       num_inputs=2, num_outputs=3, n_basis=4, spline_degree=3)

  open(newunit=unit, file='test_kan_layer.tmp', &
       status='replace', action='write')
  write(unit,'("KAN")')
  call kan_layer1%print_to_unit(unit)
  write(unit,'("END KAN")')
  close(unit)

  open(newunit=unit, file='test_kan_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_kan_layer(unit)
  close(unit)

  select type(read_layer)
  type is (kan_layer_type)
     if (.not. read_layer%name .eq. 'kan') then
        success = .false.
        write(0,*) 'read KAN layer has wrong name'
     end if
     if (read_layer%n_basis .ne. 4) then
        success = .false.
        write(0,*) 'read KAN layer has wrong n_basis'
     end if
     if (read_layer%spline_degree .ne. 3) then
        success = .false.
        write(0,*) 'read KAN layer has wrong spline_degree'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not kan_layer_type'
  end select

  open(newunit=unit, file='test_kan_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! Test 9: SiLU base activation computes correct values
!   silu(x) = x / (1 + exp(-x))
!   silu(0) = 0, silu(large) ≈ large, silu(-large) ≈ 0
!-------------------------------------------------------------------------------
  write(*,*) "Test 9: SiLU activation correctness"
  silu_values: block
    type(network_type) :: network
    real(real32), dimension(1,3) :: x_in
    real(real32), allocatable, dimension(:,:) :: y_out
    real(real32) :: expected_silu
    integer :: i

    ! Build PyKAN-style KAN with known scale_base
    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=4, spline_degree=3, &
         use_base_activation=.true.))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', &
         metrics=['loss'], &
         batch_size=3, &
         verbose=0 &
    )

    ! Set scale_base=1, scale_sp=0, weights=0, bias=0
    ! So output = 1 * silu(x) + 0 * spline(x) + 0
    select type(layer => network%model(1)%layer)
    type is(kan_layer_type)
       layer%params(1)%val(:,1) = 0.0_real32  ! weights=0
       layer%params(2)%val(:,1) = 0.0_real32  ! bias=0
       layer%params(3)%val(:,1) = 1.0_real32  ! scale_base=1
       layer%params(4)%val(:,1) = 0.0_real32  ! scale_sp=0
    end select

    x_in(1,1) = 0.0_real32
    x_in(1,2) = 1.0_real32
    x_in(1,3) = -2.0_real32

    y_out = network%predict(input=x_in)

    ! Check silu(0) = 0
    if(abs(y_out(1,1)) .gt. 1.0E-5)then
       success = .false.
       write(0,*) 'SiLU(0) should be 0, got:', y_out(1,1)
    end if

    ! Check silu(1) = 1/(1+exp(-1)) ≈ 0.7311
    expected_silu = 1.0_real32 / (1.0_real32 + exp(-1.0_real32))
    if(abs(y_out(1,2) - expected_silu) .gt. 1.0E-4)then
       success = .false.
       write(0,*) 'SiLU(1) expected', expected_silu, 'got:', y_out(1,2)
    end if

    ! Check silu(-2) = -2/(1+exp(2)) ≈ -0.2384
    expected_silu = -2.0_real32 / (1.0_real32 + exp(2.0_real32))
    if(abs(y_out(1,3) - expected_silu) .gt. 1.0E-4)then
       success = .false.
       write(0,*) 'SiLU(-2) expected', expected_silu, 'got:', y_out(1,3)
    end if

  end block silu_values


!-------------------------------------------------------------------------------
! Test 10: Base + spline combination correctness
!   With scale_base=0, scale_sp=1, behaviour should match spline-only
!-------------------------------------------------------------------------------
  write(*,*) "Test 10: Base+spline combination (default scale)"
  base_spline_combo: block
    type(network_type) :: net_plain, net_base
    real(real32), dimension(2,1) :: x_test
    real(real32), allocatable, dimension(:,:) :: y_plain, y_base

    ! Plain KAN (no base activation)
    call net_plain%add(kan_layer_type( &
         num_inputs=2, num_outputs=3, n_basis=5, spline_degree=3))
    call net_plain%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], batch_size=1, verbose=0)

    ! PyKAN-style KAN with default scales (base=0, sp=1)
    call net_base%add(kan_layer_type( &
         num_inputs=2, num_outputs=3, n_basis=5, spline_degree=3, &
         use_base_activation=.true.))
    call net_base%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], batch_size=1, verbose=0)

    ! Copy weights from plain to base (so spline part is identical)
    select type(lp => net_plain%model(1)%layer)
    type is(kan_layer_type)
       select type(lb => net_base%model(1)%layer)
       type is(kan_layer_type)
          lb%params(1)%val = lp%params(1)%val  ! copy weights
          lb%params(2)%val = lp%params(2)%val  ! copy bias
          ! scale_base already 0, scale_sp already 1
       end select
    end select

    x_test(:,1) = [0.5_real32, -0.3_real32]
    y_plain = net_plain%predict(input=x_test)
    y_base  = net_base%predict(input=x_test)

    if(any(abs(y_plain - y_base) .gt. 1.0E-5))then
       success = .false.
       write(0,*) 'Base+spline with default scales differs from plain KAN'
       write(0,*) '  Plain:', y_plain(:,1)
       write(0,*) '  Base: ', y_base(:,1)
    end if

  end block base_spline_combo


!-------------------------------------------------------------------------------
! Test 11: Gradients propagate through scale_base and scale_sp
!-------------------------------------------------------------------------------
  write(*,*) "Test 11: Gradient propagation through scale params"
  grad_scale: block
    type(network_type) :: network
    type(array_type), pointer :: loss
    real(real32), dimension(2,1) :: x_grad
    real(real32), dimension(3,1) :: y_grad

    call network%add(kan_layer_type( &
         num_inputs=2, num_outputs=3, n_basis=4, spline_degree=3, &
         use_base_activation=.true.))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', &
         metrics=['loss'], &
         batch_size=1, &
         verbose=0 &
    )

    x_grad(:,1) = [0.5_real32, -0.3_real32]
    y_grad(:,1) = [0.1_real32, 0.2_real32, 0.3_real32]

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(source=y_grad)

    call network%forward(x_grad)
    loss => network%loss_eval(1, 1)
    call loss%grad_reverse()

    select type(layer => network%model(1)%layer)
    type is(kan_layer_type)
       ! Check scale_base gradients (params(3))
       if(.not.associated(layer%params(3)%grad))then
          success = .false.
          write(0,*) 'scale_base gradients not computed'
       end if
       ! Check scale_sp gradients (params(4))
       if(.not.associated(layer%params(4)%grad))then
          success = .false.
          write(0,*) 'scale_sp gradients not computed'
       end if
    class default
       success = .false.
       write(0,*) 'model(1) is not kan_layer_type in scale grad test'
    end select

    call loss%nullify_graph()
    loss => null()

  end block grad_scale


!-------------------------------------------------------------------------------
! Test 12: PyKAN-style KAN learns sin(x)
!-------------------------------------------------------------------------------
  write(*,*) "Test 12: PyKAN-style KAN learning sin(x)"
  learn_sin_base: block
    type(network_type) :: network
    type(array_type), pointer :: loss
    real(real32), dimension(1,1) :: x, y
    real(real32) :: loss_start, loss_end, loss_val
    integer :: n
    integer, parameter :: num_iterations = 2000
    integer :: seed_size_l
    integer, allocatable :: seed_l(:)

    seed_size_l = 8
    allocate(seed_l(seed_size_l))
    seed_l = 42
    call random_seed(put=seed_l)

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=10, spline_degree=3, &
         use_base_activation=.true.))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', &
         metrics=['loss'], &
         batch_size=1, &
         verbose=0 &
    )

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(array_shape=[1,1])

    loss_start = 0.0_real32
    loss_end = 0.0_real32

    do n = 1, num_iterations
       call random_number(x)
       x = x * 6.2832_real32
       y(1,1) = (sin(x(1,1)) + 1.0_real32) / 2.0_real32

       network%expected_array(1,1)%val = y

       call network%set_batch_size(1)
       call network%forward(x)
       loss => network%loss_eval(1, 1)
       loss_val = loss%val(1,1)

       if(n .le. 10) loss_start = loss_start + loss_val / 10.0_real32
       if(n .gt. num_iterations - 10) &
            loss_end = loss_end + loss_val / 10.0_real32

       call loss%grad_reverse()
       call network%update()
       call loss%nullify_graph()
       loss => null()
    end do

    write(*,'(A,F10.6)') '  Start loss (avg first 10):  ', loss_start
    write(*,'(A,F10.6)') '  End loss   (avg last 10):   ', loss_end

    if(loss_end .ge. loss_start)then
       success = .false.
       write(0,*) 'PyKAN-style KAN failed to decrease loss on sin(x) task'
    end if

    deallocate(seed_l)

  end block learn_sin_base


!-------------------------------------------------------------------------------
! Test 13: Full-batch training mode works
!   KAN layers use sample-dependent basis_matrix, tested with manual loop
!-------------------------------------------------------------------------------
  write(*,*) "Test 13: Full-batch training mode"
  fullbatch_train: block
    type(network_type) :: network
    type(array_type), pointer :: loss
    real(real32), dimension(1,50) :: x_tr, y_tr
    real(real32), allocatable, dimension(:,:) :: y_pred
    real(real32) :: mse_before, mse_after
    real(real32), dimension(1,1) :: x_s, y_s
    integer :: i, n, epoch
    integer :: seed_size_l
    integer, allocatable :: seed_l(:)

    seed_size_l = 8
    allocate(seed_l(seed_size_l))
    seed_l = 42
    call random_seed(put=seed_l)

    ! Generate training data
    do i = 1, 50
       x_tr(1,i) = -1.0_real32 + 2.0_real32 * real(i - 1, real32) / 49.0_real32
       y_tr(1,i) = (sin(3.14159_real32 * x_tr(1,i)) + 1.0_real32) / 2.0_real32
    end do

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=8, spline_degree=3))
    call network%compile( &
         optimiser=adam_optimiser_type(learning_rate=0.01), &
         loss_method='mse', accuracy_method='mse', &
         metrics=['loss'], batch_size=1, verbose=0)

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(array_shape=[1,1])

    ! Evaluate before training
    y_pred = network%predict(input=x_tr)
    mse_before = sum((y_pred - y_tr)**2) / 50.0_real32

    ! Train with stochastic steps for multiple epochs
    do epoch = 1, 10
       do n = 1, 50
          x_s(1,1) = x_tr(1,n)
          y_s(1,1) = y_tr(1,n)
          network%expected_array(1,1)%val = y_s
          call network%set_batch_size(1)
          call network%forward(x_s)
          loss => network%loss_eval(1, 1)
          call loss%grad_reverse()
          call network%update()
          call loss%nullify_graph()
          loss => null()
       end do
    end do

    ! Evaluate after training
    y_pred = network%predict(input=x_tr)
    mse_after = sum((y_pred - y_tr)**2) / 50.0_real32

    write(*,'(A,F10.6)') '  MSE before:  ', mse_before
    write(*,'(A,F10.6)') '  MSE after:   ', mse_after

    if(mse_after .ge. mse_before)then
       success = .false.
       write(0,*) 'Training did not reduce loss'
    end if

    deallocate(seed_l)

  end block fullbatch_train


!-------------------------------------------------------------------------------
! Test 14: File I/O with base activation
!-------------------------------------------------------------------------------
  write(*,*) "Test 14: File I/O with base activation"
  io_base: block
    class(base_layer_type), allocatable :: kan_ba, read_ba
    integer :: unit_ba

    kan_ba = kan_layer_type( &
         num_inputs=2, num_outputs=3, n_basis=4, spline_degree=3, &
         use_base_activation=.true.)

    open(newunit=unit_ba, file='test_kan_layer_ba.tmp', &
         status='replace', action='write')
    write(unit_ba,'("KAN")')
    call kan_ba%print_to_unit(unit_ba)
    write(unit_ba,'("END KAN")')
    close(unit_ba)

    open(newunit=unit_ba, file='test_kan_layer_ba.tmp', &
         status='old', action='read')
    read(unit_ba,*) ! Skip first line
    read_ba = read_kan_layer(unit_ba)
    close(unit_ba)

    select type(read_ba)
    type is(kan_layer_type)
       if(.not. read_ba%use_base_activation)then
          success = .false.
          write(0,*) 'read KAN layer lost use_base_activation flag'
       end if
       if(read_ba%n_basis .ne. 4)then
          success = .false.
          write(0,*) 'read KAN layer (base) has wrong n_basis'
       end if
       ! Verify scale_base and scale_sp were read
       if(size(read_ba%params) .lt. 4)then
          success = .false.
          write(0,*) 'read KAN layer (base) missing params 3/4'
       end if
    class default
       success = .false.
       write(0,*) 'read layer (base) is not kan_layer_type'
    end select

    open(newunit=unit_ba, file='test_kan_layer_ba.tmp', status='old')
    close(unit_ba, status='delete')

  end block io_base


!-------------------------------------------------------------------------------
! Test 15: Number of parameters with base activation
!-------------------------------------------------------------------------------
  write(*,*) "Test 15: Parameter count with base activation"
  param_count: block
    type(kan_layer_type) :: layer_plain, layer_base
    integer :: expected_plain, expected_base

    layer_plain = kan_layer_type( &
         num_inputs=3, num_outputs=5, n_basis=4, spline_degree=3)
    layer_base = kan_layer_type( &
         num_inputs=3, num_outputs=5, n_basis=4, spline_degree=3, &
         use_base_activation=.true.)

    ! Plain: weights(5*3*4) + bias(5) = 65
    expected_plain = 5 * 3 * 4 + 5
    if(layer_plain%get_num_params() .ne. expected_plain)then
       success = .false.
       write(0,*) 'Plain KAN wrong param count:', &
            layer_plain%get_num_params(), 'expected:', expected_plain
    end if

    ! Base: weights(5*3*4) + bias(5) + scale_base(5*3) + scale_sp(5) = 85
    expected_base = 5 * 3 * 4 + 5 + 5 * 3 + 5
    if(layer_base%get_num_params() .ne. expected_base)then
       success = .false.
       write(0,*) 'Base KAN wrong param count:', &
            layer_base%get_num_params(), 'expected:', expected_base
    end if

  end block param_count


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_kan_layer passed all tests'
  else
     write(0,*) 'test_kan_layer failed one or more tests'
     stop 1
  end if

end program test_kan_layer
