program test_kan_layer
  use coreutils, only: real32
  use athena, only: &
       kan_layer_type, &
       base_layer_type, &
       network_type, &
       sgd_optimiser_type
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
  kan_layer1 = kan_layer_type(num_inputs=3, num_outputs=5, n_basis=4)

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
     write(0,*) 'KAN layer (deferred) has wrong input_shape'
  end if

  if(any(kan_layer2%output_shape .ne. [10]))then
     success = .false.
     write(0,*) 'KAN layer (deferred) has wrong output_shape'
  end if


!-------------------------------------------------------------------------------
! Test 3: Forward pass produces correct output shape
!-------------------------------------------------------------------------------
  write(*,*) "Test 3: Forward pass output shape"
  forward_shape: block
    type(network_type) :: network
    real(real32), dimension(2,1) :: x_fwd
    real(real32), allocatable, dimension(:,:) :: y_out

    call network%add(kan_layer_type(num_inputs=2, num_outputs=3, n_basis=5))
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
       write(0,*) 'KAN forward output has wrong first dimension:', &
            size(y_out, 1)
    end if
    if(size(y_out, 2) .ne. 1)then
       success = .false.
       write(0,*) 'KAN forward output has wrong second dimension:', &
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
         num_inputs=2, num_outputs=3, n_basis=4))
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

    ! Check that gradients exist on all parameters via the network
    select type(layer => network%model(1)%layer)
    type is(kan_layer_type)
       ! Check weights gradients (params(3))
       if(.not.associated(layer%params(3)%grad))then
          success = .false.
          write(0,*) 'KAN weight gradients not computed'
       end if

       ! Check centres gradients (params(1))
       if(.not.associated(layer%params(1)%grad))then
          success = .false.
          write(0,*) 'KAN centre gradients not computed'
       end if

       ! Check bandwidths gradients (params(2))
       if(.not.associated(layer%params(2)%grad))then
          success = .false.
          write(0,*) 'KAN bandwidth gradients not computed'
       end if

       ! Check bias gradients (params(4))
       if(.not.associated(layer%params(4)%grad))then
          success = .false.
          write(0,*) 'KAN bias gradients not computed'
       end if
    class default
       success = .false.
       write(0,*) 'model(1) is not kan_layer_type'
    end select

    call loss%nullify_graph()
    loss => null()

  end block gradient_check


!-------------------------------------------------------------------------------
! Test 5: Parameters update during optimisation (network training)
!-------------------------------------------------------------------------------
  write(*,*) "Test 5: Parameter update in network"
  training: block
    type(network_type) :: network
    type(array_type), pointer :: loss
    real(real32), dimension(1,1) :: x, y
    real(real32), allocatable :: params_before(:), params_after(:)
    integer :: n

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=5))
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
       allocate(params_before(size(layer%params(3)%val(:,1))))
       params_before = layer%params(3)%val(:,1)
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
       allocate(params_after(size(layer%params(3)%val(:,1))))
       params_after = layer%params(3)%val(:,1)
    class default
       success = .false.
       write(0,*) 'model(1) is not kan_layer_type after training'
    end select

    ! Check that weights changed
    if(all(abs(params_before - params_after) .lt. 1.E-12))then
       success = .false.
       write(0,*) 'KAN weights did not update during training'
    end if

    call loss%nullify_graph()
    loss => null()
    if(allocated(params_before)) deallocate(params_before)
    if(allocated(params_after)) deallocate(params_after)

  end block training


!-------------------------------------------------------------------------------
! Test 6: Learn sin(x) regression
!-------------------------------------------------------------------------------
  write(*,*) "Test 6: Learning sin(x)"
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
         num_inputs=1, num_outputs=1, n_basis=10))
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
       if(n .gt. num_iterations - 10) loss_end = loss_end + loss_val / 10.0_real32

       call loss%grad_reverse()
       call network%update()
       call loss%nullify_graph()
       loss => null()
    end do

    write(*,'(A,F10.6)') '  Start loss (avg first 10):  ', loss_start
    write(*,'(A,F10.6)') '  End loss   (avg last 10):   ', loss_end

    if(loss_end .ge. loss_start)then
       success = .false.
       write(0,*) 'KAN layer failed to decrease loss on sin(x) task'
    end if

    deallocate(seed)

  end block learn_sin


!-------------------------------------------------------------------------------
! Test 7: File I/O
!-------------------------------------------------------------------------------
  write(*,*) "Test 7: File I/O"
  kan_layer1 = kan_layer_type(num_inputs=2, num_outputs=3, n_basis=4)

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
  class default
     success = .false.
     write(0,*) 'read layer is not kan_layer_type'
  end select

  open(newunit=unit, file='test_kan_layer.tmp', status='old')
  close(unit, status='delete')


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
