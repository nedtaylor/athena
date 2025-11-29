program test_activation_layer
  !! Unit tests for the activation layer module
  use coreutils, only: real32, test_error_handling
  use athena__actv_layer, only: actv_layer_type, read_actv_layer
  use athena__base_layer, only: base_layer_type
  use diffstruc, only: array_type, operator(-)
  implicit none

  type(actv_layer_type), target :: actv_layer
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: batch_size = 2
  integer, parameter :: width = 4
  integer, parameter :: height = 3
  integer, parameter :: depth = 2
  integer, parameter :: channels = 3
  logical :: success = .true.
  real(real32), parameter :: tol = 1.0e-6_real32

  ! Test data for different ranks
  type(array_type) :: input(1,1)
  type(array_type), pointer :: output, loss

  integer :: i, j, k, l, m
  integer :: unit

  ! Random seed setup
  integer :: seed_size
  integer, allocatable, dimension(:) :: seed


  character(10), dimension(11) :: activation_functions

  activation_functions= [ &
       "none      ", "linear    ", "relu      ", "leaky_relu", &
       "sigmoid   ", "tanh      ", "swish     ", "softmax   ", &
       "gaussian  ", "piecewise ", "invalid   " ]

!-------------------------------------------------------------------------------
! Initialise random number generator with a seed
!-------------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=42)
  call random_seed(put = seed)


!-------------------------------------------------------------------------------
! Test 1D activation layer setup and properties
!-------------------------------------------------------------------------------
  write(*,*) "Testing 1D activation layer..."

  ! Test with ReLU activation
  actv_layer = actv_layer_type( &
       activation = "relu", &
       input_shape = [width], &
       batch_size = batch_size &
  )

  ! Check layer properties
  if (.not. actv_layer%name .eq. 'actv') then
     success = .false.
     write(0,*) 'activation layer has wrong name'
  end if

  if (.not. actv_layer%type .eq. 'actv') then
     success = .false.
     write(0,*) 'activation layer has wrong type'
  end if

  if (any(actv_layer%input_shape .ne. [width])) then
     success = .false.
     write(0,*) 'activation layer (1D) has wrong input_shape'
  end if

  if (any(actv_layer%output_shape .ne. [width])) then
     success = .false.
     write(0,*) 'activation layer (1D) has wrong output_shape'
  end if

  if (actv_layer%input_rank .ne. 1) then
     success = .false.
     write(0,*) 'activation layer (1D) has wrong input_rank'
  end if

  if (actv_layer%output_rank .ne. 1) then
     success = .false.
     write(0,*) 'activation layer (1D) has wrong output_rank'
  end if

  if (actv_layer%batch_size .ne. batch_size) then
     success = .false.
     write(0,*) 'activation layer (1D) has wrong batch_size'
  end if


!-------------------------------------------------------------------------------
! Test 1D forward pass with ReLU
!-------------------------------------------------------------------------------
  ! Initialize test input with some negative and positive values
  call input(1,1)%allocate(array_shape = [width, batch_size])
  input(1,1)%val(:,1) = [-2.0, -1.0, 1.0, 2.0]
  input(1,1)%val(:,2) = [-1.5, 0.5, 1.5, 2.5]
  call input(1,1)%set_requires_grad(.true.)
  input%is_temporary = .false.

  ! Run forward pass
  call actv_layer%forward(input)
  output => actv_layer%output(1,1)

  ! Check ReLU activation (negative values should be zero)
  if (any(abs(output%val(:,1) - [0.0, 0.0, 1.0, 2.0]) .gt. tol)) then
     success = .false.
     write(0,*) 'activation layer (1D) ReLU forward pass incorrect'
     write(0,*) 'Expected: [0.0, 0.0, 1.0, 2.0]'
     write(0,*) 'Got: ', output%val(:,1)
  end if

  if (any(abs(output%val(:,2) - [0.0, 0.5, 1.5, 2.5]) .gt. tol)) then
     success = .false.
     write(0,*) 'activation layer (1D) ReLU forward pass incorrect for batch 2'
  end if


!-------------------------------------------------------------------------------
! Test 1D backward pass with ReLU
!-------------------------------------------------------------------------------
  ! Initialize gradient
  loss => output - 1._real32

  ! Run backward pass
  call loss%grad_reverse()

  ! Check ReLU derivative (should be 0 for negative inputs, 1 for positive)
  if(associated(input(1,1)%grad))then
     if (any(abs(input(1,1)%grad%val(:,1) - [0.0, 0.0, 1.0, 1.0]) .gt. tol)) then
        success = .false.
        write(0,*) 'activation layer (1D) ReLU backward pass incorrect'
        write(0,*) 'Expected: [0.0, 0.0, 1.0, 1.0]'
        write(0,*) 'Got: ', input(1,1)%grad%val(:,1)
     end if
  else
     success = .false.
     write(0,*) 'activation layer (1D) has not set di type correctly'
  end if
  call actv_layer%output(1,1)%nullify_graph()
  call input(1,1)%deallocate()
  deallocate(loss)


!-------------------------------------------------------------------------------
! Test 2D activation layer with sigmoid
!-------------------------------------------------------------------------------
  write(*,*) "Testing 2D activation layer..."

  actv_layer = actv_layer_type( &
       activation = "sigmoid", &
       input_shape = [width, height], &
       batch_size = batch_size &
  )

  ! Check layer properties
  if (any(actv_layer%input_shape .ne. [width, height])) then
     success = .false.
     write(0,*) 'activation layer (2D) has wrong input_shape'
  end if

  if (actv_layer%input_rank .ne. 2) then
     success = .false.
     write(0,*) 'activation layer (2D) has wrong input_rank'
  end if

  ! Test forward pass
  call input(1,1)%allocate(array_shape = [width, height, batch_size])
  call random_number(input(1,1)%val)
  input(1,1)%val = input(1,1)%val * 4.0_real32 - 2.0_real32  ! Scale to [-2, 2]
  call input(1,1)%set_requires_grad(.true.)
  input%is_temporary = .false.

  call actv_layer%forward(input)
  output => actv_layer%output(1,1)

  ! Check that sigmoid output is in (0,1)
  if (any(output%val .le. 0.0_real32) .or. any(output%val .ge. 1.0_real32)) then
     success = .false.
     write(0,*) 'activation layer (2D) sigmoid output not in correct range'
  end if

  ! Test backward pass
  loss => output - 1._real32
  call loss%grad_reverse()

  ! Check that backward pass produces reasonable values
  if(associated(input(1,1)%grad))then
     if (any(input(1,1)%grad%val .lt. 0.0_real32) .or. &
          any(input(1,1)%grad%val .gt. 0.25_real32) &
     ) then
        success = .false.
        write(0,*) 'activation layer (2D) sigmoid backward pass out of range'
     end if
  else
     success = .false.
     write(0,*) 'activation layer (2D) has not set di type correctly'
  end if
  call actv_layer%output(1,1)%nullify_graph()
  call input(1,1)%deallocate()
  deallocate(loss)


!-------------------------------------------------------------------------------
! Test 3D activation layer with tanh
!-------------------------------------------------------------------------------
  write(*,*) "Testing 3D activation layer..."

  actv_layer = actv_layer_type( &
       activation = "tanh", &
       input_shape = [width, height, depth], &
       batch_size = batch_size &
  )

  ! Test forward pass
  call input(1,1)%allocate(array_shape=[width, height, depth, batch_size])
  call random_number(input(1,1)%val)
  input(1,1)%val = input(1,1)%val * 2.0_real32 - 1.0_real32  ! Scale to [-1, 1]

  call actv_layer%forward(input)
  output => actv_layer%output(1,1)

  ! Check that tanh output is in (-1,1)
  if(any(output%val .le. -1.0_real32) .or. any(output%val .ge. 1.0_real32)) then
     success = .false.
     write(0,*) 'activation layer (3D) tanh output not in correct range'
  end if
  call input(1,1)%deallocate()


!-------------------------------------------------------------------------------
! Test 4D activation layer with linear
!-------------------------------------------------------------------------------
  write(*,*) "Testing 4D activation layer..."

  actv_layer = actv_layer_type( &
       activation = "linear", &
       input_shape = [width, height, depth, channels], &
       batch_size = batch_size &
  )

  ! Test forward pass
  call input(1,1)%allocate(array_shape=[width, height, depth, channels, batch_size])
  call random_number(input(1,1)%val)

  call actv_layer%forward(input)
  output => actv_layer%output(1,1)

  ! Check that linear activation with scale 2.0 doubles the input
  if(any(abs(output%val - 2.0_real32 * input(1,1)%val) .gt. tol)) then
     success = .false.
     write(0,*) 'activation layer (4D) linear activation incorrect'
  end if
  call input(1,1)%deallocate()


!-------------------------------------------------------------------------------
! Test activation layer with 'none' activation
!-------------------------------------------------------------------------------
  write(*,*) "Testing activation layer with 'none' activation..."

  actv_layer = actv_layer_type( &
       activation_function = "none", &
       input_shape = [width], &
       batch_size = batch_size &
  )

  call input(1,1)%allocate(array_shape=[width, batch_size])
  call random_number(input(1,1)%val)

  call actv_layer%forward(input)
  output => actv_layer%output(1,1)

  ! Check that 'none' activation returns input unchanged
  if(any(abs(output%val - input(1,1)%val) .gt. tol)) then
     success = .false.
     write(0,*) 'activation layer none activation incorrect'
  end if
  call input(1,1)%deallocate()


!-------------------------------------------------------------------------------
! Test setting batch size after initialization
!-------------------------------------------------------------------------------
  write(*,*) "Testing batch size modification..."

  actv_layer = actv_layer_type( &
       activation_function = "relu", &
       input_shape = [width] &
  )

  call actv_layer%set_batch_size(batch_size)

  if (actv_layer%batch_size .ne. batch_size) then
     success = .false.
     write(0,*) 'activation layer set_batch_size failed'
  end if


!-------------------------------------------------------------------------------
! Test rank setting
!-------------------------------------------------------------------------------
  write(*,*) "Testing rank setting..."

  call actv_layer%set_rank(3, 3)

  if (actv_layer%input_rank .ne. 3 .or. actv_layer%output_rank .ne. 3) then
     success = .false.
     write(0,*) 'activation layer set_rank failed'
  end if


!-------------------------------------------------------------------------------
! Test hyperparameters setting
!-------------------------------------------------------------------------------
  write(*,*) "Testing hyperparameters setting..."

  call actv_layer%set_hyperparams( &
       activation = "swish", &
       verbose = 0 &
  )

  if (.not. allocated(actv_layer%transfer)) then
     success = .false.
     write(0,*) 'activation layer transfer function not allocated'
  end if

  if (abs(actv_layer%transfer%scale - 1.5_real32) .gt. tol) then
     success = .false.
     write(0,*) 'activation layer transfer scale incorrect'
  end if


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_actv_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("ACTV")')
  call actv_layer%print_to_unit(unit)
  write(unit,'("END ACTV")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_actv_layer.tmp', status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_actv_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (actv_layer_type)
     if (.not. read_layer%name .eq. 'actv') then
        success = .false.
        write(0,*) 'read activation layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not actv_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_actv_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! Test with different activation functions
!-------------------------------------------------------------------------------
  write(*,*) "Testing different activation functions..."

  ! Test each activation function type

  do i = 1, size(activation_functions)
     actv_layer = actv_layer_type( &
          activation_function = trim(activation_functions(i)), &
          input_shape = [width], &
          batch_size = 1 &
     )

     if(.not. allocated(actv_layer%transfer)) then
        success = .false.
        write(0,*) 'activation function not allocated for: ', &
             trim(activation_functions(i))
     end if

     ! Test that we can do forward pass without errors
     call input(1,1)%allocate(array_shape=[width, 1])
     input(1,1)%val(:,1) = [0.1, 0.2, 0.3, 0.4]

     call actv_layer%forward(input)
     output => actv_layer%output(1,1)

     if(.not. allocated(output%val)) then
        success = .false.
        write(0,*) 'output not allocated for activation: ', &
             trim(activation_functions(i))
     end if
     call input(1,1)%deallocate()
  end do


!-------------------------------------------------------------------------------
! Test error conditions
!-------------------------------------------------------------------------------
  write(*,*) "Testing error conditions..."

  test_error_handling = .true.  ! Enable error handling for tests
  ! Test invalid rank setting
  call actv_layer%set_rank(0, 1)
  ! This should trigger a warning but not necessarily fail

  call actv_layer%set_rank(1, 0)
  ! This should trigger a warning but not necessarily fail
  test_error_handling = .false.  ! Enable error handling for tests


!-------------------------------------------------------------------------------
! Check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if (success) then
     write(*,*) 'test_activation_layer passed all tests'
  else
     write(0,*) 'test_activation_layer failed one or more tests'
     stop 1
  end if

end program test_activation_layer
