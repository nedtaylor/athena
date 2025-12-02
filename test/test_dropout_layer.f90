program test_dropout_layer
  use coreutils, only: real32
  use athena, only: &
       dropout_layer_type, &
       base_layer_type
  use diffstruc, only: array_type
  use athena__dropout_layer, only: read_dropout_layer
  implicit none

  class(base_layer_type), allocatable, target :: drop_layer
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: num_channels = 3, num_inputs = 6
  integer :: unit
  type(array_type) :: input(1,1)
  real, allocatable, dimension(:,:) :: output_data
  real, allocatable, dimension(:,:,:) :: input_3d, output_3d, gradient_3d
  real, allocatable, dimension(:) :: output_1d
  real, parameter :: tol = 1.E-7
  logical :: success = .true.
  class(array_type), pointer :: output => null()
  class(array_type), pointer :: gradient => null()

  integer :: i, j, output_width
  real, parameter :: max_value = 3.0

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

!-------------------------------------------------------------------------------
! Initialise random number generator with a seed
!-------------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=0)
  call random_seed(put = seed)


!-------------------------------------------------------------------------------
! set up dropout layer
!-------------------------------------------------------------------------------
  drop_layer = dropout_layer_type( &
       rate = 0.0, &
       num_masks = 1, &
       input_shape = [num_inputs] &
  )

  !! check layer name
  if(.not. drop_layer%name .eq. 'dropout')then
     success = .false.
     write(0,*) 'dropout layer has wrong name'
  end if

  !! check layer type
  select type(drop_layer)
  type is(dropout_layer_type)
     !! check input shape
     if(any(drop_layer%input_shape .ne. [num_inputs]))then
        success = .false.
        write(0,*) 'dropout layer has wrong input_shape'
     end if

     !! check output shape
     if(any(drop_layer%output_shape .ne. [num_inputs]))then
        success = .false.
        write(0,*) 'dropout layer has wrong output shape'
     end if

     if(any(.not.drop_layer%mask))then
        success = .false.
        write(0,*) 'dropout layer has wrong mask, should all be true for &
             &rate = 0.0'
     end if
  class default
     success = .false.
     write(0,*) 'dropout layer has wrong type'
  end select


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  allocate(input_3d(num_inputs, 1, 1), source = 0.0)
  input_3d = max_value

  drop_layer = dropout_layer_type( &
       rate = 0.5, &
       num_masks = 1, &
       input_shape = [num_inputs] &
  )

  call input(1,1)%allocate(array_shape=[num_inputs, 1, 1])
  call input(1,1)%set_requires_grad(.true.)
  call input(1,1)%set(input_3d)

  !! run forward pass
  call drop_layer%forward(input)
  call drop_layer%extract_output(output_3d)


  !! check outputs have expected value
  select type(drop_layer)
  type is(dropout_layer_type)
     if(any(abs(merge(input_3d(:,1,1),0.0,drop_layer%mask(:,1)) / &
          ( 1.E0 - drop_layer%rate ) - output_3d(:,1,1)).gt.tol))then
        success = .false.
        write(0,*) 'dropout layer forward pass failed: mask incorrectly applied'
        write(0,*) merge(input_3d(:,1,1),0.0,drop_layer%mask(:,1)) / &
             ( 1.E0 - drop_layer%rate )
        write(0,*) output_3d(:,1,1)
     end if
  end select
  deallocate(input_3d)

  !! run backward pass
  output => drop_layer%output(1,1)
  allocate(output%grad)
  call output%grad%allocate(array_shape=[output%shape, size(output%val, 2)])
  output%grad%val = output%val
  call output%grad_reverse()
  gradient => input(1,1)%grad
  call gradient%extract(gradient_3d)

  !! check gradient has expected value
  select type(drop_layer)
  type is(dropout_layer_type)
     if(any(abs( &
          merge(output_3d(:,1,1),0.0,drop_layer%mask(:,1)) / &
          (1.E0 - drop_layer%rate) - &
          gradient_3d(:,1,1)).gt.tol))then
        success = .false.
        write(0,*) &
             'dropout layer backward pass failed: mask incorrectly applied'
        write(0,*)  merge(output_3d(:,1,1),0.0,drop_layer%mask(:,1)) / &
             (1.E0 - drop_layer%rate)
        write(0,*)  gradient_3d(:,1,1)
        write(0,*)  output_3d(:,1,1)
     end if
  end select

  deallocate(output_3d, gradient_3d)
  call input(1,1)%deallocate()
  call input(1,1)%reset_graph()

  !! check 1d and 2d output are consistent
  call drop_layer%extract_output(output_1d)
  call drop_layer%extract_output(output_data)
  if(any(abs(output_1d - reshape(output_data, [size(output_data)])) .gt. 1.E-6))then
     success = .false.
     write(0,*) 'output_1d and output_2d are not consistent'
  end if
  deallocate(output_1d, output_data)

!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_dropout_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("DROPOUT")')
  call drop_layer%print_to_unit(unit)
  write(unit,'("END DROPOUT")')
  close(unit)
  write(*,*) "Wrote dropout layer to file"

  ! Read layer from file
  open(newunit=unit, file='test_dropout_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_dropout_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (dropout_layer_type)
     if (.not. read_layer%name .eq. 'dropout') then
        success = .false.
        write(0,*) 'read dropout layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not dropout_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_dropout_layer.tmp', status='old')
  close(unit, status='delete')

!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_dropout_layer passed all tests'
  else
     write(0,*) 'test_dropout_layer failed one or more tests'
     stop 1
  end if

end program test_dropout_layer
