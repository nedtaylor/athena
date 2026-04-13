program test_dropblock2d_layer
  use coreutils, only: real32
  use athena, only: &
       dropblock2d_layer_type, &
       base_layer_type
  use diffstruc, only: array_type
  use athena__dropblock2d_layer, only: read_dropblock2d_layer
  implicit none

  class(base_layer_type), allocatable, target :: db_layer
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: num_channels = 3, width = 6
  integer :: unit
  type(array_type) :: input(1,1)
  real, allocatable, dimension(:,:,:,:) :: input_4d, output_4d, gradient_4d
  real, allocatable, dimension(:) :: output_1d
  real, allocatable, dimension(:,:) :: output_2d
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
! set up layer
!-------------------------------------------------------------------------------
  db_layer = dropblock2d_layer_type( &
       rate = 0.0, &
       block_size = 5, &
       input_shape = [width, width, num_channels] &
  )

  !! check layer name
  if(.not. db_layer%name .eq. 'dropblock2d')then
     success = .false.
     write(0,*) 'dropblock2d layer has wrong name'
  end if

  !! check layer type
  select type(db_layer)
  type is(dropblock2d_layer_type)
     !! check input shape
     if(any(db_layer%input_shape .ne. [width,width,num_channels]))then
        success = .false.
        write(0,*) 'dropblock2d layer has wrong input_shape'
     end if

     !! check output shape
     if(any(db_layer%output_shape .ne. [width,width,num_channels]))then
        success = .false.
        write(0,*) 'dropblock2d layer has wrong output shape'
     end if

     if(any(.not.db_layer%mask))then
        success = .false.
        write(0,*) 'dropblock2d layer has wrong mask, should all be true for &
             &rate = 0.0'
     end if
  class default
     success = .false.
     write(0,*) 'dropblock2d layer has wrong type'
  end select


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  allocate(input_4d(width, width, num_channels, 1), source = max_value)

  db_layer = dropblock2d_layer_type( &
       rate = 0.5, &
       block_size = 5, &
       input_shape = [width, width, num_channels] &
  )

  call input(1,1)%allocate(array_shape=[width, width, num_channels, 1])
  call input(1,1)%set_requires_grad(.true.)
  call input(1,1)%set(input_4d)

  !! run forward pass
  call db_layer%forward(input)
  call db_layer%extract_output(output_4d)

  !! check outputs have expected value
  select type(db_layer)
  type is(dropblock2d_layer_type)
     if(any( &
          abs( &
               merge(input_4d(:,:,1,1),0.0,db_layer%mask) / &
               ( 1.E0 - db_layer%rate ) - &
               output_4d(:,:,1,1) &
          ) .gt. tol) &
     )then
        success = .false.
        write(0,*) &
             'dropblock2d layer forward pass failed: mask incorrectly applied'
     end if
  end select


!-------------------------------------------------------------------------------
! test backward pass and check expected output
!-------------------------------------------------------------------------------
  !! run backward pass
  output => db_layer%output(1,1)
  allocate(output%grad)
  call output%grad%allocate(array_shape=[output%shape, size(output%val, 2)])
  output%grad%val = output%val
  call output%grad_reverse()
  gradient => input(1,1)%grad
  call gradient%extract(gradient_4d)

  !! check gradient has expected value
  select type(db_layer)
  type is(dropblock2d_layer_type)
     if(any( &
          abs( &
               merge(output_4d(:,:,1,1),0.0,db_layer%mask) / &
               (1.E0 - db_layer%rate) - &
               gradient_4d(:,:,1,1) &
          ) .gt. tol ) &
     )then
        success = .false.
        write(0,*) 'dropblock2d layer backward pass failed: mask &
             &incorrectly applied'
        write(0,*)  merge(output_4d(:,:,1,1),0.0,db_layer%mask) / &
             (1.E0 - db_layer%rate)
        write(0,*)  gradient_4d(:,:,1,1)
     end if
     deallocate(input_4d)
  end select

  deallocate(output_4d, gradient_4d)
  call input(1,1)%deallocate()
  call input(1,1)%reset_graph()


!-------------------------------------------------------------------------------
! check output request using rank 1 and rank 2 arrays is consistent
!-------------------------------------------------------------------------------
  call db_layer%extract_output(output_1d)
  call db_layer%extract_output(output_2d)
  if(any(abs(output_1d - reshape(output_2d, [size(output_2d)])) .gt. 1.E-6))then
     success = .false.
     write(0,*) 'output_1d and output_2d are not consistent'
  end if
  deallocate(output_1d, output_2d)


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_dropblock2d_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("DROPBLOCK2D")')
  call db_layer%print_to_unit(unit)
  write(unit,'("END DROPBLOCK2D")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_dropblock2d_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_dropblock2d_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (dropblock2d_layer_type)
     if(.not. read_layer%name .eq. 'dropblock2d')then
        success = .false.
        write(0,*) 'read dropblock2d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not dropblock2d_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_dropblock2d_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_dropblock2d_layer passed all tests'
  else
     write(0,*) 'test_dropblock2d_layer failed one or more tests'
     stop 1
  end if

end program test_dropblock2d_layer
