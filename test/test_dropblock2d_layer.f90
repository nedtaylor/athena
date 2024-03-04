program test_dropblock2d_layer
  use athena, only: &
       dropblock2d_layer_type, &
       base_layer_type
  implicit none

  class(base_layer_type), allocatable :: db_layer
  integer, parameter :: num_channels = 3, width = 6
  real, allocatable, dimension(:,:,:,:) :: input_data, output, gradient
  real, parameter :: tol = 1.E-7
  logical :: success = .true.

  integer :: i, j, output_width
  real, parameter :: max_value = 3.0

  integer :: seed_size = 1
  integer, allocatable, dimension(:) :: seed

  !! Initialize random number generator with a seed
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=0)
  call random_seed(put = seed)

  !! set up dropblock2d layer
  db_layer = dropblock2d_layer_type( &
       rate = 0.0, &
       block_size = 5, &
       input_shape = [width, width, num_channels], &
       batch_size = 1 &
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
        write(0,*) 'dropblock2d layer has wrong output_shape'
     end if

     !! check batch size
     if(db_layer%batch_size .ne. 1)then
        success = .false.
        write(0,*) 'dropblock2d layer has wrong batch size'
     end if

     if(any(.not.db_layer%mask))then
        success = .false.
        write(0,*) 'dropblock2d layer has wrong mask, should all be true for rate = 0.0'
     end if
  class default
     success = .false.
     write(0,*) 'dropblock2d layer has wrong type'
  end select

  !! initialise sample input
  allocate(input_data(width, width, num_channels, 1), source = 0.0)
  input_data = max_value

  db_layer = dropblock2d_layer_type( &
      rate = 0.5, &
      block_size = 5, &
      input_shape = [width, width, num_channels], &
      batch_size = 1 &
      )
  !! run forward pass
  call db_layer%forward(input_data)
  call db_layer%get_output(output)


  !! check outputs have expected value
  select type(db_layer)
  type is(dropblock2d_layer_type)
    if(any(abs(merge(input_data(:,:,1,1),0.0,db_layer%mask)-output(:,:,1,1)).gt.tol))then
      success = .false.
      write(*,*) 'dropblock2d layer forward pass failed: mask incorrectly applied'
    end if
  end select

  !! run backward pass
  allocate(gradient, source = output)
  call db_layer%backward(input_data, gradient)

  !! check gradient has expected value
  select type(db_layer)
  type is(dropblock2d_layer_type)
    if(any(abs(merge(gradient(:,:,1,1),0.0,db_layer%mask)-db_layer%di(:,:,1,1)).gt.tol))then
      success = .false.
      write(*,*) 'dropblock2d layer backward pass failed: mask incorrectly applied'
    end if
  end select


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_dropblock2d_layer passed all tests'
  else
     write(0,*) 'test_dropblock2d_layer failed one or more tests'
     stop 1
  end if

end program test_dropblock2d_layer