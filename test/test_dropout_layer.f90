program test_dropout_layer
  use athena, only: &
       dropout_layer_type, &
       base_layer_type
  implicit none

  class(base_layer_type), allocatable :: drop_layer
  integer, parameter :: num_channels = 3, num_inputs = 6
  real, allocatable, dimension(:,:) :: input_data, output, gradient
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

  !! set up dropout layer
  drop_layer = dropout_layer_type( &
       rate = 0.0, &
       num_masks = 1, &
       input_shape = [num_inputs], &
       batch_size = 1 &
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
        write(0,*) 'dropout layer has wrong output_shape'
     end if

     !! check batch size
     if(drop_layer%batch_size .ne. 1)then
        success = .false.
        write(0,*) 'dropout layer has wrong batch size'
     end if

     if(any(.not.drop_layer%mask))then
        success = .false.
        write(0,*) 'dropout layer has wrong mask, should all be true for rate = 0.0'
     end if
  class default
     success = .false.
     write(0,*) 'dropout layer has wrong type'
  end select

  !! initialise sample input
  allocate(input_data(num_inputs, 1), source = 0.0)
  input_data = max_value

  drop_layer = dropout_layer_type( &
      rate = 0.5, &
      num_masks = 1, &
      input_shape = [num_inputs], &
      batch_size = 1 &
      )
  !! run forward pass
  call drop_layer%forward(input_data)
  call drop_layer%get_output(output)


  !! check outputs have expected value
  select type(drop_layer)
  type is(dropout_layer_type)
    if(any(abs(merge(input_data(:,1),0.0,drop_layer%mask(:,1)) / &
         ( 1.E0 - drop_layer%rate ) - output(:,1)).gt.tol))then
      success = .false.
      write(*,*) 'dropout layer forward pass failed: mask incorrectly applied'
    end if
  end select

  !! run backward pass
  allocate(gradient, source = output)
  call drop_layer%backward(input_data, gradient)

  !! check gradient has expected value
  select type(drop_layer)
  type is(dropout_layer_type)
    if(any(abs(merge(gradient(:,1),0.0,drop_layer%mask(:,1)) - &
         drop_layer%di(:,1)).gt.tol))then
      success = .false.
      write(*,*) 'dropout layer backward pass failed: mask incorrectly applied'
    end if
  end select


  !! check for any fails
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_dropout_layer passed all tests'
  else
     write(0,*) 'test_dropout_layer failed one or more tests'
     stop 1
  end if

end program test_dropout_layer