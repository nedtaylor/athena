program test_conv2d_layer
  use athena, only: &
       conv2d_layer_type, &
       input3d_layer_type, &
       base_layer_type, &
       learnable_layer_type
  implicit none

  class(base_layer_type), allocatable :: conv_layer, input_layer
  integer, parameter :: num_filters = 32, kernel_size = 3
  real, allocatable, dimension(:,:,:,:) :: input_data, output
  real, parameter :: tol = 1.E-7
  logical :: success = .true.


  !! set up conv2d layer
  conv_layer = conv2d_layer_type( &
       num_filters = num_filters, &
       kernel_size = kernel_size &
       )

  !! check layer name
  if(.not. conv_layer%name .eq. 'conv2d')then
     success = .false.
     write(0,*) 'conv2d layer has wrong name'
  end if

  !! check layer type
  select type(conv_layer)
  type is(conv2d_layer_type)
     !! check default layer transfer/activation function
     if(conv_layer%transfer%name .ne. 'none')then
        success = .false.
        write(0,*) 'conv2d layer has wrong transfer: '//conv_layer%transfer%name
     end if

     !! check number of filters
     if(conv_layer%num_filters .ne. num_filters)then
        success = .false.
        write(0,*) 'conv2d layer has wrong num_filters'
     end if

     !! check kernel size
     if(all(conv_layer%knl .ne. kernel_size))then
        success = .false.
        write(0,*) 'conv2d layer has wrong kernel_size'
     end if

     !! check input shape allocated
     if(allocated(conv_layer%input_shape))then
        success = .false.
        write(0,*) 'conv2d layer shape should not be allocated yet'
     end if
  class default
     success = .false.
     write(0,*) 'conv2d layer has wrong type'
  end select


  !! check layer input and output shape based on input layer
  !! conv2d layer: 32 x 32 pixel image, 3 channels
  input_layer = input3d_layer_type([32,32,3])
  call conv_layer%init(input_layer%input_shape)
  select type(conv_layer)
  type is(conv2d_layer_type)
    if(any(conv_layer%input_shape .ne. [32,32,3]))then
       success = .false.
       write(0,*) 'conv2d layer has wrong input_shape'
    end if
    if(any(conv_layer%output_shape .ne. [30,30,num_filters]))then
       success = .false.
       write(0,*) 'conv2d layer has wrong output_shape'
    end if
  end select


  !! initialise sample input
  !! conv2d layer: 3x3 pixel image, 1 channel
  allocate(input_data(3, 3, 1, 1), source = 0.0)
  input_layer = input3d_layer_type([3,3,1], batch_size=1)
  conv_layer = conv2d_layer_type( &
       num_filters = 1, &
       kernel_size = 3, &
       activation_function = 'sigmoid' &
       )
  call conv_layer%init(input_layer%input_shape, batch_size=1)

  !! set input data in input_layer
  select type (current => input_layer)
  type is(input3d_layer_type)
     call current%set(input_data)
  end select
  call input_layer%get_output(output)

  !! run forward pass
  call conv_layer%forward(input_data)
  call conv_layer%get_output(output)

  !! check outputs have expected value
  if (any(abs(output - 0.5).gt.tol)) then
    success = .false.
    write(*,*) 'conv2d layer with zero input and sigmoid activation must return outputs all equal to 0.5'
    write(*,*) output
  end if

  !! check for any fails
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_conv2d_layer passed all tests'
  else
     write(*,*) 'test_conv2d_layer failed one or more tests'
     stop 1
  end if

end program test_conv2d_layer