program test_conv2d_network
  use athena, only: &
       network_type, &
       conv2d_layer_type, &
       base_optimiser_type
  implicit none

  type(network_type) :: network

  integer :: i
  integer, parameter :: num_channels = 3
  integer, parameter :: kernel_size = 3
  integer, parameter :: num_filters1 = 4
  integer, parameter :: num_filters2 = 8
  integer, parameter :: width = 7

  real, allocatable, dimension(:,:) :: output_reshaped
  real, allocatable, dimension(:,:,:,:) :: input_data, output, gradients_weight
  real, allocatable, dimension(:) :: gradients, gradients_bias
  logical :: success = .true.


!!!-----------------------------------------------------------------------------
!!! set up network
!!!-----------------------------------------------------------------------------
  call network%add(conv2d_layer_type( &
       input_shape=[width, width, num_channels], &
       num_filters = num_filters1, &
       kernel_size = kernel_size, &
       kernel_initialiser = "ones", &
       activation_function = "linear" &
       ))
  call network%add(conv2d_layer_type( &
       num_filters = num_filters2, &
       kernel_size = kernel_size, &
       kernel_initialiser = "ones", &
       activation_function = "linear" &
       ))
    
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=1.0), &
       loss_method="mse", metrics=["loss"], verbose=1, &
       batch_size=1)

  if(network%num_layers.ne.3)then
    success = .false.
    write(0,*) "conv2d network should have 3 layers"
  end if

  call network%set_batch_size(1)
  allocate(input_data(width, width, num_channels, 1))
  input_data = 0.0

  call network%forward(input_data)
  call network%model(3)%layer%get_output(output)

  if(any(shape(output).ne.[width-4,width-4,num_filters2,1]))then
     success = .false.
     write(0,*) "conv2d network output shape should be [28,28,32]"
  end if


!!!-----------------------------------------------------------------------------
!!! check gradients
!!!-----------------------------------------------------------------------------
  output = 0.E0
  output(:(width-4)/2,:,:,:) = 1.E0
  input_data = 0.E0
  input_data(:(width)/2,:,:,:) = 1.E0
  call network%forward(input_data)
  output_reshaped = reshape(output, [(width-4)**2*num_filters2,1])
  call network%backward(output_reshaped)
  select type(current => network%model(3)%layer)
  type is(conv2d_layer_type)
     gradients = current%get_gradients()
     gradients_weight = &
          reshape(&
          gradients(:kernel_size**2*num_filters1*num_filters2), &
          [kernel_size,kernel_size,num_filters1,num_filters2])
     gradients_bias = &
          gradients(kernel_size**2*num_filters1*num_filters2+1:)
     if(size(gradients).ne.&
          (kernel_size**2*num_filters1 + 1) * num_filters2)then
        success = .false.
        write(0,*) "conv2d network gradients size should be ", &
             ( kernel_size**2 * num_filters1 + 1 ) * num_filters2
     end if
     if(any(abs(gradients).lt.1.E-6))then
        success = .false.
        write(0,*) "conv2d network gradients should not be zero"
     end if
     if(any(abs(gradients_weight(1,:,:,:) - &
          gradients_weight(1,1,1,1)).gt.1.E-6))then
        success = .false.
        write(0,*) "conv2d network gradients first column should be equivalent"
     end if
     if(any(abs(gradients_weight(2,:,:,:) - &
          gradients_weight(2,1,1,1)).gt.1.E-6))then
        success = .false.
        write(0,*) "conv2d network gradients second column should be equivalent"
     end if
     if(any(abs(gradients_weight(3,:,:,:) - &
          gradients_weight(3,1,1,1)).gt.1.E-6))then
        success = .false.
        write(0,*) "conv2d network gradients third column should be equivalent"
     end if
     if(any(abs(gradients_bias(:)-gradients_bias(1)).gt.1.E-6))then
        success = .false.
        write(0,*) "conv2d network gradients bias should be equivalent"
     end if
  class default
     success = .false.
     write(0,*) "conv2d network layer should be conv2d_layer_type"
  end select

!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_conv2d_network passed all tests'
  else
     write(0,*) 'test_conv2d_network failed one or more tests'
     stop 1
  end if

end program test_conv2d_network