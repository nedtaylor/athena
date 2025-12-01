program test_conv2d_network
  use coreutils, only: real32
  use athena, only: &
       network_type, &
       conv2d_layer_type, &
       base_optimiser_type
  use diffstruc, only: array_type
  implicit none

  type(network_type) :: network

  integer :: i
  integer, parameter :: num_channels = 3
  integer, parameter :: kernel_size1 = 3
  integer, parameter :: kernel_size2 = 4
  integer, parameter :: num_filters1 = 4
  integer, parameter :: num_filters2 = 8
  integer, parameter :: width = 8
  integer :: output_width

  real, allocatable, dimension(:,:,:,:) :: data_tmp, gradients_weight
  real, allocatable, dimension(:) :: gradients, gradients_bias
  logical :: success = .true.
  type(array_type) :: input(1,1)
  type(array_type), dimension(:,:), allocatable :: output


  output_width = width - kernel_size1 + 1 - kernel_size2 + 1

!-------------------------------------------------------------------------------
! set up network
!-------------------------------------------------------------------------------
  call network%add(conv2d_layer_type( &
       input_shape=[width, width, num_channels], &
       num_filters = num_filters1, &
       kernel_size = kernel_size1, &
       kernel_initialiser = "ones", &
       activation = "linear" &
  ))
  call network%add(conv2d_layer_type( &
       num_filters = num_filters2, &
       kernel_size = kernel_size2, &
       kernel_initialiser = "ones", &
       activation = "linear" &
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
  call input(1,1)%allocate([width, width, num_channels, 1], source=0._real32)
  call input(1,1)%set_requires_grad(.true.)

  call network%forward_generic2d(input)
  output = network%get_output()
  write(*,*) "c"

  if( any( &
       [output(1,1)%shape,size(output(1,1)%val,2)] .ne. &
       [output_width,output_width,num_filters2,1] &
  ) )then
     success = .false.
     write(0,*) "conv2d network output shape should be [3,3,8]"
     write(0,*) "output shape is ", shape(output)
  end if
  call input(1,1)%deallocate()


!-------------------------------------------------------------------------------
! check gradients
!-------------------------------------------------------------------------------
  allocate(data_tmp(width, width, num_channels, 1))
  data_tmp = 0.E0
  data_tmp(:(width)/2,:,:,:) = 1.E0
  call input(1,1)%allocate([width, width, num_channels, 1], source=0._real32)
  call input(1,1)%set_requires_grad(.true.)
  call input(1,1)%set(data_tmp)
  write(*,*) input(1,1)%val
  call network%forward_generic2d(input)
  write(*,*) "e"
  call network%model(network%leaf_vertices(1))%layer%output(1,1)%grad_reverse()
  select type(current => network%model(network%leaf_vertices(1))%layer)
  type is(conv2d_layer_type)
     gradients = current%get_gradients()
     gradients_weight = &
          reshape(&
               gradients(:kernel_size2**2*num_filters1*num_filters2), &
               [kernel_size2,kernel_size2,num_filters1,num_filters2])
     gradients_bias = &
          gradients(kernel_size2**2*num_filters1*num_filters2+1:)
     if(size(gradients).ne.&
          (kernel_size2**2*num_filters1 + 1) * num_filters2)then
        success = .false.
        write(0,*) "conv2d network gradients size should be ", &
             ( kernel_size2**2 * num_filters1 + 1 ) * num_filters2
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

!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_conv2d_network passed all tests'
  else
     write(0,*) 'test_conv2d_network failed one or more tests'
     stop 1
  end if

end program test_conv2d_network
