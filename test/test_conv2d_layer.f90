program test_conv2d_layer
  use athena, only: &
       conv2d_layer_type, &
       input3d_layer_type, &
       base_layer_type, &
       learnable_layer_type
  implicit none

  class(base_layer_type), allocatable :: conv_layer, conv_layer1, conv_layer2
  class(base_layer_type), allocatable :: input_layer
  integer, parameter :: num_filters = 32, kernel_size = 3
  real, allocatable, dimension(:,:,:,:) :: input_data, output
  real, parameter :: tol = 1.E-7
  logical :: success = .true.

  real, allocatable, dimension(:) :: params
  real, allocatable, dimension(:,:) :: outputs


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
     if(any(conv_layer%knl .ne. kernel_size))then
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
    write(*,*) 'conv2d layer with zero input and sigmoid activation must &
         &return outputs all equal to 0.5'
    write(*,*) output
  end if


!!!-----------------------------------------------------------------------------
!!! handle layer parameters gradients, and outputs
!!!-----------------------------------------------------------------------------
  select type(conv_layer)
  class is(learnable_layer_type)
     !! check layer parameter handling
     params = conv_layer%get_params()
     if(size(params) .eq. 0)then
        success = .false.
        write(0,*) 'conv2d layer has wrong number of parameters'
     end if
     params = 1.E0
     call conv_layer%set_params(params)
     params = conv_layer%get_params()
     if(any(abs(params - 1.E0).gt.tol))then
        success = .false.
        write(0,*) 'conv2d layer has wrong parameters'
     end if

     !! check layer gradient handling
     params = conv_layer%get_gradients() 
     if(size(params) .eq. 0)then
        success = .false.
        write(0,*) 'conv2d layer has wrong number of gradients'
     end if
     params = 1.E0
     call conv_layer%set_gradients(params)
     params = conv_layer%get_gradients()
     if(any(abs(params - 1.E0).gt.tol))then
        success = .false.
        write(0,*) 'conv2d layer has wrong gradients'
     end if
     call conv_layer%set_gradients(10.E0)
     params = conv_layer%get_gradients()
     if(any(abs(params - 10.E0).gt.tol))then
        success = .false.
        write(0,*) 'conv2d layer has wrong gradients'
     end if

     !! check layer output handling
     call conv_layer%get_output(params)
     if(size(params) .ne. product(conv_layer%output_shape))then
        success = .false.
        write(0,*) 'conv2d layer has wrong number of outputs'
     end if
     call conv_layer%get_output(outputs)
     if(any(shape(outputs) .ne. conv_layer%output_shape))then
        success = .false.
        write(0,*) 'conv2d layer has wrong number of outputs'
     end if
  end select


!!!-----------------------------------------------------------------------------
!!! check layer operations
!!!-----------------------------------------------------------------------------
  conv_layer1 = conv2d_layer_type( &
       num_filters = 1, &
       kernel_size = 3, &
       activation_function = 'sigmoid' &
       )
  call conv_layer1%init(input_layer%input_shape, batch_size=1)
  conv_layer2 = conv2d_layer_type( &
       num_filters = 1, &
       kernel_size = 3, &
       activation_function = 'sigmoid' &
       )
  call conv_layer2%init(input_layer%input_shape, batch_size=1)
  select type(conv_layer1)
  type is(conv2d_layer_type)
     select type(conv_layer2)
     type is(conv2d_layer_type)
        conv_layer = conv_layer1 + conv_layer2
        select type(conv_layer)
        type is(conv2d_layer_type)
           !! check layer addition
           call compare_conv2d_layers(&
                conv_layer, conv_layer1, success, conv_layer2)

           !! check layer reduction
           conv_layer = conv_layer1
           call conv_layer%reduce(conv_layer2)
           call compare_conv2d_layers(&
                conv_layer, conv_layer1, success, conv_layer2)

           !! check layer merge
           conv_layer = conv_layer1
           call conv_layer%merge(conv_layer2)
           call compare_conv2d_layers(&
                conv_layer, conv_layer1, success, conv_layer2)
        class default
            success = .false.
            write(0,*) 'conv2d layer has wrong type'
        end select
     class default
        success = .false.
        write(0,*) 'conv2d layer has wrong type'
     end select
  class default
     success = .false.
     write(0,*) 'conv2d layer has wrong type'
  end select


!!!-----------------------------------------------------------------------------
!!! Check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_conv2d_layer passed all tests'
  else
     write(0,*) 'test_conv2d_layer failed one or more tests'
     stop 1
  end if


contains

  subroutine compare_conv2d_layers(layer1, layer2, success, layer3)
     type(conv2d_layer_type), intent(in) :: layer1, layer2
     logical, intent(inout) :: success
     type(conv2d_layer_type), optional, intent(in) :: layer3

     if(all(layer1%knl .ne. layer2%knl))then
        success = .false.
        write(0,*) 'conv2d layer has wrong kernel_size'
     end if
     if(layer1%num_filters .ne. layer2%num_filters)then
        success = .false.
        write(0,*) 'conv2d layer has wrong num_filters'
     end if
     if(layer1%transfer%name .ne. 'sigmoid')then
        success = .false.
        write(0,*) 'conv2d layer has wrong transfer: '//&
             layer1%transfer%name
     end if
     if(present(layer3))then
        if(any(abs(layer1%dw-layer2%dw-layer3%dw).gt.tol))then
           success = .false.
           write(0,*) 'conv2d layer has wrong weights'
        end if
        if(any(abs(layer1%db-layer2%db-layer3%db).gt.tol))then
           success = .false.
           write(0,*) 'conv2d layer has wrong weights'
        end if
     end if

  end subroutine compare_conv2d_layers

end program test_conv2d_layer