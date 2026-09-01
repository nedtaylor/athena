program resnet_block_example
  use athena
!   use diffstruc, only: operator(+)
  use resnet_block_module
  implicit none

  type(network_type) :: net_simple, net_deep

  print *, "========================================="
  print *, "Test 1: Simple ResNet with one block"
  print *, "========================================="
  call test_simple_resnet()

contains

  subroutine test_simple_resnet()
    !! Test simple ResNet with one residual block
    type(network_type) :: net
    type(adam_optimiser_type) :: optimiser
    type(cce_loss_type) :: loss
    class(base_layer_type), allocatable :: block_layer
    integer :: layer_id
    type(array_type) :: input_data(1,1)
    type(array_type), pointer :: output_data(:,:)

    ! Image dimensions: 28x28 grayscale images
    ! Data format: [width, height, channels]
    integer, parameter :: width = 28, height = 28, channels = 1
    integer, parameter :: num_classes = 10

    ! Build ResNet architecture
    ! Initial conv layer
    call net%add(conv2d_layer_type( &
         input_shape=[width, height, channels], &
         num_filters=64, kernel_size=3, padding="same"))
    call net%add(batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))
    call net%add(actv_layer_type(activation="relu"))

    ! First residual block (64 filters)
    allocate(block_layer, source=resnet_block_constructor())
    call net%add(block_layer)

    ! Pooling and output
    call net%add(maxpool2d_layer_type(pool_size=2))
    call net%add(flatten_layer_type(input_rank=3))
    call net%add(full_layer_type(num_outputs=num_classes, activation="softmax"))

    ! Compile
    optimiser = adam_optimiser_type(learning_rate=0.001_real32)
    loss = cce_loss_type()
    call net%compile(optimiser=optimiser, loss_method=loss, verbose=3, check_shapes=.false.)

    ! Print summary
    call net%print_summary()

    ! Run a forward pass with dummy input data
    call input_data(1,1)%allocate([width, height, channels, 1])
    output_data => net%forward_eval(input_data)

    ! Update the network (dummy update for testing)
    call net%update()

    print *, "Simple ResNet test completed successfully!"

  end subroutine test_simple_resnet

end program resnet_block_example
