program resnet_example
  !! Residual Network (ResNet) architecture demonstration
  !!
  !! This example demonstrates building and testing ResNet architectures,
  !! which use skip connections to enable training of very deep networks.
  !!
  !! ## Residual Learning
  !!
  !! Instead of learning a direct mapping \( \mathcal{H}(\mathbf{x}) \), ResNets
  !! learn a residual function:
  !! $$\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$$
  !!
  !! The output of a residual block is:
  !! $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
  !!
  !! where \( \mathcal{F}(\mathbf{x}, \{W_i\}) \) represents the layers in the block.
  !!
  !! ## Residual Block Structure
  !!
  !! A typical residual block consists of:
  !! 1. Conv layer → BatchNorm → ReLU
  !! 2. Conv layer → BatchNorm
  !! 3. Add skip connection: output = block_output + input
  !! 4. ReLU activation
  !!
  !! Mathematically:
  !! $$\mathbf{y} = \text{ReLU}\left(\text{BN}(W_2 * \text{ReLU}(\text{BN}(W_1 * \mathbf{x}))) + \mathbf{x}\right)$$
  !!
  !! ## Advantages
  !!
  !! - **Addresses vanishing gradients**: Skip connections provide direct gradient flow
  !! - **Enables deep networks**: Can train networks with 50+ layers
  !! - **Identity mapping**: Network can learn identity if needed (\( \mathcal{F} = 0 \))
  !! - **Better optimisation**: Easier to optimise than plain deep networks
  !!
  !! ## Reference
  !!
  !! He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
  use athena
  implicit none

  type(network_type) :: net_simple, net_deep

  print *, "========================================="
  print *, "Test 1: Simple ResNet with one block"
  print *, "========================================="
  call test_simple_resnet()

  print *, ""
  print *, "========================================="
  print *, "Test 2: Deeper ResNet (CIFAR-10 style)"
  print *, "========================================="
  call test_cifar10_resnet()

contains

  subroutine test_simple_resnet()
    !! Test simple ResNet with one residual block
    type(network_type) :: net
    type(adam_optimiser_type) :: optimiser
    type(cce_loss_type) :: loss
    integer :: layer_id

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
    layer_id = net%num_layers  ! Save for skip connection
    call net%add(conv2d_layer_type( &
         num_filters=64, kernel_size=3, padding="same"))
    call net%add(batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))
    call net%add(actv_layer_type(activation="relu"))
    call net%add(conv2d_layer_type( &
         num_filters=64, kernel_size=3, padding="same"))
    call net%add(batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))
    ! Add skip connection and activation
    call net%add( &
         actv_layer_type(activation="relu"), &
         input_list = [layer_id, -1], operator = "+" &
    )

    ! Pooling and output
    call net%add(maxpool2d_layer_type(pool_size=2))
    call net%add(flatten_layer_type(input_rank=3))
    call net%add(full_layer_type(num_outputs=num_classes, activation="softmax"))

    ! Compile
    optimiser = adam_optimiser_type(learning_rate=0.001_real32)
    loss = cce_loss_type()
    call net%compile(optimiser=optimiser, loss_method=loss)

    ! Print summary
    call net%print_summary()

    print *, "Simple ResNet test completed successfully!"

  end subroutine test_simple_resnet


  subroutine test_cifar10_resnet()
    !! Test ResNet for CIFAR-10 style images (32x32 RGB)
    type(network_type) :: net
    type(adam_optimiser_type) :: optimiser
    type(cce_loss_type) :: loss
    integer :: i

    ! Data format: [width, height, channels]
    integer, parameter :: width = 32, height = 32, channels = 3
    integer, parameter :: num_classes = 10

    ! Initial conv (no pooling for small images)
    call net%add(input_layer_type(input_shape=[width, height, channels]))
    call net%add(conv2d_layer_type( &
         input_shape=[width, height, channels], &
         num_filters=16, kernel_size=3, padding="same"))
    call net%add(batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))
    call net%add(actv_layer_type(activation="relu"))

    ! Stage 1: 16 filters (3 blocks)
    do i = 1, 3
       call add_residual_block(net, 16)
    end do

    ! Stage 2: 32 filters with stride (1 block with stride, 2 without)
    call add_residual_block_with_projection(net, 32, stride=2)
    do i = 1, 2
       call add_residual_block(net, 32)
    end do

    ! Stage 3: 64 filters with stride (1 block with stride, 2 without)
    call add_residual_block_with_projection(net, 64, stride=2)
    do i = 1, 2
       call add_residual_block(net, 64)
    end do

    ! Global average pooling
    call net%add(avgpool2d_layer_type(pool_size=8))
    call net%add(flatten_layer_type(input_rank=3))
    call net%add(full_layer_type(num_outputs=num_classes, activation="softmax"))

    ! Compile
    optimiser = adam_optimiser_type(learning_rate=0.001_real32)
    loss = cce_loss_type()
    call net%compile(optimiser=optimiser, loss_method=loss)

    ! Print summary
    call net%print_summary()

    print *, "CIFAR-10 style ResNet test completed successfully!"

  end subroutine test_cifar10_resnet


  subroutine add_residual_block(net, num_filters)
    !! Add a basic residual block (identity shortcut)
    type(network_type), intent(inout) :: net
    integer, intent(in) :: num_filters
    integer :: skip_id

    ! Save layer ID for skip connection
    skip_id = net%num_layers

    ! First conv layer in block
    call net%add(conv2d_layer_type( &
         num_filters=num_filters, kernel_size=3, padding="same"))
    call net%add(batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))
    call net%add(actv_layer_type(activation="relu"))

    ! Second conv layer in block
    call net%add(conv2d_layer_type( &
         num_filters=num_filters, kernel_size=3, padding="same"))
    call net%add(batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))

    ! Skip connection with addition and activation
    call net%add(actv_layer_type(activation="relu"), &
         input_list=[skip_id, -1], operator = "+")

  end subroutine add_residual_block


  subroutine add_residual_block_with_projection(net, num_filters, stride)
    !! Add a residual block with projection shortcut for dimension change
    type(network_type), intent(inout) :: net
    integer, intent(in) :: num_filters, stride
    integer :: skip_id, main_path_id

    ! Save layer ID for skip connection
    skip_id = net%num_layers

    ! Main path with strided convolution
    call net%add(conv2d_layer_type( &
         num_filters=num_filters, kernel_size=3, &
         stride=stride, padding="same"))
    call net%add(batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))
    call net%add(actv_layer_type(activation="relu"))

    call net%add(conv2d_layer_type( &
         num_filters=num_filters, kernel_size=3, padding="same"))
    call net%add(batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))
    main_path_id = net%num_layers

    ! Projection shortcut (1x1 conv to match dimensions)
    call net%add( &
         conv2d_layer_type( &
              num_filters=num_filters, kernel_size=1, stride=stride &
         ), &
         input_list=[skip_id])
    call net%add(batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))

    ! Add skip connection and activation
    call net%add(actv_layer_type(activation="relu"), &
         input_list=[-1, main_path_id], operator = "+")

  end subroutine add_residual_block_with_projection

end program resnet_example
