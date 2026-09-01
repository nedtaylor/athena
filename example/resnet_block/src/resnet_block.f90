module resnet_block_module
  use athena
  use diffstruc, only: operator(+)
  implicit none

  type, extends(block_layer_type) :: resnet_block
   contains
     procedure :: forward_eval => forward_block
     procedure :: set_hyperparameters => set_hyperparameters_block
  end type resnet_block

  interface resnet_block
     module function resnet_block_constructor() result(block)
       implicit none
       type(resnet_block) :: block
     end function resnet_block_constructor
  end interface

contains

  module function resnet_block_constructor() result(block)
    !! Constructor for the ResNet block
    implicit none
    type(resnet_block) :: block

    allocate(block%layers(6))
    write(*,*) "Constructing ResNet block"
    ! Initialize layers in the ResNet block
    allocate(block%layers(1)%layer, source=conv2d_layer_type( &
         num_filters=64, kernel_size=3, padding="same"))
    allocate(block%layers(2)%layer, source=batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))
    allocate(block%layers(3)%layer, source=actv_layer_type(activation="relu"))
    allocate(block%layers(4)%layer, source=conv2d_layer_type( &
         num_filters=64, kernel_size=3, padding="same"))
    allocate(block%layers(5)%layer, source=batchnorm2d_layer_type( &
         gamma_initialiser="ones", &
         beta_initialiser="zeros", &
         moving_mean_initialiser="zeros", &
         moving_variance_initialiser="ones"))
    allocate(block%layers(6)%layer, source=actv_layer_type(activation="relu"))


    call block%set_hyperparameters()

  end function resnet_block_constructor


  function forward_block(this, input) result(output)
    !! Forward pass for the ResNet block
    implicit none
    class(resnet_block), intent(inout), target :: this
    class(array_type), dimension(:,:), intent(in) :: input
    type(array_type), pointer :: output(:,:)

    type(array_type), pointer :: x(:,:) => null(), x1(:,:) => null(), ptr

    x => this%layers(1)%layer%forward_eval(input)
    x => this%layers(2)%layer%forward_eval(x)
    x => this%layers(3)%layer%forward_eval(x)
    x => this%layers(4)%layer%forward_eval(x)
    x => this%layers(5)%layer%forward_eval(x)

    ! Add skip connection
    allocate(x1(size(input,1), size(input,2)))
    ptr => x(1,1) + input(1,1)
    call x1(1,1)%assign_and_deallocate_source(ptr)
    output => this%layers(6)%layer%forward_eval(x)

  end function forward_block

  subroutine set_hyperparameters_block(this, verbose)
    !! Set hyperparameters for the ResNet block
    implicit none
    class(resnet_block), intent(inout) :: this
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    write(*,*) "Setting hyperparameters for ResNet block"
    this%name = "resnet_block"
    this%type = "blck"
    this%num_layers = 6
    this%input_rank = 3
    this%output_rank = 3

  end subroutine set_hyperparameters_block

end module resnet_block_module
