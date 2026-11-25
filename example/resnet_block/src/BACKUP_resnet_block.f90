module BACKUP_resnet_block_module
  use athena
!   use diffstruc, only: operator(+)
  implicit none

  type, extends(block_layer_type) :: resnet_block
     type(actv_layer_type) :: activation_layer
     type(batchnorm2d_layer_type) :: batchnorm_layer1, batchnorm_layer2
     type(conv2d_layer_type) :: conv_layer1, conv_layer2
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

    write(*,*) "Constructing ResNet block"
    ! Initialize layers in the ResNet block
    block%conv_layer1 = conv2d_layer_type( &
         input_shape=[1, 1, 1])

      block%batchnorm_layer1 = batchnorm2d_layer_type( &
           gamma_initialiser="ones", &
             beta_initialiser="zeros", &
             moving_mean_initialiser="zeros", &
             moving_variance_initialiser="ones")

        block%activation_layer = actv_layer_type(activation="relu")

          block%conv_layer2 = conv2d_layer_type()

            block%batchnorm_layer2 = batchnorm2d_layer_type( &
                 gamma_initialiser="ones", &
                   beta_initialiser="zeros", &
                   moving_mean_initialiser="zeros", &
                   moving_variance_initialiser="ones")

              call block%set_hyperparameters()

            end function resnet_block_constructor


            function forward_block(this, input) result(output)
              !! Forward pass for the ResNet block
              implicit none
              class(resnet_block), intent(inout), target :: this
              class(array_type), dimension(:,:), intent(in) :: input
              type(array_type), pointer :: output(:,:)

              type(array_type), pointer :: x(:,:) => null()

              x => this%conv_layer1%forward_eval(input)
              x => this%batchnorm_layer1%forward_eval(x)
              x => this%activation_layer%forward_eval(x)
              x => this%conv_layer2%forward_eval(x)
              x => this%batchnorm_layer2%forward_eval(x)

              ! Add skip connection
!     x(1,1) => x(1,1) + input(1,1)
              output => this%activation_layer%forward_eval(x)

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
              this%num_layers = 5
              this%input_rank = 3

            end subroutine set_hyperparameters_block

          end module BACKUP_resnet_block_module
