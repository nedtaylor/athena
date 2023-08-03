!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module full_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  use custom_types, only: activation_type, initialiser_type
  implicit none
  

!!!------------------------------------------------------------------------
!!! fully connected network layer type
!!!------------------------------------------------------------------------
  type, extends(base_layer_type) :: full_layer_type
     integer :: num_inputs
     integer :: num_outputs
     real(real12), allocatable, dimension(:,:) :: weight, weight_incr
     real(real12), allocatable, dimension(:,:) :: dw ! gradient of weight
     real(real12), allocatable, dimension(:) :: output
     real(real12), allocatable, dimension(:) :: di ! gradient of input (i.e. delta)
     !! then include m and v

     class(activation_type), allocatable :: transfer
   contains
     procedure :: forward  => forward_rank
     procedure :: backward => backward_rank
     procedure :: forward_1d
     procedure :: backward_1d
  end type full_layer_type


  interface full_layer_type
     module function layer_setup( &
          num_inputs, num_outputs, &
          activation_function, activation_scale, &
          kernel_initialiser, bias_initialiser) result(layer)
       integer, intent(in) :: num_inputs, num_outputs
       real(real12), optional, intent(in) :: activation_scale
       character(*), optional, intent(in) :: activation_function, &
            kernel_initialiser, bias_initialiser
       type(full_layer_type) :: layer
     end function layer_setup
  end interface full_layer_type


  private
  public :: full_layer_type


contains

!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select rank(input); rank(1)
       call forward_1d(this, input)
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(input); rank(1)
    select rank(gradient); rank(1)
       call backward_1d(this, input, gradient)
    end select
    end select
  end subroutine backward_rank
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  module function layer_setup( &
       num_inputs, num_outputs, &
       activation_scale, activation_function, &
       kernel_initialiser, bias_initialiser) result(layer)
    use activation,  only: activation_setup
    use initialiser, only: initialiser_setup
    implicit none
    integer, intent(in) :: num_inputs, num_outputs
    real(real12), optional, intent(in) :: activation_scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser, bias_initialiser
    
    type(full_layer_type) :: layer

    real(real12) :: scale
    character(len=10) :: t_activation_function, initialiser_name
    class(initialiser_type), allocatable :: initialiser


    layer%num_inputs  = num_inputs
    layer%num_outputs = num_outputs

    !!--------------------------------------------------------------------------
    !! set activation and derivative functions based on input name
    !!--------------------------------------------------------------------------
    if(present(activation_function))then
       t_activation_function = activation_function
    else
       t_activation_function = "none"
    end if
    if(present(activation_scale))then
       scale = activation_scale
    else
       scale = 1._real12
    end if
       
    allocate(layer%transfer, source=activation_setup(t_activation_function, scale))



    allocate(layer%dw(layer%num_inputs+1,layer%num_outputs))
    allocate(layer%output(layer%num_outputs))
    allocate(layer%di(layer%num_inputs+1)) ! +1 account for bias
    

    !!--------------------------------------------------------------------------
    !! initialise kernels and biases
    !!--------------------------------------------------------------------------
    allocate(layer%weight(layer%num_inputs+1,layer%num_outputs))
    allocate(layer%weight_incr(layer%num_inputs+1,layer%num_outputs))
    if(present(kernel_initialiser))then
       initialiser_name = kernel_initialiser
    else
       initialiser_name = "he_uniform"
    end if
    initialiser = initialiser_setup(initialiser_name)
    call initialiser%initialise(layer%weight(:layer%num_inputs,:), &
         fan_in=layer%num_inputs+1, fan_out=layer%num_outputs)
    if(present(bias_initialiser))then
       initialiser_name = bias_initialiser
    else
       initialiser_name= "zeros"
    end if
    initialiser = initialiser_setup(initialiser_name)
    call initialiser%initialise(layer%weight(layer%num_inputs+1,:), &
         fan_in=layer%num_inputs+1, fan_out=layer%num_outputs)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_1d(this, input)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real12), dimension(this%num_inputs), intent(in) :: input


    !! generate outputs from weights, biases, and inputs    
    this%output = this%transfer%activate(&
         this%weight(this%num_inputs+1,:) + &
         matmul(input,this%weight(:this%num_inputs,:))&
         )

  end subroutine forward_1d
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine backward_1d(this, input, gradient)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real12), dimension(this%num_inputs), intent(in) :: input
    real(real12), dimension(this%num_outputs), intent(in) :: gradient !was output_gradients
    !! NOTE, gradient is di, not dw


    !! the delta values are the error multipled by the derivative ...
    !! ... of the transfer function
    !! final layer: error (delta) = activ_diff (g') * error
    !! other layer: error (delta) = activ_diff (g') * sum(weight(l+1)*error(l+1))
    this%di = gradient * &
         this%transfer%differentiate([this%output])

    !! define the input to the neuron
    this%dw(:this%num_inputs,:) = matmul(&
         reshape(this%di, [size(this%di,dim=1), 1]),&
         reshape(input, [1, size(input,dim=1)])&
         )

    !! bias weight gradient
    !! ... as the bias neuron = 1._real12, then gradient of the bias ...
    !! ... is just the delta (error), no need to multiply by 1._real12
    this%dw(this%num_inputs+1,:) = &
         this%di * this%transfer%differentiate([1._real12])

    !! the errors are summed from the delta of the ...
    !! ... 'child' node * 'child' weight
    this%di = matmul(&
         this%weight(:this%num_inputs,:),&
         this%di&
         )

  end subroutine backward_1d
!!!#############################################################################


end module full_layer
!!!#############################################################################
