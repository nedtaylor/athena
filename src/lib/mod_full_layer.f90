!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module full_layer
  use constants, only: real12
  use base_layer, only: learnable_layer_type
  use custom_types, only: activation_type, initialiser_type, clip_type
  implicit none
  

!!!-----------------------------------------------------------------------------
!!! fully connected network layer type
!!!-----------------------------------------------------------------------------
  type, extends(learnable_layer_type) :: full_layer_type
     integer :: num_inputs
     integer :: num_outputs
     type(clip_type) :: clip
     real(real12), allocatable, dimension(:,:) :: weight, weight_incr
     real(real12), allocatable, dimension(:,:) :: dw ! weight gradient
     real(real12), allocatable, dimension(:) :: output, z !output and activation
     real(real12), allocatable, dimension(:) :: di ! input gradient (i.e. delta)
     class(activation_type), allocatable :: transfer
   contains
     procedure, pass(this) :: print => print_full
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, pass(this) :: update
     procedure, private, pass(this) :: forward_1d
     procedure, private, pass(this) :: backward_1d
  end type full_layer_type


!!!-----------------------------------------------------------------------------
!!! interface for layer set up
!!!-----------------------------------------------------------------------------
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
!!! forward propagation assumed rank handler
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
!!! backward propagation assumed rank handler
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


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up and initialise network layer
!!!#############################################################################
  module function layer_setup( &
       num_inputs, num_outputs, &
       activation_scale, activation_function, &
       kernel_initialiser, bias_initialiser, &
       clip_dict, clip_min, clip_max, clip_norm) result(layer)
    use activation,  only: activation_setup
    use initialiser, only: initialiser_setup
    implicit none
    integer, intent(in) :: num_inputs, num_outputs
    real(real12), optional, intent(in) :: activation_scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser, bias_initialiser
    type(clip_type), optional, intent(in) :: clip_dict
    real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm
    
    type(full_layer_type) :: layer

    real(real12) :: scale
    character(len=10) :: t_activation_function
    character(len=14) :: initialiser_name
    class(initialiser_type), allocatable :: initialiser



    !!--------------------------------------------------------------------------
    !! set up clipping limits
    !!--------------------------------------------------------------------------
    if(present(clip_dict))then
       layer%clip = clip_dict
       if(present(clip_min).or.present(clip_max).or.present(clip_norm))then
          write(*,*) "Multiple clip options provided to conv2d layer"
          write(*,*) "Ignoring all bar clip_dict"
       end if
    else
       if(present(clip_min))then
          layer%clip%l_min_max = .true.
          layer%clip%min = clip_min
       end if
       if(present(clip_max))then
          layer%clip%l_min_max = .true.
          layer%clip%max = clip_max
       end if
       if(present(clip_norm))then
          layer%clip%l_norm = .true.
          layer%clip%norm = clip_norm
       end if
    end if


    layer%num_inputs  = num_inputs
    layer%input_shape = [layer%num_inputs]
    layer%num_outputs = num_outputs
    layer%output_shape = [layer%num_outputs]

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
    write(*,'("FC activation function: ",A)') trim(t_activation_function)
    allocate(layer%transfer, &
         source=activation_setup(t_activation_function, scale))
    

    !!--------------------------------------------------------------------------
    !! allocate weight, weight steps (velocities), output, and activation
    !!--------------------------------------------------------------------------
    allocate(layer%weight(layer%num_inputs+1,layer%num_outputs))

    allocate(layer%weight_incr, mold=layer%weight)
    allocate(layer%output(layer%num_outputs), source=0._real12)
    allocate(layer%z, mold=layer%output)
    layer%weight_incr = 0._real12
    layer%output = 0._real12
    layer%z = 0._real12

    allocate(layer%dw(layer%num_inputs+1,layer%num_outputs), source=0._real12)
    allocate(layer%di(layer%num_inputs), source=0._real12)
    layer%dw = 0._real12
    layer%di = 0._real12


    !!--------------------------------------------------------------------------
    !! initialise weights (kernels)
    !!--------------------------------------------------------------------------
    if(present(kernel_initialiser))then
       initialiser_name = kernel_initialiser
    elseif(trim(t_activation_function).eq."selu")then
       initialiser_name = "lecun_normal"
    elseif(index(t_activation_function,"elu").ne.0)then
       initialiser_name = "he_uniform"
    else
       initialiser_name = "glorot_uniform"
    end if
    write(*,'("FC kernel initialiser: ",A)') trim(initialiser_name)
    allocate(initialiser, source=initialiser_setup(initialiser_name))
    call initialiser%initialise(layer%weight(:layer%num_inputs,:), &
         fan_in=layer%num_inputs+1, fan_out=layer%num_outputs)
    deallocate(initialiser)

    !! initialise biases
    !!--------------------------------------------------------------------------
    if(present(bias_initialiser))then
       initialiser_name = bias_initialiser
    else
       initialiser_name= "zeros"
    end if
    write(*,'("FC bias initialiser: ",A)') trim(initialiser_name)
    allocate(initialiser, source=initialiser_setup(initialiser_name))
    call initialiser%initialise(layer%weight(layer%num_inputs+1,:), &
         fan_in=layer%num_inputs+1, fan_out=layer%num_outputs)
    deallocate(initialiser)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_full(this, file)
    implicit none
    class(full_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    integer :: unit


    !! open file with new unit
    !!--------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    !! write convolution initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("FULL")')
    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') this%transfer%scale

    !! write fully connected weights and biases
    !!--------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))', advance="no") this%weight
    write(unit,'("END WEIGHTS")')
    write(unit,'("END FULL")')

    !! close unit
    !!--------------------------------------------------------------------------
    close(unit)

  end subroutine print_full
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_1d(this, input)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real12), dimension(this%num_inputs), intent(in) :: input


    !! generate outputs from weights, biases, and inputs
    this%z = this%weight(this%num_inputs+1,:) + &
      matmul(input,this%weight(:this%num_inputs,:))
      
    !! apply activation function to activation
    this%output = this%transfer%activate(this%z)

  end subroutine forward_1d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!! method : gradient descent
!!!#############################################################################
  pure subroutine backward_1d(this, input, gradient)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real12), dimension(this%num_inputs,1), intent(in) :: input
    real(real12), dimension(this%num_outputs), intent(in) :: gradient

    real(real12), dimension(1,this%num_outputs) :: delta
    real(real12), dimension(this%num_inputs, this%num_outputs) :: dw


    real(real12), dimension(1) :: bias_diff
    bias_diff = this%transfer%differentiate([1._real12])

    !! the delta values are the error multipled by the derivative ...
    !! ... of the transfer function
    !! delta(l) = g'(a) * dE/dI(l)
    !! delta(l) = differential of activation * error from next layer
    delta(1,:) = gradient * this%transfer%differentiate(this%z)

    !! partial derivatives of error wrt weights
    !! dE/dW = o/p(l-1) * delta
    dw = matmul(input, delta)

    !! the errors are summed from the delta of the ...
    !! ... 'child' node * 'child' weight
    !! dE/dI(l-1) = sum(weight(l) * delta(l))
    !! this prepares dE/dI for when it is passed into the previous layer
    this%di = matmul(this%weight(:this%num_inputs,:), delta(1,:))

    !! sum weights and biases errors to use in batch gradient descent
    this%dw(:this%num_inputs,:) = this%dw(:this%num_inputs,:) + dw
    this%dw(this%num_inputs+1,:) = this%dw(this%num_inputs+1,:) + delta(1,:) * &
         bias_diff(1)

  end subroutine backward_1d
!!!#############################################################################


!!!#############################################################################
!!! update the weights based on how much error the node is responsible for
!!!#############################################################################
  pure subroutine update(this, optimiser, batch_size)
    use optimiser, only: optimiser_type
    use normalisation, only: gradient_clip
    implicit none
    class(full_layer_type), intent(inout) :: this
    type(optimiser_type), intent(in) :: optimiser
    integer, optional, intent(in) :: batch_size


    !! normalise by number of samples
    if(present(batch_size)) this%dw = this%dw/batch_size

    !! apply gradient clipping
    if(this%clip%l_min_max) call gradient_clip(size(this%dw),&
         this%dw,&
         clip_min=this%clip%min,clip_max=this%clip%max)
    if(this%clip%l_norm) &
         call gradient_clip(size(this%dw),&
         this%dw,&
         clip_norm=this%clip%norm)

    !! update the layer weights and bias using gradient descent
    call optimiser%optimise(&
         this%weight, &
         this%weight_incr, &
         this%dw)

    !! reset gradients
    this%di = 0._real12
    this%dw = 0._real12

  end subroutine update
!!!#############################################################################

end module full_layer
!!!#############################################################################
