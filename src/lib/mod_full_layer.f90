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
     procedure, pass(this) :: init => init_full
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
          num_outputs, num_inputs, &
          activation_function, activation_scale, &
          kernel_initialiser, bias_initialiser) result(layer)
       integer, intent(in) :: num_outputs
       integer, optional, intent(in) :: num_inputs
       real(real12), optional, intent(in) :: activation_scale
       character(*), optional, intent(in) :: activation_function, &
            kernel_initialiser, bias_initialiser
       type(full_layer_type) :: layer
     end function layer_setup
  end interface full_layer_type


  private
  public :: full_layer_type
  public :: read_full_layer


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
!!! set up layer
!!!#############################################################################
  module function layer_setup( &
       num_outputs, num_inputs, &
       activation_scale, activation_function, &
       kernel_initialiser, bias_initialiser, &
       clip_dict, clip_min, clip_max, clip_norm) result(layer)
    use activation,  only: activation_setup
    use initialiser, only: get_default_initialiser
    implicit none
    integer, intent(in) :: num_outputs
    integer, optional, intent(in) :: num_inputs
    real(real12), optional, intent(in) :: activation_scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser, bias_initialiser
    type(clip_type), optional, intent(in) :: clip_dict
    real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm
    
    type(full_layer_type) :: layer

    real(real12) :: scale
    character(len=10) :: t_activation_function


    !!--------------------------------------------------------------------------
    !! set up clipping limits
    !!--------------------------------------------------------------------------
    if(present(clip_dict))then
       layer%clip = clip_dict
       if(present(clip_min).or.present(clip_max).or.present(clip_norm))then
          write(*,*) "Multiple clip options provided to full layer"
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
    !! define weights (kernels) and biases initialisers
    !!--------------------------------------------------------------------------
    if(present(kernel_initialiser))then
       layer%kernel_initialiser = kernel_initialiser
    else
       layer%kernel_initialiser = get_default_initialiser(t_activation_function)
    end if
    write(*,'("FC kernel initialiser: ",A)') trim(layer%kernel_initialiser)
    if(present(bias_initialiser))then
       layer%bias_initialiser = bias_initialiser
    else
       layer%bias_initialiser = get_default_initialiser(&
            t_activation_function, is_bias=.true.)       
    end if
    write(*,'("FC bias initialiser: ",A)') trim(layer%bias_initialiser)


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    layer%num_outputs = num_outputs
    layer%output_shape = [layer%num_outputs]
    if(present(num_inputs)) call layer%init(input_shape=[num_inputs])

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_full(this, input_shape)
    use initialiser, only: initialiser_setup
    implicit none
    class(full_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape

    class(initialiser_type), allocatable :: initialiser


    !!--------------------------------------------------------------------------
    !! initialise number of inputs
    !!--------------------------------------------------------------------------
    if(size(input_shape,dim=1).eq.1)then
       this%input_shape = input_shape
       this%num_inputs = input_shape(1)
    else
       write(*,*) "WARNING: reshaping input_shape to 1D for full layer"
       this%num_inputs  = product(input_shape)
       this%input_shape = [this%num_inputs]
       !stop "ERROR: invalid size of input_shape in full, expected (1)"
    end if


    !!--------------------------------------------------------------------------
    !! allocate weight, weight steps (velocities), output, and activation
    !!--------------------------------------------------------------------------
    allocate(this%weight(this%num_inputs+1,this%num_outputs))

    allocate(this%weight_incr, mold=this%weight)
    allocate(this%output(this%num_outputs), source=0._real12)
    allocate(this%z, mold=this%output)
    this%weight_incr = 0._real12
    this%output = 0._real12
    this%z = 0._real12

    allocate(this%dw(this%num_inputs+1,this%num_outputs), source=0._real12)
    allocate(this%di(this%num_inputs), source=0._real12)
    this%dw = 0._real12
    this%di = 0._real12


    !!--------------------------------------------------------------------------
    !! initialise weights (kernels)
    !!--------------------------------------------------------------------------
    allocate(initialiser, source=initialiser_setup(this%kernel_initialiser))
    call initialiser%initialise(this%weight(:this%num_inputs,:), &
         fan_in=this%num_inputs+1, fan_out=this%num_outputs)
    deallocate(initialiser)

    !! initialise biases
    !!--------------------------------------------------------------------------
    allocate(initialiser, source=initialiser_setup(this%bias_initialiser))
    call initialiser%initialise(this%weight(this%num_inputs+1,:), &
         fan_in=this%num_inputs+1, fan_out=this%num_outputs)
    deallocate(initialiser)

  end subroutine init_full
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


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


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  function read_full_layer(unit) result(layer)
    use infile_tools, only: assign_val, assign_vec
    use misc, only: to_lower, icount
    implicit none
    integer, intent(in) :: unit

    class(full_layer_type), allocatable :: layer

    integer :: stat
    integer :: i, j, k, c, itmp1
    integer :: num_inputs, num_outputs
    real(real12) :: activation_scale
    character(256) :: buffer, tag
    character(:), allocatable :: activation_function

    logical :: found_weights
    real(real12), allocatable, dimension(:) :: data_list


    !! loop over tags in layer card
    found_weights = .false.
    tag_loop: do

       !! check for end of file
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(0,*) "ERROR: file hit error (EoF?) before encountering END FULL"
          write(0,*) "Exiting..."
          stop
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       !! check for end of convolution card
       if(trim(adjustl(buffer)).eq."END FULL")then
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       !! read parameters from save file
       select case(trim(tag))
       case("NUM_INPUTS")
          call assign_val(buffer, num_inputs, itmp1)
       case("NUM_OUTPUTS")
          call assign_val(buffer, num_outputs, itmp1)
       case("ACTIVATION_FUNCTION")
          call assign_val(buffer, activation_function, itmp1)
       case("ACTIVATION_SCALE")
          call assign_val(buffer, activation_scale, itmp1)
       case("WEIGHTS")
          found_weights = .true.
          exit tag_loop
       case default
          !! don't look for "e" due to scientific notation of numbers
          !! ... i.e. exponent (E+00)
          if(scan(to_lower(trim(adjustl(buffer))),&
               'abcdfghijklmnopqrstuvwxyz').eq.0)then
             cycle tag_loop
          elseif(tag(:3).eq.'END')then
             cycle tag_loop
          end if
          stop "Unrecognised line in input file: "//trim(adjustl(buffer))
       end select
    end do tag_loop

    !! set transfer activation function

    layer = full_layer_type( &
         num_outputs = num_outputs, num_inputs = num_inputs, &
         activation_function = activation_function, &
         activation_scale = activation_scale, &
         kernel_initialiser="zeros", bias_initialiser="zeros")

    !! check if WEIGHTS card was found
    if(.not.found_weights)then
      write(0,*) "WARNING: WEIGHTS card in FULL not found"
    else
       !! allocate convolutional layer and read weights
       layer%weight_incr = 0._real12
       layer%weight = 0._real12

       do i=1,num_outputs
          allocate(data_list(num_inputs), source=0._real12)
          c = 1
          k = 1
          data_concat_loop: do while(c.le.num_inputs)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop
          layer%weight(:,i) = data_list
          deallocate(data_list)
       end do

       !! check for end of weights card
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          write(*,*) trim(adjustl(buffer))
          stop "ERROR: END WEIGHTS not where expected"
       end if
    end if

    !! check for end of layer card
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END FULL")then
       write(*,*) trim(adjustl(buffer))
       stop "ERROR: END FULL not where expected"
    end if

  end function read_full_layer
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
