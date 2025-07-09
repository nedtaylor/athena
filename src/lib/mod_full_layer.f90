module athena__full_layer
  !! Module containing implementation of a fully connected layer
  !!
  !! This module implements a fully connected (aka dense) layer for a
  !! neural network.
  !! Attribution statement:
  !! The get_num_params procedure is based on code from the
  !! neural-fortran library
  !! https://github.com/modern-fortran/neural-fortran/blob/main/src/nf/nf_layer.f90
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: activation_type, initialiser_type, &
       array2d_type
  implicit none


  private

  public :: full_layer_type
  public :: read_full_layer


  type, extends(learnable_layer_type) :: full_layer_type
     !! Type for fully connected (aka dense) layer with overloaded procedures
     integer :: num_inputs
     !! Number of inputs
     integer :: num_outputs
     !! Number of outputs
     real(real32), pointer :: weight(:,:) => null()
     !! Pointer to weights (kernels)
     real(real32), pointer :: dw(:,:,:) => null()
     !! Pointer to weight gradients
     real(real32), allocatable, dimension(:,:) :: z
     !! Activation values
   contains
     procedure, pass(this) :: get_num_params => get_num_params_full
     !! Get the number of parameters for fully connected layer
     procedure, pass(this) :: set_hyperparams => set_hyperparams_full
     !! Set the hyperparameters for fully connected layer
     procedure, pass(this), private :: &
          set_ptrs_hyperparams => set_ptrs_hyperparams_full
     !! Set the pointers to hyperparameters
     procedure, pass(this) :: init => init_full
     !! Initialise fully connected layer
     procedure, pass(this) :: set_batch_size => set_batch_size_full
     !! Set the batch size for fully connected layer
     procedure, pass(this) :: print => print_full
     !! Print the layer to a file
     procedure, pass(this) :: read => read_full
     !! Read the layer from a file
     procedure, pass(this) :: forward  => forward_rank
     !! Forward propagation for fully connected layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward propagation for fully connected layer
     procedure, private, pass(this) :: forward_2d
     !! Forward propagation for 2D input
     procedure, private, pass(this) :: backward_2d
     !! Backward propagation for 2D input
     final :: finalise_full
     !! Finalise fully connected layer
  end type full_layer_type

  interface full_layer_type
     !! Interface for setting up the fully connected layer
     module function layer_setup( &
          num_outputs, num_inputs, batch_size, &
          activation_function, activation_scale, &
          kernel_initialiser, bias_initialiser &
     ) result(layer)
       !! Setup a fully connected layer
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: num_inputs
       !! Number of inputs
       integer, optional, intent(in) :: batch_size
       !! Batch size
       real(real32), optional, intent(in) :: activation_scale
       !! Activation scale
       character(*), optional, intent(in) :: activation_function, &
            kernel_initialiser, bias_initialiser
       !! Activation function, kernel initialiser, and bias initialiser
       type(full_layer_type) :: layer
       !! Instance of the fully connected layer
     end function layer_setup
  end interface full_layer_type



contains

!###############################################################################
  subroutine finalise_full(this)
    !! Finalise fully connected layer
    implicit none

    ! Arguments
    type(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer

    if(associated(this%weight)) nullify(this%weight)
    if(associated(this%dw)) nullify(this%dw)
    if(allocated(this%z)) deallocate(this%z)
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(allocated(this%di)) deallocate(this%di)

  end subroutine finalise_full
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure function get_num_params_full(this) result(num_params)
    !! Get the number of parameters for fully connected layer
    !!
    !! This function calculates the number of parameters for a fully connected
    !! layer.
    !! This procedure is based on code from the neural-fortran library
    implicit none

    ! Arguments
    class(full_layer_type), intent(in) :: this
    !! Instance of the fully connected layer
    integer :: num_params
    !! Number of parameters

    num_params = ( this%num_inputs + 1 )* this%num_outputs

  end function get_num_params_full
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine forward_rank(this, input)
    !! Forward propagation for fully connected layer
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values

    select rank(input); rank(2)
       call forward_2d(this, input)
    end select
  end subroutine forward_rank
!###############################################################################


!###############################################################################
  pure subroutine backward_rank(this, input, gradient)
    !! Backward propagation for fully connected layer
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient values

    select rank(input)
    rank(2)
       select rank(gradient); rank(2)
          call backward_2d(this, input, gradient)
       end select
    end select
  end subroutine backward_rank
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       num_outputs, num_inputs, &
       batch_size, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser, verbose &
  ) result(layer)
    !! Setup a fully connected layer
    implicit none

    ! Arguments
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, optional, intent(in) :: num_inputs
    !! Number of inputs
    integer, optional, intent(in) :: batch_size
    !! Batch size
    real(real32), optional, intent(in) :: activation_scale
    !! Activation scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser, bias_initialiser
    !! Activation function, kernel initialiser, and bias initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(full_layer_type) :: layer
    !! Instance of the fully connected layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    real(real32) :: scale = 1._real32
    !! Activation scale
    character(len=10) :: activation_function_ = "none"
    !! Activation function

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set activation and derivative functions based on input name
    !---------------------------------------------------------------------------
    if(present(activation_function)) activation_function_ = activation_function
    if(present(activation_scale)) scale = activation_scale


    !---------------------------------------------------------------------------
    ! Define weights (kernels) and biases initialisers
    !---------------------------------------------------------------------------
    if(present(kernel_initialiser)) layer%kernel_initialiser =kernel_initialiser
    if(present(bias_initialiser)) layer%bias_initialiser = bias_initialiser


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         num_outputs = num_outputs, &
         activation_function = activation_function_, &
         activation_scale = scale, &
         kernel_initialiser = layer%kernel_initialiser, &
         bias_initialiser = layer%bias_initialiser, &
         verbose = verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise batch size
    !---------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(num_inputs)) call layer%init(input_shape=[num_inputs])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_full( &
       this, num_outputs, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    !! Set the hyperparameters for fully connected layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    integer, intent(in) :: num_outputs
    !! Number of outputs
    character(*), intent(in) :: activation_function
    !! Activation function
    real(real32), intent(in) :: activation_scale
    !! Activation scale
    character(*), intent(in) :: kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level


    this%name = "full"
    this%type = "full"
    this%input_rank = 1
    this%num_outputs = num_outputs
    allocate(this%transfer, &
         source=activation_setup(activation_function, activation_scale))
    if(trim(kernel_initialiser).eq.'') &
         this%kernel_initialiser = get_default_initialiser(activation_function)
    if(trim(bias_initialiser).eq.'') &
         this%bias_initialiser = get_default_initialiser( &
              activation_function, &
              is_bias=.true. &
         )
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("FULL activation function: ",A)') &
               trim(activation_function)
          write(*,'("FULL kernel initialiser: ",A)') &
               trim(this%kernel_initialiser)
          write(*,'("FULL bias initialiser: ",A)') &
               trim(this%bias_initialiser)
       end if
    end if

  end subroutine set_hyperparams_full
!###############################################################################


!###############################################################################
  subroutine set_ptrs_hyperparams_full(this)
    !! Set the pointers to hyperparameters
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout), target :: this
    !! Instance of the fully connected layer

    if(allocated(this%params)) &
         this%weight(1:this%num_outputs,1:this%num_inputs+1) => this%params
    if(allocated(this%dp)) &
         this%dw(1:this%num_outputs,1:this%num_inputs,1:this%batch_size) => &
         this%dp

  end subroutine set_ptrs_hyperparams_full
!###############################################################################


!###############################################################################
  subroutine init_full(this, input_shape, batch_size, verbose)
    !! Initialise fully connected layer
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    class(initialiser_type), allocatable :: initialiser_
    !! Initialiser


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise number of inputs
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%num_inputs = this%input_shape(1)
    this%output_shape = [this%num_outputs]
    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Allocate weight, weight steps (velocities), output, and activation
    !---------------------------------------------------------------------------
    if(allocated(this%params)) deallocate(this%params)
    allocate(this%params(this%num_params), source=0._real32)


    !---------------------------------------------------------------------------
    ! Initialise weights (kernels)
    !---------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%kernel_initialiser))
    call initialiser_%initialise( &
         this%params(:this%num_params-this%num_outputs), &
         fan_in = this%num_inputs + 1, fan_out = this%num_outputs, &
         spacing = [ this%num_outputs ] &
    )
    deallocate(initialiser_)

    ! Initialise biases
    !---------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%bias_initialiser))
    call initialiser_%initialise( &
         this%params(this%num_params-this%num_outputs+1:), &
         fan_in=this%num_inputs+1, fan_out=this%num_outputs &
    )
    deallocate(initialiser_)


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_full
!###############################################################################


!###############################################################################
  subroutine set_batch_size_full(this, batch_size, verbose)
    !! Set the batch size for fully connected layer
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout), target :: this
    integer, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Set weights and biases pointers to params array
    !---------------------------------------------------------------------------
    this%weight(1:this%num_outputs,1:this%num_inputs+1) => this%params


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(allocated(this%output)) deallocate(this%output)
       allocate(this%output(1,1), source=array2d_type())
       call this%output(1,1)%allocate( &
            [this%num_outputs, this%batch_size], &
            source=0._real32 &
       )
       if(allocated(this%z)) deallocate(this%z)
       select type(output => this%output(1,1))
       type is (array2d_type)
          allocate( this%z, source = output%val )
       end select
       if(allocated(this%dp)) deallocate(this%dp)
       allocate( &
            this%dp( &
                 this%num_params - this%num_outputs, &
                 this%batch_size &
            ), source=0._real32 &
       )
       this%dw(1:this%num_outputs,1:this%num_inputs,1:this%batch_size) => &
            this%dp
       if(allocated(this%db)) deallocate(this%db)
       allocate(this%db(this%num_outputs, this%batch_size), source=0._real32)
       if(allocated(this%di)) deallocate(this%di)
       allocate(this%di(1,1), source=array2d_type())
       call this%di(1,1)%allocate( &
            [this%num_inputs, this%batch_size], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_full
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_full(this, file)
    !! Print fully connected layer to file
    use athena__misc, only: to_upper
    implicit none

    ! Arguments
    class(full_layer_type), intent(in) :: this
    !! Instance of the fully connected layer
    character(*), intent(in) :: file
    !! File name

    ! Local variables
    integer :: i
    !! Loop index
    integer :: unit
    !! Unit number


    ! Open file with new unit
    !---------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(A)') to_upper(trim(this%name))
    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') this%transfer%scale


    ! Write fully connected weights and biases
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    do i=1,this%num_inputs+1
       write(unit,'(5(E16.8E2))') this%weight(:,i)
    end do
    write(unit,'("END WEIGHTS")')
    write(unit,'("END FULL")')


    ! Close unit
    !---------------------------------------------------------------------------
    close(unit)

  end subroutine print_full
!###############################################################################


!###############################################################################
  subroutine read_full(this, unit, verbose)
    !! Read fully connected layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat
    !! Status of read
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: i, j, k, c, itmp1
    !! Loop indices
    integer :: num_inputs, num_outputs
    !! Number of inputs and outputs
    real(real32) :: activation_scale
    !! Activation scale
    logical :: found_weights = .false.
    !! Boolean whether weights card was found
    character(14) :: kernel_initialiser='', bias_initialiser=''
    !! Initialisers
    character(20) :: activation_function
    !! Activation function
    character(256) :: buffer, tag, err_msg
    !! Buffer and tag

    real(real32), allocatable, dimension(:) :: data_list
    !! List of data


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
    tag_loop: do

       ! Check for end of file
       !------------------------------------------------------------------------
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg,'("file encountered error (EoF?) before END ",A)') &
               to_upper(this%name)
          call stop_program(err_msg)
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       ! Check for end of layer card
       !------------------------------------------------------------------------
       if(trim(adjustl(buffer)).eq."END "//to_upper(trim(this%name)))then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       ! Read parameters from file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("NUM_INPUTS")
          call assign_val(buffer, num_inputs, itmp1)
       case("NUM_OUTPUTS")
          call assign_val(buffer, num_outputs, itmp1)
       case("ACTIVATION")
          call assign_val(buffer, activation_function, itmp1)
       case("ACTIVATION_SCALE")
          call assign_val(buffer, activation_scale, itmp1)
       case("KERNEL_INITIALISER")
          call assign_val(buffer, kernel_initialiser, itmp1)
       case("BIAS_INITIALISER")
          call assign_val(buffer, bias_initialiser, itmp1)
       case("WEIGHTS")
          found_weights = .true.
          kernel_initialiser = 'zeros'
          bias_initialiser   = 'zeros'
          exit tag_loop
       case default
          ! Don't look for "e" due to scientific notation of numbers
          ! ... i.e. exponent (E+00)
          if(scan(to_lower(trim(adjustl(buffer))),&
               'abcdfghijklmnopqrstuvwxyz').eq.0)then
             cycle tag_loop
          elseif(tag(:3).eq.'END')then
             cycle tag_loop
          end if
          write(err_msg,'("Unrecognised line in input file: ",A)') &
               trim(adjustl(buffer))
          call stop_program(err_msg)
          return
       end select
    end do tag_loop


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    call this%set_hyperparams( &
         num_outputs = num_outputs, &
         activation_function = activation_function, &
         activation_scale = activation_scale, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[num_inputs])


    ! Check if WEIGHTS card was found
    !---------------------------------------------------------------------------
    if(.not.found_weights)then
       write(0,*) "WARNING: WEIGHTS card in "//to_upper(trim(this%name))//" not found"
    else
       do i=1,num_inputs+1
          allocate(data_list((num_outputs)), source=0._real32)
          c = 1
          k = 1
          data_concat_loop: do while(c.le.num_outputs)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop
          this%weight(:,i) = data_list
          deallocate(data_list)
       end do

       ! Check for end of weights card
       !------------------------------------------------------------------------
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          write(0,*) trim(adjustl(buffer))
          call stop_program("END WEIGHTS not where expected")
          return
       end if
    end if


    !---------------------------------------------------------------------------
    ! Check for end of layer card
    !---------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_full
!###############################################################################


!###############################################################################
  function read_full_layer(unit, verbose) result(layer)
    !! Read fully connected layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the fully connected layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=full_layer_type(num_outputs=0))
    call layer%read(unit, verbose=verbose_)

  end function read_full_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine forward_2d(this, input)
    !! Forward propagation for 2D input
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    real(real32), dimension(this%num_inputs, this%batch_size), &
         intent(in) :: input
    !! Input values

    ! Local variables
    integer :: s
    !! Loop index


    ! Generate outputs from weights, biases, and inputs
    !---------------------------------------------------------------------------
    do concurrent(s=1:this%batch_size)
       this%z(:,s) = this%weight(:,this%num_inputs+1) + &
            matmul(this%weight(:,:this%num_inputs),input(:,s))
    end do

    ! Apply activation function to activation
    !---------------------------------------------------------------------------
    this%output(1,1)%val(:,:) = this%transfer%activate(this%z)

  end subroutine forward_2d
!###############################################################################


!###############################################################################
!!! backward propagation
!!! method : gradient descent
!###############################################################################
  pure subroutine backward_2d(this, input, gradient)
    !! Backward propagation for 2D input
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    real(real32), dimension(this%num_inputs, this%batch_size), &
         intent(in) :: input
    !! Input values
    real(real32), dimension(this%num_outputs, this%batch_size), &
         intent(in) :: gradient
    !! Gradient values

    ! Local variables
    real(real32), dimension(this%num_outputs, this%batch_size) :: grad_dz
    !! Gradient multiplied by differential of Z (aka delta values)
    real(real32), dimension(1) :: bias_diff
    !! Differential of bias

    ! Loop variables
    integer :: s, j
    !! Loop indices


    bias_diff = this%transfer%differentiate([1._real32])


    ! Get gradient multiplied by differential of Z
    !---------------------------------------------------------------------------
    ! The grad_dz values are the error multipled by the derivative ...
    ! ... of the transfer function
    ! grad_dz(l) = g'(a) * dE/dI(l)
    ! grad_dz(l) = differential of activation * error from next layer
    grad_dz = gradient * this%transfer%differentiate(this%z)
    this%db(:,:) = this%db(:,:) + grad_dz * bias_diff(1)


    ! Update weights
    !---------------------------------------------------------------------------
    do concurrent(s=1:this%batch_size)
       !! partial derivatives of error wrt weights
       !! dE/dW = o/p(l-1) * grad_dz
       do j = 1, this%num_inputs
          this%dw(:,j,s) = this%dw(:,j,s) + input(j,s) * grad_dz(:,s)
       end do
       !! the errors are summed from the grad_dz of the ...
       !! ... 'child' node * 'child' weight
       !! dE/dI(l-1) = sum(weight(l) * grad_dz(l))
       !! this prepares dE/dI for when it is passed into the previous layer
       this%di(1,1)%val(:,s) = &
            matmul(grad_dz(:,s), this%weight(:,:this%num_inputs))
    end do

  end subroutine backward_2d
!###############################################################################

end module athena__full_layer
