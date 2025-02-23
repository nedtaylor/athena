module athena__deepset_layer
  !! Module containing the implementation of a deep set layer
  use athena__constants, only: real32
  use athena__base_layer, only: learnable_layer_type
  use athena__misc_types, only: activation_type, initialiser_type
  implicit none


  private

  public :: deepset_layer_type
  public :: read_deepset_layer


  type, extends(learnable_layer_type) :: deepset_layer_type
     !! Type for deep set layer
     integer :: num_inputs, num_addit_inputs = 0
     !! Number of inputs and additional inputs
     integer :: num_outputs
     !! Number of outputs
     real(real32) :: lambda, gamma, bias
     !! Lambda, gamma, and bias
     real(real32), allocatable, dimension(:) :: dg, dl, db
     !! Weight gradients
     real(real32), allocatable, dimension(:,:) :: output
     !! Output and activation
     real(real32), allocatable, dimension(:,:) :: di
     !! Input gradient (i.e. delta)
   contains
     procedure, pass(this) :: get_num_params => get_num_params_deepset
     !! Get the number of parameters for deep set layer
     procedure, pass(this) :: get_params => get_params_deepset
     !! Get the parameters for deep set layer
     procedure, pass(this) :: set_params => set_params_deepset
     !! Set the parameters for deep set layer
     procedure, pass(this) :: get_gradients => get_gradients_deepset
     !! Get the gradients for deep set layer
     procedure, pass(this) :: set_gradients => set_gradients_deepset
     !! Set the gradients for deep set layer
     procedure, pass(this) :: get_output => get_output_deepset
     !! Get the output for deep set layer
     procedure, pass(this) :: print => print_deepset
     !! Print the deep set layer to file
     procedure, pass(this) :: init => init_deepset
     !! Initialise the deep set layer
     procedure, pass(this) :: set_batch_size => set_batch_size_deepset
     !! Set the batch size for deep set layer
     procedure, pass(this) :: forward  => forward_rank
     !! Forward propagation handler for deep set layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward propagation handler for deep set layer
     procedure, private, pass(this) :: forward_2d
     !! Forward propagation for 2D input
     procedure, private, pass(this) :: backward_2d
     !! Backward propagation for 2D input
  end type deepset_layer_type


  interface deepset_layer_type
     !! Interface for setting up the deep set layer
     module function layer_setup( &
          input_shape, batch_size, &
          num_inputs, &
          lambda, gamma &
     ) result(layer)
       !! Set up the deep set layer
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: num_inputs
       !! Number of inputs
       integer, optional, intent(in) :: batch_size
       !! Batch size
       real(real32), optional, intent(in) :: lambda, gamma
       !! Lambda and gamma
       type(deepset_layer_type) :: layer
       !! Instance of the deep set layer
     end function layer_setup
  end interface deepset_layer_type



contains

!###############################################################################
  pure function get_num_params_deepset(this) result(num_params)
    !! Get the number of parameters for deep set layer
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(in) :: this
    !! Instance of the deep set layer
    integer :: num_params
    !! Number of learnable parameters

    num_params = 3

  end function get_num_params_deepset
!###############################################################################


!###############################################################################
  pure function get_params_deepset(this) result(params)
    !! Get the parameters for deep set layer
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(in) :: this
    !! Instance of the deep set layer
    real(real32), dimension(this%num_params) :: params
    !! Learnable parameters

    params = [this%lambda, this%gamma, this%bias]

  end function get_params_deepset
!###############################################################################


!###############################################################################
  subroutine set_params_deepset(this, params)
    !! Set the parameters for deep set layer
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(inout) :: this
    !! Instance of the deep set layer
    real(real32), dimension(this%num_params), intent(in) :: params
    !! Learnable parameters

    this%lambda = params(1)
    this%gamma  = params(2)
    this%bias   = params(3)

  end subroutine set_params_deepset
!###############################################################################


!###############################################################################
  pure function get_gradients_deepset(this, clip_method) result(gradients)
    !! Get the gradients for deep set layer
    use athena__clipper, only: clip_type
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(in) :: this
    !! Instance of the deep set layer
    type(clip_type), optional, intent(in) :: clip_method
    !! Method for clipping gradients
    real(real32), allocatable, dimension(:) :: gradients
    !! Gradients

    gradients = [ this%dl/this%batch_size, &
         this%dg/this%batch_size, &
         this%db/this%batch_size ]

    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)

  end function get_gradients_deepset
!###############################################################################


!###############################################################################
  subroutine set_gradients_deepset(this, gradients)
    !! Set the gradients for deep set layer
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(inout) :: this
    !! Instance of the deep set layer
    real(real32), dimension(..), intent(in) :: gradients
    !! Gradients

    select rank(gradients)
    rank(0)
       this%dl = gradients * this%batch_size
       this%dg = gradients * this%batch_size
       this%db = gradients * this%batch_size
    rank(1)
       this%dl = gradients(1) * this%batch_size
       this%dg = gradients(2) * this%batch_size
       this%db = gradients(3) * this%batch_size
    end select

  end subroutine set_gradients_deepset
!###############################################################################


!###############################################################################
  pure subroutine get_output_deepset(this, output)
    !! Get the output for deep set layer
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(in) :: this
    !! Instance of the deep set layer
    real(real32), allocatable, dimension(..), intent(out) :: output
    !! Output

    select rank(output)
    rank(1)
       output = reshape(this%output, [size(this%output)])
    rank(2)
       output = this%output
    end select

  end subroutine get_output_deepset
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine forward_rank(this, input)
    !! Forward propagation handler for deep set layer
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(inout) :: this
    !! Instance of the deep set layer
    real(real32), dimension(..), intent(in) :: input
    !! Input

    select rank(input); rank(2)
       call forward_2d(this, input)
    end select
  end subroutine forward_rank
!###############################################################################


!###############################################################################
  pure subroutine backward_rank(this, input, gradient)
    !! Backward propagation handler for deep set layer
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(inout) :: this
    !! Instance of the deep set layer
    real(real32), dimension(..), intent(in) :: input
    !! Input
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient

    select rank(input)
    rank(2)
       select rank(gradient)
       rank(2)
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
       input_shape, batch_size, &
       num_inputs, &
       lambda, gamma &
  ) result(layer)
    !! Set up the deep set layer
    implicit none

    ! Arguments
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: num_inputs
    !! Number of inputs
    integer, optional, intent(in) :: batch_size
    !! Batch size
    real(real32), optional, intent(in) :: lambda
    !! Lambda
    real(real32), optional, intent(in) :: gamma
    !! Gamma

    type(deepset_layer_type) :: layer
    !! Instance of the deep set layer


    layer%name = "deepset"
    layer%input_rank = 1


    !---------------------------------------------------------------------------
    ! Initialise batch size
    !---------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Set up lambda and gamma
    !---------------------------------------------------------------------------
    if(present(lambda))then
       layer%lambda = lambda
    else
       layer%lambda = 1._real32
    end if
    if(present(gamma))then
       layer%gamma = gamma
    else
       layer%gamma = 1._real32
    end if


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape))then
       call layer%init(input_shape=input_shape)
    elseif(present(num_inputs))then
       call layer%init(input_shape=[num_inputs])
    end if

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine init_deepset(this, input_shape, batch_size, verbose)
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(inout) :: this
    !! Instance of the deep set layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise number of inputs
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%output_shape = [this%num_outputs]


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_deepset
!###############################################################################


!###############################################################################
  subroutine set_batch_size_deepset(this, batch_size, verbose)
    !! Set the batch size for deep set layer
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(inout), target :: this
    !! Instance of the deep set layer
    integer, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(allocated(this%output)) deallocate(this%output)
       allocate(this%output( &
            this%input_shape(1), &
            this%batch_size), source=0._real32)
       if(allocated(this%dl)) deallocate(this%dl)
       allocate(this%dl(this%batch_size), source=0._real32)
       if(allocated(this%dg)) deallocate(this%dg)
       allocate(this%dg(this%batch_size), source=0._real32)
       if(allocated(this%di)) deallocate(this%di)
       allocate(this%di( &
            this%input_shape(1), &
            this%batch_size), source=0._real32)
    end if

  end subroutine set_batch_size_deepset
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_deepset(this, file)
    !! Print the deep set layer to file
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(in) :: this
    !! Instance of the deep set layer
    character(*), intent(in) :: file
    !! File name

    ! Local variables
    integer :: unit
    !! Unit number


    ! Open file with new unit
    !---------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'("DEEPSET")')
    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs
    write(unit,'(3X,"LAMBDA = ",F0.9)') this%lambda
    write(unit,'(3X,"GAMMA = ",F0.9)') this%gamma
    write(unit,'(3X,"BIAS = ",F0.9)') this%bias
    write(unit,'("END DEEPSET")')


    ! Close unit
    !---------------------------------------------------------------------------
    close(unit)

  end subroutine print_deepset
!###############################################################################


!###############################################################################
  subroutine read_deepset(this, unit, verbose)
    !! Read a deep set layer from a file
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, icount
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(inout) :: this
    !! Instance of the deep set layer
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level


    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: stat
    !! Status of read
    integer :: itmp1
    !! Temporary integer
    integer :: num_inputs, num_outputs
    !! Number of inputs and outputs
    real(real32) :: lambda, gamma, bias
    !! Lambda, gamma, and bias
    character(256) :: buffer, tag, err_msg
    !! Buffer for reading lines and tags


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    ! Loop over tags in layer card
    !--------------------------------------------------------------------------
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
       if(trim(adjustl(buffer)).eq."END DEEPSET")then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       ! Read parameters from save file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("NUM_INPUTS")
          call assign_val(buffer, num_inputs, itmp1)
       case("NUM_OUTPUTS")
          call assign_val(buffer, num_outputs, itmp1)
       case("LAMBDA")
          call assign_val(buffer, lambda, itmp1)
       case("GAMMA")
          call assign_val(buffer, gamma, itmp1)
       case("BIAS")
          call assign_val(buffer, bias, itmp1)
       case default
          !! Don't look for "e" due to scientific notation of numbers
          !! ... i.e. exponent (E+00)
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
    !--------------------------------------------------------------------------
    call layer%set_hyperparams( &
         num_outputs = num_outputs, &
         lambda = lambda, &
         gamma = gamma, &
         bias = bias, &
         verbose = verbose_ &
    )
    call layer%init(input_shape=[num_inputs])


    ! Check for end of layer card
    !--------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_deepset
!###############################################################################


!###############################################################################
  function read_deepset_layer(unit, verbose) result(layer)
    !! Read deepset layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the deepset layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=deepset_layer_type())
    call layer%read(unit, verbose=verbose_)

  end function read_deepset_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine forward_2d(this, input)
    !! Forward propagation for 2D input
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(inout) :: this
    !! Instance of the deep set layer
    real(real32), dimension(this%num_inputs, this%batch_size), &
         intent(in) :: input

    integer :: s

    !! Generate outputs from weights (lambda and gamma), biases, and inputs
    do concurrent(s=1:this%batch_size)
       this%output(:,s) = this%lambda * this%output(:,s) + &
            this%gamma * sum(input(:,s), dim=1) + this%bias
    end do

  end subroutine forward_2d
!###############################################################################


!###############################################################################
  pure subroutine backward_2d(this, input, gradient)
    !! Backward propagation for 2D input
    implicit none

    ! Arguments
    class(deepset_layer_type), intent(inout) :: this
    !! Instance of the deep set layer
    real(real32), dimension(this%num_inputs, this%batch_size), &
         intent(in) :: input
    !! Input values
    real(real32), dimension(this%num_outputs, this%batch_size), &
         intent(in) :: gradient
    !! Gradient values

    ! Local variables
    integer :: s
    !! Loop index


    do concurrent(s=1:this%batch_size)
       this%db(s) = sum(gradient(:,s))
       this%dl(s) = dot_product(input(:,s), gradient(:,s))
       this%dg(s)  = sum(gradient(:,s) * sum(input(:,s), dim=1))

       this%di(:,s) = this%lambda * gradient(:,s) + &
            this%gamma * sum(gradient(:,s), dim=1)
    end do

  end subroutine backward_2d
!###############################################################################

end module athena__deepset_layer
