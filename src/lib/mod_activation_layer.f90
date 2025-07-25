module athena__actv_layer
  !! Module containing implementation of the activation layer
  !!
  !! This module wraps different activation functions into a layer type
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: activation_type, &
       array1d_type, &
       array2d_type, &
       array3d_type, &
       array4d_type, &
       array5d_type
  implicit none


  private

  public :: actv_layer_type
  public :: read_actv_layer


  type, extends(base_layer_type) :: actv_layer_type
     !! Layer type for activation layers
     class(activation_type), allocatable :: transfer
     !! Activation function
   contains
     procedure, pass(this) :: set_rank => set_rank_actv
     !! Set the input and output ranks of the layer
     procedure, pass(this) :: set_hyperparams => set_hyperparams_actv
     !! Set hyperparameters
     procedure, pass(this) :: init => init_actv
     !! Initialise layer
     procedure, pass(this) :: set_batch_size => set_batch_size_actv
     !! Set batch size
     procedure, pass(this) :: print_to_unit => print_to_unit_actv
     !! Print layer to unit
     procedure, pass(this) :: read => read_actv
     !! Read layer from file
     procedure, pass(this) :: forward  => forward_rank
     !! Forward propagation
     procedure, pass(this) :: backward => backward_rank
     !! Backward propagation
     procedure, pass(this), private :: forward_assumed_rank
     !! Forward propagation assumed rank handler
     procedure, pass(this), private :: backward_assumed_rank
     !! Backward propagation assumed rank handler
  end type actv_layer_type


  interface actv_layer_type
     !! Interface for the activation layer type
     module function layer_setup( &
          activation_function, activation_scale, &
          input_shape, batch_size, &
          verbose &
     ) result(layer)
       !! Set up the activation layer
       character(*), intent(in) :: activation_function
       !! Activation function name
       real(real32), optional, intent(in) :: activation_scale
       !! Activation function scale
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(actv_layer_type) :: layer
       !! Instance of the activation layer
     end function layer_setup
  end interface actv_layer_type



contains

!###############################################################################
  subroutine forward_rank(this, input)
    !! Forward propagation handler for activation layer
    implicit none
    class(actv_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input

    call this%forward_assumed_rank(input)
  end subroutine forward_rank
!###############################################################################


!###############################################################################
  subroutine backward_rank(this, input, gradient)
    !! Backward propagation handler for activation layer
    implicit none
    class(actv_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input
    real(real32), dimension(..), intent(in) :: gradient

    call this%backward_assumed_rank(input, gradient)
  end subroutine backward_rank
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       activation_function, activation_scale, &
       input_shape, batch_size, &
       verbose &
  ) result(layer)
    !! Set up the activation layer
    use athena__activation,  only: activation_setup
    implicit none

    ! Arguments
    character(*), intent(in) :: activation_function
    !! Activation function name
    real(real32), optional, intent(in) :: activation_scale
    !! Activation function scale
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    type(actv_layer_type) :: layer
    !! Instance of the activation layer

    ! Local variables
    real(real32) :: activation_scale_
    !! Activation function scale
    integer :: verbose_
    !! Verbosity level


    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! set hyperparameters
    !---------------------------------------------------------------------------
    activation_scale_ = 1._real32
    if(present(activation_scale)) activation_scale_ = activation_scale
    call layer%set_hyperparams( &
         activation_function = activation_function, &
         activation_scale = activation_scale_, &
         verbose = verbose_ &
    )


    !---------------------------------------------------------------------------
    ! initialise batch size
    !---------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape)) call layer%init( &
         input_shape=input_shape, &
         verbose=verbose_ &
    )

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_actv( &
       this, &
       activation_function, &
       activation_scale, &
       input_rank, &
       verbose &
  )
    !! Set hyperparameters for activation layer
    use athena__activation,  only: activation_setup
    use athena__misc, only: to_lower
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout) :: this
    !! Instance of the activation layer
    integer, optional, intent(in) :: input_rank
    !! Input rank
    character(*), intent(in) :: activation_function
    !! Activation function name
    real(real32), intent(in) :: activation_scale
    !! Activation function scale
    integer, optional, intent(in) :: verbose
    !! Verbosity level


    this%name = "actv"
    this%type = "actv"
    this%input_rank = 0
    if(present(input_rank)) this%input_rank = input_rank
    this%output_rank = this%input_rank
    if(allocated(this%transfer)) deallocate(this%transfer)
    allocate(this%transfer, &
         source=activation_setup(activation_function, activation_scale) &
    )
    this%subtype = trim(to_lower(activation_function))

    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("ACTV activation function: ",A)') &
               trim(activation_function)
       end if
    end if

  end subroutine set_hyperparams_actv
!###############################################################################


!###############################################################################
  subroutine init_actv(this, input_shape, batch_size, verbose)
    !! Initialise activation layer
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout) :: this
    !! Instance of the activation layer
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
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    this%input_rank = size(input_shape, dim=1)
    this%output_rank = this%input_rank
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%output_shape = this%input_shape


    !---------------------------------------------------------------------------
    ! initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_actv
!###############################################################################


!###############################################################################
  subroutine set_batch_size_actv(this, batch_size, verbose)
    !! Set batch size for activation layer
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout), target :: this
    !! Instance of the activation layer
    integer, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    verbose_ = 0
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(allocated(this%output)) deallocate(this%output)
       if(this%use_graph_input)then
          allocate(this%output(2,this%batch_size), source=array2d_type())
          call stop_program( &
               "Graph input not supported for activation layer" &
          )
          return
       else
          select case(size(this%input_shape))
          case(1)
             this%input_rank = 1
             allocate(this%output(1,1), source=array2d_type())
             call this%output(1,1)%allocate( &
                  array_shape = [ &
                       this%input_shape(1), this%batch_size &
                  ], &
                  source=0._real32 &
             )
          case(2)
             this%input_rank = 2
             allocate(this%output(1,1), source=array3d_type())
             call this%output(1,1)%allocate( &
                  array_shape = [ &
                       this%input_shape(1), &
                       this%input_shape(2), &
                       this%batch_size ], &
                  source=0._real32 &
             )
          case(3)
             this%input_rank = 3
             allocate(this%output(1,1), source=array4d_type())
             call this%output(1,1)%allocate( &
                  array_shape = [ &
                       this%input_shape(1), &
                       this%input_shape(2), &
                       this%input_shape(3), this%batch_size &
                  ], &
                  source=0._real32 &
             )
          case(4)
             this%input_rank = 4
             allocate(this%output(1,1), source=array5d_type())
             call this%output(1,1)%allocate( &
                  array_shape = [ &
                       this%input_shape(1), &
                       this%input_shape(2), &
                       this%input_shape(3), &
                       this%input_shape(4), this%batch_size &
                  ], &
                  source=0._real32 &
             )
          case default
             call stop_program('Activation layer only supports input ranks 1-4')
             return
          end select
          allocate(this%di(1,1), source=this%output(1,1))
       end if
    end if

  end subroutine set_batch_size_actv
!###############################################################################


!###############################################################################
  subroutine set_rank_actv(this, input_rank, output_rank)
    !! Set the input and output ranks of the activation layer
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout) :: this
    !! Instance of the activation layer
    integer, intent(in) :: input_rank
    !! Input rank
    integer, intent(in) :: output_rank
    !! Output rank

    this%input_rank = input_rank
    this%output_rank = output_rank
    if(this%input_rank.ne.this%output_rank)then
       call stop_program("Warning: Activation layer input and output ranks differ")
       return
    end if
    if(this%input_rank.lt.1) then
       write(*,*) "Error: Activation layer input rank must be at least 1"
       call stop_program("Invalid activation layer input rank")
       return
    end if
    if(this%output_rank.lt.1) then
       write(*,*) "Error: Activation layer output rank must be at least 1"
       call stop_program("Invalid activation layer output rank")
       return
    end if

  end subroutine set_rank_actv
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_actv(this, unit)
    !! Print activation layer to unit
    use athena__misc, only: to_upper
    implicit none

    ! Arguments
    class(actv_layer_type), intent(in) :: this
    !! Instance of the activation layer
    integer, intent(in) :: unit
    !! File unit


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    write(unit,'(3X,"ACTIVATION_FUNCTION = ",A)') this%transfer%name
    write(unit,'(3X,"ACTIVATION_SCALE = ",1ES20.10)') this%transfer%scale

  end subroutine print_to_unit_actv
!###############################################################################


!###############################################################################
  subroutine read_actv(this, unit, verbose)
    !! Read activation layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout) :: this
    !! Instance of the activation layer
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: stat
    !! File status
    integer :: itmp1
    !! Temporary integer
    real(real32) :: activation_scale
    !! Activation scale
    integer, dimension(3) :: input_shape
    !! Input shape
    character(256) :: buffer, tag, err_msg
    !! Buffer for reading lines, tag for identifying lines, error message
    character(20) :: activation_function
    !! Activation function name


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

       ! Read parameters from save file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("INPUT_SHAPE")
          call assign_vec(buffer, input_shape, itmp1)
       case("ACTIVATION_FUNCTION")
          call assign_val(buffer, activation_function, itmp1)
       case("ACTIVATION_SCALE")
          call assign_val(buffer, activation_scale, itmp1)
       case default
          !! don't look for "e" due to scientific notation of numbers
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
    !---------------------------------------------------------------------------
    call this%set_hyperparams( &
         activation_function = activation_function, &
         activation_scale = activation_scale &
    )
    call this%init(input_shape = input_shape)


    ! Check for end of layer card
    !---------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_actv
!###############################################################################


!###############################################################################
  function read_actv_layer(unit, verbose) result(layer)
    !! Read activation layer from file
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the activation layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=actv_layer_type("none"))
    call layer%read(unit, verbose=verbose_)

  end function read_actv_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_assumed_rank(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout) :: this
    !! Instance of the activation layer
    real(real32), dimension(..), intent(in), target :: input
    !! Input values

    ! Local variables
    real(real32), pointer :: input_ptr(:,:)
    !! Input pointer

    select rank(input)
    rank(1)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(2)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(3)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(4)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(5)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    end select
    this%output(1,1)%val(:,:) = this%transfer%activate(input_ptr)

  end subroutine forward_assumed_rank
!###############################################################################


!###############################################################################
  subroutine backward_assumed_rank(this, input, gradient)
    !! Backward propagation
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout) :: this
    !! Instance of the activation layer
    real(real32), dimension(..), intent(in), target :: input
    !! Input values
    real(real32), dimension(..), intent(in), target :: gradient
    !! Gradient values

    ! Local variables
    real(real32), pointer :: input_ptr(:,:), gradient_ptr(:,:)
    !! Input and gradient pointers

    select rank(gradient)
    rank(1)
       gradient_ptr(1:product(this%input_shape),1:this%batch_size) => gradient
    rank(2)
       gradient_ptr(1:product(this%input_shape),1:this%batch_size) => gradient
    rank(3)
       gradient_ptr(1:product(this%input_shape),1:this%batch_size) => gradient
    rank(4)
       gradient_ptr(1:product(this%input_shape),1:this%batch_size) => gradient
    rank(5)
       gradient_ptr(1:product(this%input_shape),1:this%batch_size) => gradient
    end select

    select rank(input)
    rank(1)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(2)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(3)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(4)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(5)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    end select
    this%di(1,1)%val(:,:) = &
         gradient_ptr * this%transfer%differentiate(input_ptr)

  end subroutine backward_assumed_rank
!###############################################################################

end module athena__actv_layer
