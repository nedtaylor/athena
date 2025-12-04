module athena__actv_layer
  !! Module containing implementation of the activation layer
  !!
  !! This module wraps various activation functions into a layer type,
  !! applying element-wise non-linear transformations to inputs.
  !!
  !! Mathematical operation:
  !!   y = σ(x)
  !!
  !! where σ is one of: relu, sigmoid, tanh, softmax, linear, etc.
  !!
  !! Properties:
  !!   - No learnable parameters (fixed non-linearity)
  !!   - Element-wise operation (preserves shape)
  !!   - Enables networks to learn non-linear functions
  !!   - Choice of activation affects gradient flow and convergence
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: base_actv_type, &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use diffstruc, only: array_type
  implicit none


  private

  public :: actv_layer_type
  public :: read_actv_layer, create_from_onnx_actv_layer


  type, extends(base_layer_type) :: actv_layer_type
     !! Layer type for activation layers
     class(base_actv_type), allocatable :: activation
     !! Activation function
   contains
     procedure, pass(this) :: set_rank => set_rank_actv
     !! Set the input and output ranks of the layer
     procedure, pass(this) :: set_hyperparams => set_hyperparams_actv
     !! Set hyperparameters
     procedure, pass(this) :: init => init_actv
     !! Initialise layer
     procedure, pass(this) :: print_to_unit => print_to_unit_actv
     !! Print layer to unit
     procedure, pass(this) :: read => read_actv
     !! Read layer from file
     procedure, pass(this) :: build_from_onnx => build_from_onnx_actv
     !! Build activation layer from ONNX node and initialiser

     procedure, pass(this) :: forward => forward_actv
     !! Forward propagation derived type handler

  end type actv_layer_type


  interface actv_layer_type
     !! Interface for the activation layer type
     module function layer_setup( &
          activation, &
          input_shape, &
          verbose &
     ) result(layer)
       !! Set up the activation layer
       class(*), intent(in) :: activation
       !! Activation function
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(actv_layer_type) :: layer
       !! Instance of the activation layer
     end function layer_setup
  end interface actv_layer_type



contains

!###############################################################################
  module function layer_setup( &
       activation, &
       input_shape, &
       verbose &
  ) result(layer)
    !! Set up the activation layer
    use athena__activation,  only: activation_setup
    implicit none

    ! Arguments
    class(*), intent(in) :: activation
    !! Activation function
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    type(actv_layer_type) :: layer
    !! Instance of the activation layer

    ! Local variables
    class(base_actv_type), allocatable :: activation_
    !! Activation function
    integer :: verbose_
    !! Verbosity level


    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Set activation function
    !---------------------------------------------------------------------------
    activation_ = activation_setup(activation)


    !---------------------------------------------------------------------------
    ! set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         activation = activation_, &
         verbose = verbose_ &
    )


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
       activation, &
       input_rank, &
       verbose &
  )
    !! Set hyperparameters for activation layer
    use athena__activation,  only: activation_setup
    use coreutils, only: to_lower
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout) :: this
    !! Instance of the activation layer
    integer, optional, intent(in) :: input_rank
    !! Input rank
    class(base_actv_type), allocatable, intent(in) :: activation
    !! Activation function
    integer, optional, intent(in) :: verbose
    !! Verbosity level


    this%name = "actv"
    this%type = "actv"
    this%input_rank = 0
    if(present(input_rank)) this%input_rank = input_rank
    this%output_rank = this%input_rank
    if(.not.allocated(activation))then
       this%activation = activation_setup("none")
    else
       this%activation = activation
    end if
    this%subtype = trim(to_lower(this%activation%name))

    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("ACTV activation function: ",A)') &
               trim(this%activation%name)
       end if
    end if

  end subroutine set_hyperparams_actv
!###############################################################################


!###############################################################################
  subroutine init_actv(this, input_shape, verbose)
    !! Initialise activation layer
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout) :: this
    !! Instance of the activation layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    this%input_rank = size(input_shape, dim=1)
    this%output_rank = this%input_rank
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%output_shape = this%input_shape


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program( &
            "Graph input not supported for activation layer" &
       )
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))

  end subroutine init_actv
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
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(actv_layer_type), intent(in) :: this
    !! Instance of the activation layer
    integer, intent(in) :: unit
    !! File unit


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if

  end subroutine print_to_unit_actv
!###############################################################################


!###############################################################################
  subroutine read_actv(this, unit, verbose)
    !! Read activation layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use coreutils, only: to_lower, to_upper
    use athena__activation, only: read_activation
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
    integer :: itmp1, iline
    !! Temporary integer and line counter
    character(20) :: activation_name
    !! Activation function name
    class(base_actv_type), allocatable :: activation
    !! Activation function
    integer, dimension(3) :: input_shape
    !! Input shape
    character(256) :: buffer, tag, err_msg
    !! Buffer for reading lines, tag for identifying lines, error message


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
    iline = 0
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
       iline = iline + 1

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       ! Read parameters from save file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("INPUT_SHAPE")
          call assign_vec(buffer, input_shape, itmp1)
       case("ACTIVATION")
          iline = iline - 1
          backspace(unit)
          activation = read_activation(unit, iline)
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
         activation = activation &
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


!###############################################################################
  subroutine build_from_onnx_actv(this, node, initialisers, value_info, verbose )
    !! Read ONNX attributes for activation layer
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout) :: this
    !! Instance of the activation layer
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info
    integer, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

  end subroutine build_from_onnx_actv
!###############################################################################


!###############################################################################
  function create_from_onnx_actv_layer(node, initialisers, value_info, verbose) &
       result(layer)
    !! Build activation layer from attributes and return layer
    use coreutils, only: to_lower
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the activation layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=actv_layer_type(to_lower(trim(node%op_type))))
    call layer%build_from_onnx(node, initialisers, value_info, verbose=verbose_)

  end function create_from_onnx_actv_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_actv(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(actv_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    integer :: i, s
    !! Loop indices
    type(array_type), pointer :: ptr
    !! Pointer array

    do s = 1, size(input, 2)
       do i = 1, size(input, 1)
          ptr => this%activation%apply(input(i,s))
          call this%output(i,s)%assign_and_deallocate_source(ptr)
       end do
    end do

  end subroutine forward_actv
!###############################################################################

end module athena__actv_layer
