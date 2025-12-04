module athena__reshape_layer
  !! Module containing implementation of a reshape layer
  !!
  !! This module implements a general reshape layer that can transform tensors
  !! between arbitrary shapes while preserving the total number of elements.
  !! Unlike flatten (which only converts to 1D), reshape allows any target shape.
  !!
  !! Mathematical operation:
  !!   Reshape: (d1, d2, ..., dn) -> (d1', d2', ..., dm')
  !!   where: d1 * d2 * ... * dn = d1' * d2' * ... * dm'
  !!
  !! Examples:
  !!   - (28, 28) -> (784)          [flatten]
  !!   - (784) -> (28, 28)          [unflatten]
  !!   - (64, 32, 32) -> (64, 1024) [spatial to sequence]
  !!   - (100, 50) -> (10, 10, 50)  [add spatial dimension]
  !!
  !! Properties:
  !!   - No learnable parameters (pure reshape operation)
  !!   - Preserves all information (bijective mapping)
  !!   - No computation beyond memory reorganization
  !!   - Gradients flow unchanged (chain rule applies directly)
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: base_layer_type
  use diffstruc, only: array_type, reshape
  use athena__misc_types, only: &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  implicit none


  private

  public :: reshape_layer_type
  public :: read_reshape_layer, create_from_onnx_reshape_layer


  type, extends(base_layer_type) :: reshape_layer_type
     !! Type for reshape layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_reshape
     !! Set hyperparameters for reshape layer
     procedure, pass(this) :: init => init_reshape
     !! Initialise reshape layer
     procedure, pass(this) :: print_to_unit => print_to_unit_reshape
     !! Print reshape layer to unit
     procedure, pass(this) :: read => read_reshape
     !! Read reshape layer from file
     procedure, pass(this) :: build_from_onnx => build_from_onnx_reshape
     !! Build reshape layer from ONNX node and initialisers

     procedure, pass(this) :: forward => forward_reshape
     !! Forward propagation derived type handler

  end type reshape_layer_type

  interface reshape_layer_type
     !! Interface for setting up the reshape layer
     module function layer_setup( &
          output_shape, input_shape, verbose &
     ) result(layer)
       !! Set up the reshape layer
       integer, dimension(:), intent(in) :: output_shape
       !! Target output shape (excluding batch dimension)
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape (excluding batch dimension)
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(reshape_layer_type) :: layer
       !! Instance of the reshape layer
     end function layer_setup
  end interface reshape_layer_type



contains

!###############################################################################
  module function layer_setup( &
       output_shape, input_shape, verbose &
  ) result(layer)
    !! Set up the reshape layer
    implicit none

    ! Arguments
    integer, dimension(:), intent(in) :: output_shape
    !! Target output shape (excluding batch dimension)
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape (excluding batch dimension)
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(reshape_layer_type) :: layer
    !! Instance of the reshape layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams(output_shape, verbose_)


    !---------------------------------------------------------------------------
    ! Initialise layer
    !---------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape, verbose_)

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_reshape(this, output_shape, verbose)
    !! Set hyperparameters for reshape layer
    implicit none

    ! Arguments
    class(reshape_layer_type), intent(inout) :: this
    !! Instance of the reshape layer
    integer, dimension(:), intent(in) :: output_shape
    !! Output rank
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose

    this%type = "rshp"
    this%name = "reshape"
    this%input_rank = 0
    this%output_shape = output_shape
    this%output_rank = size(output_shape)

    if(verbose_ .gt. 0) write(*,'("  Setting up reshape layer")')

  end subroutine set_hyperparams_reshape
!###############################################################################


!###############################################################################
  subroutine init_reshape(this, input_shape, verbose)
    !! Initialise reshape layer
    implicit none

    ! Arguments
    class(reshape_layer_type), intent(inout) :: this
    !! Instance of the reshape layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: input_size, output_size
    !! Total number of elements
    integer :: i
    !! Loop index

    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Set input shape
    !---------------------------------------------------------------------------
    this%input_rank = size(input_shape)
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    allocate(this%input_shape, source=input_shape)


    !---------------------------------------------------------------------------
    ! Validate reshape compatibility
    !---------------------------------------------------------------------------
    input_size = product(input_shape)

    output_size = product(this%output_shape)

    if(input_size .ne. output_size)then
       write(*,'("ERROR: Reshape layer - incompatible shapes")')
       write(*,'("  Input shape has ",I0," elements")') input_size
       write(*,'("  Output shape has ",I0," elements")') output_size
       call stop_program("Reshape layer shape mismatch")
    end if


    !---------------------------------------------------------------------------
    ! Print layer info
    !---------------------------------------------------------------------------
    if(verbose_ .gt. 0)then
       write(*,'("  Reshape layer initialised")')
       write(*,'("    Input shape:  ",*(I0," x "))') this%input_shape
       write(*,'("    Output shape: ",*(I0," x "))') this%output_shape
    end if

  end subroutine init_reshape
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_reshape(this, unit)
    !! Print reshape layer to unit
    implicit none

    ! Arguments
    class(reshape_layer_type), intent(in) :: this
    !! Instance of the reshape layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    character(100) :: fmt
    !! Format string


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"INPUT_RANK = ",I0)') this%input_rank
    write(fmt,'("(3X,""INPUT_SHAPE ="",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    write(fmt,'("(3X,""OUTPUT_SHAPE ="",",I0,"(1X,I0))")') size(this%output_shape)
    write(unit,fmt) this%output_shape

  end subroutine print_to_unit_reshape
!###############################################################################


!###############################################################################
  subroutine read_reshape(this, unit, verbose)
    !! Read reshape layer from file
    use athena__tools_infile, only: assign_val, assign_vec, get_val
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(reshape_layer_type), intent(inout) :: this
    !! Instance of the reshape layer
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat, verbose_ = 0
    !! File status and verbosity level
    integer :: itmp1 = 0
    !! Temporary integer
    integer :: input_rank = 0
    !! Input rank
    integer, dimension(:), allocatable :: input_shape, output_shape
    !! Input shape
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message


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
          itmp1 = icount(get_val(buffer))
          allocate(input_shape(itmp1), source=0)
          call assign_vec(buffer, input_shape, itmp1)
       case("OUTPUT_SHAPE")
          itmp1 = icount(get_val(buffer))
          allocate(output_shape(itmp1), source=0)
          call assign_vec(buffer, output_shape, itmp1)
       case default
          ! Don't look for "e" due to scientific notation of numbers
          ! ... i.e. exponent (E+00)
          if( &
               scan( &
                    to_lower(trim(adjustl(buffer))),  &
                    'abcdfghijklmnopqrstuvwxyz' &
               ) .eq. 0 &
          )then
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

    if(.not.allocated(output_shape))then
       call stop_program('("Reshape layer missing OUTPUT_SHAPE")')
       return
    end if


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    call this%set_hyperparams(output_shape = output_shape, verbose = verbose_)
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


  end subroutine read_reshape
!###############################################################################


!###############################################################################
  function read_reshape_layer(unit, verbose) result(layer)
    !! Read reshape layer from file
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, intent(in), optional :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the reshape layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=reshape_layer_type(output_shape=[0]))
    call layer%read(unit, verbose=verbose_)

  end function read_reshape_layer
!###############################################################################


!###############################################################################
  subroutine build_from_onnx_reshape(this, node, initialisers, value_info, verbose)
    !! Build reshape layer from ONNX node and initialiser
    implicit none

    ! Arguments
    class(reshape_layer_type), intent(inout) :: this
    !! Instance of the reshape layer
    type(onnx_node_type), intent(in) :: node
    !! ONNX node
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialisers
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value infos
    integer, intent(in) :: verbose
    !! Verbosity level

  end subroutine build_from_onnx_reshape
!###############################################################################


!###############################################################################
  function create_from_onnx_reshape_layer( &
       node, initialisers, value_info, verbose &
  ) result(layer)
    !! Build reshape layer from ONNX node and initialiser
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! ONNX node
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialisers
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value infos
    integer, intent(in), optional :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the reshape layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=reshape_layer_type(output_shape=[0]))
    call layer%build_from_onnx(node, initialisers, value_info, verbose=verbose_)

  end function create_from_onnx_reshape_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_reshape(this, input)
    !! Forward propagation derived type handler
    implicit none

    ! Arguments
    class(reshape_layer_type), intent(inout) :: this
    !! Instance of the reshape layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input array

    type(array_type), pointer :: ptr => null()

    ! Reshape input
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    ptr => reshape(input(1,1), this%output_shape)
    call this%output(1,1)%assign_and_deallocate_source(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_reshape
!###############################################################################

end module athena__reshape_layer
