module athena__pad2d_layer
  !! Module containing implementation of a 2D padding layer
  !!
  !! This module implements padding for 2D spatial data (images), adding values
  !! around borders to control output dimensions or prepare for convolution.
  !!
  !! Operation: Extends spatial dimensions at boundaries
  !!   Adds p_top, p_bottom rows and p_left, p_right columns
  !!
  !! Padding modes:
  !!   - 'constant': pad with fixed value (typically 0)
  !!   - 'replicate': repeat edge values
  !!   - 'reflect': mirror values at boundaries
  !!
  !! Common use: Preserve spatial dimensions through convolution,
  !! handle boundary effects in CNNs
  !! Shape: (W,H,C) -> (W+p_l+p_r, H+p_t+p_b, C)
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: pad_layer_type, base_layer_type
  use diffstruc, only: array_type
  use athena__diffstruc_extd, only: pad2d
  use athena__misc_types, only: &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  implicit none


  private

  public :: pad2d_layer_type
  public :: read_pad2d_layer


  type, extends(pad_layer_type) :: pad2d_layer_type
     !! Type for 2D padding layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_pad2d
     !! Set hyperparameters for 2D padding layer
     procedure, pass(this) :: read => read_pad2d
     !! Read 2D padding layer from file

     procedure, pass(this) :: forward => forward_pad2d
     !! Forward propagation derived type handler

  end type pad2d_layer_type

  interface pad2d_layer_type
     !! Interface for setting up the 2D padding layer
     module function layer_setup( &
          padding, method, &
          input_shape, &
          verbose &
     ) result(layer)
       !! Set up the 2D padding layer
       integer, dimension(:), intent(in) :: padding
       !! Padding sizes
       character(*), intent(in) :: method
       !! Padding method
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(pad2d_layer_type) :: layer
       !! Instance of the 2D padding layer
     end function layer_setup
  end interface pad2d_layer_type



contains

!###############################################################################
  module function layer_setup( &
       padding, method, &
       input_shape, &
       verbose) result(layer)
    !! Set up the 2D padding layer
    implicit none

    ! Arguments
    integer, dimension(:), intent(in) :: padding
    !! Padding sizes
    character(*), intent(in) :: method
    !! Padding method
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(pad2d_layer_type) :: layer
    !! Instance of the 2D padding layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer, dimension(2) :: padding_2d
    !! 2D padding sizes

    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Initialise padding sizes
    !---------------------------------------------------------------------------
    select case(size(padding))
    case(1)
       padding_2d = [padding(1), padding(1)]
    case(2)
       padding_2d = padding
    case default
       call stop_program("Invalid padding size")
    end select


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams(padding=padding_2d, method=method, verbose=verbose_)


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_pad2d(this, padding, method, verbose)
    !! Set hyperparameters for 2D padding layer
    use coreutils, only: to_lower
    implicit none

    ! Arguments
    class(pad2d_layer_type), intent(inout) :: this
    !! Instance of the 2D padding layer
    integer, dimension(2), intent(in) :: padding
    !! Padding sizes
    character(*), intent(in) :: method
    !! Padding method
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "pad2d"
    this%type = "pad"
    this%input_rank = 3
    this%output_rank = 3
    this%pad = padding
    if(allocated(this%facets)) deallocate(this%facets)
    allocate(this%facets(this%input_rank - 1))
    this%facets(1)%rank = 2
    this%facets(1)%nfixed_dims = 1
    this%facets(2)%rank = 2
    this%facets(2)%nfixed_dims = 2
    select case(trim(adjustl(to_lower(method))))
    case("valid", "none")
       this%imethod = 0
    case("same", "zero", "constant", "const")
       this%imethod = 1
    case("full")
       this%imethod = 2
    case("circular", "circ")
       this%imethod = 3
    case("reflection", "reflect", "refl")
       this%imethod = 4
    case("replication", "replicate", "copy", "repl")
       this%imethod = 5
    case default
       call stop_program("Unrecognised padding method :"//method)
       return
    end select
    this%method = trim(adjustl(to_lower(method)))

  end subroutine set_hyperparams_pad2d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine read_pad2d(this, unit, verbose)
    !! Read 2D padding layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(pad2d_layer_type), intent(inout) :: this
    !! Instance of the 2D padding layer
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
    integer, dimension(2) :: padding
    !! Padding sizes
    integer, dimension(3) :: input_shape
    !! Input shape
    character(20) :: method
    !! Padding method
    character(256) :: buffer, tag, err_msg
    !! Buffer for reading lines, tag for identifying lines, error message


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
       case("PADDING")
          call assign_vec(buffer, padding, itmp1)
       case("METHOD")
          call assign_val(buffer, method, itmp1)
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
    call this%set_hyperparams(padding=padding, method=method, verbose=verbose_)
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

  end subroutine read_pad2d
!###############################################################################


!###############################################################################
  function read_pad2d_layer(unit, verbose) result(layer)
    !! Read 2D padding layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 2D padding layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=pad2d_layer_type(padding=[0,0], method="none"))
    call layer%read(unit, verbose=verbose_)

  end function read_pad2d_layer
!###############################################################################


!###############################################################################
  subroutine build_from_onnx_pad2d( &
       this, node, initialisers, value_info, verbose &
  )
    !! Read ONNX attributes for 2D padding layer
    implicit none

    ! Arguments
    class(pad2d_layer_type), intent(inout) :: this
    !! Instance of the 2D padding layer
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info
    integer, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: i
    !! Loop index
    integer, dimension(2) :: padding
    !! Padding sizes
    character(256) :: val, mode
    !! Attribute value and mode

    ! Set default values
    padding = 0
    mode = "constant"

    ! Parse ONNX attributes
    do i = 1, size(node%attributes)
       val = node%attributes(i)%val
       select case(trim(adjustl(node%attributes(i)%name)))
       case("pads")
          read(val,*) padding
       case("mode")
          mode = trim(adjustl(val))
       case default
          ! Do nothing
          write(0,*) "WARNING: Unrecognised attribute in ONNX PAD2D &
               &layer: ", trim(adjustl(node%attributes(i)%name))
       end select
    end do

    ! Check size of initialisers
    if(size(initialisers).gt.0)then
       write(0,*) "WARNING: initialisers found for ONNX PAD2D layer"
    end if

    call this%set_hyperparams( &
         padding = padding, &
         method = mode, &
         verbose = verbose &
    )

  end subroutine build_from_onnx_pad2d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_pad2d(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(pad2d_layer_type), intent(inout) :: this
    !! Instance of the 2D padding layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    type(array_type), pointer :: ptr
    !! Pointer array


    call this%output(1,1)%zero_grad()
    ptr => pad2d(input(1,1), this%facets, this%pad, this%imethod)
    call this%output(1,1)%assign_and_deallocate_source(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_pad2d
!###############################################################################

end module athena__pad2d_layer
