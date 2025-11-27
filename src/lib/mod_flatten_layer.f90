module athena__flatten_layer
  !! Module containing implementation of a 1D flattening layer
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: base_layer_type
  use diffstruc, only: array_type, pack
  use athena__misc_types, only: onnx_node_type, onnx_initialiser_type
  implicit none


  private

  public :: flatten_layer_type
  public :: read_flatten_layer, create_from_onnx_flatten_layer


  type, extends(base_layer_type) :: flatten_layer_type
     !! Type for 1D flattening layer with overloaded procedures
     integer :: num_outputs
     !! Number of outputs
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_flatten
     !! Set hyperparameters for flattening layer
     procedure, pass(this) :: init => init_flatten
     !! Initialise flattening layer
     procedure, pass(this) :: set_batch_size => set_batch_size_flatten
     !! Set batch size for flattening layer
     procedure, pass(this) :: print_to_unit => print_to_unit_flatten
     !! Print flatten layer to unit
     procedure, pass(this) :: read => read_flatten
     !! Read flattening layer from file
     procedure, pass(this) :: build_from_onnx => build_from_onnx_flatten
     !! Build flattening layer from ONNX node and initialisers

     procedure, pass(this) :: forward => forward_flatten
     !! Forward propagation derived type handler

  end type flatten_layer_type

  interface flatten_layer_type
     !! Interface for setting up the flattening layer
     module function layer_setup( &
          input_shape, batch_size, input_rank, verbose &
     ) result(layer)
       !! Set up the flattening layer
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: input_rank
       !! Input rank
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(flatten_layer_type) :: layer
       !! Instance of the flattening layer
     end function layer_setup
  end interface flatten_layer_type



contains

!###############################################################################
  module function layer_setup( &
       input_shape, batch_size, input_rank, verbose &
  ) result(layer)
    !! Set up the flattening layer
    implicit none

    ! Arguments
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: input_rank
    !! Input rank
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(flatten_layer_type) :: layer
    !! Instance of the flattening layer

    ! Local variables
    integer :: input_rank_ = 0
    !! Input rank
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    if(present(input_rank))then
       input_rank_ = input_rank
    elseif(present(input_shape))then
       input_rank_ = size(input_shape)
    else
       call stop_program( &
            "input_rank or input_shape must be provided to flatten layer" &
       )
       return
    end if
    call layer%set_hyperparams(input_rank_, verbose_)


    !---------------------------------------------------------------------------
    ! Initialise batch size
    !---------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_flatten(this, input_rank, verbose)
    !! Set hyperparameters for flattening layer
    implicit none

    ! Arguments
    class(flatten_layer_type), intent(inout) :: this
    !! Instance of the flattening layer
    integer, intent(in) :: input_rank
    !! Input rank
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "flatten"
    this%type = "flat"
    this%input_rank = input_rank
    this%output_rank = 1
  end subroutine set_hyperparams_flatten
!###############################################################################


!###############################################################################
  subroutine init_flatten(this, input_shape, batch_size, verbose)
    !! Initialise flattening layer
    implicit none

    ! Arguments
    class(flatten_layer_type), intent(inout) :: this
    !! Instance of the flattening layer
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
    ! Initialise input shape
    !---------------------------------------------------------------------------
    this%input_rank = size(input_shape)
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! Initialise output shape
    !---------------------------------------------------------------------------
    this%num_outputs = product(this%input_shape)
    if(allocated(this%output))then
       if(this%output(1,1)%allocated) call this%output(1,1)%deallocate()
    end if
    allocate(this%output(1,1))
    this%output_shape = [this%num_outputs]


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_flatten
!###############################################################################


!###############################################################################
  subroutine set_batch_size_flatten(this, batch_size, verbose)
    !! Set batch size for flattening layer
    implicit none

    ! Arguments
    class(flatten_layer_type), intent(inout), target :: this
    !! Instance of the flattening layer
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
       if(this%use_graph_input)then
          call stop_program( &
               "Graph input not supported for flatten layer" &
          )
          return
       else
          if(allocated(this%output)) deallocate(this%output)
          allocate( this%output(1,1) )
          call this%output(1,1)%allocate( &
               array_shape = [ this%num_outputs, this%batch_size ], &
               source=0._real32 &
          )
       end if
    end if

  end subroutine set_batch_size_flatten
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_flatten(this, unit)
    !! Print flatten layer to unit
    implicit none

    ! Arguments
    class(flatten_layer_type), intent(in) :: this
    !! Instance of the flatten layer
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

  end subroutine print_to_unit_flatten
!###############################################################################


!###############################################################################
  subroutine read_flatten(this, unit, verbose)
    !! Read flattening layer from file
    use athena__tools_infile, only: assign_val, assign_vec, get_val
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(flatten_layer_type), intent(inout) :: this
    !! Instance of the flattening layer
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
    integer, dimension(:), allocatable :: input_shape
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
       case("INPUT_RANK")
          call assign_val(buffer, input_rank, itmp1)
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

    if(allocated(input_shape))then
       if(input_rank.eq.0)then
          input_rank = size(input_shape)
       elseif(input_rank.ne.size(input_shape))then
          write(err_msg,'("input_rank (",I0,") does not match input_shape (",I0,")")') &
               input_rank, size(input_shape)
          call stop_program(err_msg)
          return
       end if
    elseif(input_rank.eq.0)then
       write(err_msg,'("input_rank must be provided if input_shape is not")')
       call stop_program(err_msg)
       return
    end if


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    call this%set_hyperparams(input_rank = input_rank, verbose = verbose_)
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

  end subroutine read_flatten
!###############################################################################


!###############################################################################
  function read_flatten_layer(unit, verbose) result(layer)
    !! Read flattening layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the flattening layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=flatten_layer_type(input_rank=1))
    call layer%read(unit, verbose=verbose_)

  end function read_flatten_layer
!###############################################################################


!###############################################################################
  subroutine build_from_onnx_flatten(this, node, initialisers, verbose )
    !! Read ONNX attributes for flattening layer
    implicit none

    ! Arguments
    class(flatten_layer_type), intent(inout) :: this
    !! Instance of the flattening layer
    type(onnx_node_type), intent(in) :: node
    !! Instance of ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! Instance of ONNX initialiser information
    integer, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

  end subroutine build_from_onnx_flatten
!###############################################################################


!###############################################################################
  function create_from_onnx_flatten_layer(node, initialisers, verbose) result(layer)
    !! Build flattening layer from attributes and return layer
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! Instance of ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! Instance of ONNX initialiser information
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the flattening layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=flatten_layer_type(input_rank=0))
    call layer%build_from_onnx(node, initialisers, verbose=verbose_)

  end function create_from_onnx_flatten_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_flatten(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(flatten_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    type(array_type), pointer :: ptr => null()


    ! Flatten input
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    ptr => pack(input(1,1), dim = 1)
    call this%output(1,1)%assign_and_deallocate_source(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_flatten
!###############################################################################

end module athena__flatten_layer
