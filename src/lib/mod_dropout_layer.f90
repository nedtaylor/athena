module athena__dropout_layer
  !! Module containing implementation of a dropout layer
  !!
  !! This module implements dropout regularisation, randomly zeroing elements
  !! during training to prevent overfitting and co-adaptation of neurons.
  !!
  !! Mathematical operation (training):
  !!   y_i = { 0                if r_i < p
  !!         { x_i / (1-p)      otherwise
  !!
  !! where r_i ~ U(0,1) is random, p is dropout probability
  !!
  !! Scaling by 1/(1-p) maintains expected value: E[y] = E[x]
  !!
  !! Inference: acts as identity (no dropout applied)
  !!   y = x
  !!
  !! Benefits: Prevents overfitting, ensemble effect, forces redundancy
  !! Typical p values: 0.2-0.5 (higher dropout for larger networks)
  !! Reference: Srivastava et al. (2014), JMLR
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: drop_layer_type, base_layer_type
  use diffstruc, only: array_type, operator(*)
  use athena__misc_types, only: &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use athena__diffstruc_extd, only: merge_over_channels
  implicit none


  private

  public :: dropout_layer_type
  public :: read_dropout_layer, create_from_onnx_dropout_layer


  type, extends(drop_layer_type) :: dropout_layer_type
     !! Type for dropout layer with overloaded procedures
     integer :: idx = 0
     !! Temporary index of sample (doesn't need to be accurate)
     integer :: num_masks
     !! Number of unique masks = number of samples in batch
     logical, allocatable, dimension(:,:) :: mask
     !! Mask for dropout
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_dropout
     !! Set hyperparameters for dropout layer
     procedure, pass(this) :: init => init_dropout
     !! Initialise dropout layer
     procedure, pass(this) :: print_to_unit => print_to_unit_dropout
     !! Print dropout layer to unit
     procedure, pass(this) :: read => read_dropout
     !! Read dropout layer from file

     procedure, pass(this) :: forward => forward_dropout
     !! Forward propagation derived type handler

     procedure, pass(this) :: generate_mask => generate_dropout_mask
     !! Generate dropout mask
  end type dropout_layer_type

  interface dropout_layer_type
     !! Interface for setting up the dropout layer
     module function layer_setup( &
          rate, num_masks, &
          input_shape) result(layer)
       !! Set up the dropout layer
       integer, intent(in) :: num_masks
       !! Number of unique masks
       real(real32), intent(in) :: rate
       !! Drop rate
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       type(dropout_layer_type) :: layer
       !! Instance of the dropout layer
     end function layer_setup
  end interface dropout_layer_type



contains

!###############################################################################
  module function layer_setup( &
       rate, num_masks, &
       input_shape) result(layer)
    !! Set up the dropout layer
    implicit none

    ! Arguments
    integer, intent(in) :: num_masks
    !! Number of unique masks
    real(real32), intent(in) :: rate
    !! Drop rate
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape

    type(dropout_layer_type) :: layer
    !! Instance of the dropout layer


    !---------------------------------------------------------------------------
    ! Initialise hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams(rate, num_masks)


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!###############################################################################


!###############################################################################
  pure subroutine set_hyperparams_dropout(this, rate, num_masks)
    !! Set hyperparameters for dropout layer
    implicit none

    ! Arguments
    class(dropout_layer_type), intent(inout) :: this
    !! Instance of the dropout layer
    real(real32), intent(in) :: rate
    !! Drop rate
    integer, intent(in) :: num_masks
    !! Number of unique masks

    this%name = "dropout"
    this%type = "drop"
    this%input_rank = 1
    this%output_rank = 1

    this%num_masks = num_masks
    this%rate = rate

  end subroutine set_hyperparams_dropout
!###############################################################################


!###############################################################################
  subroutine init_dropout(this, input_shape, verbose)
    !! Initialise dropout layer
    implicit none

    ! Arguments
    class(dropout_layer_type), intent(inout) :: this
    !! Instance of the dropout layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! Set up number of channels, width, height
    !---------------------------------------------------------------------------
    allocate(this%output_shape(2))
    this%output_shape = this%input_shape


    !---------------------------------------------------------------------------
    ! Allocate mask
    !---------------------------------------------------------------------------
    allocate(this%mask(this%input_shape(1), this%num_masks), source=.true.)


    !---------------------------------------------------------------------------
    ! Generate mask
    !---------------------------------------------------------------------------
    call this%generate_mask()


    !---------------------------------------------------------------------------
    ! Generate mask
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program( &
            "Graph input not supported for dropout layer" &
       )
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate( this%output(1,1) )

  end subroutine init_dropout
!###############################################################################


!###############################################################################
  subroutine generate_dropout_mask(this)
    !! Generate dropout mask
    implicit none

    ! Arguments
    class(dropout_layer_type), intent(inout) :: this
    !! Instance of the dropout layer

    ! Local variables
    real(real32), allocatable, dimension(:,:) :: mask_real
    !! Real mask

    ! Generate masks
    !---------------------------------------------------------------------------
    allocate(mask_real(size(this%mask,1), size(this%mask,2)))
    call random_number(mask_real)  !  Generate random values in [0..1]
    this%mask = mask_real > this%rate

    this%idx = 0

  end subroutine generate_dropout_mask
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_dropout(this, unit)
    !! Print dropout layer to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(dropout_layer_type), intent(in) :: this
    !! Instance of the dropout layer
    integer, intent(in) :: unit
    !! File unit


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    write(unit,'(3X,"RATE = ",F0.9)') this%rate
    write(unit,'(3X,"NUM_MASKS = ",I0)') this%num_masks

  end subroutine print_to_unit_dropout
!###############################################################################


!###############################################################################
  subroutine read_dropout(this, unit, verbose)
    !! Read dropout layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(dropout_layer_type), intent(inout) :: this
    !! Instance of the dropout layer
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
    integer :: num_masks
    !! Number of unique masks
    real(real32) :: rate
    !! Drop rate
    integer, dimension(1) :: input_shape
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
          call assign_vec(buffer, input_shape, itmp1)
       case("RATE")
          call assign_val(buffer, rate, itmp1)
       case("NUM_MASKS")
          call assign_val(buffer, num_masks, itmp1)
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
    call this%set_hyperparams(rate = rate, num_masks = num_masks)
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

  end subroutine read_dropout
!###############################################################################


!###############################################################################
  function read_dropout_layer(unit, verbose) result(layer)
    !! Read dropout layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the base layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=dropout_layer_type(rate=0._real32, num_masks=0))
    call layer%read(unit, verbose=verbose_)

  end function read_dropout_layer
!###############################################################################


!###############################################################################
  subroutine build_from_onnx_dropout( &
       this, node, initialisers, value_info, verbose &
  )
    !! Read ONNX attributes for dropout layer
    implicit none

    ! Arguments
    class(dropout_layer_type), intent(inout) :: this
    !! Instance of the dropout layer
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
    real(real32) :: rate
    !! Dropout rate
    integer :: num_masks
    !! Number of masks (batch size)
    character(256) :: val
    !! Attribute value

    ! Set default values
    rate = 0.5_real32
    num_masks = 1

    ! Parse ONNX attributes
    do i = 1, size(node%attributes)
       val = node%attributes(i)%val
       select case(trim(adjustl(node%attributes(i)%name)))
       case("ratio")
          read(val,*) rate
       case default
          ! Do nothing
          write(0,*) "WARNING: Unrecognised attribute in ONNX DROPOUT layer: ", &
               trim(adjustl(node%attributes(i)%name))
       end select
    end do

    ! Check size of initialisers is zero
    if(size(initialisers).ne.0)then
       write(0,*) "WARNING: initialisers not used for ONNX DROPOUT layer"
    end if

    call this%set_hyperparams( &
         rate = rate, &
         num_masks = num_masks &
    )

  end subroutine build_from_onnx_dropout
!###############################################################################


!###############################################################################
  function create_from_onnx_dropout_layer( &
       node, initialisers, value_info, verbose &
  ) result(layer)
    !! Build dropout layer from attributes and return layer
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info information
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the dropout layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose

    allocate(layer, source=dropout_layer_type(rate=0._real32, num_masks=0))
    call layer%build_from_onnx(node, initialisers, value_info, verbose=verbose_)

  end function create_from_onnx_dropout_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_dropout(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(dropout_layer_type), intent(inout) :: this
    !! Instance of the dropout layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    real(real32) :: rtmp1
    !! Temporary variable
    type(array_type), pointer :: ptr
    !! Pointer array


    rtmp1 = 1._real32 - this%rate
    select case(this%inference)
    case(.true.)
       ! Do not perform the drop operation
       ptr => input(1,1) * rtmp1
    case default
       ! Perform the drop operation
       this%idx = this%idx + 1

       rtmp1 = 1._real32 / rtmp1
       ptr => merge_over_channels( input(1,1), 0._real32, this%mask) * rtmp1
    end select
    call this%output(1,1)%assign_and_deallocate_source(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_dropout
!###############################################################################

end module athena__dropout_layer
