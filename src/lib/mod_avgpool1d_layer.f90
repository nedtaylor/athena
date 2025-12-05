module athena__avgpool1d_layer
  !! Module containing implementation of a 1D average pooling layer
  !!
  !! This module implements 1D average pooling for downsampling sequential data
  !! by computing mean values within pooling windows.
  !!
  !! Mathematical operation:
  !!   output[i,k] = (1/pool_size) * sum_{m∈[0,pool_size)} input[i*stride+m, k]
  !!
  !! where:
  !!   i is the output position along sequence
  !!   k is the channel index
  !!   pool_size is the pooling window length
  !!   stride controls the step size
  !!
  !! Smoother downsampling than max pooling, preserves more information.
  !! Shape: (length, channels) -> (length//stride, channels)
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: pool_layer_type, base_layer_type
  use athena__pad1d_layer, only: pad1d_layer_type
  use diffstruc, only: array_type
  use athena__misc_types, only: &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use athena__diffstruc_extd, only: avgpool1d
  implicit none


  private

  public :: avgpool1d_layer_type
  public :: read_avgpool1d_layer


  type, extends(pool_layer_type) :: avgpool1d_layer_type
     !! Type for 1D average pooling layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_avgpool1d
     !! Set hyperparameters for 1D average pooling layer
     procedure, pass(this) :: read => read_avgpool1d
     !! Read 1D average pooling layer from file

     procedure, pass(this) :: forward => forward_avgpool1d
     !! Forward propagation derived type handler

  end type avgpool1d_layer_type

  interface avgpool1d_layer_type
     !! Interface for setting up the 1D average pooling layer
     module function layer_setup( input_shape, &
          pool_size, stride, padding, verbose ) result(layer)
       !! Set up the 1D average pooling layer
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, dimension(..), optional, intent(in) :: pool_size
       !! Pool size
       integer, dimension(..), optional, intent(in) :: stride
       !! Stride
       character(*), optional, intent(in) :: padding
       !! Padding
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(avgpool1d_layer_type) :: layer
       !! Instance of the 1D average pooling layer
     end function layer_setup
  end interface avgpool1d_layer_type



contains

!###############################################################################
  module function layer_setup( &
       input_shape, &
       pool_size, stride, padding, verbose) result(layer)
    !! Set up the 1D average pooling layer
    implicit none

    ! Arguments
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, dimension(..), optional, intent(in) :: pool_size
    !! Pool size
    integer, dimension(..), optional, intent(in) :: stride
    !! Stride
    character(*), optional, intent(in) :: padding
    !! Padding
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(avgpool1d_layer_type) :: layer
    !! Instance of the 1D average pooling layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer, dimension(1) :: pool_size_, stride_
    !! Pool size and stride
    character(len=20) :: padding_
    !! Padding

    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Set up pool size
    !---------------------------------------------------------------------------
    if(present(pool_size))then
       select rank(pool_size)
       rank(0)
          pool_size_ = pool_size
       rank(1)
          pool_size_ = pool_size
       end select
    else
       pool_size_ = 2
    end if


    !---------------------------------------------------------------------------
    ! Set up stride
    !---------------------------------------------------------------------------
    if(present(stride))then
       select rank(stride)
       rank(0)
          stride_ = stride
       rank(1)
          stride_ = stride
       end select
    else
       stride_ = 2
    end if


    !---------------------------------------------------------------------------
    ! Set up padding
    !---------------------------------------------------------------------------
    if(present(padding))then
       padding_ = padding
    else
       padding_ = "valid"
    end if


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         pool_size=pool_size_, stride=stride_, &
         padding=padding_, verbose=verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_avgpool1d( &
       this, pool_size, stride, padding, verbose &
  )
    !! Set hyperparameters for 1D average pooling layer
    use coreutils, only: to_lower
    implicit none

    ! Arguments
    class(avgpool1d_layer_type), intent(inout) :: this
    !! Instance of the 1D average pooling layer
    integer, dimension(1), intent(in) :: pool_size
    !! Pool size
    integer, dimension(1), intent(in) :: stride
    !! Stride
    character(*), optional, intent(in) :: padding
    !! Padding
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    character(len=20) :: padding_

    this%name = "avgpool1d"
    this%type = "pool"
    this%subtype = "avg"
    this%input_rank = 2
    this%output_rank = 2
    if(allocated(this%pool)) deallocate(this%pool)
    if(allocated(this%strd)) deallocate(this%strd)
    allocate( &
         this%pool(this%input_rank-1), &
         this%strd(this%input_rank-1) &
    )
    this%pool = pool_size
    this%strd = stride

    ! Handle padding
    if(present(padding))then
       padding_ = trim(adjustl(padding))
    else
       padding_ = "valid"
    end if

    select case(trim(adjustl(to_lower(padding_))))
    case("valid", "none", "")
    case default
       this%pad_layer = pad1d_layer_type( &
            padding = [ (this%pool-1)/2 ], &
            method = padding_ &
       )
    end select

  end subroutine set_hyperparams_avgpool1d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine read_avgpool1d(this, unit, verbose)
    !! Read 1D average pooling layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(avgpool1d_layer_type), intent(inout) :: this
    !! Instance of the 1D average pooling layer
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
    integer, dimension(1) :: pool_size, stride
    !! Pool size and stride
    integer, dimension(2) :: input_shape
    !! Input shape
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
       case("POOL_SIZE")
          call assign_vec(buffer, pool_size, itmp1)
       case("STRIDE")
          call assign_vec(buffer, stride, itmp1)
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
    call this%set_hyperparams(pool_size=pool_size, stride=stride)
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

  end subroutine read_avgpool1d
!###############################################################################


!###############################################################################
  function read_avgpool1d_layer(unit, verbose) result(layer)
    !! Read 1D average pooling layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 1D average pooling layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=avgpool1d_layer_type())
    call layer%read(unit, verbose=verbose_)

  end function read_avgpool1d_layer
!###############################################################################


!###############################################################################
  subroutine build_from_onnx_avgpool1d( &
       this, node, initialisers, value_info, verbose &
  )
    !! Read ONNX attributes for 1D average pooling layer
    implicit none

    ! Arguments
    class(avgpool1d_layer_type), intent(inout) :: this
    !! Instance of the 1D average pooling layer
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
    integer, dimension(1) :: stride, pool_size, padding
    !! Stride, kernel size, and padding
    character(256) :: val
    !! Attribute value
    character(20) :: padding_method
    !! Padding method

    ! Set default values
    stride = 1
    pool_size = 2
    padding = 0

    do i = 1, size(node%attributes)
       val = node%attributes(i)%val
       select case(trim(adjustl(node%attributes(i)%name)))
       case("kernel_shape")
          read(val,*) pool_size
       case("strides")
          read(val,*) stride
       case("pads")
          read(val,*) padding
       case default
          ! Do nothing
          write(0,*) "WARNING: Unrecognised attribute in ONNX AVGPOOL1D ", &
               "layer: ", trim(adjustl(node%attributes(i)%name))
       end select
    end do

    ! Check size of initialisers is zero
    if(size(initialisers).ne.0)then
       write(0,*) "WARNING: initialisers not used for ONNX AVGPOOL1D layer"
    end if

    ! Convert integer padding to character method
    if(any(padding.gt.0))then
       padding_method = "constant"
    else
       padding_method = "valid"
    end if

    call this%set_hyperparams( &
         stride = stride, &
         pool_size = pool_size, &
         padding = padding_method &
    )

  end subroutine build_from_onnx_avgpool1d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_avgpool1d(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(avgpool1d_layer_type), intent(inout) :: this
    !! Instance of the 1D average pooling layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    type(array_type), pointer :: ptr
    !! Pointer array


    call this%output(1,1)%zero_grad()
    select case(allocated(this%pad_layer))
    case(.true.)
       call this%pad_layer%forward(input)
       ptr => avgpool1d(this%pad_layer%output(1,1), this%pool(1), this%strd(1))
    case default
       ptr => avgpool1d(input(1,1), this%pool(1), this%strd(1))
    end select
    call this%output(1,1)%assign_and_deallocate_source(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_avgpool1d
!###############################################################################

end module athena__avgpool1d_layer
