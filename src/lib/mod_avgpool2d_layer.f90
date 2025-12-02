module athena__avgpool2d_layer
  !! Module containing implementation of a 2D average pooling layer
  !!
  !! This module implements 2D average pooling for downsampling by computing
  !! mean values within pooling windows.
  !!
  !! Mathematical operation:
  !!   output[i,j,k] = (1/N) Σ_{m,n} input[i*stride+m, j*stride+n, k]
  !!
  !! where:
  !!   (i,j) are output spatial coordinates
  !!   k is the channel index
  !!   N = pool_h * pool_w is the window size
  !!   (m,n) iterate over the pooling window
  !!
  !! Provides smooth downsampling by averaging.
  !! Shape: (width, height, channels) -> (width//stride, height//stride, channels)
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: pool_layer_type, base_layer_type
  use diffstruc, only: array_type
  use athena__diffstruc_extd, only: avgpool2d
  implicit none


  private

  public :: avgpool2d_layer_type
  public :: read_avgpool2d_layer


  type, extends(pool_layer_type) :: avgpool2d_layer_type
     !! Type for 2D average pooling layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_avgpool2d
     !! Set hyperparameters for 2D average pooling layer
     procedure, pass(this) :: set_batch_size => set_batch_size_avgpool2d
     !! Set batch size for 2D average pooling layer
     procedure, pass(this) :: read => read_avgpool2d
     !! Read 2D average pooling layer from file

     procedure, pass(this) :: forward => forward_avgpool2d
     !! Forward propagation derived type handler

  end type avgpool2d_layer_type

  interface avgpool2d_layer_type
     !! Interface for setting up the 2D average pooling layer
     module function layer_setup( &
          input_shape, batch_size, &
          pool_size, stride, verbose ) result(layer)
       !! Set up the 2D average pooling layer
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, dimension(..), optional, intent(in) :: pool_size
       !! Pool size
       integer, dimension(..), optional, intent(in) :: stride
       !! Stride
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(avgpool2d_layer_type) :: layer
       !! Instance of the 2D average pooling layer
     end function layer_setup
  end interface avgpool2d_layer_type



contains

!###############################################################################
  module function layer_setup( &
       input_shape, batch_size, &
       pool_size, stride, verbose) result(layer)
    !! Set up the 2D average pooling layer
    implicit none

    ! Arguments
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, dimension(..), optional, intent(in) :: pool_size
    !! Pool size
    integer, dimension(..), optional, intent(in) :: stride
    !! Stride
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(avgpool2d_layer_type) :: layer
    !! Instance of the 2D average pooling layer

    !! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer, dimension(2) :: pool_size_, stride_
    !! Pool size and stride

    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Set up pool size
    !---------------------------------------------------------------------------
    if(present(pool_size))then
       select rank(pool_size)
       rank(0)
          pool_size_ = pool_size
       rank(1)
          pool_size_(1) = pool_size(1)
          if(size(pool_size,dim=1).eq.1)then
             pool_size_(2) = pool_size(1)
          elseif(size(pool_size,dim=1).eq.2)then
             pool_size_(2) = pool_size(2)
          end if
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
          stride_(1) = stride(1)
          if(size(stride,dim=1).eq.1)then
             stride_(2) = stride(1)
          elseif(size(stride,dim=1).eq.2)then
             stride_(2) = stride(2)
          end if
       end select
    else
       stride_ = 2
    end if


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         pool_size=pool_size_, stride=stride_, verbose=verbose_ &
    )


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
  subroutine set_hyperparams_avgpool2d(this, pool_size, stride, verbose)
    !! Set hyperparameters for 2D average pooling layer
    implicit none

    ! Arguments
    class(avgpool2d_layer_type), intent(inout) :: this
    !! Instance of the 2D average pooling layer
    integer, dimension(2), intent(in) :: pool_size
    !! Pool size
    integer, dimension(2), intent(in) :: stride
    !! Stride
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "avgpool2d"
    this%type = "pool"
    this%subtype = "average"
    this%input_rank = 3
    this%output_rank = 3
    if(allocated(this%pool)) deallocate(this%pool)
    if(allocated(this%strd)) deallocate(this%strd)
    allocate( &
         this%pool(this%input_rank-1), &
         this%strd(this%input_rank-1) &
    )
    this%pool = pool_size
    this%strd = stride

  end subroutine set_hyperparams_avgpool2d
!###############################################################################


!###############################################################################
  subroutine set_batch_size_avgpool2d(this, batch_size, verbose)
    !! Set batch size for 2D average pooling layer
    implicit none

    ! Arguments
    class(avgpool2d_layer_type), intent(inout), target :: this
    !! Instance of the 2D average pooling layer
    integer, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    !! Local variables
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
               "Graph input not supported for 2D average pooling layer" &
          )
          return
       end if
       if(allocated(this%output)) deallocate(this%output)
       allocate( this%output(1,1) )
       call this%output(1,1)%allocate( &
            array_shape = [ &
                 this%output_shape(1), &
                 this%output_shape(2), this%num_channels, &
                 this%batch_size ], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_avgpool2d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine read_avgpool2d(this, unit, verbose)
    !! Read 2D average pooling layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(avgpool2d_layer_type), intent(inout) :: this
    !! Instance of the 2D average pooling layer
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
    integer, dimension(2) :: pool_size, stride
    !! Pool size and stride
    integer, dimension(3) :: input_shape
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

  end subroutine read_avgpool2d
!###############################################################################


!###############################################################################
  function read_avgpool2d_layer(unit, verbose) result(layer)
    !! Read 2D average pooling layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 2D average pooling layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=avgpool2d_layer_type())
    call layer%read(unit, verbose=verbose_)

  end function read_avgpool2d_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_avgpool2d(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(avgpool2d_layer_type), intent(inout) :: this
    !! Instance of the 2D average pooling layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    type(array_type), pointer :: ptr
    !! Pointer array


    call this%output(1,1)%zero_grad()
    ptr => avgpool2d(input(1,1), this%pool, this%strd)
    call this%output(1,1)%assign_and_deallocate_source(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_avgpool2d
!###############################################################################

end module athena__avgpool2d_layer
