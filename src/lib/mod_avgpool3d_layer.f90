module athena__avgpool3d_layer
  !! Module containing implementation of a 3D average pooling layer
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__base_layer, only: pool_layer_type, base_layer_type
  use athena__misc_types, only: array5d_type
  implicit none


  private

  public :: avgpool3d_layer_type
  public :: read_avgpool3d_layer


  type, extends(pool_layer_type) :: avgpool3d_layer_type
     !! Type for 3D average pooling layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_avgpool3d
     !! Set hyperparameters for 3D average pooling layer
     procedure, pass(this) :: init => init_avgpool3d
     !! Initialise 3D average pooling layer
     procedure, pass(this) :: set_batch_size => set_batch_size_avgpool3d
     !! Set batch size for 3D average pooling layer
     procedure, pass(this) :: read => read_avgpool3d
     !! Read 3D average pooling layer from file
     procedure, pass(this) :: forward  => forward_rank
     !! Forward propagation handler for 3D average pooling layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward propagation handler for 3D average pooling layer
     procedure, private, pass(this) :: forward_5d
     !! Forward propagation for 5D input
     procedure, private, pass(this) :: backward_5d
     !! Backward propagation for 5D input
  end type avgpool3d_layer_type

  interface avgpool3d_layer_type
     !! Interface for setting up the 3D average pooling layer
     module function layer_setup( &
          input_shape, batch_size, &
          pool_size, stride, verbose) result(layer)
       !! Set up the 3D average pooling layer
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
       type(avgpool3d_layer_type) :: layer
       !! Instance of the 3D average pooling layer
     end function layer_setup
  end interface avgpool3d_layer_type



contains

!###############################################################################
  pure subroutine forward_rank(this, input)
    !! Forward propagation handler for 3D average pooling layer
    implicit none

    ! Arguments
    class(avgpool3d_layer_type), intent(inout) :: this
    !! Instance of the 3D average pooling layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values

    select rank(input)
    rank(2)
       call forward_5d(this, input)
    rank(5)
       call forward_5d(this, input)
    end select
  end subroutine forward_rank
!###############################################################################


!###############################################################################
  pure subroutine backward_rank(this, input, gradient)
    !! Backward propagation handler for 3D average pooling layer
    implicit none

    ! Arguments
    class(avgpool3d_layer_type), intent(inout) :: this
    !! Instance of the 3D average pooling layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient values

    select rank(input)
    rank(2)
       select rank(gradient)
       rank(2)
          call backward_5d(this, input, gradient)
       end select
    rank(5)
       select rank(gradient)
       rank(5)
         call backward_5d(this, input, gradient)
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
       pool_size, stride, verbose) result(layer)
    !! Set up the 3D average pooling layer
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
    
    type(avgpool3d_layer_type) :: layer
    !! Instance of the 3D average pooling layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer, dimension(3) :: pool_size_, stride_
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
             pool_size_(2:) = pool_size(1)
          elseif(size(pool_size,dim=1).eq.3)then
             pool_size_(2:) = pool_size(2:)
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
             stride_(2:) = stride(1)
          elseif(size(stride,dim=1).eq.3)then
             stride_(2:) = stride(2:)
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
  subroutine set_hyperparams_avgpool3d(this, pool_size, stride, verbose)
    !! Set hyperparameters for 3D average pooling layer
    implicit none

    ! Arguments
    class(avgpool3d_layer_type), intent(inout) :: this
    !! Instance of the 3D average pooling layer
    integer, dimension(3), intent(in) :: pool_size
    !! Pool size
    integer, dimension(3), intent(in) :: stride
    !! Stride
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "avgpool3d"
    this%type = "pool"
    this%input_rank = 4
    allocate( &
         this%pool(this%input_rank-1), &
         this%strd(this%input_rank-1) &
    )
    this%pool = pool_size
    this%strd = stride

  end subroutine set_hyperparams_avgpool3d
!###############################################################################


!###############################################################################
  subroutine init_avgpool3d(this, input_shape, batch_size, verbose)
    !! Initialise 3D average pooling layer
    implicit none

    ! Arguments
    class(avgpool3d_layer_type), intent(inout) :: this
    !! Instance of the 3D average pooling layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    ! Initialise optional arguments
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size

    ! Initialise input shape
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)

    ! Set up number of channels, width, height
    this%num_channels = this%input_shape(4)
    if(allocated(this%output))then
       if(this%output%allocated) call this%output%deallocate()
    end if
    this%output = array5d_type()
    this%output%shape(4) = this%input_shape(4)
    this%output%shape(:3) = &
         floor( (this%input_shape(:3) - this%pool)/real(this%strd)) + 1

    ! Initialise batch size-dependent arrays
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_avgpool3d
!###############################################################################


!###############################################################################
  subroutine set_batch_size_avgpool3d(this, batch_size, verbose)
    !! Set batch size for 3D average pooling layer
    implicit none

    ! Arguments
    class(avgpool3d_layer_type), intent(inout), target :: this
    !! Instance of the 3D average pooling layer
    integer, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    ! Initialise optional arguments
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size

    ! Allocate arrays
    if(allocated(this%input_shape))then
       if(.not.allocated(this%output)) this%output = array5d_type()
       if(this%output%allocated) call this%output%deallocate(keep_shape=.true.)
       call this%output%allocate( array_shape = [ &
            this%output%shape(1), &
            this%output%shape(2), &
            this%output%shape(3), this%num_channels, &
            this%batch_size ], &
            source=0._real32 &
       )
       if(.not.allocated(this%di)) this%di = array5d_type()
       if(this%di%allocated) call this%di%deallocate()
       call this%di%allocate( array_shape = [ &
            this%input_shape(1), &
            this%input_shape(2), &
            this%input_shape(3), &
            this%input_shape(4), &
            this%batch_size ], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_avgpool3d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine read_avgpool3d(this, unit, verbose)
    !! Read 3D average pooling layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(avgpool3d_layer_type), intent(inout) :: this
    !! Instance of the 3D average pooling layer
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
    integer, dimension(3) :: pool_size, stride
    !! Pool size and stride
    integer, dimension(4) :: input_shape
    !! Input shape
    character(256) :: buffer, tag, err_msg
    !! Buffer for reading lines, tag for identifying lines, error message

    if(present(verbose)) verbose_ = verbose

    ! Loop over tags in layer card
    tag_loop: do

       ! Check for end of file
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          call stop_program( &
               "file encountered error (EoF?) before END AVGPOOL3D" &
          )
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       ! Check for end of layer card
       if(trim(adjustl(buffer)).eq."END AVGPOOL3D")then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       ! Read parameters from save file
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

    ! Set transfer activation function
    call this%set_hyperparams(pool_size=pool_size, stride=stride)
    call this%init(input_shape = input_shape)

    ! Check for end of layer card
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END AVGPOOL3D")then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_avgpool3d
!###############################################################################


!###############################################################################
  function read_avgpool3d_layer(unit, verbose) result(layer)
    !! Read 3D average pooling layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 3D average pooling layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=avgpool3d_layer_type())
    call layer%read(unit, verbose=verbose_)

  end function read_avgpool3d_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine forward_5d(this, input)
    !! Forward propagation for 5D input
    implicit none

    ! Arguments
    class(avgpool3d_layer_type), intent(inout) :: this
    !! Instance of the 3D average pooling layer
    real(real32), dimension( &
         this%input_shape(1), &
         this%input_shape(2), &
         this%input_shape(3), &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: input
    !! Input values

    ! Local variables
    integer :: i, j, k, m, s
    !! Loop indices
    integer, dimension(3) :: stride_idx
    !! Stride index

    select type(output => this%output)
    type is (array5d_type)
       ! Perform the pooling operation
       do concurrent(&
            s = 1:this%batch_size, &
            m = 1:this%num_channels, &
            k = 1:this%output%shape(3), &
            j = 1:this%output%shape(2), &
            i = 1:this%output%shape(1))
          stride_idx = ([i,j,k] - 1) * this%strd + 1
          output%val_ptr(i, j, k, m, s) = sum(&
               input( &
               stride_idx(1):stride_idx(1)+this%pool(1)-1, &
               stride_idx(2):stride_idx(2)+this%pool(2)-1, &
               stride_idx(3):stride_idx(3)+this%pool(3)-1, m, s)) / &
               product(this%pool)
       end do
    end select

  end subroutine forward_5d
!###############################################################################


!###############################################################################
  pure subroutine backward_5d(this, input, gradient)
    !! Backward propagation for 5D input
    implicit none

    ! Arguments
    class(avgpool3d_layer_type), intent(inout) :: this
    !! Instance of the 3D average pooling layer
    real(real32), dimension( &
         this%input_shape(1), &
         this%input_shape(2), &
         this%input_shape(3), &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: input
    !! Input values
    real(real32), &
         dimension(&
         this%output%shape(1), &
         this%output%shape(2), &
         this%output%shape(3), &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: gradient
    !! Gradient values

    ! Local variables
    integer :: i, j, k, m, s
    !! Loop indices
    integer, dimension(3) :: stride_idx
    !! Stride index

    select type(di => this%di)
    type is (array5d_type)
       di%val_ptr = 0._real32
       ! Compute gradients for input feature map
       do concurrent( &
            s = 1:this%batch_size, &
            m = 1:this%num_channels, &
            k = 1:this%output%shape(3), &
            j = 1:this%output%shape(2), &
            i = 1:this%output%shape(1))
          stride_idx = ([i,j,k] - 1) * this%strd
          ! Compute gradients for input feature map
          di%val_ptr( &
               stride_idx(1)+1:stride_idx(1)+this%pool(1), &
               stride_idx(2)+1:stride_idx(2)+this%pool(2), &
               stride_idx(3)+1:stride_idx(3)+this%pool(3), m, s) = &
               di%val_ptr( &
               stride_idx(1)+1:stride_idx(1)+this%pool(1), &
               stride_idx(2)+1:stride_idx(2)+this%pool(2), &
               stride_idx(3)+1:stride_idx(3)+this%pool(3), m, s) + &
               gradient(i, j, k, m, s) / product(this%pool)
       end do
    end select

  end subroutine backward_5d
!###############################################################################

end module athena__avgpool3d_layer