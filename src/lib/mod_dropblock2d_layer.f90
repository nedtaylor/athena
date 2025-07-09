module athena__dropblock2d_layer
  !! Module containing implementation of a 2D dropblock layer
  !!
  !! This module contains the implementation of a 2D dropblock layer
  !! for use in neural networks.
  !! DropBlock reference: https://arxiv.org/pdf/1810.12890.pdf
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__base_layer, only: drop_layer_type, base_layer_type
  use athena__misc_types, only: array4d_type
  implicit none


  private

  public :: dropblock2d_layer_type
  public :: read_dropblock2d_layer


  type, extends(drop_layer_type) :: dropblock2d_layer_type
     !! Type for 2D dropblock layer with overloaded procedures
     integer :: block_size, half
     !! Block size and half block size
     !! Block size is the width of the block to drop (typical = 5)
     real(real32) :: gamma
     !! Number of activation units to drop
     integer :: num_channels
     !! Number of channels
     logical, allocatable, dimension(:,:) :: mask
     !! Mask for dropblock
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_dropblock2d
     !! Set hyperparameters for 2D dropblock layer
     procedure, pass(this) :: init => init_dropblock2d
     !! Initialise 2D dropblock layer
     procedure, pass(this) :: set_batch_size => set_batch_size_dropblock2d
     !! Set batch size for 2D dropblock layer
     procedure, pass(this) :: print => print_dropblock2d
     !! Print 2D dropblock layer to file
     procedure, pass(this) :: read => read_dropblock2d
     !! Read 2D dropblock layer from file
     procedure, pass(this) :: forward  => forward_rank
     !! Forward propagation handler for 2D dropblock layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward propagation handler for 2D dropblock layer
     procedure, private, pass(this) :: forward_4d
     !! Forward propagation for 4D input
     procedure, private, pass(this) :: backward_4d
     !! Backward propagation for 4D input
     procedure, pass(this) :: generate_mask => generate_bernoulli_mask
     !! Generate Bernoulli mask
  end type dropblock2d_layer_type

  interface dropblock2d_layer_type
     !! Interface for setting up the 2D dropblock layer
     module function layer_setup( &
          rate, block_size, &
          input_shape, batch_size, &
          verbose ) result(layer)
       !! Set up the 2D dropblock layer
       real(real32), intent(in) :: rate
       !! Drop rate
       integer, intent(in) :: block_size
       !! Block size
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(dropblock2d_layer_type) :: layer
       !! Instance of the 2D dropblock layer
     end function layer_setup
  end interface dropblock2d_layer_type



contains

!###############################################################################
  pure subroutine forward_rank(this, input)
    !! Forward propagation handler for 2D dropblock layer
    implicit none

    ! Arguments
    class(dropblock2d_layer_type), intent(inout) :: this
    !! Instance of the 2D dropblock layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values

    select rank(input)
    rank(2)
       call forward_4d(this, input)
    rank(4)
       call forward_4d(this, input)
    end select
  end subroutine forward_rank
!###############################################################################


!###############################################################################
  pure subroutine backward_rank(this, input, gradient)
    !! Backward propagation handler for 2D dropblock layer
    implicit none

    ! Arguments
    class(dropblock2d_layer_type), intent(inout) :: this
    !! Instance of the 2D dropblock layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient values

    select rank(input)
    rank(2)
       select rank(gradient)
       rank(2)
          call backward_4d(this, input, gradient)
       end select
    rank(4)
       select rank(gradient)
       rank(2)
          call backward_4d(this, input, gradient)
       rank(4)
          call backward_4d(this, input, gradient)
       end select
    end select
  end subroutine backward_rank
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       rate, block_size, &
       input_shape, batch_size, &
       verbose ) result(layer)
    !! Set up the 2D dropblock layer
    implicit none

    ! Arguments
    real(real32), intent(in) :: rate
    !! Drop rate
    integer, intent(in) :: block_size
    !! Block size
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(dropblock2d_layer_type) :: layer
    !! Instance of the 2D dropblock layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Initialise hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams(rate, block_size, verbose=verbose_)


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
  pure subroutine set_hyperparams_dropblock2d(this, rate, block_size, verbose)
    !! Set hyperparameters for 2D dropblock layer
    implicit none

    ! Arguments
    class(dropblock2d_layer_type), intent(inout) :: this
    !! Instance of the 2D dropblock layer
    real(real32), intent(in) :: rate
    !! Drop rate
    integer, intent(in) :: block_size
    !! Block size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "dropblock2d"
    this%type = "drop"
    this%input_rank = 3

    this%rate = rate
    this%block_size = block_size
    this%half = (this%block_size-1)/2

  end subroutine set_hyperparams_dropblock2d
!###############################################################################


!###############################################################################
  subroutine init_dropblock2d(this, input_shape, batch_size, verbose)
    !! Initialise 2D dropblock layer
    implicit none

    ! Arguments
    class(dropblock2d_layer_type), intent(inout) :: this
    !! Instance of the 2D dropblock layer
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
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! Set up number of channels, width, height
    !---------------------------------------------------------------------------
    this%num_channels = this%input_shape(3)
    allocate(this%output_shape(2))
    this%output_shape = this%input_shape


    !---------------------------------------------------------------------------
    ! Set gamma
    !---------------------------------------------------------------------------
    ! original paper uses keep_prob, we use drop_rate
    ! drop_rate = 1 - keep_prob
    this%gamma = ( this%rate/this%block_size**2._real32 ) * &
         this%input_shape(1) / &
         (this%input_shape(1) - this%block_size + 1._real32) * &
         this%input_shape(2) / &
         (this%input_shape(2) - this%block_size + 1._real32)
    allocate(this%mask( &
         this%input_shape(1), &
         this%input_shape(2)), source=.true.)


    !---------------------------------------------------------------------------
    ! Generate mask
    !---------------------------------------------------------------------------
    call this%generate_mask()


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_dropblock2d
!###############################################################################


!###############################################################################
  subroutine set_batch_size_dropblock2d(this, batch_size, verbose)
    !! Set batch size for 2D dropblock layer
    implicit none

    ! Arguments
    class(dropblock2d_layer_type), intent(inout), target :: this
    !! Instance of the 2D dropblock layer
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
               "Graph input not supported for 2D dropblock layer" &
          )
          return
       end if
       if(allocated(this%output)) deallocate(this%output)
       allocate( this%output(1,1), source = array4d_type() )
       call this%output(1,1)%allocate( &
            array_shape = [ &
                 this%output_shape(1), &
                 this%output_shape(2), this%num_channels, &
                 this%batch_size ], &
            source=0._real32 &
       )
       if(allocated(this%di)) deallocate(this%di)
       allocate( this%di(1,1), source = array4d_type() )
       call this%di(1,1)%allocate( source = this%output(1,1) )
    end if


  end subroutine set_batch_size_dropblock2d
!###############################################################################


!###############################################################################
  subroutine generate_bernoulli_mask(this)
    !! Generate Bernoulli mask
    implicit none

    ! Arguments
    class(dropblock2d_layer_type), intent(inout) :: this
    !! Instance of the 2D dropblock layer

    ! Local variables
    real(real32), allocatable, dimension(:,:) :: mask_real
    !! Real mask
    integer :: i, j
    !! Loop indices
    integer, dimension(2) :: ilim, jlim
    !! Limits for mask


    ! Generate Bernoulli mask
    !---------------------------------------------------------------------------
    ! assume random number already seeded and don't need to again
    allocate(mask_real(size(this%mask,1), size(this%mask,2)))
    call random_number(mask_real)  ! Generate random values in [0..1]

    this%mask = .true. ! 1 = keep

    ! Apply threshold to create binary mask
    !---------------------------------------------------------------------------
    do j = 1 + this%half, size(this%mask, dim=2) - this%half
       do i = 1 + this%half, size(this%mask, dim=1) - this%half
          if(mask_real(i, j).lt.this%gamma)then
             ilim(:) = [ &
                  max(i - this%half, lbound(this%mask,1)), &
                  min(i + this%half, ubound(this%mask,1)) ]
             jlim(:) = [ &
                  max(j - this%half, lbound(this%mask,2)), &
                  min(j + this%half, ubound(this%mask,2)) ]
             this%mask(ilim(1):ilim(2), jlim(1):jlim(2)) = .false. ! 0 = drop
          end if
       end do
    end do

  end subroutine generate_bernoulli_mask
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_dropblock2d(this, file)
    !! Print 2D dropblock layer to file
    use athena__misc, only: to_upper
    implicit none

    ! Arguments
    class(dropblock2d_layer_type), intent(in) :: this
    !! Instance of the 2D dropblock layer
    character(*), intent(in) :: file
    !! File name

    ! Local variables
    integer :: unit
    !! File unit


    ! Open file with new unit
    !---------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(A)') to_upper(trim(this%name))
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    write(unit,'(3X,"RATE = ",F0.9)') this%rate
    write(unit,'(3X,"BLOCK_SIZE = ",I0)') this%block_size
    write(unit,'("END ",A)') to_upper(trim(this%name))


    ! Close unit
    !---------------------------------------------------------------------------
    close(unit)

  end subroutine print_dropblock2d
!###############################################################################


!###############################################################################
  subroutine read_dropblock2d(this, unit, verbose)
    !! Read 2D dropblock layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(dropblock2d_layer_type), intent(inout) :: this
    !! Instance of the 2D dropblock layer
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat, verbose_ = 0
    !! File status and verbosity level
    integer :: itmp1
    !! Temporary integer
    integer :: block_size
    !! Block size
    real(real32) :: rate
    !! Drop rate
    integer, dimension(3) :: input_shape
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
       case("BLOCK_SIZE")
          call assign_val(buffer, block_size, itmp1)
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
    call this%set_hyperparams( &
         rate = rate, block_size = block_size, &
         verbose = verbose_ &
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

  end subroutine read_dropblock2d
!###############################################################################


!###############################################################################
!!! read layer from file and return layer
!###############################################################################
  function read_dropblock2d_layer(unit, verbose) result(layer)
    !! Read 2D dropblock layer from file and return layer
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
    allocate(layer, source=dropblock2d_layer_type(rate=0._real32, block_size=0))
    call layer%read(unit, verbose=verbose_)

  end function read_dropblock2d_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
!!! forward propagation
!###############################################################################
  pure subroutine forward_4d(this, input)
    !! Forward propagation for 4D input
    implicit none

    ! Arguments
    class(dropblock2d_layer_type), intent(inout) :: this
    !! Instance of the 2D dropblock layer
    real(real32), &
         dimension( &
              this%input_shape(1), &
              this%input_shape(2), &
              this%num_channels, this%batch_size), &
         intent(in) :: input
    !! Input values

    ! Local variables
    integer :: m, s
    !! Loop indices


    select type(output => this%output(1,1))
    type is (array4d_type)
       select case(this%inference)
       case(.true.)
          ! do not perform drop operation
          output%val_ptr = input * ( 1._real32 - this%rate )
       case default
          ! perform the drop operation
          do concurrent(m = 1:this%num_channels, s = 1:this%batch_size)
             output%val_ptr(:,:,m,s) = &
                  merge(input(:,:,m,s), 0._real32, this%mask)
          end do
       end select
    end select

  end subroutine forward_4d
!###############################################################################


!###############################################################################
!!! backward propagation
!###############################################################################
  pure subroutine backward_4d(this, input, gradient)
    !! Backward propagation for 4D input
    implicit none

    ! Arguments
    class(dropblock2d_layer_type), intent(inout) :: this
    !! Instance of the 2D dropblock layer
    real(real32), &
         dimension( &
              this%input_shape(1), &
              this%input_shape(2), &
              this%num_channels, this%batch_size), &
         intent(in) :: input
    !! Input values
    real(real32), &
         dimension(&
              this%output_shape(1), &
              this%output_shape(2), &
              this%num_channels, this%batch_size), &
         intent(in) :: gradient
    !! Gradient values

    ! Local variables
    integer :: m, s
    !! Loop indices

    ! compute gradients for input feature map
    !---------------------------------------------------------------------------
    select type(di => this%di(1,1))
    type is (array4d_type)
       do concurrent(m = 1:this%num_channels, s=1:this%batch_size)
          di%val_ptr(:,:,m,s) = merge(gradient(:,:,m,s), 0._real32, this%mask)
       end do
    end select

  end subroutine backward_4d
!###############################################################################

end module athena__dropblock2d_layer
