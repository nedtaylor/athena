module athena__flatten_layer
  !! Module containing implementation of a 1D flattening layer
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: &
       array1d_type, &
       array2d_type, &
       array3d_type, &
       array4d_type, &
       array5d_type
  implicit none


  private

  public :: flatten_layer_type
  public :: read_flatten_layer


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
     procedure, pass(this) :: read => read_flatten
     !! Read flattening layer from file
     procedure, pass(this) :: forward  => forward_rank
     !! Forward propagation handler for flattening layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward propagation handler for flattening layer
     procedure, pass(this) :: set_addit_input
     !! Set additional input for flattening layer
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
  pure subroutine forward_rank(this, input)
    !! Forward propagation handler for flattening layer
    implicit none

    ! Arguments
    class(flatten_layer_type), intent(inout) :: this
    !! Instance of the flattening layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values

   !  select type(output => this%output)
   !  type is (array2d_type)
       !! can make this even easier by just setting output%val_ptr = input, and then output%val is already rank 2
   !  end select
    select rank(input)
    rank(2)
       this%output%val(:this%num_outputs, :this%batch_size) = input
    rank(3)
       this%output%val(:this%num_outputs, :this%batch_size) = &
            reshape(input, [this%num_outputs, this%batch_size])
    rank(4)
       this%output%val(:this%num_outputs, :this%batch_size) = &
            reshape(input, [this%num_outputs, this%batch_size])
    rank(5)
       this%output%val(:this%num_outputs, :this%batch_size) = &
            reshape(input, [this%num_outputs, this%batch_size])
    rank(6)
       this%output%val(:this%num_outputs, :this%batch_size) = &
            reshape(input, [this%num_outputs, this%batch_size])
    end select
  end subroutine forward_rank
!###############################################################################


!###############################################################################
  pure subroutine backward_rank(this, input, gradient)
    !! Backward propagation handler for flattening layer
    implicit none

    ! Arguments
    class(flatten_layer_type), intent(inout) :: this
    !! Instance of the flattening layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient values

    select rank(gradient)
    rank(2)
       select type(di => this%di)
       type is (array1d_type)
          di%val_ptr = reshape(gradient(:this%num_outputs,:), &
               [size(di%val_ptr)] )
       type is (array2d_type)
          di%val_ptr = reshape(gradient(:this%num_outputs,:), &
               [this%input_shape(1), this%batch_size] )
       type is (array3d_type)
          di%val_ptr = reshape(gradient(:this%num_outputs,:), &
               [this%input_shape(1), this%input_shape(2), this%batch_size] )
       type is (array4d_type)
          di%val_ptr = reshape(gradient(:this%num_outputs,:), [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%input_shape(3), &
               this%batch_size ] )
       type is (array5d_type)
          di%val_ptr = reshape(gradient(:this%num_outputs,:), [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%input_shape(3), &
               this%input_shape(4), &
               this%batch_size ] )
       end select
    end select
  end subroutine backward_rank
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


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
       if(this%output%allocated) call this%output%deallocate()
    end if
    this%output = array2d_type()
    this%output%shape = [this%num_outputs]


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
       if(.not.allocated(this%output)) this%output = array2d_type()
       if(this%output%allocated) call this%output%deallocate(keep_shape=.true.)
       call this%output%allocate(array_shape = [ &
            (this%num_outputs), this%batch_size ], &
            source=0._real32 &
       )
       if(allocated(this%di)) deallocate(this%di)
       select case(size(this%input_shape))
       case(1)
          this%input_rank = 1
          this%di = array2d_type()
          call this%di%allocate( array_shape = [ &
               this%input_shape(1), this%batch_size ], &
               source=0._real32 &
          )
       case(2)
          this%input_rank = 2
          this%di = array3d_type()
          call this%di%allocate( array_shape = [ &
               this%input_shape(1), &
               this%input_shape(2), this%batch_size ], &
               source=0._real32 &
          )
       case(3)
          this%input_rank = 3
          this%di = array4d_type()
          call this%di%allocate( array_shape = [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%input_shape(3), this%batch_size ], &
               source=0._real32 &
          )
       case(4)
          this%input_rank = 4
          this%di = array5d_type()
          call this%di%allocate( array_shape = [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%input_shape(3), &
               this%input_shape(4), this%batch_size ], &
               source=0._real32 &
          )
       end select
    end if

  end subroutine set_batch_size_flatten
!###############################################################################


!###############################################################################
  subroutine read_flatten(this, unit, verbose)
    !! Read flattening layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
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

    if(input_rank.eq.0.and.allocated(input_shape))then
       input_rank = size(input_shape)
    else
       call stop_program( &
            "input_rank or input_shape must be provided to flatten layer" &
       )
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
    allocate(layer, source=flatten_layer_type())
    call layer%read(unit, verbose=verbose_)

  end function read_flatten_layer
!###############################################################################


!###############################################################################
  pure subroutine set_addit_input(this, addit_input)
    !! Set additional input for flattening layer
    implicit none

    ! Arguments
    class(flatten_layer_type), intent(inout) :: this
    !! Instance of the flattening layer
    real(real32), dimension(:,:), intent(in) :: addit_input
    !! Additional input

    select type(output => this%output)
    type is (array2d_type)
       output%val_ptr(this%num_outputs+1:, :) = addit_input
    end select
  end subroutine set_addit_input
!###############################################################################

end module athena__flatten_layer
