module athena__add_layer
  !! Module containing implementation of a add layer
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: merge_layer_type, base_layer_type
  use athena__diffstruc_extd, only: array_ptr_type, add_layers
  use diffstruc, only: array_type, operator(+)
  implicit none


  private

  public :: add_layer_type
  public :: read_add_layer


  type, extends(merge_layer_type) :: add_layer_type
     !! Type for add layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_add
     !! Set the hyperparameters for add layer
     procedure, pass(this) :: init => init_add
     !! Initialise add layer
     procedure, pass(this) :: set_batch_size => set_batch_size_add
     !! Set the batch size for add layer
     procedure, pass(this) :: print_to_unit => print_to_unit_add
     !! Print the layer to a file
     procedure, pass(this) :: read => read_add
     !! Read the layer from a file

     procedure, pass(this) :: calc_input_shape => calc_input_shape_add
     !! Calculate input shape based on shapes of input layers

     procedure, pass(this) :: combine => combine_add
  end type add_layer_type

  interface add_layer_type
     !! Interface for setting up the add layer
     module function layer_setup( &
          input_layer_ids, batch_size, input_rank, verbose &
     ) result(layer)
       !! Setup a add layer
       integer, dimension(:), intent(in) :: input_layer_ids
       !! Input layer IDs
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: input_rank
       !! Input rank
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(add_layer_type) :: layer
     end function layer_setup
  end interface add_layer_type



contains

!###############################################################################
  module function layer_setup( &
       input_layer_ids, batch_size, input_rank, verbose &
  ) result(layer)
    !! Setup a add layer
    implicit none

    ! Arguments
    integer, dimension(:), intent(in) :: input_layer_ids
    !! Input layer IDs
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: input_rank
    !! Input rank
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(add_layer_type) :: layer
    !! Instance of the add layer

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
    else
       call stop_program( &
            "input_rank or input_shape must be provided to concat layer" &
       )
       return
    end if
    call layer%set_hyperparams( &
         input_layer_ids = input_layer_ids, &
         input_rank = input_rank_, &
         verbose = verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise batch size
    !---------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_add( &
       this, &
       input_layer_ids, &
       input_rank, &
       verbose &
  )
    !! Set the hyperparameters for add layer
    implicit none

    ! Arguments
    class(add_layer_type), intent(inout) :: this
    !! Instance of the add layer
    integer, dimension(:), intent(in) :: input_layer_ids
    !! Input layer IDs
    integer, intent(in) :: input_rank
    !! Input rank
    integer, optional, intent(in) :: verbose
    !! Verbosity level


    this%name = "add"
    this%type = "merg"
    this%merge_mode = 1 ! pointwise mode
    this%input_layer_ids = input_layer_ids
    this%input_rank = input_rank
    this%output_rank = input_rank

  end subroutine set_hyperparams_add
!###############################################################################


!###############################################################################
  subroutine init_add(this, input_shape, batch_size, verbose)
    !! Initialise add layer
    implicit none

    ! Arguments
    class(add_layer_type), intent(inout) :: this
    !! Instance of the add layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: i
    !! Loop index
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
    this%output_shape = this%input_shape


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_add
!###############################################################################


!###############################################################################
  subroutine set_batch_size_add(this, batch_size, verbose)
    !! Set the batch size for add layer
    implicit none

    ! Arguments
    class(add_layer_type), intent(inout), target :: this
    integer, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(allocated(this%output)) deallocate(this%output)
       if(.not.this%use_graph_input)then
          allocate(this%output(1,1))
          this%input_rank = size(this%input_shape)
          this%output_rank = size(this%output_shape)
          call this%output(1,1)%allocate( &
               [ this%output_shape, this%batch_size ], &
               source=0._real32 &
          )
       else
          allocate(this%output(2,this%batch_size))
       end if
    end if

  end subroutine set_batch_size_add
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_add(this, unit)
    !! Print add layer to unit
    implicit none

    ! Arguments
    class(add_layer_type), intent(in) :: this
    !! Instance of the add layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    integer :: i
    !! Loop index
    character(100) :: fmt


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"INPUT_RANK = ",I0)') this%input_rank
    write(fmt,'("(3X,""INPUT_SHAPE ="",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    write(fmt,'("(3X,""INPUT_LAYER_IDS ="",",I0,"(1X,I0))")') size(this%input_layer_ids)
    write(unit,fmt) this%input_layer_ids

  end subroutine print_to_unit_add
!###############################################################################


!###############################################################################
  subroutine read_add(this, unit, verbose)
    !! Read add layer from file
    use athena__tools_infile, only: assign_val, assign_vec, get_val
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(add_layer_type), intent(inout) :: this
    !! Instance of the add layer
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat, verbose_ = 0
    !! File status and verbosity level
    integer :: itmp1 = 0
    !! Temporary integer
    integer :: input_rank = 0
    !! Input rank
    integer, dimension(:), allocatable :: input_shape, input_layer_ids
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

       ! Read parameters from file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("INPUT_SHAPE")
          itmp1 = icount(get_val(buffer))
          allocate(input_shape(itmp1), source=0)
          call assign_vec(buffer, input_shape, itmp1)
       case("INPUT_RANK")
          call assign_val(buffer, input_rank, itmp1)
       case("INPUT_LAYER_IDS")
          itmp1 = icount(get_val(buffer))
          allocate(input_layer_ids(itmp1), source=0)
          call assign_vec(buffer, input_layer_ids, itmp1)
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
    call this%set_hyperparams( &
         input_layer_ids = input_layer_ids, &
         input_rank = input_rank, &
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

  end subroutine read_add
!###############################################################################


!###############################################################################
  function read_add_layer(unit, verbose) result(layer)
    !! Read add layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the add layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=add_layer_type(input_layer_ids=[0,0]))
    call layer%read(unit, verbose=verbose_)

  end function read_add_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  function calc_input_shape_add(this, input_shapes) result(input_shape)
    !! Calculate input shape based on shapes of input layers
    implicit none

    ! Arguments
    class(add_layer_type), intent(in) :: this
    !! Instance of the layer
    integer, dimension(:,:), intent(in) :: input_shapes
    !! Input shapes
    integer, allocatable, dimension(:) :: input_shape
    !! Calculated input shape

    ! Local variables
    integer :: i

    ! Check that all input shapes are the same
    do i = 2, size(input_shapes, 2), 1
       if(any(input_shapes(:,1).ne.input_shapes(:,i)))then
          call stop_program("All input shapes to add layer must be the same")
          return
       end if
    end do

    ! Set input shape
    input_shape = input_shapes(:,1)

  end function calc_input_shape_add
!###############################################################################



!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine combine_add(this, input_list)
    !! Forward propagation for 2D input
    implicit none

    ! Arguments
    class(add_layer_type), intent(inout) :: this
    !! Instance of the add layer
    type(array_ptr_type), dimension(:), intent(in) :: input_list
    !! Input values

    ! Local variables
    integer :: i, j, s
    !! Loop index
    type(array_type), pointer :: ptr
    !! Pointer array

    do s = 1, size(input_list(1)%array, 2)
       index_loop: do i = 1, size(input_list(1)%array, 1)
          do j = 1, size(input_list,1)
             if(.not.input_list(j)%array(i,s)%allocated) cycle index_loop
          end do
          ptr => add_layers(input_list, i, s)
          call this%output(i,s)%zero_grad()
          call this%output(i,s)%assign_and_deallocate_source(ptr)
          this%output(i,s)%is_temporary = .false.
       end do index_loop
    end do

  end subroutine combine_add
!###############################################################################

end module athena__add_layer
