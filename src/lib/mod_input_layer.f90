module athena__input_layer
  !! Module containing procedures for an input layer
  !!
  !! This module implements the input layer which serves as the entry point
  !! for data into a neural network. It handles data conversion and batching.
  !!
  !! Operation: Accepts external data and converts to internal array_type format
  !!   - Validates input shape matches specified dimensions
  !!   - Handles both dense arrays and graph-structured data
  !!   - No learnable parameters (pass-through layer)
  !!
  !! Properties:
  !!   - First layer in any network architecture
  !!   - Defines expected input shape for subsequent layers
  !!   - Supports multiple input sources in multi-input networks
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: base_layer_type
  use graphstruc, only: graph_type
  use diffstruc, only: array_type
  implicit none


  private

  public :: input_layer_type
  public :: read_input_layer


  type, extends(base_layer_type) :: input_layer_type
     !! Type for an input layer
     integer :: index = 1
     !! Index of the layer
     integer :: num_outputs
     !! Number of outputs
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_input
     !! Set hyperparameters
     procedure, pass(this) :: init => init_input
     !! Initialise layer
     procedure, pass(this) :: set_batch_size => set_batch_size_input
     !! Set batch size
     procedure, pass(this) :: print_to_unit => print_to_unit_input
     !! Print layer to unit
     procedure, pass(this) :: read => read_input
     !! Read layer from file
     procedure, pass(this) :: set_input_real
     !! Set input values
     procedure, pass(this) :: set_input_graph
     !! Set input values
     generic :: set => set_input_real, set_input_graph
     !! Generic interface for setting input values

     procedure, pass(this) :: forward => forward_input
     !! Forward propagation derived type handler

  end type input_layer_type

  interface input_layer_type
     !! Interface for an input layer
     module function layer_setup( &
          input_shape, batch_size, index, use_graph_input, verbose &
     ) result(layer)
       !! Set up layer
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Shape of the input data
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: index
       !! Index of the layer
       logical, optional, intent(in) :: use_graph_input
       !! Use graph input
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(input_layer_type) :: layer
       !! Instance of the input layer
     end function layer_setup
  end interface input_layer_type



contains

!###############################################################################
  module function layer_setup( &
       input_shape, batch_size, index, use_graph_input, verbose &
  ) result(layer)
    !! Set up layer
    implicit none

    ! Arguments
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Shape of the input data
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: index
    !! Index of the layer
    logical, optional, intent(in) :: use_graph_input
    !! Use graph input
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(input_layer_type) :: layer
    !! Instance of the input layer

    ! Local variables
    integer :: index_ = 1
    !! Index of the layer
    integer :: verbose_ = 0
    !! Verbosity level
    logical :: use_graph_input_ = .false.
    !! Use graph input


    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    if(present(index)) index_ = index
    if(present(use_graph_input)) use_graph_input_ = use_graph_input
    call layer%set_hyperparams( &
         index = index_, &
         use_graph_input = use_graph_input_, &
         verbose = verbose_ &
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
  subroutine set_hyperparams_input( &
       this, &
       input_rank, &
       index, &
       use_graph_input, &
       verbose &
  )
    !! Set hyperparameters for an input layer
    implicit none

    ! Arguments
    class(input_layer_type), intent(inout) :: this
    !! Instance of the input layer
    integer, optional, intent(in) :: input_rank
    !! Rank of the input data
    integer, optional, intent(in) :: index
    !! Index of the layer
    logical, optional, intent(in) :: use_graph_input
    !! Use graph input
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "input"
    this%type = "inpt"
    this%input_rank = 0
    if(present(input_rank)) this%input_rank = input_rank
    this%output_rank = this%input_rank
    if(present(index)) this%index = index
    if(present(use_graph_input))then
       this%use_graph_input = use_graph_input
       this%use_graph_output = use_graph_input
    end if

  end subroutine set_hyperparams_input
!###############################################################################


!###############################################################################
  subroutine init_input(this, input_shape, batch_size, verbose)
    !! Initialise an input layer
    implicit none

    ! Arguments
    class(input_layer_type), intent(inout) :: this
    !! Instance of the input layer
    integer, dimension(:), intent(in) :: input_shape
    !! Shape of the input data
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
    this%input_rank = size(input_shape, dim=1)
    this%output_rank = this%input_rank
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%output_shape = this%input_shape
    this%num_outputs = product(this%input_shape)


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_input
!###############################################################################


!###############################################################################
  subroutine set_batch_size_input(this, batch_size, verbose)
    !! Set batch size for an input layer
    implicit none

    ! Arguments
    class(input_layer_type), intent(inout), target :: this
    !! Instance of the input layer
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
       if(allocated(this%output)) deallocate(this%output)
       if(this%use_graph_input)then
          allocate(this%output(2,this%batch_size))
       else
          select case(size(this%input_shape))
          case(1)
             this%input_rank = 1
             allocate(this%output(1,1))
             call this%output(1,1)%allocate( &
                  array_shape = [ &
                       this%input_shape(1), this%batch_size ], &
                  source=0._real32 &
             )
          case(2)
             this%input_rank = 2
             allocate(this%output(1,1))
             call this%output(1,1)%allocate( &
                  array_shape = [ &
                       this%input_shape(1), &
                       this%input_shape(2), &
                       this%batch_size ], &
                  source=0._real32 &
             )
          case(3)
             this%input_rank = 3
             allocate(this%output(1,1))
             call this%output(1,1)%allocate( &
                  array_shape = [ &
                       this%input_shape(1), &
                       this%input_shape(2), &
                       this%input_shape(3), this%batch_size ], &
                  source=0._real32 &
             )
          case(4)
             this%input_rank = 4
             allocate(this%output(1,1))
             call this%output(1,1)%allocate( &
                  array_shape = [ &
                       this%input_shape(1), &
                       this%input_shape(2), &
                       this%input_shape(3), &
                       this%input_shape(4), this%batch_size ], &
                  source=0._real32 &
             )
          case default
             call stop_program('Input layer only supports input ranks 1-4')
             return
          end select
       end if
    end if
    this%output_rank = this%input_rank

  end subroutine set_batch_size_input
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_input(this, unit)
    !! Print input layer to unit
    implicit none

    ! Arguments
    class(input_layer_type), intent(in) :: this
    !! Instance of the input layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    integer :: t
    !! Loop index
    character(100) :: fmt
    !! Format string


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"INPUT_RANK = ",I0)') this%input_rank
    write(fmt,'("(3X,""INPUT_SHAPE ="",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    write(unit,'(3X,"USE_GRAPH_INPUT = ",L1)') this%use_graph_input

  end subroutine print_to_unit_input
!###############################################################################


!###############################################################################
  subroutine read_input(this, unit, verbose)
    !! Read an input layer from a file
    use athena__tools_infile, only: assign_val, assign_vec, get_val
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(input_layer_type), intent(inout) :: this
    !! Instance of the input layer
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: stat
    !! File status
    integer :: itmp1= 0
    !! Temporary integer

    ! Local variables
    integer :: input_rank = 0
    !! Rank of the input data
    integer, dimension(:), allocatable :: input_shape
    !! Shape of the input data
    logical :: use_graph_input = .false.
    !! Use graph input
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
       if(trim(adjustl(buffer)).eq."END INPUT")then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       ! Read parameters from save file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("INPUT_RANK")
          call assign_val(buffer, input_rank, itmp1)
       case("INPUT_SHAPE")
          itmp1 = icount(get_val(buffer))
          allocate(input_shape(itmp1))
          call assign_vec(buffer, input_shape, itmp1)
       case("USE_GRAPH_INPUT")
          call assign_val(buffer, use_graph_input, itmp1)
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
    if(.not.allocated(input_shape))then
       write(err_msg,'("No input shape found in ",A)') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    call this%set_hyperparams( &
         input_rank = input_rank, &
         use_graph_input = use_graph_input, &
         verbose = verbose_ &
    )
    call this%init(input_shape = input_shape)


    ! Check for end of layer card
    !---------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END INPUT")then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_input
!###############################################################################


!###############################################################################
  function read_input_layer(unit, verbose) result(layer)
    !! Read an input layer from a file
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the input layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=input_layer_type())
    call layer%read(unit, verbose=verbose_)

  end function read_input_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_input(this, input)
    !! Forward propagation for an input layer
    implicit none

    ! Arguments
    class(input_layer_type), intent(inout) :: this
    !! Instance of the input layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data

    ! Local variables
    integer :: i, j
    !! Loop indices

    do i = 1, size(input, 1)
       do j = 1, size(input, 2)
          if(.not.input(i,j)%allocated)then
             call stop_program('Input to input layer not allocated')
             return
          end if
          call this%output(i,j)%assign_shallow( input(i,j) )
          this%output(i,j)%is_temporary = .false.
       end do
    end do

  end subroutine forward_input
!###############################################################################


!###############################################################################
  pure subroutine set_input_real(this, input)
    !! Set input values for an input layer
    implicit none

    ! Arguments
    class(input_layer_type), intent(inout) :: this
    !! Instance of the input layer
    real(real32), dimension(..), intent(in) :: input
    !! Input data
    !dimension(this%batch_size * this%num_outputs), intent(in) :: input

    call this%output(1,1)%set( input )
    this%output(1,1)%is_temporary = .false.

  end subroutine set_input_real
!-------------------------------------------------------------------------------
  subroutine set_input_graph(this, input)
    !! Set input values for an input layer
    implicit none

    ! Arguments
    class(input_layer_type), intent(inout) :: this
    !! Instance of the input layer
    type(graph_type), dimension(:), intent(in) :: input
    !! Input data
    !dimension(this%batch_size * this%num_outputs), intent(in) :: input

    integer :: s

    do s = 1, this%batch_size
       if(this%output(1,s)%allocated) call this%output(1,s)%deallocate()
       if(this%output(2,s)%allocated) call this%output(2,s)%deallocate()
       call this%output(1,s)%allocate( &
            array_shape = [ &
                 input(s)%num_vertex_features, input(s)%num_vertices &
            ] &
       )
       call this%output(1,s)%zero_grad()
       call this%output(1,s)%set_requires_grad(.false.)
       call this%output(1,s)%set( input(s)%vertex_features )
       this%output(1,s)%is_temporary = .false.
       if(input(s)%num_edge_features.le.0) cycle
       call this%output(2,s)%allocate( &
            array_shape = [ &
                 input(s)%num_edge_features, input(s)%num_edges &
            ] &
       )
       call this%output(2,s)%zero_grad()
       call this%output(2,s)%set_requires_grad(.false.)
       call this%output(2,s)%set( input(s)%edge_features )
       this%output(2,s)%is_temporary = .false.
    end do

  end subroutine set_input_graph
!###############################################################################

end module athena__input_layer
