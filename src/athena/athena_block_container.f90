module athena__block_container
  !! Module containing implementation of a block container
  !!
  !! The block container is a data structure that holds a collection of layers
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__initialiser_data, only: data_init_type
  use athena__container_layer, only: container_layer_type, list_of_layer_types
  use diffstruc, only: array_type
  implicit none


  private

  public :: block_container_type
  public :: read_block_container


  type, extends(learnable_layer_type) :: block_container_type
     !! Type for block container with overloaded procedures
     integer :: num_layers
     !! Number of layers in the block
     type(container_layer_type), allocatable, dimension(:) :: layers
   contains
     procedure, pass(this) :: get_num_params => get_num_params_block
     !! Get the number of parameters for block container
     procedure, pass(this) :: set_hyperparams => set_hyperparams_block
     !! Set the hyperparameters for block container
     procedure, pass(this) :: init => init_block
     !! Initialise block container
     procedure, pass(this) :: print_to_unit => print_to_unit_block
     !! Print the layer to a file
     procedure, pass(this) :: read => read_block
     !! Read the layer from a file

     procedure, pass(this) :: forward => forward_block
     !! Forward propagation derived type handler

     final :: finalise_block
     !! Finalise block container
  end type block_container_type

  interface block_container_type
     !! Interface for setting up the block container
     module function block_setup( &
          layers, input_shape, verbose &
     ) result(layer)
       !! Setup a fully connected layer
       type(container_layer_type), dimension(:), intent(in) :: layers
       !! Array of layers to include in the block
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(block_container_type) :: layer
       !! Instance of the block container
     end function block_setup
  end interface block_container_type



contains

!###############################################################################
  subroutine finalise_block(this)
    !! Finalise block container
    implicit none

    ! Arguments
    type(block_container_type), intent(inout) :: this
    !! Instance of the block container

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(allocated(this%layers)) deallocate(this%layers)

  end subroutine finalise_block
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure function get_num_params_block(this) result(num_params)
    !! Get the number of parameters for block container
    implicit none

    ! Arguments
    class(block_container_type), intent(in) :: this
    !! Instance of the block container
    integer :: num_params
    !! Number of parameters

    ! Local variables
    integer :: l
    !! Loop index

    do l = 1, size(this%layers)
       num_params = num_params + this%layers(l)%layer%get_num_params()
    end do

  end function get_num_params_block
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function block_setup(layers, input_shape, verbose) result(layer)
    !! Setup a block container
    implicit none

    ! Arguments
    type(container_layer_type), dimension(:), intent(in) :: layers
    !! Array of layers to include in the block
    !!! NEED TO ADD OPTIONAL ARGUMENT FOR WHICH LAYER GOES INTO WHICH
    !!! NEED TO ADD OPTIONAL ARGUMENT SAYING WHICH LAYERS ARE THE OUTPUTS OF THE BLOCK
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(block_container_type) :: layer
    !! Instance of the block container

    ! Local variables
    integer :: l
    !! Loop index
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set activation functions based on input name
    !---------------------------------------------------------------------------
    allocate(layer%layers(size(layers)))
    do l = 1, size(layers)
       layer%layers(l) = layers(l)
    end do


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         verbose = verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function block_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_block( &
       this, &
       verbose &
  )
    !! Set the hyperparameters for block container
    implicit none

    ! Arguments
    class(block_container_type), intent(inout) :: this
    !! Instance of the block container
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "block"
    this%type = "blck"
    this%num_layers = size(this%layers)

    ! get the rank of the input and output shape from the first layer in the block
    this%input_rank = this%layers(1)%layer%input_rank
    this%output_rank = this%layers(size(this%layers))%layer%output_rank

  end subroutine set_hyperparams_block
!###############################################################################


!###############################################################################
  subroutine init_block(this, input_shape, verbose)
    !! Initialise block container
    implicit none

    ! Arguments
    class(block_container_type), intent(inout) :: this
    !! Instance of the fully connected layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: l
    !! Loop index
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Initialise number of inputs
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%num_params = this%get_num_params()
    do l = 1, size(this%layers)
       call this%layers(l)%layer%init(input_shape=this%input_shape, verbose=verbose_)
       this%input_shape = this%layers(l)%layer%output_shape
    end do
    this%output_shape = this%layers(size(this%layers))%layer%output_shape


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    ! set this based on the number of output layers
    allocate(this%output(1,1))

  end subroutine init_block
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_block(this, unit)
    !! Print block container to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(block_container_type), intent(in) :: this
    !! Instance of the fully connected layer
    integer, intent(in) :: unit
    !! File unit
    integer :: l
    !! Loop index


    write(unit,'(3X,"NUM_LAYERS = ",I0)') this%num_layers
    do l = 1, size(this%layers)
       write(unit,'(A)') to_upper(trim(this%layers(l)%layer%name))
       call this%layers(l)%layer%print_to_unit(unit)
       write(unit,'("END ",A)') to_upper(trim(this%layers(l)%layer%name))
    end do

  end subroutine print_to_unit_block
!###############################################################################


!###############################################################################
  subroutine read_block(this, unit, verbose)
    !! Read block container from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(block_container_type), intent(inout) :: this
    !! Instance of the block container
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat
    !! Status of read
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: i, j, k, c, itmp1, iline, num_params
    !! Loop variables and temporary integer
    integer :: num_layers
    !! Number of layers
    integer :: layer_index
    !! Index of layer in list_of_layer_types
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    character(20) :: name
    !! Name of the layer
    integer :: param_line, final_line
    !! Parameter line number
    type(container_layer_type), allocatable, dimension(:) :: layers
    !! Array of layers


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
    iline = 0
    param_line = 0
    final_line = 0
    num_layers = 0
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
          final_line = iline
          backspace(unit)
          exit tag_loop
       end if
       iline = iline + 1

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       ! Read parameters from file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("NUM_LAYERS")
          ! call assign_val(buffer, num_layers, itmp1)
          ! this%num_layers = num_layers
          ! num_layers = 0
       case default
          ! check if there is no = sign in the line, if so, it is probably a layer card
          if(scan(buffer,"=").eq.0)then
             name = trim(adjustl(to_lower(tag)))
             layer_index = &
                  findloc( &
                       [ list_of_layer_types(:)%name ], &
                       name, &
                       dim = 1 &
                  )

             if(.not.allocated(this%layers))then
                this%layers = [container_layer_type()]
                this%num_layers = 1
             else
                allocate(layers(size(this%layers,dim=1)+1))
                do i = 1, size(this%layers,dim=1)
                   allocate(layers(i)%layer, source=this%layers(i)%layer)
                end do
                call move_alloc(layers, this%layers)
                this%num_layers = this%num_layers + 1
             end if
             allocate( &
                  this%layers(size(this%layers,dim=1))%layer, &
                  source=list_of_layer_types(layer_index)%read_ptr( &
                       unit, verbose=verbose_ &
                  ) &
             )

             cycle tag_loop
          elseif(scan(to_lower(trim(adjustl(buffer))),&
               'abcdfghijklmnopqrstuvwxyz').eq.0)then
             ! Don't look for "e" due to scientific notation of numbers
             ! ... i.e. exponent (E+00)
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
         verbose = verbose_ &
    )
    ! call this%init(input_shape=[num_inputs]) !!! NEED TO SET INPUT SHAPE


    !---------------------------------------------------------------------------
    ! Check for end of layer card
    !---------------------------------------------------------------------------
    call move(unit, final_line - iline, iostat=stat)
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_block
!###############################################################################


!###############################################################################
  function read_block_container(unit, verbose) result(layer)
    !! Read fully connected layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the fully connected layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=block_container_type([container_layer_type()]))
    call layer%read(unit, verbose=verbose_)

  end function read_block_container
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_block(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(block_container_type), intent(inout) :: this
    !! Instance of the fully connected layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    integer :: l

    call this%layers(1)%layer%forward(input)
    do l = 2, size(this%layers), 1
       call this%layers(l)%layer%forward(this%layers(l-1)%layer%output)
    end do
    call this%output(1,1)%assign(this%layers(size(this%layers))%layer%output(1,1))
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_block
!###############################################################################

end module athena__block_container
