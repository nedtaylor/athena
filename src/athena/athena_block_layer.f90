module athena__block_layer
  !! Module containing implementation of a block container
  !!
  !! The block container is a data structure that holds a collection of layers
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__clipper, only: clip_type
  use athena__container_layer, only: container_layer_type, &
       list_of_layer_types, allocate_list_of_layer_types
  use athena__diffstruc_extd, only: batchnorm_array_type
  use diffstruc, only: array_type, operator(*)
  use diffstruc, only: array_type, discard_sources
  implicit none


  private

  public :: block_layer_type
  public :: read_block_layer


  type, extends(learnable_layer_type) :: block_layer_type
     !! Type for block container with overloaded procedures
     integer :: num_layers
     !! Number of layers in the block
     type(container_layer_type), allocatable, dimension(:) :: layers
   contains
     procedure, pass(this) :: get_num_params => get_num_params_block
     !! Get the number of parameters for block container
     procedure, pass(this) :: get_params => get_params_block
     !! Get the parameters of all child layers
     procedure, pass(this) :: set_params => set_params_block
     !! Set the parameters of all child layers
     procedure, pass(this) :: get_gradients => get_gradients_block
     !! Get the gradients of all child layers
     procedure, pass(this) :: set_gradients => set_gradients_block
     !! Set the gradients of all child layers
     procedure, pass(this) :: reset_gradients => reset_gradients_block
     !! Reset the gradients of all child layers
     procedure, pass(this) :: reduce => reduce_block
     !! Merge another block into this one
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

     procedure, pass(this) :: discard_history => discard_history_block
     !! Release expression history held by the block and its children
     procedure, pass(this) :: nullify_graph => nullify_graph_block
     !! Release graph data held by the block and its children
     procedure, pass(this) :: reset_state => reset_state_block
     !! Reset persistent state in the block and its children

     final :: finalise_block
     !! Finalise block container
  end type block_layer_type

  interface block_layer_type
     !! Interface for setting up the block container
     module function layer_setup( &
          layers, input_shape, verbose &
     ) result(layer)
       !! Setup a fully connected layer
       type(container_layer_type), dimension(:), optional, intent(in) :: layers
       !! Array of layers to include in the block
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(block_layer_type) :: layer
       !! Instance of the block container
     end function layer_setup
  end interface block_layer_type



contains

!###############################################################################
  subroutine reduce_block(this, input)
    !! Merge another block into this one
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
    !! Instance of the block container
    class(learnable_layer_type), intent(in) :: input
    !! Other layer to merge

    ! Local variables
    integer :: l
    !! Loop index

    select type(source => input)
    class is(block_layer_type)
       if(.not.allocated(this%layers).or..not.allocated(source%layers))then
          call stop_program("reduce_block: unallocated layer lists")
          return
       end if
       if(size(this%layers).ne.size(source%layers))then
          call stop_program("reduce_block: incompatible block sizes")
          return
       end if
       do l = 1, size(this%layers)
          if(.not.allocated(this%layers(l)%layer).or. &
               .not.allocated(source%layers(l)%layer))then
             call stop_program("reduce_block: unallocated child layer")
             return
          end if
          select type(child => this%layers(l)%layer)
          class is(learnable_layer_type)
             select type(other => source%layers(l)%layer)
             class is(block_layer_type)
                select type(child)
                class is(block_layer_type)
                   call child%reduce(other)
                class default
                   call stop_program( &
                        "reduce_block: child layer types are incompatible" &
                   )
                   return
                end select
             class is(learnable_layer_type)
                call child%reduce(other)
             class default
                call stop_program( &
                     "reduce_block: child layer types are incompatible" &
                )
                return
             end select
          class default
             select type(other => source%layers(l)%layer)
             class is(learnable_layer_type)
                call stop_program( &
                     "reduce_block: child layer types are incompatible" &
                )
                return
             end select
          end select
       end do
    class default
       call stop_program("reduce_block: cannot reduce a block with a non-block")
       return
    end select

  end subroutine reduce_block
!###############################################################################


!###############################################################################
  subroutine discard_history_block(this)
    !! Release expression history held by the block and its children
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
    !! Instance of the block container

    ! Local variables
    integer :: l, last_child
    !! Loop index and final child index
    logical :: aliases_last_child
    !! Whether the block output is the last child's view

    if(allocated(this%layers))then
       do l = 1, size(this%layers)
          if(allocated(this%layers(l)%layer)) &
               call this%layers(l)%layer%discard_history()
       end do
    end if

    ! The sequential block publishes a non-owning view of the last child. That
    ! child has just been cleaned up above, so cleaning the view again would
    ! double-release the same operands.
    aliases_last_child = .false.
    if(allocated(this%output))then
       if(size(this%output,1).ge.1 .and. size(this%output,2).ge.1)then
          if(allocated(this%layers))then
             last_child = size(this%layers)
             if(last_child.gt.0)then
                if(allocated(this%layers(last_child)%layer))then
                   if(allocated(this%layers(last_child)%layer%output))then
                      aliases_last_child = &
                           this%output(1,1)%id.ne.0 .and. &
                           this%output(1,1)%id.eq. &
                           this%layers(last_child)%layer%output(1,1)%id
                   end if
                end if
             end if
          end if
          if(.not.aliases_last_child) &
               call discard_sources(this%output(1,1), leaves=.true.)
       end if
    end if

  end subroutine discard_history_block
!###############################################################################


!###############################################################################
  subroutine nullify_graph_block(this)
    !! Release graph data held by the block and its children
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
    !! Instance of the block container

    ! Local variables
    integer :: l, last_child
    !! Loop index and final child index
    logical :: aliases_last_child
    !! Whether the block output is the last child's view

    if(allocated(this%layers))then
       do l = 1, size(this%layers)
          if(allocated(this%layers(l)%layer)) &
               call this%layers(l)%layer%nullify_graph()
       end do
    end if

    ! The block output may be the last child's view; that graph has already
    ! been released above.
    aliases_last_child = .false.
    if(allocated(this%output))then
       if(size(this%output,1).ge.1 .and. size(this%output,2).ge.1)then
          if(allocated(this%layers))then
             last_child = size(this%layers)
             if(last_child.gt.0)then
                if(allocated(this%layers(last_child)%layer))then
                   if(allocated(this%layers(last_child)%layer%output))then
                      aliases_last_child = &
                           this%output(1,1)%id.ne.0 .and. &
                           this%output(1,1)%id.eq. &
                           this%layers(last_child)%layer%output(1,1)%id
                   end if
                end if
             end if
          end if
          if(.not.aliases_last_child) call this%output(1,1)%nullify_graph()
       end if
    end if

  end subroutine nullify_graph_block
!###############################################################################


!###############################################################################
  subroutine reset_state_block(this)
    !! Reset persistent state in the block and its children
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
    !! Instance of the block container

    ! Local variables
    integer :: l
    !! Loop index

    if(.not.allocated(this%layers)) return
    do l = 1, size(this%layers)
       if(allocated(this%layers(l)%layer)) &
            call this%layers(l)%layer%reset_state()
    end do

  end subroutine reset_state_block
!###############################################################################


!###############################################################################
  subroutine finalise_block(this)
    !! Finalise block container
    implicit none

    ! Arguments
    type(block_layer_type), intent(inout) :: this
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
    class(block_layer_type), intent(in) :: this
    !! Instance of the block container
    integer :: num_params
    !! Number of parameters

    ! Local variables
    integer :: l
    !! Loop index

    num_params = 0
    if(.not.allocated(this%layers)) return
    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer)) cycle
       select type(layer => this%layers(l)%layer)
       class is(learnable_layer_type)
          num_params = num_params + layer%get_num_params()
       end select
    end do

  end function get_num_params_block
!###############################################################################


!###############################################################################
  pure function get_params_block(this) result(params)
    !! Get the parameters of all child layers
    implicit none

    ! Arguments
    class(block_layer_type), intent(in) :: this
    !! Instance of the block container
    real(real32), dimension(this%num_params) :: params
    !! Parameters

    ! Local variables
    integer :: l, start_idx, end_idx, n
    !! Loop index and parameter offsets

    params = 0._real32
    if(this%get_num_params().ne.this%num_params) return
    if(.not.allocated(this%layers)) return
    end_idx = 0
    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer)) cycle
       select type(layer => this%layers(l)%layer)
       class is(learnable_layer_type)
          n = layer%get_num_params()
          if(n.le.0) cycle
          start_idx = end_idx + 1
          end_idx = start_idx + n - 1
          params(start_idx:end_idx) = layer%get_params()
       end select
    end do

  end function get_params_block
!###############################################################################


!###############################################################################
  subroutine set_params_block(this, params)
    !! Set the parameters of all child layers
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
    !! Instance of the block container
    real(real32), dimension(this%num_params), intent(in) :: params
    !! Parameters

    ! Local variables
    integer :: l, start_idx, end_idx, n
    !! Loop index and parameter offsets

    if(.not.allocated(this%layers)) return
    if(this%num_params.ne.this%get_num_params())then
       call stop_program( &
            "set_params_block: num_params is out of sync with the layers" &
       )
       return
    end if
    end_idx = 0
    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer)) cycle
       select type(layer => this%layers(l)%layer)
       class is(learnable_layer_type)
          n = layer%get_num_params()
          if(n.le.0) cycle
          start_idx = end_idx + 1
          end_idx = start_idx + n - 1
          call layer%set_params(params(start_idx:end_idx))
       end select
    end do

  end subroutine set_params_block
!###############################################################################


!###############################################################################
  pure function get_gradients_block(this, clip_method) result(gradients)
    !! Get the gradients of all child layers
    implicit none

    ! Arguments
    class(block_layer_type), intent(in) :: this
    !! Instance of the block container
    type(clip_type), optional, intent(in) :: clip_method
    !! Clip method
    real(real32), dimension(this%num_params) :: gradients
    !! Gradients

    ! Local variables
    integer :: l, start_idx, end_idx, n
    !! Loop index and gradient offsets

    gradients = 0._real32
    if(this%get_num_params().ne.this%num_params) return
    if(.not.allocated(this%layers)) return
    end_idx = 0
    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer)) cycle
       select type(layer => this%layers(l)%layer)
       class is(learnable_layer_type)
          n = layer%get_num_params()
          if(n.le.0) cycle
          start_idx = end_idx + 1
          end_idx = start_idx + n - 1
          gradients(start_idx:end_idx) = layer%get_gradients()
       end select
    end do

    if(present(clip_method)) call clip_method%apply(size(gradients), gradients)

  end function get_gradients_block
!###############################################################################


!###############################################################################
  subroutine set_gradients_block(this, gradients)
    !! Set the gradients for all child layers
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
    !! Instance of the block container
    real(real32), dimension(..), intent(in) :: gradients
    !! Gradients

    if(.not.allocated(this%layers)) return
    if(this%num_params.ne.this%get_num_params())then
       call stop_program( &
            "set_gradients_block: num_params is out of sync with the layers" &
       )
       return
    end if

    select rank(gradients)
    rank(0)
       call set_gradients_scalar(this, gradients)
    rank(1)
       call set_gradients_from_flat(this, gradients)
    rank default
       call stop_program( &
            "set_gradients_block: gradients must have rank 0 or 1" &
       )
       return
    end select

  end subroutine set_gradients_block
!###############################################################################


!###############################################################################
  subroutine set_gradients_scalar(this, gradient)
    !! Set a scalar gradient for all child layers
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
    !! Instance of the block container
    real(real32), intent(in) :: gradient
    !! Scalar gradient

    ! Local variables
    integer :: l
    !! Loop index

    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer)) cycle
       select type(layer => this%layers(l)%layer)
       class is(learnable_layer_type)
          call layer%set_gradients(gradient)
       end select
    end do

  end subroutine set_gradients_scalar
!###############################################################################


!###############################################################################
  subroutine set_gradients_from_flat(this, flat)
    !! Set block gradients from a flat gradient vector
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
    !! Instance of the block container
    real(real32), dimension(:), intent(in) :: flat
    !! Flat gradient vector

    ! Local variables
    integer :: l, start_idx, end_idx, n
    !! Loop index and gradient offsets

    if(this%num_params.ne.this%get_num_params())then
       call stop_program( &
            "set_gradients_from_flat: num_params is out of sync with layers" &
       )
       return
    end if

    if(size(flat).ne.this%num_params)then
       call stop_program( &
            "set_gradients_from_flat: gradient size does not match block" &
       )
       return
    end if

    end_idx = 0
    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer)) cycle
       select type(layer => this%layers(l)%layer)
       class is(learnable_layer_type)
          n = layer%get_num_params()
          if(n.le.0) cycle
          start_idx = end_idx + 1
          end_idx = start_idx + n - 1
          call layer%set_gradients(flat(start_idx:end_idx))
       end select
    end do

  end subroutine set_gradients_from_flat
!###############################################################################


!###############################################################################
  subroutine reset_gradients_block(this)
    !! Reset the gradients of all child layers
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
    !! Instance of the block container

    ! Local variables
    integer :: l
    !! Loop index

    if(.not.allocated(this%layers)) return
    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer)) cycle
       select type(child => this%layers(l)%layer)
       class is(learnable_layer_type)
          call child%reset_gradients()
       end select
    end do

  end subroutine reset_gradients_block
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup(layers, input_shape, verbose) result(layer)
    !! Setup a block container
    implicit none

    ! Arguments
    type(container_layer_type), dimension(:), optional, intent(in) :: layers
    !! Array of layers to include in the block
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(block_layer_type) :: layer
    !! Instance of the block container

    ! Local variables
    integer :: l
    !! Loop index
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Validate and copy the layers
    !---------------------------------------------------------------------------
    if(present(layers))then
       do l = 1, size(layers)
          if(.not.allocated(layers(l)%layer))then
             call stop_program( &
                  "layer_setup: block contains an unallocated layer" &
             )
             return
          end if
       end do
       allocate(layer%layers(size(layers)))
       do l = 1, size(layers)
          allocate(layer%layers(l)%layer, source=layers(l)%layer)
       end do
    else
       allocate(layer%layers(0))
    end if


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

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_block( &
       this, &
       verbose &
  )
    !! Set the hyperparameters for block container
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
    !! Instance of the block container
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: l
    !! Loop index

    if(present(verbose))then
       if(verbose.gt.0)then
          write(*,*) "Setting hyperparameters for block container"
       end if
    end if
    this%name = "block"
    this%type = "blck"
    this%num_layers = 0
    if(allocated(this%layers)) this%num_layers = size(this%layers)

    ! get the rank of the input and output shape from the first and last
    ! allocated layers in the block
    if(this%num_layers.lt.1) return
    do l = 1, this%num_layers
       if(.not.allocated(this%layers(l)%layer)) cycle
       this%input_rank = this%layers(l)%layer%input_rank
       exit
    end do
    do l = this%num_layers, 1, -1
       if(.not.allocated(this%layers(l)%layer)) cycle
       this%output_rank = this%layers(l)%layer%output_rank
       exit
    end do

  end subroutine set_hyperparams_block
!###############################################################################


!###############################################################################
  subroutine init_block(this, input_shape, verbose)
    !! Initialise block container
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
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
    integer, dimension(:), allocatable :: layer_input_shape
    !! Input shape for the current layer


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Validate the layers
    !---------------------------------------------------------------------------
    if(.not.allocated(this%layers))then
       call stop_program("init_block: block contains no layers")
       return
    end if
    if(size(this%layers).lt.1)then
       call stop_program("init_block: block contains no layers")
       return
    end if
    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer))then
          call stop_program( &
               "init_block: block contains an unallocated layer" &
          )
          return
       end if
    end do


    !---------------------------------------------------------------------------
    ! Initialise number of inputs and chain the layer shapes
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(size(this%input_shape).ne.size(input_shape))then
          call stop_program( &
               "init_block: block re-init input rank does not match existing input_shape" &
          )
          return
       end if
       this%input_rank = size(this%input_shape)
    else
       this%input_rank = size(input_shape)
       call this%set_shape(input_shape)
    end if
    if(.not.allocated(this%input_shape))then
       call stop_program("init_block: failed to set the block input shape")
       return
    end if
    allocate(layer_input_shape, source=this%input_shape)
    do l = 1, size(this%layers)
       call this%layers(l)%layer%init( &
            input_shape=layer_input_shape, verbose=verbose_ &
       )
       if(.not.allocated(this%layers(l)%layer%output_shape))then
          call stop_program( &
               "init_block: layer has no output shape after initialisation" &
          )
          return
       end if
       layer_input_shape = this%layers(l)%layer%output_shape
    end do
    this%output_shape = this%layers(size(this%layers))%layer%output_shape
    this%output_rank = size(this%output_shape)
    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    call this%discard_history()
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
    class(block_layer_type), intent(in) :: this
    !! Instance of the fully connected layer
    integer, intent(in) :: unit
    !! File unit
    integer :: l
    !! Loop index
    character(100) :: fmt
    !! Format string


    write(unit,'(3X,"NUM_LAYERS = ",I0)') this%num_layers
    if(allocated(this%input_shape))then
       write(fmt,'("(3X,""INPUT_SHAPE ="",",I0,"(1X,I0))")') &
            size(this%input_shape)
       write(unit,fmt) this%input_shape
    end if
    if(.not.allocated(this%layers)) return
    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer)) cycle
       write(unit,'(A)') to_upper(trim(this%layers(l)%layer%name))
       call this%layers(l)%layer%print_to_unit(unit)
       write(unit,'("END ",A)') to_upper(trim(this%layers(l)%layer%name))
    end do

  end subroutine print_to_unit_block
!###############################################################################


!###############################################################################
  subroutine read_block(this, unit, verbose)
    !! Read block container from file
    use athena__tools_infile, only: assign_val
    use coreutils, only: to_lower, to_upper
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout) :: this
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
    integer :: i, l, itmp1, layer_index, n, num_layers_expected
    !! Loop variables and temporary integers
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    character(20) :: name
    !! Name of the layer
    type(container_layer_type), allocatable, dimension(:) :: layers
    !! Temporary array of layers


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose

    if(.not.allocated(this%name)) this%name = "block"
    this%type = "blck"
    if(allocated(this%layers)) deallocate(this%layers)
    if(.not.allocated(list_of_layer_types)) call allocate_list_of_layer_types()
    if(size(list_of_layer_types).lt.1) call allocate_list_of_layer_types()


    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
    num_layers_expected = -1
    itmp1 = 0
    tag_loop: do

       ! Check for end of file
       !------------------------------------------------------------------------
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg,'("file encountered error (EoF?) before END ",A)') &
               to_upper(trim(this%name))
          call stop_program(err_msg)
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       ! Check for end of layer card
       !------------------------------------------------------------------------
       if(trim(adjustl(buffer)).eq."END "//to_upper(trim(this%name)))then
          exit tag_loop
       end if

       tag = trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag = trim(tag(:scan(tag,"=")-1))

       ! Read parameters from file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("NUM_LAYERS")
          call assign_val(buffer, num_layers_expected, itmp1)
       case("INPUT_SHAPE")
          ! the input shape is derived from the first layer
       case default
          if(tag(:3).eq.'END')then
             cycle tag_loop
          elseif(scan(to_lower(trim(adjustl(buffer))),&
               'abcdfghijklmnopqrstuvwxyz').eq.0)then
             ! Don't look for "e" due to scientific notation of numbers
             ! ... i.e. exponent (E+00)
             cycle tag_loop
          elseif(scan(buffer,"=").eq.0)then
             ! no = sign, so this is a layer card
             name = trim(adjustl(to_lower(tag)))
             layer_index = &
                  findloc( &
                       [ list_of_layer_types(:)%name ], &
                       name, &
                       dim = 1 &
                  )
             if(layer_index.eq.0)then
                write(err_msg,'("Unrecognised layer type in block: ",A)') &
                     trim(name)
                call stop_program(err_msg)
                return
             end if

             if(.not.allocated(this%layers)) allocate(this%layers(0))
             n = size(this%layers)
             allocate(layers(n+1))
             do i = 1, n
                if(allocated(this%layers(i)%layer))then
                   allocate(layers(i)%layer, source=this%layers(i)%layer)
                end if
             end do
             call move_alloc(layers, this%layers)
             allocate( &
                  this%layers(n+1)%layer, &
                  source=list_of_layer_types(layer_index)%read_ptr( &
                       unit, verbose=verbose_ &
                  ) &
             )
             this%num_layers = n + 1

             cycle tag_loop
          end if
          write(err_msg,'("Unrecognised line in input file: ",A)') &
               trim(adjustl(buffer))
          call stop_program(err_msg)
          return
       end select
    end do tag_loop


    !---------------------------------------------------------------------------
    ! Validate the layers
    !---------------------------------------------------------------------------
    if(.not.allocated(this%layers))then
       call stop_program("read_block: block contains no layers")
       return
    end if
    if(size(this%layers).lt.1)then
       call stop_program("read_block: block contains no layers")
       return
    end if
    if(num_layers_expected.ge.0)then
       if(num_layers_expected.ne.size(this%layers))then
          write(err_msg,'("NUM_LAYERS does not match the layers in block: "&
               &,I0," /= ",I0)') &
               num_layers_expected, size(this%layers)
          call stop_program(err_msg)
          return
       end if
    end if
    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer))then
          call stop_program( &
               "read_block: block contains an unallocated layer" &
          )
          return
       end if
    end do


    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call this%set_hyperparams(verbose = verbose_)


    !---------------------------------------------------------------------------
    ! Check that adjacent layers have consistent shapes
    !---------------------------------------------------------------------------
    do l = 2, size(this%layers)
       if(.not.allocated(this%layers(l-1)%layer%output_shape))then
          call stop_program( &
               "read_block: preceding layer has no output shape" &
          )
          return
       end if
       if(.not.allocated(this%layers(l)%layer%input_shape))then
          call stop_program("read_block: layer has no input shape")
          return
       end if
       if( &
            size(this%layers(l-1)%layer%output_shape).ne. &
            size(this%layers(l)%layer%input_shape) &
       )then
          call stop_program( &
               "read_block: adjacent layer shapes are incompatible" &
          )
          return
       end if
       if(any( &
            this%layers(l-1)%layer%output_shape.ne. &
            this%layers(l)%layer%input_shape &
       ))then
          call stop_program( &
               "read_block: adjacent layer shapes are incompatible" &
          )
          return
       end if
    end do

    n = size(this%layers)


    !---------------------------------------------------------------------------
    ! Set the block input and output shapes
    !---------------------------------------------------------------------------
    if(.not.allocated(this%layers(1)%layer%input_shape))then
       call stop_program("read_block: first layer has no input shape")
       return
    end if
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    allocate(this%input_shape, source=this%layers(1)%layer%input_shape)
    this%input_rank = size(this%input_shape)

    if(.not.allocated(this%layers(n)%layer%output_shape))then
       call stop_program("read_block: last layer has no output shape")
       return
    end if
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    allocate(this%output_shape, source=this%layers(n)%layer%output_shape)
    this%output_rank = size(this%output_shape)

    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))

  end subroutine read_block
!###############################################################################


!###############################################################################
  function read_block_layer(unit, verbose) result(layer)
    !! Read block layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the block layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(block_layer_type :: layer)
    call layer%read(unit, verbose=verbose_)

  end function read_block_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_block(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(block_layer_type), intent(inout), target :: this
    !! Instance of the fully connected layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    integer :: l
    !! Loop index
    type(array_type), pointer :: ptr
    !! Pointer array
    logical, allocatable, dimension(:) :: inference_save
    !! Saved inference flags


    !---------------------------------------------------------------------------
    ! Validate the layers and output
    !---------------------------------------------------------------------------
    if(.not.allocated(this%layers))then
       call stop_program("forward_block: block contains no layers")
       return
    end if
    if(size(this%layers).lt.1)then
       call stop_program("forward_block: block contains no layers")
       return
    end if
    if(.not.allocated(this%input_shape))then
       call stop_program("forward_block: block has no input shape")
       return
    end if
    do l = 1, size(this%layers)
       if(.not.allocated(this%layers(l)%layer))then
          call stop_program( &
               "forward_block: block contains an unallocated layer" &
          )
          return
       end if
    end do
    if(.not.allocated(this%output)) allocate(this%output(1,1))


    !---------------------------------------------------------------------------
    ! Propagate the inference flag for the duration of the forward pass
    !---------------------------------------------------------------------------
    allocate(inference_save(size(this%layers)))
    do l = 1, size(this%layers)
       inference_save(l) = this%layers(l)%layer%inference
       this%layers(l)%layer%inference = this%inference
    end do
    call this%layers(1)%layer%forward(input)
    do l = 2, size(this%layers), 1
       call this%layers(l)%layer%forward(this%layers(l-1)%layer%output)
    end do
    do l = 1, size(this%layers)
       this%layers(l)%layer%inference = inference_save(l)
    end do
    deallocate(inference_save)


    !---------------------------------------------------------------------------
    ! Expose the final output as a distinct temporary expression
    !---------------------------------------------------------------------------
    ! Use select type so the identity multiply receives a non-polymorphic
    ! array_type actual (a polymorphic component actual crashes gfortran).
    select type(last_output => this%layers(size(this%layers))%layer%output(1,1))
    type is(array_type)
       ptr => last_output * 1._real32
    type is(batchnorm_array_type)
       ptr => last_output * 1._real32
    class default
       call stop_program("forward_block: unsupported output type")
       return
    end select
    call this%output(1,1)%assign_replacing_history(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_block
!###############################################################################

end module athena__block_layer
