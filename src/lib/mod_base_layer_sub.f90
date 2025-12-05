submodule(athena__base_layer) athena__base_layer_submodule
  !! Submodule containing the implementation of the base layer types
  !!
  !! This submodule contains the implementation of the base layer types
  !! used in the ATHENA library. The base layer types are the abstract
  !! types from which all other layer types are derived. The submodule
  !! contains the implementation of the procedures that are common to
  !! all layer types, such as setting the input shape, getting the
  !! number of parameters, and printing the layer to a file.
  !!
  !! The following procedures are based on code from the neural-fortran library
  !! https://github.com/modern-fortran/neural-fortran/blob/main/src/nf/nf_layer.f90
  !! procedures:
  !! - get_num_params*
  !! - get_params*
  !! - set_params*
  !! - get_gradients*
  !! - set_gradients*
  use coreutils, only: stop_program, print_warning, to_lower, to_upper, icount
  use athena__tools_infile, only: assign_val, assign_vec
  use athena__diffstruc_extd, only: batchnorm_array_type

contains

!###############################################################################
  module subroutine print_base(this, file, unit, print_header_footer)
    !! Print the layer and wrapping info to a file
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    character(*), optional, intent(in) :: file
    !! File name
    integer, optional, intent(in) :: unit
    !! Unit number
    logical, optional, intent(in) :: print_header_footer
    !! Boolean whether to print header and footer

    ! Local variables
    integer :: unit_
    !! Unit number
    logical :: filename_provided
    !! Boolean whether file is
    logical :: print_header_footer_
    !! Boolean whether to print header and footer


    ! Open file with new unit
    !---------------------------------------------------------------------------
    filename_provided = .false.
    if(present(file).and.present(unit))then
       call stop_program("print_base: both file and unit specified")
    elseif(present(file))then
       filename_provided = .true.
       open(newunit=unit_, file=trim(file), access='append')
    elseif(present(unit))then
       unit_ = unit
    else
       call stop_program("print_base: neither file nor unit specified")
    end if
    print_header_footer_ = .true.
    if(present(print_header_footer)) print_header_footer_ = print_header_footer


    ! Write card
    !---------------------------------------------------------------------------
    if(print_header_footer_) write(unit_,'(A)') to_upper(trim(this%name))
    call this%print_to_unit(unit_)
    if(print_header_footer_) write(unit_,'("END ",A)') to_upper(trim(this%name))


    ! Close unit
    !---------------------------------------------------------------------------
    if(filename_provided) close(unit_)

  end subroutine print_base
!-------------------------------------------------------------------------------
  module subroutine print_to_unit_base(this, unit)
    !! Print the layer to a file
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    integer, intent(in) :: unit
    !! File unit

    return
  end subroutine print_to_unit_base
!-------------------------------------------------------------------------------
  module subroutine print_to_unit_pool(this, unit)
    !! Print pooling layer to a file
    implicit none

    ! Arguments
    class(pool_layer_type), intent(in) :: this
    !! Instance of the layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    character(100) :: fmt
    !! Format string

    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(fmt,'("(3X,""INPUT_SHAPE ="",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    if(all(this%pool.eq.this%pool(1)))then
       write(unit,'(3X,"POOL_SIZE =",1X,I0)') this%pool(1)
    else
       write(fmt,'("(3X,""STRIDE ="",",I0,"(1X,I0))")') size(this%pool)
       write(unit,fmt) this%pool
    end if
    if(all(this%strd.eq.this%strd(1)))then
       write(unit,'(3X,"STRIDE =",1X,I0)') this%strd(1)
    else
       write(fmt,'("(3X,""STRIDE ="",",I0,"(1X,I0))")') size(this%strd)
       write(unit,fmt) this%strd
    end if

  end subroutine print_to_unit_pool
!-------------------------------------------------------------------------------
  module subroutine print_to_unit_pad(this, unit)
    !! Print padding layer to a file
    implicit none

    ! Arguments
    class(pad_layer_type), intent(in) :: this
    !! Instance of the layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    character(100) :: fmt
    !! Format string

    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(fmt,'("(3X,""INPUT_SHAPE ="",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    write(fmt,'("(3X,""PADDING ="",",I0,"(1X,I0))")') size(this%pad)
    write(unit,fmt) this%pad
    write(unit,'(3X,"METHOD = ",A)') trim(this%method)

  end subroutine print_to_unit_pad
!-------------------------------------------------------------------------------
  module subroutine print_to_unit_batch(this, unit)
    !! Print 3D batch normalisation layer to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(batch_layer_type), intent(in) :: this
    !! Instance of batch normalisation layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    integer :: m
    !! Loop index
    character(100) :: fmt
    !! Format string


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(fmt,'("(3X,""INPUT_SHAPE = "",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    write(unit,'(3X,"MOMENTUM = ",F0.9)') this%momentum
    write(unit,'(3X,"EPSILON = ",F0.9)') this%epsilon
    write(unit,'(3X,"NUM_CHANNELS = ",I0)') this%num_channels
    write(unit,'(3X,"GAMMA_INITIALISER = ",A)') trim(this%kernel_init%name)
    write(unit,'(3X,"BETA_INITIALISER = ",A)') trim(this%bias_init%name)
    write(unit,'(3X,"MOVING_MEAN_INITIALISER = ",A)') &
         trim(this%moving_mean_init%name)
    write(unit,'(3X,"MOVING_VARIANCE_INITIALISER = ",A)') &
         trim(this%moving_variance_init%name)
    write(unit,'("GAMMA")')
    do m = 1, this%num_channels
       write(unit,'(5(E16.8E2))') this%params(1)%val(m,1)
    end do
    write(unit,'("END GAMMA")')
    write(unit,'("BETA")')
    do m = 1, this%num_channels
       write(unit,'(5(E16.8E2))') this%params(1)%val(this%num_channels+m,1)
    end do
    write(unit,'("END BETA")')

  end subroutine print_to_unit_batch
!###############################################################################


!###############################################################################
  module function get_attributes_base(this) result(attributes)
    !! Get the attributes of the layer (for ONNX export)
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Attributes of the layer

    ! Allocate attributes array
    allocate(attributes(0))
    ! attributes(0)%name = this%name
    ! attributes(0)%val = this%get_type_name()
    ! attributes(0)%type = ""

  end function get_attributes_base
!-------------------------------------------------------------------------------
  module function get_attributes_conv(this) result(attributes)
    !! Get the attributes of a convolutional layer (for ONNX export)
    implicit none

    ! Arguments
    class(conv_layer_type), intent(in) :: this
    !! Instance of the layer
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Attributes of the layer

    ! Local variables
    character(256) :: buffer, fmt
    !! Buffer for formatting

    ! Allocate attributes array
    allocate(attributes(2))
    attributes(1)%name = "kernel_shape"
    write(fmt,'("(",I0,"(1X,I0))")') size(this%knl)
    write(buffer,fmt) this%knl
    attributes(1)%val = trim(adjustl(buffer))
    attributes(1)%type = "ints"

    attributes(2)%name = "strides"
    write(fmt,'("(",I0,"(1X,I0))")') size(this%stp)
    write(buffer,fmt) this%stp
    attributes(2)%val = trim(adjustl(buffer))
    attributes(2)%type = "ints"

  end function get_attributes_conv
!-------------------------------------------------------------------------------
  module function get_attributes_pool(this) result(attributes)
    !! Get the attributes of a pooling layer (for ONNX export)
    implicit none

    ! Arguments
    class(pool_layer_type), intent(in) :: this
    !! Instance of the layer
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Attributes of the layer

    ! Local variables
    character(256) :: buffer, fmt
    !! Buffer for formatting

    ! Allocate attributes array
    allocate(attributes(2))
    attributes(1)%name = "kernel_shape"
    write(fmt,'("(",I0,"(1X,I0))")') size(this%pool)
    write(buffer,fmt) this%pool
    attributes(1)%val = trim(adjustl(buffer))
    attributes(1)%type = "ints"

    attributes(2)%name = "strides"
    write(fmt,'("(",I0,"(1X,I0))")') size(this%strd)
    write(buffer,fmt) this%strd
    attributes(2)%val = trim(adjustl(buffer))
    attributes(2)%type = "ints"

  end function get_attributes_pool
!-------------------------------------------------------------------------------
  module function get_attributes_batch(this) result(attributes)
    !! Get the attributes of a batch normalisation layer (for ONNX export)
    implicit none

    ! Arguments
    class(batch_layer_type), intent(in) :: this
    !! Instance of the layer
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Attributes of the layer

    ! Local variables
    character(256) :: buffer, fmt
    !! Buffer for formatting

    ! Allocate attributes array
    allocate(attributes(4))
    attributes(1)%name = "epsilon"
    write(buffer,'("(",F0.6,")")') this%epsilon
    attributes(1)%val = trim(adjustl(buffer))
    attributes(1)%type = "float"

    attributes(2)%name = "momentum"
    write(buffer,'("(",F0.6,")")') this%momentum
    attributes(2)%val = trim(adjustl(buffer))
    attributes(2)%type = "float"

    attributes(3)%name = "scale"
    write(fmt,'("(",I0,"(1X,I0))")') this%num_channels
    write(buffer,fmt) this%params(1)%val(1:this%num_channels,1)
    attributes(3)%val = trim(adjustl(buffer))
    attributes(3)%type = "float"

    attributes(4)%name = "B"
    write(fmt,'("(",I0,"(1X,I0))")') this%num_channels
    write(buffer,fmt) this%params(1)%val(this%num_channels+1:2*this%num_channels,1)
    attributes(4)%val = trim(adjustl(buffer))
    attributes(4)%type = "float"

  end function get_attributes_batch
!###############################################################################


!###############################################################################
  module subroutine build_from_onnx_base( &
       this, node, initialisers, value_info, verbose &
  )
    !! Build layer from ONNX node and initialiser
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer
    type(onnx_node_type), intent(in) :: node
    !! ONNX node
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialisers
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info
    integer, intent(in) :: verbose
    !! Verbosity level

    write(0,*) "build_from_onnx_base: " // &
         trim(this%name) // " layer cannot be built from ONNX"

  end subroutine build_from_onnx_base
!###############################################################################


!###############################################################################
  module subroutine set_rank_base(this, input_rank, output_rank)
    !! Set the input and output ranks of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, intent(in) :: input_rank
    !! Input rank
    integer, intent(in) :: output_rank
    !! Output rank

    !---------------------------------------------------------------------------
    ! Set input and output ranks
    !---------------------------------------------------------------------------
    call stop_program("set_rank_base: this layer cannot have its rank set")

  end subroutine set_rank_base
!###############################################################################


!###############################################################################
  module subroutine set_shape_base(this, input_shape)
    !! Set the input shape of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    character(len=100) :: err_msg
    !! Error message

    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(size(input_shape,dim=1).eq.this%input_rank)then
       this%input_shape = input_shape
    else
       write(err_msg,'("Invalid size of input_shape in ",A,&
            &" expected (",I0,"), got (",I0,")")')  &
            trim(this%name), this%input_rank, size(input_shape,dim=1)
       call stop_program(err_msg)
       return
    end if

  end subroutine set_shape_base
!###############################################################################


!###############################################################################
  module subroutine extract_output_base(this, output)
    !! Get the output of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    real(real32), allocatable, dimension(..), intent(out) :: output
    !! Output of the Layer

    if(size(this%output).gt.1)then
       call print_warning("extract_output_base: output has more than one"&
            &" sample, cannot extract")
       return
    end if

    call this%output(1,1)%extract(output)

  end subroutine extract_output_base
!###############################################################################


!###############################################################################
  pure module function get_num_params_base(this) result(num_params)
    !! Get the number of parameters in the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    integer :: num_params
    !! Number of parameters

    ! No parameters in the base layer
    num_params = 0

  end function get_num_params_base
!-------------------------------------------------------------------------------
  pure module function get_num_params_conv(this) result(num_params)
    !! Get the number of parameters in convolutional layer
    implicit none

    ! Arguments
    class(conv_layer_type), intent(in) :: this
    !! Instance of the layer
    integer :: num_params
    !! Number of parameters

    ! num_filters x num_channels x kernel_size + num_biases
    ! num_biases = num_filters
    num_params = this%num_filters * this%num_channels * product(this%knl) + &
         this%num_filters

  end function get_num_params_conv
!-------------------------------------------------------------------------------
  pure module function get_num_params_batch(this) result(num_params)
    !! Get the number of parameters in batch normalisation layer
    implicit none

    ! Arguments
    class(batch_layer_type), intent(in) :: this
    !! Instance of the layer
    integer :: num_params
    !! Number of parameters

    ! num_filters x num_channels x kernel_size + num_biases
    ! num_biases = num_filters
    num_params = 2 * this%num_channels

  end function get_num_params_batch
!###############################################################################


!###############################################################################
  module subroutine forward_base(this, input)
    !! Forward pass for the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer
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
          this%output(i,j) = input(i,j)
       end do
    end do

  end subroutine forward_base

  module function forward_eval_base(this, input) result(output)
    !! Forward pass of layer and return output for evaluation
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout), target :: this
    !! Instance of the layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data
    type(array_type), pointer :: output(:,:)
    !! Output data

    call this%forward(input)
    output => this%output
  end function forward_eval_base

  module subroutine set_graph_base(this, graph)
    !! Set the graph structure of the input data
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer
    type(graph_type), dimension(:), intent(in) :: graph
    !! Graph structure of input data

    ! Local variables
    integer :: s
    !! Loop index

    if(allocated(this%graph))then
       if(size(this%graph).ne.size(graph))then
          deallocate(this%graph)
          allocate(this%graph(size(graph)))
       end if
    else
       allocate(this%graph(size(graph)))
    end if
    do s = 1, size(graph)
       this%graph(s)%adj_ia = graph(s)%adj_ia
       this%graph(s)%adj_ja = graph(s)%adj_ja
       this%graph(s)%edge_weights = graph(s)%edge_weights
       this%graph(s)%num_edges = graph(s)%num_edges
       this%graph(s)%num_vertices = graph(s)%num_vertices
    end do

  end subroutine set_graph_base
!-------------------------------------------------------------------------------
  module subroutine nullify_graph_base(this)
    !! Nullify the forward pass data of the layer to free memory
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer

    ! Local variables
    integer :: i, j
    !! Loop indices

    do i = 1, size(this%output,1)
       do j = 1, size(this%output,2)
          call this%output(i,j)%nullify_graph()
       end do
    end do

  end subroutine nullify_graph_base
!###############################################################################


!###############################################################################
  module subroutine reduce_learnable(this, input)
    !! Merge two learnable layers via summation
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(inout) :: this
    !! Instance of the layer
    class(learnable_layer_type), intent(in) :: input
    !! Instance of a layer

    ! Local variables
    integer :: i
    !! Loop index

    if(allocated(this%params).and.allocated(input%params))then
       if(size(this%params).ne.size(input%params))then
          call stop_program("reduce_learnable: incompatible parameter sizes")
          return
       end if
       do i = 1, size(this%params,1)
          this%params(i) = this%params(i) + input%params(i)
          if(associated(this%params(i)%grad).and.&
               associated(input%params(i)%grad))then
             this%params(i)%grad = this%params(i)%grad + &
                  input%params(i)%grad
          end if
       end do
    else
       call stop_program("reduce_learnable: unallocated parameter arrays")
       return
    end if

  end subroutine reduce_learnable
!###############################################################################


!###############################################################################
  module function add_learnable(a, b) result(output)
    !! Add two learnable layers together
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: a, b
    !! Instances of layers
    class(learnable_layer_type), allocatable :: output
    !! Output layer

    ! Local variables
    integer :: i
    !! Loop index

    output = a
    if(allocated(a%params).and.allocated(b%params))then
       if(size(a%params).ne.size(b%params))then
          call stop_program("add_learnable: incompatible parameter sizes")
          return
       end if
       do i = 1, size(a%params,1)
          output%params(i)%grad => null()
          output%params(i) = a%params(i) + b%params(i)
          if(associated(a%params(i)%grad).and.&
               associated(b%params(i)%grad))then
             allocate(output%params(i)%grad)
             output%params(i)%grad = a%params(i)%grad + &
                  b%params(i)%grad
          end if
       end do
    else
       call stop_program("add_learnable: unallocated parameter arrays")
       return
    end if

  end function add_learnable
!###############################################################################


!###############################################################################
  pure module function get_params(this) result(params)
    !! Get the learnable parameters of the layer
    !!
    !! This function returns the learnable parameters of the layer
    !! as a single array.
    !! This has been modified from the neural-fortran library
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: this
    !! Instance of the layer
    real(real32), dimension(this%num_params) :: params
    !! Learnable parameters

    ! Local variables
    integer :: i, start_idx, end_idx
    !! Loop indices

    start_idx = 0
    end_idx = 0
    do i = 1, size(this%params)
       start_idx = end_idx + 1
       end_idx = start_idx + size(this%params(i)%val,1) - 1
       params(start_idx:end_idx) = this%params(i)%val(:,1)
    end do

  end function get_params
!###############################################################################


!###############################################################################
  module subroutine set_params(this, params)
    !! Set the learnable parameters of the layer
    !!
    !! This function sets the learnable parameters of the layer
    !! from a single array.
    !! This has been modified from the neural-fortran library
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(inout) :: this
    !! Instance of the layer
    real(real32), dimension(this%num_params), intent(in) :: params
    !! Learnable parameters

    ! Local variables
    integer :: i, start_idx, end_idx
    !! Loop indices

    if(.not.allocated(this%params)) then
       call stop_program("set_params: params not allocated")
       return
    end if
    start_idx = 0
    end_idx = 0
    do i = 1, size(this%params)
       start_idx = end_idx + 1
       end_idx = start_idx + size(this%params(i)%val,1) - 1
       this%params(i)%val(:,1) = params(start_idx:end_idx)
    end do

  end subroutine set_params
!###############################################################################


!###############################################################################
  pure module function get_gradients(this, clip_method) result(gradients)
    !! Get the gradients of the layer
    !!
    !! This function returns the gradients of the layer as a single array.
    !! This has been modified from the neural-fortran library
    use athena__clipper, only: clip_type
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: this
    !! Instance of the layer
    type(clip_type), optional, intent(in) :: clip_method
    !! Method to clip the gradients
    real(real32), dimension(this%num_params) :: gradients
    !! Gradients of the layer

    ! Local variables
    integer :: i, start_idx, end_idx
    !! Loop indices

    if(.not.allocated(this%params)) then
       return
    end if
    start_idx = 0
    end_idx = 0
    do i = 1, size(this%params)
       start_idx = end_idx + 1
       end_idx = start_idx + size(this%params(i)%val,1) - 1
       if(.not.associated(this%params(i)%grad)) then
          gradients(start_idx:end_idx) = 0._real32
       else
          gradients(start_idx:end_idx) = this%params(i)%grad%val(:,1)
       end if
    end do

    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)

  end function get_gradients
!###############################################################################


!###############################################################################
  module subroutine set_gradients(this, gradients)
    !! Set the gradients of the layer
    !!
    !! This function sets the gradients of the layer from a single array.
    !! This has been modified from the neural-fortran library
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(inout) :: this
    !! Instance of the layer
    real(real32), dimension(..), intent(in) :: gradients
    !! Gradients of the layer

    ! Local variables
    integer :: i, start_idx, end_idx
    !! Loop indices

    start_idx = 0
    end_idx = 0
    select rank(gradients)
    rank(0)
       do i = 1, size(this%params)
          if(.not.associated(this%params(i)%grad)) then
             this%params(i)%grad => this%params(i)%create_result()
          end if
          this%params(i)%grad%val(:,1) = gradients
       end do
    rank(1)
       do i = 1, size(this%params)
          if(.not.associated(this%params(i)%grad)) then
             this%params(i)%grad => this%params(i)%create_result()
          end if
          start_idx = end_idx + 1
          end_idx = start_idx + size(this%params(i)%val,1) - 1
          this%params(i)%grad%val(:,1) = gradients(start_idx:end_idx)
       end do
    end select

  end subroutine set_gradients
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module subroutine init_pad(this, input_shape, verbose)
    !! Initialise padding layer
    implicit none

    ! Arguments
    class(pad_layer_type), intent(inout) :: this
    !! Instance of the padding layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
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


    !---------------------------------------------------------------------------
    ! Initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    if(.not.allocated(this%orig_bound)) then
       allocate(this%orig_bound(2,this%input_rank-1))
       allocate(this%dest_bound(2,this%input_rank-1))
    end if
    do i = 1, this%input_rank - 1
       this%orig_bound(:,i) = [ 1, this%input_shape(i) ]
       this%dest_bound(:,i) = [ 1, this%input_shape(i) + this%pad(i) * 2 ]
       call this%facets(i)%setup_bounds( &
            length = this%input_shape(:this%input_rank-1), &
            pad = this%pad, &
            imethod = this%imethod &
       )
    end do


    !---------------------------------------------------------------------------
    ! Set up number of channels, width, height
    !---------------------------------------------------------------------------
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    allocate( this%output_shape(this%input_rank) )
    this%output_shape(this%input_rank) = this%input_shape(this%input_rank)
    this%output_shape(:this%input_rank-1) = &
         this%input_shape(:this%input_rank-1) + this%pad(:) * 2


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program("Graph input not supported for padding layer")
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate( this%output(1,1) )

  end subroutine init_pad
!###############################################################################


!###############################################################################
  module subroutine init_pool(this, input_shape, verbose)
    !! Initialise pooling layer
    implicit none

    ! Arguments
    class(pool_layer_type), intent(inout) :: this
    !! Instance of the pooling layer
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
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    allocate( this%output_shape(this%input_rank) )
    this%output_shape(this%input_rank) = this%input_shape(this%input_rank)
    this%output_shape(:this%input_rank-1) = &
         floor( &
              ( &
                   this%input_shape(:this%input_rank-1) - this%pool &
              ) / real(this%strd) &
         ) + 1


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program( &
            "Graph input not supported for pooling layer" &
       )
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate( this%output(1,1) )

  end subroutine init_pool
!###############################################################################


!###############################################################################
  module subroutine init_conv(this, input_shape, verbose)
    !! Initialise convolutional layer
    use athena__initialiser, only: initialiser_setup
    use athena__misc_types, only: base_init_type
    implicit none

    ! Arguments
    class(conv_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! initialise padding layer, if allocated
    !---------------------------------------------------------------------------
    if(allocated(this%pad_layer)) &
         call this%pad_layer%init(this%input_shape, verbose_)


    !---------------------------------------------------------------------------
    ! allocate output, activation, bias, and weight shapes
    !---------------------------------------------------------------------------
    ! NOTE: INPUT SHAPE DOES NOT INCLUDE PADDING WIDTH
    ! THIS IS HANDLED AUTOMATICALLY BY THE CODE
    ! ... provide the initial input data shape and let us deal with the padding
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    allocate( this%output_shape(this%input_rank) )
    this%output_shape(this%input_rank) = this%num_filters
    this%output_shape(:this%input_rank-1) = floor( &
         ( &
              this%input_shape(:this%input_rank-1) + 2 * this%pad - this%knl &
         ) / real(this%stp) &
    ) + 1
    this%num_params = this%get_num_params()
    allocate(this%weight_shape(this%input_rank + 1,1))
    this%weight_shape(:,1) = [ this%knl, this%num_channels, this%num_filters ]
    this%bias_shape = [this%num_filters]

    if(allocated(this%params)) deallocate(this%params)
    allocate(this%params(2))
    call this%params(1)%allocate([this%weight_shape(:,1), 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%is_sample_dependent = .false.
    call this%params(2)%allocate([this%bias_shape, 1])
    call this%params(2)%set_requires_grad(.true.)
    this%params(2)%is_sample_dependent = .false.


    !---------------------------------------------------------------------------
    ! initialise weights (kernels)
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params(1)%val(:,1), &
         fan_in = product(this%knl)+1, fan_out = 1, &
         spacing = [ this%knl, this%num_channels, this%num_filters ] &
    )

    ! initialise biases
    !---------------------------------------------------------------------------
    call this%bias_init%initialise( &
         this%params(2)%val(:,1), &
         fan_in = product(this%knl)+1, fan_out = 1 &
    )


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program( &
            "Graph input not supported for convolutional layer" &
       )
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate( this%output(1,1) )

  end subroutine init_conv
!###############################################################################


!###############################################################################
  module subroutine init_batch(this, input_shape, verbose)
    !! Initialise batch normalisation layer
    use athena__initialiser, only: initialiser_setup
    use athena__misc_types, only: base_init_type
    implicit none

    ! Arguments
    class(batch_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    integer :: verbose_ = 0


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! set up number of channels, width, height
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output_shape(this%input_rank))
    if(size(this%input_shape).eq.1)then
       this%output_shape(1) = this%input_shape(1)
       this%output_shape(2) = 1
    else
       this%output_shape = this%input_shape
    end if
    this%num_channels = this%input_shape(this%input_rank)
    this%num_params = this%get_num_params()
    allocate(this%params(1))
    call this%params(1)%allocate([2 * this%num_channels, 1])
    call this%params(1)%set_requires_grad(.true.)
    allocate(this%weight_shape(1,1))
    this%weight_shape(:,1) = [ this%num_channels ]
    this%bias_shape = [this%num_channels]


    !---------------------------------------------------------------------------
    ! allocate mean and variance
    !---------------------------------------------------------------------------
    allocate(this%mean(this%num_channels), source=0._real32)
    allocate(this%variance, source=this%mean)


    !---------------------------------------------------------------------------
    ! initialise gamma
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise(this%params(1)%val(1:this%num_channels,1), &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)

    ! initialise beta
    !---------------------------------------------------------------------------
    call this%bias_init%initialise(this%params(1)%val(this%num_channels+1:,1), &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)


    !---------------------------------------------------------------------------
    ! initialise moving mean
    !---------------------------------------------------------------------------
    call this%moving_mean_init%initialise(this%mean, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)

    ! initialise moving variance
    !---------------------------------------------------------------------------
    call this%moving_variance_init%initialise(this%variance, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program( &
            "Graph input not supported for batch normalisation layer" &
       )
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate( batchnorm_array_type :: this%output(1,1) )

  end subroutine init_batch
!###############################################################################

end submodule athena__base_layer_submodule
