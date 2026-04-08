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
  use coreutils, only: stop_program, print_warning

contains

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
    allocate(attributes(3))
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

    attributes(3)%name = "dilations"
    write(fmt,'("(",I0,"(1X,I0))")') size(this%dil)
    write(buffer,fmt) this%dil
    attributes(3)%val = trim(adjustl(buffer))
    attributes(3)%type = "ints"

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
  module subroutine emit_onnx_nodes_base( &
       this, prefix, &
       nodes, num_nodes, max_nodes, &
       inits, num_inits, max_inits, &
       input_name, is_last_layer, format &
  )
    !! Default implementation: no-op (standard layers are handled by write_onnx)
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    character(*), intent(in) :: prefix
    !! Prefix for node names
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    !! ONNX nodes
    integer, intent(inout) :: num_nodes
    !! Number of ONNX nodes
    integer, intent(in) :: max_nodes
    !! Maximum number of ONNX nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    !! ONNX initialisers
    integer, intent(inout) :: num_inits
    !! Number of ONNX initialisers
    integer, intent(in) :: max_inits
    !! Maximum number of ONNX initialisers
    character(*), optional, intent(in) :: input_name
    !! Name of the input tensor from the previous layer
    logical, optional, intent(in) :: is_last_layer
    !! Whether this is the last non-input layer
    integer, optional, intent(in) :: format
    !! Export format selector

    ! Default: do nothing. Standard layers are handled directly by write_onnx.
  end subroutine emit_onnx_nodes_base
!###############################################################################


!###############################################################################
  module subroutine emit_onnx_graph_inputs_base( &
       this, prefix, &
       graph_inputs, num_inputs &
  )
    !! Default implementation: no-op (standard layers don't add graph inputs)
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    character(*), intent(in) :: prefix
    !! Prefix for input names
    type(onnx_tensor_type), intent(inout), dimension(:) :: graph_inputs
    !! ONNX graph inputs
    integer, intent(inout) :: num_inputs
    !! Number of ONNX graph inputs

    ! Default: do nothing. Standard input layers are handled directly.
  end subroutine emit_onnx_graph_inputs_base
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
!-------------------------------------------------------------------------------
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
!###############################################################################


!###############################################################################
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
!###############################################################################


!###############################################################################
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

end submodule athena__base_layer_submodule
