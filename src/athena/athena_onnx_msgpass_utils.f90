module athena__onnx_msgpass_utils
  !! Shared ONNX builder helpers for message-passing layers.
  !!
  !! This module factors out the repeated edge-index extraction,
  !! scatter accumulation, weight export, and output naming logic used by
  !! the Duvenaud and Kipf message-passing layers.
  use coreutils, only: real32
  use athena__misc_types, only: onnx_node_type, onnx_initialiser_type, &
       onnx_tensor_type
  use athena__onnx_utils, only: emit_node, emit_squeeze_node, &
       emit_constant_int64, emit_constant_float, &
       emit_constant_of_shape_float, emit_activation_node, &
       col_to_row_major_2d
  implicit none

  private

  public :: emit_msgpass_graph_inputs
  public :: emit_output_identity
  public :: get_timestep_output_name
  public :: emit_edge_index_component
  public :: emit_scatter_aggregator
  public :: emit_weight_initialiser_2d
  public :: emit_weight_initialiser_3d

  character(len=*), parameter :: onnx_axis0_attr = &
       '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]'
  character(len=*), parameter :: onnx_concat_axis0_attr = &
       '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]'
  character(len=*), parameter :: onnx_concat_axis1_attr = &
       '        "attribute": [{"name": "axis", "i": "1", "type": "INT"}]'
  character(len=*), parameter :: onnx_softmax_axis0_attr = &
       '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]'
  character(len=*), parameter :: onnx_transpose_10_attr = &
       '        "attribute": [{"name": "perm", "ints": ["1", "0"], ' // &
       '"type": "INTS"}]'
  character(len=*), parameter :: onnx_reduce_sum_attr = &
       '        "attribute": [{"name": "keepdims", "i": "0", ' // &
       '"type": "INT"}]'
  character(len=*), parameter :: onnx_cast_float_attr = &
       '        "attribute": [{"name": "to", "i": "1", "type": "INT"}]'
  character(len=*), parameter :: onnx_cast_int64_attr = &
       '        "attribute": [{"name": "to", "i": "7", "type": "INT"}]'
  character(len=*), parameter :: onnx_scatter_add_attr = &
       '        "attribute": [' // &
       '{"name": "axis", "i": "0", "type": "INT"}, ' // &
       '{"name": "reduction", "s": "YWRk", "type": "STRING"}]'

contains


!###############################################################################
  subroutine emit_msgpass_graph_inputs(prefix, input_shape, graph_inputs, &
       num_inputs)
    !! Emit the standard graph input tensors used by message-passing layers.
    !!
    !! Adds vertex features, optional edge features, edge_index, and degree.
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Input name prefix (e.g. "input_1")
    integer, dimension(:), intent(in) :: input_shape
    !! Layer input shape [num_vertex_features, num_edge_features]
    type(onnx_tensor_type), intent(inout), dimension(:) :: graph_inputs
    !! Accumulator for graph inputs
    integer, intent(inout) :: num_inputs
    !! Current number of graph inputs

    ! Vertex features: [num_nodes, nv]
    call add_graph_input_tensor( &
         graph_inputs, num_inputs, trim(prefix)//'_vertex', 1, &
         -1, 'num_nodes', input_shape(1), '')

    ! Edge features: [num_edges, ne]
    if(input_shape(2) .gt. 0)then
       call add_graph_input_tensor( &
            graph_inputs, num_inputs, trim(prefix)//'_edge', 1, &
            -1, 'num_edges', input_shape(2), '')
    end if

    ! Edge index: [3, num_csr_entries]
    call add_graph_input_tensor( &
         graph_inputs, num_inputs, trim(prefix)//'_edge_index', 7, &
         3, '', -1, 'num_csr_entries')

    ! Node degree: [num_nodes]
    call add_graph_input_tensor( &
         graph_inputs, num_inputs, trim(prefix)//'_degree', 7, &
         -1, 'num_nodes')

  end subroutine emit_msgpass_graph_inputs
!###############################################################################


!###############################################################################
  subroutine emit_output_identity(prefix, source_name, activation_name, &
       nodes, num_nodes)
    !! Emit a final Identity node using the standard ATHENA output naming.
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Layer node prefix
    character(*), intent(in) :: source_name
    !! Source tensor to rename
    character(*), intent(in) :: activation_name
    !! Final activation name used in the exported output suffix
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    !! Accumulator for ONNX nodes
    integer, intent(inout) :: num_nodes
    !! Current number of nodes

    ! Local variables
    character(:), allocatable :: suffix

    suffix = '_output'
    if(trim(activation_name) .ne. 'none')then
       suffix = '_' // trim(adjustl(activation_name)) // '_output'
    end if
    call emit_node('Identity', trim(prefix)//'_identity', &
         trim(prefix)//trim(suffix), '', nodes, num_nodes, &
         in1=trim(source_name))

  end subroutine emit_output_identity
!###############################################################################


!###############################################################################
  subroutine add_graph_input_tensor( &
       graph_inputs, num_inputs, name, elem_type, &
       dim1, dim_param1, dim2, dim_param2)
    !! Add one graph input tensor declaration to the ONNX input list.
    implicit none

    ! Arguments
    type(onnx_tensor_type), intent(inout), dimension(:) :: graph_inputs
    integer, intent(inout) :: num_inputs
    character(*), intent(in) :: name
    integer, intent(in) :: elem_type, dim1
    character(*), intent(in) :: dim_param1
    integer, optional, intent(in) :: dim2
    character(*), optional, intent(in) :: dim_param2

    num_inputs = num_inputs + 1
    graph_inputs(num_inputs)%name = trim(name)
    graph_inputs(num_inputs)%elem_type = elem_type
    if(present(dim2))then
       allocate(graph_inputs(num_inputs)%dims(2))
       allocate(graph_inputs(num_inputs)%dim_params(2))
       graph_inputs(num_inputs)%dims = [ dim1, dim2 ]
       graph_inputs(num_inputs)%dim_params(1) = dim_param1
       graph_inputs(num_inputs)%dim_params(2) = dim_param2
    else
       allocate(graph_inputs(num_inputs)%dims(1))
       allocate(graph_inputs(num_inputs)%dim_params(1))
       graph_inputs(num_inputs)%dims(1) = dim1
       graph_inputs(num_inputs)%dim_params(1) = dim_param1
    end if

  end subroutine add_graph_input_tensor
!###############################################################################


!###############################################################################
  subroutine get_timestep_output_name( &
       prefix, t, activation_name, inactive_suffix, activation_suffix, output)
    !! Build the canonical ONNX output name for one exported timestep.
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    integer, intent(in) :: t
    character(*), intent(in) :: activation_name
    character(*), intent(in) :: inactive_suffix, activation_suffix
    character(128), intent(out) :: output

    ! Local variables
    character(128) :: step_prefix

    write(step_prefix, '(A,"_t",I0)') trim(prefix), t
    if(trim(activation_name) .ne. 'none')then
       output = trim(step_prefix)
       if(len_trim(activation_suffix) .gt. 0)then
          output = trim(output) // trim(activation_suffix)
       end if
       output = trim(output) // '_' // trim(adjustl(activation_name)) // &
            '_output'
    else
       output = trim(step_prefix) // trim(inactive_suffix)
    end if

  end subroutine get_timestep_output_name
!###############################################################################


!###############################################################################
  subroutine emit_edge_index_component( &
       tp, edge_index_in, index_name, tag, component_out, &
       nodes, num_nodes)
    !! Gather one edge_index row and squeeze it into a vector.
    implicit none

    ! Arguments
    character(*), intent(in) :: tp, edge_index_in, index_name, tag
    character(128), intent(out) :: component_out
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    integer, intent(inout) :: num_nodes

    ! Local variables
    character(128) :: raw_name

    write(raw_name, '(A,"_",A,"_raw")') trim(tp), trim(tag)
    write(component_out, '(A,"_",A)') trim(tp), trim(tag)
    call emit_node('Gather', trim(tp)//'_gather_'//trim(tag), &
         trim(raw_name), onnx_axis0_attr, nodes, num_nodes, &
         in1=trim(edge_index_in), in2=trim(index_name))
    call emit_squeeze_node(trim(tp)//'_sq_'//trim(tag), &
         trim(raw_name), trim(tp)//'_idx0', trim(component_out), &
         nodes, num_nodes)

  end subroutine emit_edge_index_component
!###############################################################################


!###############################################################################
  subroutine emit_scatter_aggregator( &
       tp, vertex_in, target_in, message_in, feature_dim, &
       nodes, num_nodes, inits, num_inits, aggr_out)
    !! Emit the zero-initialise, expand, and scatter-add aggregation block.
    implicit none

    ! Arguments
    character(*), intent(in) :: tp, vertex_in, target_in, message_in
    integer, intent(in) :: feature_dim
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    integer, intent(inout) :: num_nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    integer, intent(inout) :: num_inits
    character(128), intent(out) :: aggr_out

    ! Local variables
    character(128) :: shape_name, nnodes_idx, nnodes_name
    character(128) :: feat_dim_name, aggr_shape, zeros_name
    character(128) :: target_us, axes1_name, msg_shape, target_exp

    ! Get num_nodes from shape of vertex_in.
    write(shape_name, '(A,"_vshape")') trim(tp)
    call emit_node('Shape', trim(tp)//'_shape_v', &
         trim(shape_name), '', nodes, num_nodes, &
         in1=trim(vertex_in))

    write(nnodes_idx, '(A,"_nnodes_idx")') trim(tp)
    call emit_constant_int64(trim(nnodes_idx), [0], [1], &
         nodes, num_nodes, inits, num_inits)

    write(nnodes_name, '(A,"_nnodes")') trim(tp)
    call emit_node('Gather', trim(tp)//'_gather_nn', &
         trim(nnodes_name), onnx_axis0_attr, nodes, num_nodes, &
         in1=trim(shape_name), in2=trim(nnodes_idx))

    ! Concat [num_nodes, feature_dim] to create the scatter target shape.
    write(feat_dim_name, '(A,"_feat_dim")') trim(tp)
    call emit_constant_int64(trim(feat_dim_name), [feature_dim], [1], &
         nodes, num_nodes, inits, num_inits)

    write(aggr_shape, '(A,"_aggr_shape")') trim(tp)
    call emit_node('Concat', trim(tp)//'_cat_shape', &
         trim(aggr_shape), onnx_concat_axis0_attr, nodes, num_nodes, &
         in1=trim(nnodes_name), in2=trim(feat_dim_name))

    ! ConstantOfShape creates the zero-filled aggregation buffer.
    write(zeros_name, '(A,"_zeros")') trim(tp)
    call emit_constant_of_shape_float(trim(tp)//'_zeros', &
         trim(aggr_shape), 0.0_real32, trim(zeros_name), &
         nodes, num_nodes, inits, num_inits)

    write(target_us, '(A,"_tgt_us")') trim(tp)
    write(axes1_name, '(A,"_us_ax1")') trim(tp)
    call emit_constant_int64(trim(axes1_name), [1], [1], &
         nodes, num_nodes, inits, num_inits)
    call emit_node('Unsqueeze', trim(tp)//'_us_tgt', &
         trim(target_us), '', nodes, num_nodes, &
         in1=trim(target_in), in2=trim(axes1_name))

    ! Expand target indices to match the message rank for ScatterElements.
    write(msg_shape, '(A,"_msg_shape")') trim(tp)
    call emit_node('Shape', trim(tp)//'_shape_msg', &
         trim(msg_shape), '', nodes, num_nodes, &
         in1=trim(message_in))

    write(target_exp, '(A,"_tgt_exp")') trim(tp)
    call emit_node('Expand', trim(tp)//'_expand_tgt', &
         trim(target_exp), '', nodes, num_nodes, &
         in1=trim(target_us), in2=trim(msg_shape))

    ! Scatter-add edge messages into the target-vertex slots.
    write(aggr_out, '(A,"_aggr")') trim(tp)
    call emit_node('ScatterElements', trim(tp)//'_scatter_add', &
         trim(aggr_out), onnx_scatter_add_attr, nodes, num_nodes, &
         in1=trim(zeros_name), in2=trim(target_exp), in3=trim(message_in))

  end subroutine emit_scatter_aggregator
!###############################################################################


!###############################################################################
  subroutine emit_weight_initialiser_2d( &
       name, nrows, ncols, weight_data, inits, num_inits)
    !! Store a 2D weight matrix as an ONNX initialiser in row-major order.
    implicit none

    ! Arguments
    character(*), intent(in) :: name
    integer, intent(in) :: nrows, ncols
    real(real32), intent(in) :: weight_data(:)
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    integer, intent(inout) :: num_inits

    num_inits = num_inits + 1
    inits(num_inits)%name = trim(name)
    inits(num_inits)%data_type = 1
    allocate(inits(num_inits)%dims(2))
    inits(num_inits)%dims = [ nrows, ncols ]
    allocate(inits(num_inits)%data(size(weight_data)))
    call col_to_row_major_2d(weight_data, inits(num_inits)%data, nrows, ncols)

  end subroutine emit_weight_initialiser_2d
!###############################################################################


!###############################################################################
  subroutine emit_weight_initialiser_3d( &
       name, nslices, nrows, ncols, weight_data, inits, num_inits)
    !! Store a stacked bank of 2D weight matrices as one ONNX tensor.
    implicit none

    ! Arguments
    character(*), intent(in) :: name
    integer, intent(in) :: nslices, nrows, ncols
    real(real32), intent(in) :: weight_data(:)
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    integer, intent(inout) :: num_inits

    num_inits = num_inits + 1
    inits(num_inits)%name = trim(name)
    inits(num_inits)%data_type = 1
    allocate(inits(num_inits)%dims(3))
    inits(num_inits)%dims = [ nslices, nrows, ncols ]
    allocate(inits(num_inits)%data(size(weight_data)))

    ! Transpose each 2D slice from column-major to row-major before export.
    block
      integer :: d, slice_size
      slice_size = nrows * ncols
      do d = 1, nslices
         call col_to_row_major_2d( &
              weight_data((d-1)*slice_size+1 : d*slice_size), &
              inits(num_inits)%data((d-1)*slice_size+1 : d*slice_size), &
              nrows, ncols)
      end do
    end block

  end subroutine emit_weight_initialiser_3d
!###############################################################################

end module athena__onnx_msgpass_utils
