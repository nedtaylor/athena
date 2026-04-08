submodule(athena__onnx) athena__onnx_write_submodule
  !! Submodule containing the ONNX export procedures.
  !!
  !! This submodule contains the routines that serialise ATHENA networks
  !! to the JSON representation used for ONNX interchange.
  use athena__io_utils, only: athena__version__
  use athena__base_layer, only: base_layer_type, learnable_layer_type
  use athena__onnx_nop_utils, only: emit_nop_metadata
  use athena__misc_types, only: &
       onnx_attribute_type, onnx_node_type, onnx_initialiser_type, &
       onnx_tensor_type
  use coreutils, only: to_lower, to_camel_case, stop_program

contains

!###############################################################################
  module subroutine write_onnx(file, network, format)
    !! Export a network to ONNX JSON format.
    !!
    !! GNN layers are exported as standard ONNX operators plus metadata
    !! needed to reconstruct the original ATHENA layer on import.
    use athena__onnx_utils, only: write_json_nodes, write_json_initialisers, &
         write_json_tensors
    implicit none

    ! Arguments
    class(network_type), intent(in) :: network
    !! Instance of the network
    character(*), intent(in) :: file
    !! Output file name
    class(*), optional, intent(in) :: format
    !! Export format: 'athena_abstract' (default) or 'onnx_expanded'

    ! Local variables
    type(onnx_node_type), allocatable :: nodes(:)
    !! Exported ONNX nodes
    type(onnx_initialiser_type), allocatable :: inits(:)
    !! Exported ONNX initialisers
    type(onnx_tensor_type), allocatable :: graph_inputs(:)
    !! Graph input tensor specifications
    type(onnx_tensor_type), allocatable :: graph_outputs(:)
    !! Graph output tensor specifications
    character(4096), allocatable :: gnn_metadata(:)
    !! Metadata entries required to reconstruct ATHENA GNN layers
    integer :: num_nodes, num_inits, num_inputs, num_outputs, num_gnn_meta
    !! Numbers of populated export records
    integer :: max_nodes, max_inits
    !! Pre-allocation sizes for node and initialiser storage
    integer :: ifmt
    !! Integer selector for export format


    !---------------------------------------------------------------------------
    ! Validate the export format and convert to integer selector
    !---------------------------------------------------------------------------
    ifmt = resolve_onnx_export_format(format)
    if(ifmt .eq. 0) return


    !--------------------------------------------------------------------------
    ! Initialise export storage
    !--------------------------------------------------------------------------
    call initialise_export_storage( &
         network, nodes, inits, graph_inputs, graph_outputs, gnn_metadata, &
         max_nodes, max_inits)

    num_nodes = 0
    num_inits = 0
    num_inputs = 0
    num_outputs = 0
    num_gnn_meta = 0


    !--------------------------------------------------------------------------
    ! Collect graph content
    !--------------------------------------------------------------------------
    call collect_export_nodes( &
         network, ifmt, nodes, num_nodes, max_nodes, &
         inits, num_inits, max_inits, &
         gnn_metadata, num_gnn_meta)
    call build_graph_inputs( &
         network, ifmt, graph_inputs, num_inputs)
    call build_graph_outputs( &
         network, ifmt, graph_outputs, num_outputs)


    !--------------------------------------------------------------------------
    ! Write the JSON model
    !--------------------------------------------------------------------------
    call write_onnx_json_file( &
         file, ifmt, nodes, num_nodes, inits, num_inits, &
         graph_inputs, num_inputs, graph_outputs, num_outputs, &
         gnn_metadata, num_gnn_meta)

  end subroutine write_onnx
!###############################################################################


!###############################################################################
  function resolve_onnx_export_format(format) result(ifmt)
    !! Resolve the ONNX export format into the internal integer selector.
    implicit none

    ! Arguments
    class(*), optional, intent(in) :: format
    !! Export format as a string name or integer selector

    integer :: ifmt
    !! Integer selector for the export format (1=athena_abstract, 2=onnx_expanded)

    ! Local variables
    character(32) :: format_name
    !! Normalised string representation of the requested export format
    character(128) :: err_msg
    !! Error buffer used for unsupported integer selectors

    ifmt = 1

    if(present(format))then
       select type(format)
       type is(character(*))
          format_name = to_lower(trim(adjustl(format)))
          select case(trim(format_name))
          case('athena_abstract')
             ifmt = 1
          case('onnx_expanded')
             ifmt = 2
          case default
             call stop_program('write_onnx: unrecognised export format: ' // &
                  trim(format_name))
             ifmt = 0
             return
          end select
       type is(integer)
          ifmt = format
       class default
          call stop_program('write_onnx: unrecognised export format type')
          ifmt = 0
          return
       end select
    end if

    select case(ifmt)
    case(1, 2)
       continue
    case default
       write(err_msg, '("write_onnx: unrecognised export format selector: ",I0)') ifmt
       call stop_program(err_msg)
       ifmt = 0
       return
    end select

  end function resolve_onnx_export_format
!###############################################################################


!###############################################################################
  subroutine initialise_export_storage( &
       network, nodes, inits, graph_inputs, graph_outputs, gnn_metadata, &
       max_nodes, max_inits)
    !! Allocate the working arrays used during ONNX export.
    implicit none

    ! Arguments
    class(network_type), intent(in) :: network
    !! Instance of the network
    type(onnx_node_type), allocatable, intent(out) :: nodes(:)
    !! Exported node storage
    type(onnx_initialiser_type), allocatable, intent(out) :: inits(:)
    !! Exported initialiser storage
    type(onnx_tensor_type), allocatable, intent(out) :: graph_inputs(:)
    !! Exported graph input storage
    type(onnx_tensor_type), allocatable, intent(out) :: graph_outputs(:)
    !! Exported graph output storage
    character(4096), allocatable, intent(out) :: gnn_metadata(:)
    !! Metadata storage for GNN layers
    integer, intent(out) :: max_nodes, max_inits
    !! Pre-allocation sizes

    max_nodes = network%auto_graph%num_vertices * 60
    max_inits = network%auto_graph%num_vertices * 30

    allocate(nodes(max_nodes))
    allocate(inits(max_inits))
    allocate(graph_inputs(network%auto_graph%num_vertices * 5))
    allocate(graph_outputs(network%auto_graph%num_vertices * 3))
    allocate(gnn_metadata(network%auto_graph%num_vertices))

  end subroutine initialise_export_storage
!###############################################################################


!###############################################################################
  subroutine collect_export_nodes( &
       network, ifmt, nodes, num_nodes, max_nodes, &
       inits, num_inits, max_inits, &
       gnn_metadata, num_gnn_meta)
    !! Build the ONNX nodes, initialisers and GNN metadata.
    implicit none

    ! Arguments
    class(network_type), intent(in) :: network
    !! Instance of the network
    integer, intent(in) :: ifmt
    !! Export format selector
    type(onnx_node_type), intent(inout) :: nodes(:)
    !! Exported ONNX nodes
    integer, intent(inout) :: num_nodes, max_nodes
    !! Node counter and allocation limit
    type(onnx_initialiser_type), intent(inout) :: inits(:)
    !! Exported ONNX initialisers
    integer, intent(inout) :: num_inits, max_inits
    !! Initialiser counter and allocation limit
    character(4096), intent(inout) :: gnn_metadata(:)
    !! Exported GNN metadata entries
    integer, intent(inout) :: num_gnn_meta
    !! Number of metadata entries

    ! Local variables
    integer :: i, ii, layer_id, layer_num, lid
    !! Loop index and layer identifier
    character(128) :: node_name, input_name
    !! Node name prefix and sequential input name
    logical :: is_last_layer
    !! Whether the current NOP is the last non-input layer

    if(ifmt .eq. 2)then
       layer_num = 0
       input_name = 'input'

       do i = 1, network%auto_graph%num_vertices
          layer_id = network%auto_graph%vertex(network%vertex_order(i))%id
          if(trim(network%model(layer_id)%layer%type) .eq. 'inpt') cycle

          if(trim(network%model(layer_id)%layer%type) .ne. 'nop')then
             call stop_program( &
                  'write_onnx: pytorch format supports NOP layers only')
             return
          end if

          layer_num = layer_num + 1
          write(node_name, '("layer",I0)') layer_num

          is_last_layer = .true.
          do ii = i + 1, network%auto_graph%num_vertices
             lid = network%auto_graph%vertex(network%vertex_order(ii))%id
             if(trim(network%model(lid)%layer%type) .ne. 'inpt')then
                is_last_layer = .false.
                exit
             end if
          end do

          call network%model(layer_id)%layer%emit_onnx_nodes( &
               trim(node_name), nodes, num_nodes, max_nodes, &
               inits, num_inits, max_inits, input_name=trim(input_name), &
               is_last_layer=is_last_layer, format=ifmt)

          call update_pytorch_prev_output( &
               network%model(layer_id)%layer, trim(node_name), &
               is_last_layer, input_name)
       end do

       return
    end if

    do i = 1, network%auto_graph%num_vertices
       layer_id = network%auto_graph%vertex(network%vertex_order(i))%id
       write(node_name, '("node_",I0)') network%model(layer_id)%layer%id

       select case(trim(network%model(layer_id)%layer%type))
       case('inpt')
          cycle
       case('msgp')
          call emit_gnn_input_renames( &
               network, layer_id, i, nodes, num_nodes)
          call network%model(layer_id)%layer%emit_onnx_nodes( &
               trim(node_name), nodes, num_nodes, max_nodes, &
               inits, num_inits, max_inits)
          call build_gnn_metadata( &
               network%model(layer_id)%layer, trim(node_name), &
               gnn_metadata, num_gnn_meta)
       case('nop')
          call emit_standard_node_json( &
               network, layer_id, i, nodes, num_nodes, max_nodes, &
               inits, num_inits, max_inits)
          call emit_nop_metadata( &
               network%model(layer_id)%layer, trim(node_name), &
               gnn_metadata, num_gnn_meta)
       case default
          call emit_standard_node_json( &
               network, layer_id, i, nodes, num_nodes, max_nodes, &
               inits, num_inits, max_inits)
       end select
    end do

  end subroutine collect_export_nodes
!###############################################################################


!###############################################################################
  subroutine update_pytorch_prev_output(layer, prefix, is_last_layer, output)
    !! Resolve the downstream tensor name after emitting one PyTorch-format NOP.
    implicit none

    class(base_layer_type), intent(in) :: layer
    character(*), intent(in) :: prefix
    logical, intent(in) :: is_last_layer
    character(128), intent(inout) :: output

    if(is_last_layer)then
       output = 'output'
       return
    end if

    select type(layer)
    class is(learnable_layer_type)
       if(trim(layer%activation%name) .ne. 'none')then
          write(output, '("/",A,"/Relu_output_0")') trim(prefix)
       else
          write(output, '("/",A,"/Transpose_1_output_0")') trim(prefix)
       end if
    class default
       write(output, '("/",A,"/Transpose_1_output_0")') trim(prefix)
    end select

  end subroutine update_pytorch_prev_output
!###############################################################################


!###############################################################################
  subroutine build_graph_inputs(network, ifmt, graph_inputs, num_inputs)
    !! Build the ONNX graph input tensor specifications.
    implicit none

    ! Arguments
    class(network_type), intent(in) :: network
    !! Instance of the network
    integer, intent(in) :: ifmt
    !! Export format selector
    type(onnx_tensor_type), intent(inout) :: graph_inputs(:)
    !! Graph input tensor specifications
    integer, intent(inout) :: num_inputs
    !! Number of graph inputs

    ! Local variables
    integer :: i, j, layer_id
    !! Loop indices and current layer identifier

    if(ifmt .eq. 2)then
       do i = 1, network%auto_graph%num_vertices
          layer_id = network%auto_graph%vertex(network%vertex_order(i))%id
          if(trim(network%model(layer_id)%layer%type) .ne. 'inpt') cycle
          num_inputs = 1
          graph_inputs(1)%name = 'input'
          graph_inputs(1)%elem_type = 1
          allocate(graph_inputs(1)%dims(2))
          graph_inputs(1)%dims = [ &
               1, network%model(layer_id)%layer%input_shape(1)]
          return
       end do
       return
    end if

    do i = 1, size(network%root_vertices, dim=1)
       layer_id = network%auto_graph%vertex(network%root_vertices(i))%id

       if(network%model(layer_id)%layer%use_graph_output)then
          num_inputs = num_inputs + 1
          write(graph_inputs(num_inputs)%name, '("input_",I0,"_vertex")') &
               network%model(layer_id)%layer%id
          graph_inputs(num_inputs)%elem_type = 1
          allocate(graph_inputs(num_inputs)%dims(2))
          allocate(graph_inputs(num_inputs)%dim_params(2))
          graph_inputs(num_inputs)%dim_params(1) = 'num_nodes'
          graph_inputs(num_inputs)%dims(1) = -1
          graph_inputs(num_inputs)%dim_params(2) = ''
          graph_inputs(num_inputs)%dims(2) = &
               network%model(layer_id)%layer%input_shape(1)

          if(network%model(layer_id)%layer%input_shape(2) .gt. 0)then
             num_inputs = num_inputs + 1
             write(graph_inputs(num_inputs)%name, '("input_",I0,"_edge")') &
                  network%model(layer_id)%layer%id
             graph_inputs(num_inputs)%elem_type = 1
             allocate(graph_inputs(num_inputs)%dims(2))
             allocate(graph_inputs(num_inputs)%dim_params(2))
             graph_inputs(num_inputs)%dim_params(1) = 'num_edges'
             graph_inputs(num_inputs)%dims(1) = -1
             graph_inputs(num_inputs)%dim_params(2) = ''
             graph_inputs(num_inputs)%dims(2) = &
                  network%model(layer_id)%layer%input_shape(2)
          end if

          num_inputs = num_inputs + 1
          write(graph_inputs(num_inputs)%name, &
               '("input_",I0,"_edge_index")') &
               network%model(layer_id)%layer%id
          graph_inputs(num_inputs)%elem_type = 7
          allocate(graph_inputs(num_inputs)%dims(2))
          allocate(graph_inputs(num_inputs)%dim_params(2))
          graph_inputs(num_inputs)%dim_params(1) = ''
          graph_inputs(num_inputs)%dims(1) = 3
          graph_inputs(num_inputs)%dim_params(2) = 'num_csr_entries'
          graph_inputs(num_inputs)%dims(2) = -1

          num_inputs = num_inputs + 1
          write(graph_inputs(num_inputs)%name, '("input_",I0,"_degree")') &
               network%model(layer_id)%layer%id
          graph_inputs(num_inputs)%elem_type = 7
          allocate(graph_inputs(num_inputs)%dims(1))
          allocate(graph_inputs(num_inputs)%dim_params(1))
          graph_inputs(num_inputs)%dim_params(1) = 'num_nodes'
          graph_inputs(num_inputs)%dims(1) = -1
       else
          num_inputs = num_inputs + 1
          write(graph_inputs(num_inputs)%name, '("input_",I0)') &
               network%model(layer_id)%layer%id
          graph_inputs(num_inputs)%elem_type = 1
          allocate(graph_inputs(num_inputs)%dims( &
               size(network%model(layer_id)%layer%input_shape) + 1))
          allocate(graph_inputs(num_inputs)%dim_params( &
               size(network%model(layer_id)%layer%input_shape) + 1))
          graph_inputs(num_inputs)%dim_params(1) = 'batch_size'
          graph_inputs(num_inputs)%dims(1) = -1

          do j = 1, size(network%model(layer_id)%layer%input_shape)
             graph_inputs(num_inputs)%dim_params(j+1) = ''
             graph_inputs(num_inputs)%dims(j+1) = &
                  network%model(layer_id)%layer%input_shape(j)
          end do
       end if
    end do

  end subroutine build_graph_inputs
!###############################################################################


!###############################################################################
  subroutine build_graph_outputs(network, ifmt, graph_outputs, num_outputs)
    !! Build the ONNX graph output tensor specifications.
    implicit none

    ! Arguments
    class(network_type), intent(in) :: network
    !! Instance of the network
    integer, intent(in) :: ifmt
    !! Export format selector
    type(onnx_tensor_type), intent(inout) :: graph_outputs(:)
    !! Graph output tensor specifications
    integer, intent(inout) :: num_outputs
    !! Number of graph outputs

    ! Local variables
    integer :: i, j, layer_id
    !! Loop indices and current layer identifier
    character(:), allocatable :: suffix
    !! Optional activation suffix for output names

    if(ifmt .eq. 2)then
       layer_id = 0
       do i = 1, network%auto_graph%num_vertices
          j = network%vertex_order(i)
          if(all(network%leaf_vertices(:) .ne. j)) cycle
          layer_id = network%auto_graph%vertex(j)%id
          exit
       end do

       if(layer_id .eq. 0) return

       num_outputs = 1
       graph_outputs(1)%name = 'output'
       graph_outputs(1)%elem_type = 1
       if(allocated(graph_outputs(1)%dims)) deallocate(graph_outputs(1)%dims)
       if(allocated(graph_outputs(1)%dim_params)) then
          deallocate(graph_outputs(1)%dim_params)
       end if
       allocate(graph_outputs(1)%dims(2))
       allocate(graph_outputs(1)%dim_params(2))
       graph_outputs(1)%dim_params(1) = 'batch_size'
       graph_outputs(1)%dims(1) = -1
       graph_outputs(1)%dim_params(2) = ''
       graph_outputs(1)%dims(2) = network%model(layer_id)%layer%output_shape(1)
       return
    end if

    do i = 1, size(network%leaf_vertices, dim=1)
       layer_id = network%auto_graph%vertex(network%leaf_vertices(i))%id
       suffix = ''

       select type(layer => network%model(layer_id)%layer)
       class is(learnable_layer_type)
          if(layer%activation%name .ne. 'none')then
             suffix = '_' // trim(adjustl(layer%activation%name))
          end if
       end select

       num_outputs = num_outputs + 1
       write(graph_outputs(num_outputs)%name, '("node_",I0,A,"_output")') &
            network%model(layer_id)%layer%id, trim(adjustl(suffix))
       graph_outputs(num_outputs)%elem_type = 1
       allocate(graph_outputs(num_outputs)%dims( &
            size(network%model(layer_id)%layer%output_shape) + 1))
       allocate(graph_outputs(num_outputs)%dim_params( &
            size(network%model(layer_id)%layer%output_shape) + 1))
       graph_outputs(num_outputs)%dim_params(1) = 'batch_size'
       graph_outputs(num_outputs)%dims(1) = -1

       do j = 1, size(network%model(layer_id)%layer%output_shape)
          graph_outputs(num_outputs)%dim_params(j+1) = ''
          graph_outputs(num_outputs)%dims(j+1) = &
               network%model(layer_id)%layer%output_shape(j)
       end do
    end do

  end subroutine build_graph_outputs
!###############################################################################


!###############################################################################
  subroutine write_onnx_json_file( &
       file, ifmt, nodes, num_nodes, inits, num_inits, &
       graph_inputs, num_inputs, graph_outputs, num_outputs, &
       gnn_metadata, num_gnn_meta)
    !! Write the collected export data to disk.
    use athena__onnx_utils, only: write_json_nodes, write_json_initialisers, &
         write_json_tensors
    implicit none

    ! Arguments
    character(*), intent(in) :: file
    !! Output file name
    integer, intent(in) :: ifmt
    !! Export format selector
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Exported ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of exported nodes
    type(onnx_initialiser_type), intent(in) :: inits(:)
    !! Exported ONNX initialisers
    integer, intent(in) :: num_inits
    !! Number of exported initialisers
    type(onnx_tensor_type), intent(in) :: graph_inputs(:)
    !! Graph input tensor specifications
    integer, intent(in) :: num_inputs
    !! Number of graph inputs
    type(onnx_tensor_type), intent(in) :: graph_outputs(:)
    !! Graph output tensor specifications
    integer, intent(in) :: num_outputs
    !! Number of graph outputs
    character(4096), intent(in) :: gnn_metadata(:)
    !! GNN metadata strings
    integer, intent(in) :: num_gnn_meta
    !! Number of metadata entries

    ! Local variables
    integer :: unit, i
    !! Output unit and loop index

    open(newunit=unit, file=file, status='replace')
    write(unit, '(A)') '{'
    if(ifmt .eq. 2)then
       write(unit, '(A)') '  "irVersion": "7",'
       write(unit, '(A)') '  "producerName": "pytorch",'
       write(unit, '(A)') '  "producerVersion": "2.7.1",'
    else
       write(unit, '(A)') '  "irVersion": "8",'
       write(unit, '(A)') '  "producerName": "Athena",'
       write(unit, '(A,A,A)') '  "producerVersion": "', &
            trim(athena__version__), '",'
    end if
    write(unit, '(A)') '  "graph": {'

    call write_json_nodes(unit, nodes, num_nodes)
    write(unit, '(A)') ','
    if(ifmt .eq. 2)then
       write(unit, '(A)') '    "name": "main_graph",'
    else
       write(unit, '(A)') '    "name": "athena_network",'
    end if
    call write_json_initialisers(unit, inits, num_inits)
    write(unit, '(A)') ','
    call write_json_tensors(unit, 'input', graph_inputs, num_inputs)
    write(unit, '(A)') ','
    call write_json_tensors(unit, 'output', graph_outputs, num_outputs)

    if(ifmt .ne. 2 .and. num_gnn_meta .gt. 0)then
       write(unit, '(A)') ','
       write(unit, '(A)') '    "metadataProps": ['
       do i = 1, num_gnn_meta
          if(i .gt. 1) write(unit, '(A)') ','
          write(unit, '(A)') trim(gnn_metadata(i))
       end do
       write(unit, '(A)') ''
       write(unit, '(A)') '    ]'
    end if

    write(unit, '(A)') '  },'
    write(unit, '(A)') '  "opsetImport": ['
    write(unit, '(A)') '    {'
    if(ifmt .eq. 2)then
       write(unit, '(A)') '      "version": "14"'
    else
       write(unit, '(A)') '      "version": "17"'
    end if
    write(unit, '(A)') '    }'
    write(unit, '(A)') '  ]'
    write(unit, '(A)') '}'
    close(unit)

  end subroutine write_onnx_json_file
!###############################################################################


!###############################################################################
  subroutine emit_standard_node_json( &
       network, layer_id, vertex_idx, &
       nodes, num_nodes, max_nodes, &
       inits, num_inits, max_inits)
    !! Emit ONNX node records for a standard, non-GNN layer.
    use athena__onnx_utils, only: emit_initialisers, build_attributes_json, &
         emit_activation_node
    implicit none

    ! Arguments
    class(network_type), intent(in) :: network
    !! Instance of the network
    integer, intent(in) :: layer_id, vertex_idx
    !! Layer identifier and vertex position
    type(onnx_node_type), intent(inout) :: nodes(:)
    !! Exported ONNX nodes
    integer, intent(inout) :: num_nodes, max_nodes
    !! Node counter and allocation limit
    type(onnx_initialiser_type), intent(inout) :: inits(:)
    !! Exported ONNX initialisers
    integer, intent(inout) :: num_inits, max_inits
    !! Initialiser counter and allocation limit

    ! Local variables
    character(128) :: node_name, layer_name, input_name
    !! Temporary strings used to build node names
    character(:), allocatable :: suffix
    !! Optional activation suffix for an input tensor name
    integer :: j, input_layer_id, n_inputs
    !! Loop index, source layer identifier and input count
    character(128), allocatable :: input_list(:)
    !! Input tensor names
    character(4096) :: attr_json
    !! Pre-formatted JSON attributes

    write(node_name, '("node_", I0)') network%model(layer_id)%layer%id

    select case(trim(network%model(layer_id)%layer%type))
    case('full')
       layer_name = 'Gemm'
    case('conv')
       layer_name = 'Conv'
    case('pool')
       layer_name = to_camel_case( &
            trim(adjustl(network%model(layer_id)%layer%subtype)) // '_' // &
            trim(adjustl(network%model(layer_id)%layer%type)), &
            capitalise_first_letter = .true.)
    case('actv')
       layer_name = to_camel_case( &
            trim(adjustl(network%model(layer_id)%layer%subtype)), &
            capitalise_first_letter = .true.)
    case('flat')
       layer_name = 'Flatten'
    case('batc')
       layer_name = 'BatchNormalization'
    case('drop')
       layer_name = 'Dropout'
    case('nop')
       layer_name = to_camel_case( &
            trim(adjustl(network%model(layer_id)%layer%name)), &
            capitalise_first_letter = .true.)
    case default
       layer_name = 'Unknown'
    end select

    n_inputs = 0
    allocate(input_list(100))

    do j = 1, network%auto_graph%num_vertices
       input_layer_id = network%auto_graph%vertex(j)%id
       if(network%auto_graph%adjacency( &
            j, network%vertex_order(vertex_idx)) .eq. 0) cycle

       if(all(network%auto_graph%adjacency(:,j) .eq. 0))then
          write(input_name, '("input_",I0)') &
               network%model(input_layer_id)%layer%id
          suffix = ''
       else
          write(input_name, '("node_",I0)') &
               network%model(input_layer_id)%layer%id
          suffix = '_output'
          select type(prev => network%model(input_layer_id)%layer)
          class is(learnable_layer_type)
             if(prev%activation%name .ne. 'none')then
                suffix = '_' // trim(adjustl(prev%activation%name)) // &
                     '_output'
             end if
          end select
       end if

       n_inputs = n_inputs + 1
       write(input_list(n_inputs), '(A,A)') trim(adjustl(input_name)), suffix
    end do

    select type(layer => network%model(layer_id)%layer)
    class is(learnable_layer_type)
       do j = 1, size(layer%params)
          n_inputs = n_inputs + 1
          write(input_list(n_inputs), '(A,"_param",I0)') trim(node_name), j
       end do
    end select

    call build_attributes_json( &
         network%model(layer_id)%layer, trim(layer_name), attr_json)

    num_nodes = num_nodes + 1
    nodes(num_nodes)%name = trim(node_name)
    nodes(num_nodes)%op_type = trim(layer_name)
    allocate(nodes(num_nodes)%inputs(n_inputs))
    nodes(num_nodes)%inputs = input_list(1:n_inputs)
    allocate(nodes(num_nodes)%outputs(1))
    write(nodes(num_nodes)%outputs(1), '(A,"_output")') trim(node_name)
    nodes(num_nodes)%attributes_json = attr_json

    select type(layer => network%model(layer_id)%layer)
    class is(learnable_layer_type)
       call emit_initialisers(layer, trim(node_name), inits, num_inits, &
            max_inits)
       if(layer%activation%name .ne. 'none')then
          call emit_activation_node( &
               layer%activation%name, trim(node_name), '', &
               nodes, num_nodes, max_nodes)
       end if
    end select

    deallocate(input_list)

  end subroutine emit_standard_node_json
!###############################################################################


!###############################################################################
  subroutine emit_gnn_input_renames( &
       network, layer_id, vertex_idx, nodes, num_nodes)
    !! Emit Identity nodes that rename GNN inputs to the expected convention.
    use athena__onnx_utils, only: emit_node
    implicit none

    ! Arguments
    class(network_type), intent(in) :: network
    !! Instance of the network
    integer, intent(in) :: layer_id, vertex_idx
    !! Layer identifier and vertex position
    type(onnx_node_type), intent(inout) :: nodes(:)
    !! Exported ONNX nodes
    integer, intent(inout) :: num_nodes
    !! Number of exported nodes

    ! Local variables
    integer :: j, input_layer_id
    !! Loop index and source layer identifier
    character(128) :: prefix, vertex_in, edge_in, edge_index_in, degree_in
    !! Temporary tensor names
    character(:), allocatable :: suffix
    !! Optional activation suffix for chained vertex inputs

    write(prefix, '("node_",I0)') network%model(layer_id)%layer%id

    vertex_in = ''
    edge_in = ''
    edge_index_in = ''
    degree_in = ''

    do j = 1, network%auto_graph%num_vertices
       input_layer_id = network%auto_graph%vertex(j)%id
       if(network%auto_graph%adjacency( &
            j, network%vertex_order(vertex_idx)) .eq. 0) cycle

       if(all(network%auto_graph%adjacency(:,j) .eq. 0))then
          write(vertex_in, '("input_",I0,"_vertex")') &
               network%model(input_layer_id)%layer%id
          write(edge_in, '("input_",I0,"_edge")') &
               network%model(input_layer_id)%layer%id
          write(edge_index_in, '("input_",I0,"_edge_index")') &
               network%model(input_layer_id)%layer%id
          write(degree_in, '("input_",I0,"_degree")') &
               network%model(input_layer_id)%layer%id
       else
          suffix = '_output'
          select type(prev => network%model(input_layer_id)%layer)
          class is(learnable_layer_type)
             if(prev%activation%name .ne. 'none')then
                suffix = '_' // trim(adjustl(prev%activation%name)) // &
                     '_output'
             end if
          end select
          write(vertex_in, '("node_",I0,A)') &
               network%model(input_layer_id)%layer%id, suffix
       end if
    end do

    if(len_trim(vertex_in) .gt. 0)then
       call emit_node('Identity', trim(prefix)//'_rename_vertex', &
            trim(prefix)//'_vertex_in', '', nodes, num_nodes, &
            in1=trim(vertex_in))
    end if

    if(len_trim(edge_in) .gt. 0)then
       call emit_node('Identity', trim(prefix)//'_rename_edge', &
            trim(prefix)//'_edge_in', '', nodes, num_nodes, &
            in1=trim(edge_in))
    end if

    if(len_trim(edge_index_in) .gt. 0)then
       call emit_node('Identity', trim(prefix)//'_rename_edge_index', &
            trim(prefix)//'_edge_index_in', '', nodes, num_nodes, &
            in1=trim(edge_index_in))
    end if

    if(len_trim(degree_in) .gt. 0)then
       call emit_node('Identity', trim(prefix)//'_rename_degree', &
            trim(prefix)//'_degree_in', '', nodes, num_nodes, &
            in1=trim(degree_in))
    end if

  end subroutine emit_gnn_input_renames
!###############################################################################


!###############################################################################
  subroutine build_gnn_metadata(layer, prefix, metadata, num_meta)
    !! Build the metadata entry required to reconstruct a GNN layer.
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: layer
    !! GNN layer instance
    character(*), intent(in) :: prefix
    !! Node prefix used for this exported layer
    character(4096), intent(inout) :: metadata(:)
    !! Metadata strings to append to
    integer, intent(inout) :: num_meta
    !! Number of metadata entries

    ! Local variables
    type(onnx_attribute_type), allocatable :: attrs(:)
    !! Layer attributes returned by polymorphic dispatch
    integer :: i
    !! Loop index
    character(2048) :: value_str
    !! Semicolon-separated metadata payload

    attrs = layer%get_attributes()
    if(.not.allocated(attrs) .or. size(attrs) .eq. 0) return

    value_str = 'subtype=' // trim(adjustl(layer%name))
    do i = 1, size(attrs)
       value_str = trim(value_str) // ';' // trim(attrs(i)%name) // '=' // &
            trim(adjustl(attrs(i)%val))
    end do

    num_meta = num_meta + 1
    write(metadata(num_meta), '(A)') &
         '      {"key": "athena_gnn_' // trim(prefix) // &
         '", "value": "' // trim(value_str) // '"}'

  end subroutine build_gnn_metadata
!###############################################################################


!###############################################################################


end submodule athena__onnx_write_submodule
