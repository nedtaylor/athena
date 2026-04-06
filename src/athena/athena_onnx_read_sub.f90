submodule(athena__onnx) athena__onnx_read_submodule
  !! Submodule containing the ONNX import procedures.
  !!
  !! This submodule contains the routines that parse the JSON ONNX
  !! representation and rebuild ATHENA networks from it.
  use athena__misc_types, only: onnx_attribute_type, onnx_node_type, &
       onnx_initialiser_type, onnx_tensor_type
  use coreutils, only: real32, to_lower
  implicit none

  integer, parameter :: MAX_ITEMS = 500
  integer, parameter :: MAX_GNN_METADATA = 100

  type :: json_parse_result_type
     type(onnx_node_type) :: nodes(MAX_ITEMS)
     integer :: num_nodes = 0
     type(onnx_initialiser_type) :: inits(MAX_ITEMS)
     integer :: num_inits = 0
     type(onnx_tensor_type) :: inputs(MAX_ITEMS)
     integer :: num_inputs = 0
     type(onnx_tensor_type) :: outputs(MAX_ITEMS)
     integer :: num_outputs = 0
     character(256) :: meta_keys(MAX_GNN_METADATA)
     character(256) :: meta_values(MAX_GNN_METADATA)
     integer :: num_meta = 0
  end type json_parse_result_type

  type :: json_node_state_type
     logical :: in_object = .false.
     logical :: in_attribute = .false.
     character(256) :: name = ''
     character(256) :: op_type = ''
     character(128), allocatable :: inputs(:)
     character(128), allocatable :: outputs(:)
     integer :: num_inputs = 0
     integer :: num_outputs = 0
     type(onnx_attribute_type), allocatable :: attrs(:)
     integer :: num_attrs = 0
  end type json_node_state_type

  type :: json_initialiser_state_type
     logical :: in_object = .false.
     character(128) :: name = ''
     integer :: data_type = 1
     integer, allocatable :: dims(:)
     character(:), allocatable :: raw_data
  end type json_initialiser_state_type

  type :: json_tensor_state_type
     logical :: in_object = .false.
     integer :: object_depth = 0
     character(128) :: name = ''
     integer :: elem_type = 1
     integer, allocatable :: dim_values(:)
  end type json_tensor_state_type

  type :: json_parser_state_type
     character(32) :: section = ''
     type(json_node_state_type) :: node
     type(json_initialiser_state_type) :: initialiser
     type(json_tensor_state_type) :: input_tensor
     type(json_tensor_state_type) :: output_tensor
  end type json_parser_state_type

contains

!###############################################################################
  module function read_onnx(file, verbose) result(network)
    !! Import a network from ONNX JSON format.
    !!
    !! The parser keeps section-specific state in small helper types so the
    !! main procedure only coordinates file I/O and dispatch.
    implicit none

    ! Arguments
    character(*), intent(in) :: file
    !! File to import the network from
    integer, intent(in), optional :: verbose
    !! Verbosity level

    type(network_type) :: network
    !! Resulting network instance

    ! Local variables
    integer :: unit, stat, verbose_
    !! File unit, I/O status and effective verbosity
    character(131072) :: line
    !! File input buffer sized for large base64-encoded initialisers
    character(:), allocatable :: trimmed
    !! Current input line with leading and trailing whitespace removed
    type(json_parse_result_type) :: parsed
    !! Parsed ONNX records collected across all sections
    type(json_parser_state_type) :: parser
    !! Section-specific parser state
    logical :: has_gnn
    !! Whether the parsed model contains ATHENA GNN metadata


    !--------------------------------------------------------------------------
    ! Initialise options and parser state
    !--------------------------------------------------------------------------
    if(present(verbose))then
       verbose_ = verbose
    else
       verbose_ = 0
    end if

    call initialise_json_parser(parser)


    !--------------------------------------------------------------------------
    ! Read and dispatch JSON lines
    !--------------------------------------------------------------------------
    open(newunit=unit, file=file, status='old', action='read', iostat=stat)
    if(stat .ne. 0)then
       write(*,*) 'ERROR: Could not open file: ', trim(file)
       return
    end if

    do
       read(unit, '(A)', iostat=stat) line
       if(stat .ne. 0) exit

       trimmed = trim(adjustl(line))
       if(len_trim(trimmed) .eq. 0) cycle

       call detect_json_section(trimmed, parser)

       select case(trim(parser%section))
       case('node')
          call parse_node_section_line(trimmed, parser%node, parsed, &
               parser%section)
       case('initializer')
          call parse_initialiser_section_line(trimmed, parser%initialiser, &
               parsed, parser%section)
       case('input')
          call parse_tensor_section_line(trimmed, parser%input_tensor, &
               parsed%inputs, parsed%num_inputs, parser%section)
       case('output')
          call parse_tensor_section_line(trimmed, parser%output_tensor, &
               parsed%outputs, parsed%num_outputs, parser%section)
       case('metadata')
          call parse_metadata_line(trimmed, parsed, parser%section)
       end select
    end do

    close(unit)

    if(verbose_ .gt. 0)then
       write(*,*) 'JSON parse: ', parsed%num_nodes, ' nodes, ', &
            parsed%num_inits, ' initialisers, ', parsed%num_inputs, &
            ' inputs, ', parsed%num_outputs, ' outputs, ', &
            parsed%num_meta, ' metadata'
    end if


    !--------------------------------------------------------------------------
    ! Build the ATHENA network from the parsed JSON records
    !--------------------------------------------------------------------------
    has_gnn = parsed%num_meta .gt. 0

    if(has_gnn)then
       call build_network_from_json_gnn( &
            network, parsed%nodes, parsed%num_nodes, &
            parsed%inits, parsed%num_inits, &
            parsed%inputs, parsed%num_inputs, &
            parsed%outputs, parsed%num_outputs, &
            parsed%meta_keys, parsed%meta_values, parsed%num_meta, &
            verbose_)
    else
       call build_network_from_json_standard( &
            network, parsed%nodes, parsed%num_nodes, &
            parsed%inits, parsed%num_inits, &
            parsed%inputs, parsed%num_inputs, verbose_)
    end if

  end function read_onnx
!###############################################################################


!###############################################################################
  subroutine initialise_json_parser(parser)
    !! Initialise the reusable parser state objects.
    implicit none

    type(json_parser_state_type), intent(out) :: parser

    parser%section = ''
    call reset_node_state(parser%node)
    call reset_initialiser_state(parser%initialiser)
    call reset_tensor_state(parser%input_tensor)
    call reset_tensor_state(parser%output_tensor)

  end subroutine initialise_json_parser
!###############################################################################


!###############################################################################
  subroutine detect_json_section(line, parser)
    !! Detect the active top-level graph section.
    implicit none

    character(*), intent(in) :: line
    type(json_parser_state_type), intent(inout) :: parser

    if(len_trim(parser%section) .gt. 0) return

    if(index(line, '"node"') .gt. 0 .and. index(line, '[') .gt. 0)then
       parser%section = 'node'
       return
    end if

    if(index(line, '"initializer"') .gt. 0 .and. index(line, '[') .gt. 0)then
       parser%section = 'initializer'
       return
    end if

    if(index(line, '"input"') .gt. 0 .and. index(line, '[') .gt. 0)then
       parser%section = 'input'
       return
    end if

    if(index(line, '"output"') .gt. 0 .and. index(line, '[') .gt. 0)then
       parser%section = 'output'
       return
    end if

    if(index(line, '"metadataProps"') .gt. 0 .and. index(line, '[') .gt. 0)then
       parser%section = 'metadata'
    end if

  end subroutine detect_json_section
!###############################################################################


!###############################################################################
  subroutine parse_node_section_line(line, state, parsed, section)
    !! Parse one line from the node section.
    implicit none

    character(*), intent(in) :: line
    type(json_node_state_type), intent(inout) :: state
    type(json_parse_result_type), intent(inout) :: parsed
    character(32), intent(inout) :: section

    if(.not.state%in_object .and. is_json_object_start(line))then
       call reset_node_state(state)
       state%in_object = .true.
       return
    end if

    if(state%in_object)then
       if(state%in_attribute)then
          if(index(line, ']') .gt. 0 .and. index(line, '"') .eq. 0)then
             state%in_attribute = .false.
             return
          end if
          if(index(line, '{') .gt. 0)then
             call parse_json_attribute(line, state%attrs, state%num_attrs)
          end if
          return
       end if

       if(index(line, '"attribute"') .gt. 0 .and. index(line, '[') .gt. 0)then
          state%in_attribute = .true.
          if(index(line, ']') .gt. 0)then
             call parse_json_attribute(line, state%attrs, state%num_attrs)
             state%in_attribute = .false.
          end if
          return
       end if

       if(index(line, '}') .gt. 0 .and. index(line, '"') .eq. 0)then
          call store_node_state(state, parsed)
          state%in_object = .false.
          return
       end if

       if(index(line, '"input"') .gt. 0)then
          call parse_json_string_array(line, '"input"', state%inputs, &
               state%num_inputs)
          return
       end if

       if(index(line, '"output"') .gt. 0)then
          call parse_json_string_array(line, '"output"', state%outputs, &
               state%num_outputs)
          return
       end if

       if(index(line, '"name"') .gt. 0)then
          call extract_json_string(line, '"name"', state%name)
          return
       end if

       if(index(line, '"opType"') .gt. 0)then
          call extract_json_string(line, '"opType"', state%op_type)
          return
       end if
    end if

    if(index(line, ']') .gt. 0 .and. .not.state%in_object) section = ''

  end subroutine parse_node_section_line
!###############################################################################


!###############################################################################
  subroutine store_node_state(state, parsed)
    !! Copy the current node state into the parsed result collection.
    implicit none

    type(json_node_state_type), intent(in) :: state
    type(json_parse_result_type), intent(inout) :: parsed

    parsed%num_nodes = parsed%num_nodes + 1
    parsed%nodes(parsed%num_nodes)%name = state%name
    parsed%nodes(parsed%num_nodes)%op_type = state%op_type
    parsed%nodes(parsed%num_nodes)%num_inputs = state%num_inputs
    parsed%nodes(parsed%num_nodes)%num_outputs = state%num_outputs

    if(state%num_inputs .gt. 0)then
       allocate(parsed%nodes(parsed%num_nodes)%inputs(state%num_inputs))
       parsed%nodes(parsed%num_nodes)%inputs = state%inputs(1:state%num_inputs)
    end if

    if(state%num_outputs .gt. 0)then
       allocate(parsed%nodes(parsed%num_nodes)%outputs(state%num_outputs))
       parsed%nodes(parsed%num_nodes)%outputs = &
            state%outputs(1:state%num_outputs)
    end if

    if(state%num_attrs .gt. 0)then
       allocate(parsed%nodes(parsed%num_nodes)%attributes(state%num_attrs))
       parsed%nodes(parsed%num_nodes)%attributes = &
            state%attrs(1:state%num_attrs)
    end if

  end subroutine store_node_state
!###############################################################################


!###############################################################################
  subroutine reset_node_state(state)
    !! Reset the reusable node parser state.
    implicit none

    type(json_node_state_type), intent(inout) :: state

    state%in_object = .false.
    state%in_attribute = .false.
    state%name = ''
    state%op_type = ''
    state%num_inputs = 0
    state%num_outputs = 0
    state%num_attrs = 0

    if(.not.allocated(state%inputs)) allocate(state%inputs(100))
    if(.not.allocated(state%outputs)) allocate(state%outputs(100))
    if(allocated(state%attrs)) deallocate(state%attrs)
    allocate(state%attrs(0))

  end subroutine reset_node_state
!###############################################################################


!###############################################################################
  subroutine parse_initialiser_section_line(line, state, parsed, section)
    !! Parse one line from the initialiser section.
    implicit none

    character(*), intent(in) :: line
    type(json_initialiser_state_type), intent(inout) :: state
    type(json_parse_result_type), intent(inout) :: parsed
    character(32), intent(inout) :: section

    integer :: pos, pos2

    if(.not.state%in_object .and. is_json_object_start(line))then
       call reset_initialiser_state(state)
       state%in_object = .true.
       return
    end if

    if(state%in_object)then
       if(index(line, '}') .gt. 0 .and. index(line, '"rawData"') .eq. 0 .and. &
            index(line, '"dims"') .eq. 0)then
          call store_initialiser_state(state, parsed)
          state%in_object = .false.
          return
       end if

       if(index(line, '"dims"') .gt. 0)then
          call parse_json_int_array_from_strings(line, state%dims)
          return
       end if

       if(index(line, '"dataType"') .gt. 0)then
          call extract_json_int(line, '"dataType"', state%data_type)
          return
       end if

       if(index(line, '"name"') .gt. 0)then
          call extract_json_string(line, '"name"', state%name)
          return
       end if

       if(index(line, '"rawData"') .gt. 0)then
          pos = index(line, '"rawData"') + 9
          pos2 = index(line(pos:), '"')
          if(pos2 .gt. 0)then
             pos = pos + pos2
             pos2 = index(line(pos:), '"')
             if(pos2 .gt. 0) state%raw_data = line(pos:pos+pos2-2)
          end if
          return
       end if
    end if

    if(index(line, ']') .gt. 0 .and. .not.state%in_object) section = ''

  end subroutine parse_initialiser_section_line
!###############################################################################


!###############################################################################
  subroutine store_initialiser_state(state, parsed)
    !! Copy the current initialiser state into the parsed result collection.
    use athena__onnx_utils, only: decode_base64_to_float32, &
         decode_base64_to_int64
    implicit none

    type(json_initialiser_state_type), intent(in) :: state
    type(json_parse_result_type), intent(inout) :: parsed

    integer :: j, n_decoded
    real(real32), allocatable :: decoded_floats(:)
    integer, allocatable :: decoded_ints(:)

    parsed%num_inits = parsed%num_inits + 1
    parsed%inits(parsed%num_inits)%name = state%name
    parsed%inits(parsed%num_inits)%data_type = state%data_type

    if(allocated(state%dims))then
       allocate(parsed%inits(parsed%num_inits)%dims(size(state%dims)))
       parsed%inits(parsed%num_inits)%dims = state%dims
    end if

    if(len_trim(state%raw_data) .eq. 0) return

    if(state%data_type .eq. 1)then
       call decode_base64_to_float32(trim(state%raw_data), decoded_floats, &
            n_decoded)
       allocate(parsed%inits(parsed%num_inits)%data(n_decoded))
       parsed%inits(parsed%num_inits)%data = decoded_floats
       deallocate(decoded_floats)
    else if(state%data_type .eq. 7)then
       call decode_base64_to_int64(trim(state%raw_data), decoded_ints, &
            n_decoded)
       allocate(parsed%inits(parsed%num_inits)%data(n_decoded))
       do j = 1, n_decoded
          parsed%inits(parsed%num_inits)%data(j) = &
               real(decoded_ints(j), real32)
       end do
       deallocate(decoded_ints)
    end if

  end subroutine store_initialiser_state
!###############################################################################


!###############################################################################
  subroutine reset_initialiser_state(state)
    !! Reset the reusable initialiser parser state.
    implicit none

    type(json_initialiser_state_type), intent(inout) :: state

    state%in_object = .false.
    state%name = ''
    state%data_type = 1
    if(allocated(state%dims)) deallocate(state%dims)
    allocate(state%dims(0))
    if(allocated(state%raw_data)) deallocate(state%raw_data)
    allocate(character(0) :: state%raw_data)

  end subroutine reset_initialiser_state
!###############################################################################


!###############################################################################
  subroutine parse_tensor_section_line(line, state, tensors, num_tensors, &
       section)
    !! Parse one line from the input or output tensor section.
    implicit none

    character(*), intent(in) :: line
    type(json_tensor_state_type), intent(inout) :: state
    type(onnx_tensor_type), intent(inout) :: tensors(:)
    integer, intent(inout) :: num_tensors
    character(32), intent(inout) :: section

    integer :: stat, dim_value
    character(256) :: tmpstr

    if(.not.state%in_object .and. is_json_object_start(line))then
       call reset_tensor_state(state)
       state%in_object = .true.
       state%object_depth = 1
       return
    end if

    if(state%in_object)then
       call update_object_depth(line, state%object_depth)
       if(state%object_depth .le. 0)then
          call store_tensor_state(state, tensors, num_tensors)
          state%in_object = .false.
          return
       end if

       if(index(line, '"name"') .gt. 0)then
          call extract_json_string(line, '"name"', state%name)
          return
       end if

       if(index(line, '"elemType"') .gt. 0)then
          call extract_json_int(line, '"elemType"', state%elem_type)
          return
       end if

       if(index(line, '"dimValue"') .gt. 0)then
          call extract_json_string(line, '"dimValue"', tmpstr)
          read(tmpstr, *, iostat=stat) dim_value
          if(stat .eq. 0) state%dim_values = [state%dim_values, dim_value]
          return
       end if

       if(index(line, '"dimParam"') .gt. 0)then
          state%dim_values = [state%dim_values, -1]
          return
       end if
    end if

    if(index(line, ']') .gt. 0 .and. .not.state%in_object) section = ''

  end subroutine parse_tensor_section_line
!###############################################################################


!###############################################################################
  subroutine store_tensor_state(state, tensors, num_tensors)
    !! Copy the current tensor state into the parsed result collection.
    implicit none

    type(json_tensor_state_type), intent(in) :: state
    type(onnx_tensor_type), intent(inout) :: tensors(:)
    integer, intent(inout) :: num_tensors

    num_tensors = num_tensors + 1
    tensors(num_tensors)%name = state%name
    tensors(num_tensors)%elem_type = state%elem_type
    allocate(tensors(num_tensors)%dims(size(state%dim_values)))
    tensors(num_tensors)%dims = state%dim_values

  end subroutine store_tensor_state
!###############################################################################


!###############################################################################
  subroutine reset_tensor_state(state)
    !! Reset the reusable tensor parser state.
    implicit none

    type(json_tensor_state_type), intent(inout) :: state

    state%in_object = .false.
    state%object_depth = 0
    state%name = ''
    state%elem_type = 1
    if(allocated(state%dim_values)) deallocate(state%dim_values)
    allocate(state%dim_values(0))

  end subroutine reset_tensor_state
!###############################################################################


!###############################################################################
  subroutine parse_metadata_line(line, parsed, section)
    !! Parse one metadataProps line.
    implicit none

    character(*), intent(in) :: line
    type(json_parse_result_type), intent(inout) :: parsed
    character(32), intent(inout) :: section

    if(index(line, '"key"') .gt. 0 .and. index(line, '"value"') .gt. 0)then
       parsed%num_meta = parsed%num_meta + 1
       call extract_json_string(line, '"key"', &
            parsed%meta_keys(parsed%num_meta))
       call extract_json_string(line, '"value"', &
            parsed%meta_values(parsed%num_meta))
    end if

    if(index(line, ']') .gt. 0) section = ''

  end subroutine parse_metadata_line
!###############################################################################


!###############################################################################
  logical function is_json_object_start(line)
    !! Return true for section object lines like `{`.
    implicit none

    character(*), intent(in) :: line

    is_json_object_start = index(line, '{') .gt. 0 .and. &
         index(line, '"') .eq. 0

  end function is_json_object_start
!###############################################################################


!###############################################################################
  subroutine update_object_depth(line, object_depth)
    !! Update a nested object depth counter from one JSON line.
    implicit none

    character(*), intent(in) :: line
    integer, intent(inout) :: object_depth

    integer :: i

    do i = 1, len_trim(line)
       if(line(i:i) .eq. '{') object_depth = object_depth + 1
       if(line(i:i) .eq. '}') object_depth = object_depth - 1
    end do

  end subroutine update_object_depth
!###############################################################################


!###############################################################################
  subroutine build_network_from_json_gnn( &
       network, nodes, num_nodes, inits, num_inits, &
       inputs, num_inputs, outputs, num_outputs, &
       meta_keys, meta_values, num_meta, verbose_)
    !! Build a network containing GNN layers from parsed JSON data.
    !!
    !! GNN layer creation is delegated to the registered creator in
    !! list_of_onnx_gnn_layer_creators, keyed by the subtype stored in
    !! the metadata value string.
    use athena__base_layer, only: base_layer_type
    use athena__full_layer, only: full_layer_type
    use athena__initialiser_data, only: data_init_type
    use athena__onnx_utils, only: row_to_col_major_2d, &
         onnx_to_athena_activation
    use athena__container_layer, only: list_of_onnx_gnn_layer_creators, &
         allocate_list_of_onnx_gnn_layer_creators
    implicit none

    type(network_type), intent(inout) :: network
    type(onnx_node_type), intent(in) :: nodes(:)
    integer, intent(in) :: num_nodes
    type(onnx_initialiser_type), intent(in) :: inits(:)
    integer, intent(in) :: num_inits
    type(onnx_tensor_type), intent(in) :: inputs(:)
    integer, intent(in) :: num_inputs
    type(onnx_tensor_type), intent(in) :: outputs(:)
    integer, intent(in) :: num_outputs
    character(256), intent(in) :: meta_keys(:), meta_values(:)
    integer, intent(in) :: num_meta
    integer, intent(in) :: verbose_

    character(64) :: gnn_subtype, msg_activation
    character(128) :: gnn_prefix
    integer :: i, k, pos, pos2, layer_index
    character(256) :: meta_str, token, key
    integer :: weight_idx, bias_idx, num_out_dense
    character(128), allocatable :: std_node_names(:)
    character(128), allocatable :: std_node_ops(:)
    integer :: num_std_nodes
    character(128) :: prev_node_base


    !--------------------------------------------------------------------------
    ! Add the GNN layer via the registered creator
    !--------------------------------------------------------------------------
    gnn_subtype = ''
    meta_str = meta_values(1)
    pos = 1
    do while(pos .le. len_trim(meta_str))
       pos2 = index(meta_str(pos:), ';')
       if(pos2 .eq. 0)then
          token = meta_str(pos:len_trim(meta_str))
          pos = len_trim(meta_str) + 1
       else
          token = meta_str(pos:pos+pos2-2)
          pos = pos + pos2
       end if
       k = index(token, '=')
       if(k .eq. 0) cycle
       key = trim(adjustl(token(1:k-1)))
       if(trim(key) .eq. 'subtype')then
          gnn_subtype = trim(adjustl(token(k+1:)))
          exit
       end if
    end do

    gnn_prefix = trim(meta_keys(1))
    pos = index(gnn_prefix, 'athena_gnn_')
    if(pos .gt. 0) gnn_prefix = gnn_prefix(pos+11:)

    if(verbose_ .gt. 0)then
       write(*,*) 'GNN layer: subtype=', trim(gnn_subtype), ' prefix=', &
            trim(gnn_prefix)
    end if

    if(.not.allocated(list_of_onnx_gnn_layer_creators))then
       call allocate_list_of_onnx_gnn_layer_creators()
    end if

    layer_index = 0
    do i = 1, size(list_of_onnx_gnn_layer_creators)
       if(trim(list_of_onnx_gnn_layer_creators(i)%gnn_subtype) .eq. &
            trim(gnn_subtype))then
          layer_index = i
          exit
       end if
    end do

    if(layer_index .eq. 0)then
       write(*,*) 'ERROR: Unknown GNN subtype: ', trim(gnn_subtype)
       return
    end if

    block
      class(base_layer_type), allocatable :: gnn_layer

      gnn_layer = list_of_onnx_gnn_layer_creators(layer_index)%create_ptr( &
           meta_keys(1), meta_values(1), inits(1:num_inits), verbose_)
      call network%add(gnn_layer)
    end block


    !--------------------------------------------------------------------------
    ! Add the remaining standard layers
    !--------------------------------------------------------------------------
    allocate(std_node_names(num_nodes))
    allocate(std_node_ops(num_nodes))
    num_std_nodes = 0

    do i = 1, num_nodes
       if(index(trim(nodes(i)%name), trim(gnn_prefix)) .eq. 1) cycle
       num_std_nodes = num_std_nodes + 1
       std_node_names(num_std_nodes) = trim(nodes(i)%name)
       std_node_ops(num_std_nodes) = trim(nodes(i)%op_type)
    end do

    i = 1
    do while(i .le. num_std_nodes)
       if(trim(std_node_ops(i)) .eq. 'Gemm')then
          weight_idx = 0
          bias_idx = 0
          prev_node_base = trim(std_node_names(i))

          do k = 1, num_inits
             if(index(trim(inits(k)%name), trim(prev_node_base)) .ne. 1) cycle
             if(.not.allocated(inits(k)%dims)) cycle
             if(size(inits(k)%dims) .ge. 2)then
                weight_idx = k
             else
                bias_idx = k
             end if
          end do

          if(weight_idx .gt. 0 .and. allocated(inits(weight_idx)%dims))then
             num_out_dense = inits(weight_idx)%dims(1)
          else
             num_out_dense = 1
          end if

          msg_activation = 'none'
          if(i + 1 .le. num_std_nodes)then
             if(index(trim(std_node_names(i+1)), trim(prev_node_base)) .eq. 1 &
                  .and. trim(std_node_ops(i+1)) .ne. 'Gemm')then
                msg_activation = onnx_to_athena_activation( &
                     trim(std_node_ops(i+1)))
                i = i + 1
             end if
          end if

          block
            type(full_layer_type) :: dense_layer
            type(data_init_type) :: k_init, b_init
            real(real32), allocatable :: col_w(:)

            if(weight_idx .gt. 0 .and. allocated(inits(weight_idx)%data))then
               allocate(col_w(size(inits(weight_idx)%data)))
               call row_to_col_major_2d( &
                    inits(weight_idx)%data, col_w, &
                    inits(weight_idx)%dims(1), inits(weight_idx)%dims(2))
               k_init = data_init_type(data = col_w)
               deallocate(col_w)
            end if

            if(bias_idx .gt. 0 .and. allocated(inits(bias_idx)%data))then
               b_init = data_init_type(data = inits(bias_idx)%data)
            end if

            dense_layer = full_layer_type( &
                 num_outputs = num_out_dense, &
                 activation = trim(msg_activation), &
                 kernel_initialiser = k_init, &
                 bias_initialiser = b_init)

            call network%add(dense_layer)
          end block
       end if

       i = i + 1
    end do

    deallocate(std_node_names, std_node_ops)

    if(verbose_ .gt. 0)then
       write(*,*) 'Network built with ', network%num_layers, ' layers'
    end if

  end subroutine build_network_from_json_gnn
!###############################################################################


!###############################################################################
  subroutine build_network_from_json_standard( &
       network, nodes, num_nodes, inits, num_inits, inputs, num_inputs, &
       verbose_)
    !! Build a standard, non-GNN network from parsed JSON data.
    !!
    !! Synthetic value_info entries are created for layers whose output shape
    !! can be inferred from initialisers or simple attributes before calling
    !! build_from_onnx.
    implicit none

    type(network_type), intent(inout) :: network
    type(onnx_node_type), intent(in) :: nodes(:)
    integer, intent(in) :: num_nodes
    type(onnx_initialiser_type), intent(in) :: inits(:)
    integer, intent(in) :: num_inits
    type(onnx_tensor_type), intent(in) :: inputs(:)
    integer, intent(in) :: num_inputs
    integer, intent(in) :: verbose_

    type(onnx_tensor_type), allocatable :: value_infos(:)
    integer :: i, j, k, num_vi, ndims, n_kernel_dims
    character(128) :: out_name
    character(32) :: op_type_name

    allocate(value_infos(num_nodes))
    num_vi = 0

    node_loop: do i = 1, num_nodes
       if(.not.allocated(nodes(i)%outputs)) cycle
       if(nodes(i)%num_outputs .lt. 1) cycle

       out_name = trim(nodes(i)%outputs(1))
       op_type_name = trim(adjustl(nodes(i)%op_type))

       do j = 1, nodes(i)%num_inputs
          do k = 1, num_inits
             if(trim(nodes(i)%inputs(j)) .ne. trim(inits(k)%name)) cycle
             if(.not.allocated(inits(k)%dims)) cycle
             if(size(inits(k)%dims) .lt. 2) cycle

             num_vi = num_vi + 1
             value_infos(num_vi)%name = out_name
             value_infos(num_vi)%elem_type = 1
             ndims = size(inits(k)%dims)

             if(op_type_name .eq. 'Conv' .and. ndims .ge. 3)then
                allocate(value_infos(num_vi)%dims(ndims))
                value_infos(num_vi)%dims(1) = 1
                value_infos(num_vi)%dims(2) = inits(k)%dims(ndims)
                value_infos(num_vi)%dims(3:ndims) = 0
             else
                allocate(value_infos(num_vi)%dims(2))
                value_infos(num_vi)%dims(1) = 1
                value_infos(num_vi)%dims(2) = inits(k)%dims(1)
             end if
             cycle node_loop
          end do
       end do

       if(op_type_name .eq. 'MaxPool' .or. op_type_name .eq. 'AvgPool')then
          n_kernel_dims = 0
          if(allocated(nodes(i)%attributes))then
             do j = 1, size(nodes(i)%attributes)
                if(trim(adjustl(nodes(i)%attributes(j)%name)) .ne. &
                     'kernel_shape') cycle
                block
                  character(256) :: kval
                  integer :: kpos, kstat, ktemp

                  kval = trim(adjustl(nodes(i)%attributes(j)%val))
                  kpos = 1
                  do while(kpos .le. len_trim(kval))
                     do while(kpos .le. len_trim(kval) .and. &
                          kval(kpos:kpos) .eq. ' ')
                        kpos = kpos + 1
                     end do
                     if(kpos .gt. len_trim(kval)) exit
                     read(kval(kpos:), *, iostat=kstat) ktemp
                     if(kstat .ne. 0) exit
                     n_kernel_dims = n_kernel_dims + 1
                     do while(kpos .le. len_trim(kval) .and. &
                          kval(kpos:kpos) .ne. ' ')
                        kpos = kpos + 1
                     end do
                  end do
                end block
                exit
             end do
          end if

          if(n_kernel_dims .gt. 0)then
             num_vi = num_vi + 1
             value_infos(num_vi)%name = out_name
             value_infos(num_vi)%elem_type = 1
             allocate(value_infos(num_vi)%dims(n_kernel_dims + 2))
             value_infos(num_vi)%dims = 0
          end if
       end if

    end do node_loop

    call network%build_from_onnx( &
         nodes(1:num_nodes), inits(1:num_inits), inputs(1:num_inputs), &
         value_infos(1:num_vi), verbose=verbose_)

  end subroutine build_network_from_json_standard
!###############################################################################


!###############################################################################
  subroutine extract_json_string(line, key, value)
    !! Extract a string value from a JSON key-value pair.
    implicit none

    character(*), intent(in) :: line, key
    character(*), intent(out) :: value

    integer :: pos, pos2, pos3

    value = ''
    pos = index(line, trim(key))
    if(pos .eq. 0) return

    pos = pos + len_trim(key)
    pos2 = index(line(pos:), '"')
    if(pos2 .eq. 0) return
    pos = pos + pos2
    pos3 = index(line(pos:), '"')
    if(pos3 .eq. 0) return
    value = line(pos:pos+pos3-2)

  end subroutine extract_json_string
!###############################################################################


!###############################################################################
  subroutine extract_json_int(line, key, value)
    !! Extract an integer value from a JSON key-value pair.
    implicit none

    character(*), intent(in) :: line, key
    integer, intent(out) :: value

    integer :: pos, pos2, stat
    character(64) :: numstr

    value = 0
    pos = index(line, trim(key))
    if(pos .eq. 0) return

    pos = pos + len_trim(key)
    pos2 = index(line(pos:), ':')
    if(pos2 .eq. 0) return
    pos = pos + pos2

    numstr = adjustl(line(pos:))
    if(numstr(1:1) .eq. '"')then
       numstr = numstr(2:)
       pos2 = index(numstr, '"')
       if(pos2 .gt. 0) numstr = numstr(1:pos2-1)
    end if
    pos2 = index(numstr, ',')
    if(pos2 .gt. 0) numstr = numstr(1:pos2-1)

    read(numstr, *, iostat=stat) value

  end subroutine extract_json_int
!###############################################################################


!###############################################################################
  subroutine parse_json_string_array(line, key, values, n)
    !! Parse a JSON string array from one line.
    implicit none

    character(*), intent(in) :: line, key
    character(128), intent(inout) :: values(:)
    integer, intent(inout) :: n

    integer :: pos, pos2, pos3

    pos = index(line, trim(key))
    if(pos .eq. 0) return

    pos = pos + len_trim(key)
    pos2 = index(line(pos:), '[')
    if(pos2 .eq. 0) return
    pos = pos + pos2

    do
       pos2 = index(line(pos:), '"')
       if(pos2 .eq. 0) exit
       pos = pos + pos2
       pos3 = index(line(pos:), '"')
       if(pos3 .eq. 0) exit
       n = n + 1
       values(n) = line(pos:pos+pos3-2)
       pos = pos + pos3
       if(index(line(pos:), ']') .gt. 0 .and. &
            index(line(pos:), '"') .eq. 0) exit
       if(index(line(pos:), ']') .gt. 0 .and. &
            index(line(pos:), ']') .lt. index(line(pos:), '"')) exit
    end do

  end subroutine parse_json_string_array
!###############################################################################


!###############################################################################
  subroutine parse_json_int_array_from_strings(line, values)
    !! Parse a JSON array of string-encoded integers.
    implicit none

    character(*), intent(in) :: line
    integer, allocatable, intent(inout) :: values(:)

    integer :: pos, pos2, pos3, stat, ival
    character(64) :: numstr

    if(allocated(values)) deallocate(values)
    allocate(values(0))

    pos = index(line, '[')
    if(pos .eq. 0) return
    pos = pos + 1

    do
       pos2 = index(line(pos:), '"')
       if(pos2 .eq. 0) exit
       pos = pos + pos2
       pos3 = index(line(pos:), '"')
       if(pos3 .eq. 0) exit
       numstr = line(pos:pos+pos3-2)
       read(numstr, *, iostat=stat) ival
       if(stat .eq. 0) values = [values, ival]
       pos = pos + pos3
       if(index(line(pos:), ']') .gt. 0 .and. &
            (index(line(pos:), '"') .eq. 0 .or. &
                 index(line(pos:), ']') .lt. index(line(pos:), '"'))) exit
    end do

  end subroutine parse_json_int_array_from_strings
!###############################################################################


!###############################################################################
  subroutine parse_json_attribute(line, attrs, n_attrs)
    !! Parse one or more JSON attribute objects from a line.
    implicit none

    character(*), intent(in) :: line
    type(onnx_attribute_type), allocatable, intent(inout) :: attrs(:)
    integer, intent(inout) :: n_attrs

    integer :: pos, brace_start, brace_end, depth, k

    pos = 1
    do while(pos .le. len_trim(line))
       brace_start = index(line(pos:), '{')
       if(brace_start .eq. 0) exit
       brace_start = pos + brace_start - 1

       depth = 0
       brace_end = 0
       do k = brace_start, len_trim(line)
          if(line(k:k) .eq. '{') depth = depth + 1
          if(line(k:k) .eq. '}') depth = depth - 1
          if(depth .eq. 0)then
             brace_end = k
             exit
          end if
       end do
       if(brace_end .eq. 0) exit

       call parse_single_json_attribute( &
            line(brace_start:brace_end), attrs, n_attrs)

       pos = brace_end + 1
    end do

  end subroutine parse_json_attribute
!###############################################################################


!###############################################################################
  subroutine parse_single_json_attribute(line, attrs, n_attrs)
    !! Parse a single JSON attribute object.
    implicit none

    character(*), intent(in) :: line
    type(onnx_attribute_type), allocatable, intent(inout) :: attrs(:)
    integer, intent(inout) :: n_attrs

    type(onnx_attribute_type) :: attr
    character(64) :: attr_type_str
    character(256) :: val_str

    attr%name = ''
    attr%type = ''
    allocate(character(0) :: attr%val)

    call extract_json_string(line, '"name"', val_str)
    attr%name = trim(val_str)

    call extract_json_string(line, '"type"', attr_type_str)
    attr%type = to_lower(trim(attr_type_str))

    select case(trim(attr%type))
    case('int')
       call extract_json_string(line, '"i"', val_str)
       attr%val = trim(val_str)
    case('float')
       call extract_json_string(line, '"f"', val_str)
       if(len_trim(val_str) .eq. 0)then
          block
            integer :: fp, fp2

            fp = index(line, '"f"')
            if(fp .gt. 0)then
               fp = fp + 3
               fp = fp + index(line(fp:), ':')
               val_str = trim(adjustl(line(fp:)))
               fp2 = scan(val_str, ',}')
               if(fp2 .gt. 0) val_str = val_str(1:fp2-1)
            end if
          end block
       end if
       attr%val = trim(val_str)
    case('ints')
       block
         integer :: ip, ip2
         character(256) :: ints_str

         ints_str = ''
         ip = index(line, '"ints"')
         if(ip .gt. 0)then
            ip = ip + 6
            ip = ip + index(line(ip:), '[') - 1
            ip2 = index(line(ip:), ']')
            if(ip2 .gt. 0)then
               ints_str = line(ip+1:ip+ip2-2)
               do ip2 = 1, len_trim(ints_str)
                  if(ints_str(ip2:ip2) .eq. ',' .or. &
                       ints_str(ip2:ip2) .eq. '"') ints_str(ip2:ip2) = ' '
               end do
            end if
         end if
         attr%val = trim(adjustl(ints_str))
       end block
    case('string')
       call extract_json_string(line, '"s"', val_str)
       attr%val = trim(val_str)
    case default
       attr%val = ''
    end select

    n_attrs = n_attrs + 1
    attrs = [attrs, attr]

  end subroutine parse_single_json_attribute
!###############################################################################

end submodule athena__onnx_read_submodule
