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
     !! Parsed ONNX nodes
     integer :: num_nodes = 0
     !! Number of valid entries in nodes
     type(onnx_initialiser_type) :: inits(MAX_ITEMS)
     !! Parsed ONNX initialisers
     integer :: num_inits = 0
     !! Number of valid entries in inits
     type(onnx_tensor_type) :: inputs(MAX_ITEMS)
     !! Parsed graph input tensors
     integer :: num_inputs = 0
     !! Number of valid entries in inputs
     type(onnx_tensor_type) :: outputs(MAX_ITEMS)
     !! Parsed graph output tensors
     integer :: num_outputs = 0
     !! Number of valid entries in outputs
     character(256) :: meta_keys(MAX_GNN_METADATA)
     !! Metadata keys read from metadataProps
     character(256) :: meta_values(MAX_GNN_METADATA)
     !! Metadata values read from metadataProps
     integer :: num_meta = 0
     !! Number of valid metadata key/value pairs
  end type json_parse_result_type

  type :: json_node_state_type
     logical :: in_object = .false.
     !! Whether parser is currently inside a node object
     logical :: in_attribute = .false.
     !! Whether parser is currently inside an attribute array
     character(256) :: name = ''
     !! Node name parsed from JSON
     character(256) :: op_type = ''
     !! Node opType parsed from JSON
     character(128), allocatable :: inputs(:)
     !! Temporary node input names
     character(128), allocatable :: outputs(:)
     !! Temporary node output names
     integer :: num_inputs = 0
     !! Number of valid input names
     integer :: num_outputs = 0
     !! Number of valid output names
     type(onnx_attribute_type), allocatable :: attrs(:)
     !! Temporary parsed node attributes
     integer :: num_attrs = 0
     !! Number of valid attribute entries
  end type json_node_state_type

  type :: json_initialiser_state_type
     logical :: in_object = .false.
     !! Whether parser is currently inside an initialiser object
     character(128) :: name = ''
     !! Initialiser tensor name
     integer :: data_type = 1
     !! ONNX dataType enum value
     integer, allocatable :: dims(:)
     !! Parsed tensor dimensions
     character(:), allocatable :: raw_data
     !! Base64 payload from rawData field
  end type json_initialiser_state_type

  type :: json_tensor_state_type
     logical :: in_object = .false.
     !! Whether parser is currently inside a tensor object
     integer :: object_depth = 0
     !! Nested JSON object depth within this tensor block
     character(128) :: name = ''
     !! Tensor name
     integer :: elem_type = 1
     !! ONNX element type enum value
     integer, allocatable :: dim_values(:)
     !! Parsed tensor dimensions (-1 for dimParam)
  end type json_tensor_state_type

  type :: json_parser_state_type
     character(32) :: section = ''
     !! Active top-level section name
     type(json_node_state_type) :: node
     !! Reusable node parser state
     type(json_initialiser_state_type) :: initialiser
     !! Reusable initialiser parser state
     type(json_tensor_state_type) :: input_tensor
     !! Reusable input tensor parser state
     type(json_tensor_state_type) :: output_tensor
     !! Reusable output tensor parser state
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

    ! Arguments
    type(json_parser_state_type), intent(out) :: parser
    !! Parser state container to initialise

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

    ! Arguments
    character(*), intent(in) :: line
    !! Current trimmed JSON line
    type(json_parser_state_type), intent(inout) :: parser
    !! Parser state with mutable active section

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

    ! Arguments
    character(*), intent(in) :: line
    !! Current JSON line to parse
    type(json_node_state_type), intent(inout) :: state
    !! Mutable node parser state
    type(json_parse_result_type), intent(inout) :: parsed
    !! Parsed ONNX content accumulated so far
    character(32), intent(inout) :: section
    !! Current top-level JSON section name

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

    ! Arguments
    type(json_node_state_type), intent(in) :: state
    !! Completed node parser state
    type(json_parse_result_type), intent(inout) :: parsed
    !! Parsed ONNX content accumulated so far

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

    ! Arguments
    type(json_node_state_type), intent(inout) :: state
    !! Node parser state to reset

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

    ! Arguments
    character(*), intent(in) :: line
    !! Current JSON line to parse
    type(json_initialiser_state_type), intent(inout) :: state
    !! Mutable parser state for the active initialiser object
    type(json_parse_result_type), intent(inout) :: parsed
    !! Parsed ONNX content accumulated so far
    character(32), intent(inout) :: section
    !! Current top-level JSON section name

    ! Local variables
    integer :: pos, pos2
    !! Temporary string positions used to slice the rawData field

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

    ! Arguments
    type(json_initialiser_state_type), intent(in) :: state
    !! Completed initialiser parse state to copy into the result object
    type(json_parse_result_type), intent(inout) :: parsed
    !! Parsed ONNX content accumulated so far

    ! Local variables
    integer :: j, n_decoded
    !! Integer loop index and decoded tensor length
    real(real32), allocatable :: decoded_floats(:)
    !! Float payload decoded from base64 rawData
    integer, allocatable :: decoded_ints(:)
    !! Int64 payload decoded from base64 rawData

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

    ! Arguments
    type(json_initialiser_state_type), intent(inout) :: state
    !! Initialiser parser state to reset

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

    ! Arguments
    character(*), intent(in) :: line
    !! Current JSON line to parse
    type(json_tensor_state_type), intent(inout) :: state
    !! Mutable tensor parser state
    type(onnx_tensor_type), intent(inout) :: tensors(:)
    !! Parsed tensor destination array
    integer, intent(inout) :: num_tensors
    !! Number of valid tensor entries in tensors
    character(32), intent(inout) :: section
    !! Current top-level JSON section name

    ! Local variables
    integer :: stat, dim_value
    !! Read status and parsed dimension value
    character(256) :: tmpstr
    !! Temporary string buffer for dimValue parsing

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

    ! Arguments
    type(json_tensor_state_type), intent(in) :: state
    !! Completed tensor parser state
    type(onnx_tensor_type), intent(inout) :: tensors(:)
    !! Parsed tensor destination array
    integer, intent(inout) :: num_tensors
    !! Number of valid tensor entries in tensors

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

    ! Arguments
    type(json_tensor_state_type), intent(inout) :: state
    !! Tensor parser state to reset

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

    ! Arguments
    character(*), intent(in) :: line
    !! Current metadata JSON line
    type(json_parse_result_type), intent(inout) :: parsed
    !! Parsed ONNX content accumulated so far
    character(32), intent(inout) :: section
    !! Current top-level JSON section name

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

    ! Arguments
    character(*), intent(in) :: line
    !! Current JSON line to classify

    is_json_object_start = index(line, '{') .gt. 0 .and. &
         index(line, '"') .eq. 0

  end function is_json_object_start
!###############################################################################


!###############################################################################
  subroutine update_object_depth(line, object_depth)
    !! Update a nested object depth counter from one JSON line.
    implicit none

    ! Arguments
    character(*), intent(in) :: line
    !! Current JSON line
    integer, intent(inout) :: object_depth
    !! Mutable object depth counter

    ! Local variables
    integer :: i
    !! Character index while scanning braces

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
    !! Standard (non-GNN) layer creation is delegated to the registered
    !! creator in list_of_onnx_layer_creators, keyed by the ONNX op_type.
    use athena__base_layer, only: base_layer_type
    use athena__container_layer, only: list_of_onnx_gnn_layer_creators, &
         allocate_list_of_onnx_gnn_layer_creators, &
         list_of_onnx_nop_layer_creators, &
         allocate_list_of_onnx_nop_layer_creators, &
         list_of_onnx_layer_creators, &
         allocate_list_of_onnx_layer_creators
    implicit none

    ! Arguments
    type(network_type), intent(inout) :: network
    !! Network to populate from parsed ONNX content
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid entries in nodes
    type(onnx_initialiser_type), intent(in) :: inits(:)
    !! Parsed ONNX initialisers
    integer, intent(in) :: num_inits
    !! Number of valid entries in inits
    type(onnx_tensor_type), intent(in) :: inputs(:)
    !! Parsed graph input tensors
    integer, intent(in) :: num_inputs
    !! Number of valid entries in inputs
    type(onnx_tensor_type), intent(in) :: outputs(:)
    !! Parsed graph output tensors
    integer, intent(in) :: num_outputs
    !! Number of valid entries in outputs
    character(256), intent(in) :: meta_keys(:), meta_values(:)
    !! Metadata keys and values from metadataProps
    integer, intent(in) :: num_meta
    !! Number of valid metadata entries
    integer, intent(in) :: verbose_
    !! Effective verbosity level

    ! Local variables
    integer, allocatable :: ordered_layer_ids(:)
    !! Sorted unique layer ids discovered from metadata and node names
    integer :: i, layer_id, meta_index, node_index
    !! Loop index and per-layer lookup indices

    if(.not.allocated(list_of_onnx_gnn_layer_creators))then
       call allocate_list_of_onnx_gnn_layer_creators()
    end if
    if(.not.allocated(list_of_onnx_nop_layer_creators))then
       call allocate_list_of_onnx_nop_layer_creators()
    end if
    if(.not.allocated(list_of_onnx_layer_creators))then
       call allocate_list_of_onnx_layer_creators()
    end if

    allocate(ordered_layer_ids(0))

    do i = 1, num_meta
       call append_unique_layer_id_from_meta_key( &
            meta_keys(i), ordered_layer_ids)
    end do

    do i = 1, num_nodes
       call append_unique_primary_layer_id(nodes(i)%name, ordered_layer_ids)
    end do

    call sort_int_array(ordered_layer_ids)

    do i = 1, size(ordered_layer_ids)
       layer_id = ordered_layer_ids(i)
       meta_index = find_metadata_for_layer_id(meta_keys, num_meta, layer_id)

       if(meta_index .gt. 0)then
          call add_gnn_layer_from_metadata( &
               network, meta_keys(meta_index), meta_values(meta_index), &
               inits, num_inits, verbose_)
          cycle
       end if

       node_index = find_primary_node_for_layer_id(nodes, num_nodes, layer_id)
       if(node_index .le. 0) cycle

       call add_standard_layer_from_onnx( &
            network, layer_id, node_index, nodes, num_nodes, &
            inits, num_inits, verbose_)
    end do

    if(allocated(ordered_layer_ids)) deallocate(ordered_layer_ids)

    if(verbose_ .gt. 0)then
       write(*,*) 'Network built with ', network%num_layers, ' layers'
    end if

  end subroutine build_network_from_json_gnn
!###############################################################################


!###############################################################################
  subroutine add_gnn_layer_from_metadata(network, meta_key, meta_value, inits, &
       num_inits, verbose_)
    !! Create one GNN or NOP layer from metadata and append it to the network.
    use athena__base_layer, only: base_layer_type
    use athena__container_layer, only: list_of_onnx_gnn_layer_creators, &
         list_of_onnx_nop_layer_creators
    implicit none

    ! Arguments
    type(network_type), intent(inout) :: network
    !! Network receiving the created layer
    character(*), intent(in) :: meta_key, meta_value
    !! Metadata key/value pair describing one layer
    type(onnx_initialiser_type), intent(in) :: inits(:)
    !! Parsed ONNX initialisers
    integer, intent(in) :: num_inits, verbose_
    !! Number of initialisers and effective verbosity level

    ! Local variables
    character(64) :: subtype_name
    !! Parsed subtype token from metadata payload
    integer :: i, layer_index
    !! Creator search index and selected creator slot

    call extract_gnn_subtype(meta_value, subtype_name)

    !--------------------------------------------------------------------------
    ! Try NOP creators first when the key uses the athena_nop_ prefix
    !--------------------------------------------------------------------------
    if(index(trim(meta_key), 'athena_nop_') .gt. 0)then
       layer_index = 0
       do i = 1, size(list_of_onnx_nop_layer_creators)
          if(trim(list_of_onnx_nop_layer_creators(i)%nop_subtype) .eq. &
               trim(subtype_name))then
             layer_index = i
             exit
          end if
       end do

       if(layer_index .eq. 0)then
          write(*,*) 'ERROR: Unknown NOP subtype: ', trim(subtype_name)
          return
       end if

       block
         class(base_layer_type), allocatable :: nop_layer

         nop_layer = list_of_onnx_nop_layer_creators(layer_index)%create_ptr(&
              meta_key, meta_value, inits(1:num_inits), verbose_)
         call network%add(nop_layer)
       end block
       return
    end if

    !--------------------------------------------------------------------------
    ! Fall back to GNN creators for athena_gnn_ prefix
    !--------------------------------------------------------------------------
    layer_index = 0
    do i = 1, size(list_of_onnx_gnn_layer_creators)
       if(trim(list_of_onnx_gnn_layer_creators(i)%gnn_subtype) .eq. &
            trim(subtype_name))then
          layer_index = i
          exit
       end if
    end do

    if(layer_index .eq. 0)then
       write(*,*) 'ERROR: Unknown GNN subtype: ', trim(subtype_name)
       return
    end if

    block
      class(base_layer_type), allocatable :: gnn_layer

      gnn_layer = list_of_onnx_gnn_layer_creators(layer_index)%create_ptr( &
           meta_key, meta_value, inits(1:num_inits), verbose_)
      call network%add(gnn_layer)
    end block

  end subroutine add_gnn_layer_from_metadata
!###############################################################################


!###############################################################################
  subroutine add_standard_layer_from_onnx( &
       network, layer_id, node_index, nodes, num_nodes, &
       inits, num_inits, verbose_)
    !! Create standard (non-GNN) layers for a given layer_id using the
    !! registered ONNX creator framework (list_of_onnx_layer_creators).
    !!
    !! Processes the primary node and any trailing activation node.
    use athena__base_layer, only: base_layer_type
    use athena__container_layer, only: list_of_onnx_layer_creators
    use athena__onnx_utils, only: row_to_col_major_2d
    implicit none

    ! Arguments
    type(network_type), intent(inout) :: network
    !! Network receiving the created layer(s)
    integer, intent(in) :: layer_id, node_index, num_nodes, num_inits
    !! Layer id, primary node index, node count and initialiser count
    integer, intent(in) :: verbose_
    !! Effective verbosity level
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    type(onnx_initialiser_type), intent(in) :: inits(:)
    !! Parsed ONNX initialisers

    ! Local variables
    integer :: j, k, layer_index, actv_index, ndims, num_matching
    !! Loop indices and creator/shape lookup values
    character(128) :: op_type_name, out_name
    !! Current ONNX op_type and output tensor name
    type(onnx_initialiser_type), allocatable :: init_list(:)
    !! Initialisers matched to the active node inputs
    type(onnx_tensor_type), allocatable :: value_info_list(:)
    !! Synthetic output shape hints passed to creator
    class(base_layer_type), allocatable :: layer
    !! Created ATHENA layer instance

    op_type_name = trim(adjustl(nodes(node_index)%op_type))

    layer_index = findloc( &
         [ list_of_onnx_layer_creators(:)%op_type ], &
         trim(op_type_name), dim = 1)

    if(layer_index .eq. 0)then
       if(verbose_ .gt. 0)then
          write(*,*) 'Skipping unsupported ONNX node in GNN import: ', &
               trim(nodes(node_index)%name), ' op=', trim(op_type_name)
       end if
       return
    end if

    num_matching = 0
    if(allocated(nodes(node_index)%inputs))then
       do j = 1, size(nodes(node_index)%inputs)
          do k = 1, num_inits
             if(trim(nodes(node_index)%inputs(j)) .eq. &
                  trim(inits(k)%name))then
                num_matching = num_matching + 1
             end if
          end do
       end do
    end if

    allocate(init_list(num_matching))
    num_matching = 0
    if(allocated(nodes(node_index)%inputs))then
       do j = 1, size(nodes(node_index)%inputs)
          do k = 1, num_inits
             if(trim(nodes(node_index)%inputs(j)) .ne. &
                  trim(inits(k)%name)) cycle

             num_matching = num_matching + 1
             init_list(num_matching)%name = inits(k)%name
             init_list(num_matching)%data_type = inits(k)%data_type

             if(allocated(inits(k)%dims))then
                allocate(init_list(num_matching)%dims(size(inits(k)%dims)))
                init_list(num_matching)%dims = inits(k)%dims
             end if

             if(allocated(inits(k)%data))then
                allocate(init_list(num_matching)%data(size(inits(k)%data)))
                if(allocated(inits(k)%dims))then
                   if(size(inits(k)%dims) .eq. 2)then
                      call row_to_col_major_2d( &
                           inits(k)%data, init_list(num_matching)%data, &
                           inits(k)%dims(1), inits(k)%dims(2))
                   else
                      init_list(num_matching)%data = inits(k)%data
                   end if
                else
                   init_list(num_matching)%data = inits(k)%data
                end if
             end if

             if(allocated(inits(k)%int_data))then
                allocate(init_list(num_matching)%int_data(size(inits(k)%int_data)))
                init_list(num_matching)%int_data = inits(k)%int_data
             end if
          end do
       end do
    end if

    allocate(value_info_list(0))
    if(allocated(nodes(node_index)%outputs) .and. &
         nodes(node_index)%num_outputs .ge. 1)then
       out_name = trim(nodes(node_index)%outputs(1))

       do j = 1, size(init_list)
          if(.not.allocated(init_list(j)%dims)) cycle
          if(size(init_list(j)%dims) .lt. 2) cycle
          ndims = size(init_list(j)%dims)

          block
            type(onnx_tensor_type) :: vi

            vi%name = out_name
            vi%elem_type = 1
            if(trim(op_type_name) .eq. 'Conv' .and. ndims .ge. 3)then
               allocate(vi%dims(ndims))
               vi%dims(1) = 1
               vi%dims(2) = init_list(j)%dims(ndims)
               vi%dims(3:ndims) = 0
            else
               allocate(vi%dims(2))
               vi%dims(1) = 1
               vi%dims(2) = init_list(j)%dims(1)
            end if

            deallocate(value_info_list)
            allocate(value_info_list(1))
            value_info_list(1)%name = vi%name
            value_info_list(1)%elem_type = vi%elem_type
            if(allocated(vi%dims))then
               allocate(value_info_list(1)%dims(size(vi%dims)))
               value_info_list(1)%dims = vi%dims
            end if
          end block
          exit
       end do
    end if

    layer = list_of_onnx_layer_creators(layer_index)%create_ptr( &
         nodes(node_index), init_list, value_info_list, verbose=verbose_)
    call network%add(layer)

    deallocate(init_list)
    deallocate(value_info_list)

    actv_index = find_activation_node_for_layer_id( &
         nodes, num_nodes, layer_id)
    if(actv_index .gt. 0)then
       op_type_name = trim(adjustl(nodes(actv_index)%op_type))
       layer_index = findloc( &
            [ list_of_onnx_layer_creators(:)%op_type ], &
            trim(op_type_name), dim = 1)
       if(layer_index .gt. 0)then
          allocate(init_list(0))
          allocate(value_info_list(0))
          if(allocated(layer)) deallocate(layer)
          layer = list_of_onnx_layer_creators(layer_index)%create_ptr( &
               nodes(actv_index), init_list, value_info_list, &
               verbose=verbose_)
          call network%add(layer)
          deallocate(init_list)
          deallocate(value_info_list)
       end if
    end if

  end subroutine add_standard_layer_from_onnx
!###############################################################################


!###############################################################################
  subroutine extract_gnn_subtype(meta_value, gnn_subtype)
    !! Extract the subtype=... token from one metadata value string.
    implicit none

    ! Arguments
    character(*), intent(in) :: meta_value
    !! Metadata payload string
    character(*), intent(out) :: gnn_subtype
    !! Extracted subtype token

    ! Local variables
    integer :: pos, pos2, k
    !! Token scanning positions and key delimiter index
    character(256) :: token, key
    !! Current token and token key

    gnn_subtype = ''
    pos = 1
    do while(pos .le. len_trim(meta_value))
       pos2 = index(meta_value(pos:), ';')
       if(pos2 .eq. 0)then
          token = meta_value(pos:len_trim(meta_value))
          pos = len_trim(meta_value) + 1
       else
          token = meta_value(pos:pos+pos2-2)
          pos = pos + pos2
       end if

       k = index(token, '=')
       if(k .eq. 0) cycle
       key = trim(adjustl(token(1:k-1)))
       if(trim(key) .eq. 'subtype')then
          gnn_subtype = trim(adjustl(token(k+1:)))
          return
       end if
    end do

  end subroutine extract_gnn_subtype
!###############################################################################


!###############################################################################
  subroutine append_unique_layer_id_from_meta_key(meta_key, ids)
    !! Append a layer id parsed from athena_gnn_node_<id> or
    !! athena_nop_node_<id> if not already present.
    implicit none

    ! Arguments
    character(*), intent(in) :: meta_key
    !! Metadata key potentially containing a layer id
    integer, allocatable, intent(inout) :: ids(:)
    !! Unique set of discovered layer ids

    ! Local variables
    integer :: layer_id, pos, stat, i
    !! Parsed id, prefix position, read status and loop index
    character(128) :: rest
    !! Metadata suffix containing the candidate id
    logical :: exists
    !! Whether the id already exists in ids

    pos = index(trim(meta_key), 'athena_gnn_node_')
    if(pos .eq. 0) pos = index(trim(meta_key), 'athena_nop_node_')
    if(pos .eq. 0) return

    rest = adjustl(trim(meta_key(pos+16:)))
    read(rest, *, iostat=stat) layer_id
    if(stat .ne. 0) return

    exists = .false.
    do i = 1, size(ids)
       if(ids(i) .eq. layer_id)then
          exists = .true.
          exit
       end if
    end do
    if(.not.exists) ids = [ids, layer_id]

  end subroutine append_unique_layer_id_from_meta_key
!###############################################################################


!###############################################################################
  subroutine append_unique_primary_layer_id(node_name, ids)
    !! Append a layer id parsed from a primary node name node_<id>.
    implicit none

    ! Arguments
    character(*), intent(in) :: node_name
    !! Node name potentially containing a primary layer id
    integer, allocatable, intent(inout) :: ids(:)
    !! Unique set of discovered layer ids

    ! Local variables
    integer :: layer_id, i
    !! Parsed id and loop index
    logical :: is_primary, exists
    !! Primary-node flag and duplicate-id flag

    call parse_primary_layer_id(node_name, layer_id, is_primary)
    if(.not.is_primary) return

    exists = .false.
    do i = 1, size(ids)
       if(ids(i) .eq. layer_id)then
          exists = .true.
          exit
       end if
    end do
    if(.not.exists) ids = [ids, layer_id]

  end subroutine append_unique_primary_layer_id
!###############################################################################


!###############################################################################
  subroutine parse_primary_layer_id(node_name, layer_id, is_primary)
    !! Parse node_<id> names and mark true only for primary layer nodes.
    implicit none

    ! Arguments
    character(*), intent(in) :: node_name
    !! Candidate ONNX node name
    integer, intent(out) :: layer_id
    !! Parsed layer id when present
    logical, intent(out) :: is_primary
    !! Whether node_name matches primary pattern node_<id>

    ! Local variables
    integer :: stat
    !! Read status for integer parse
    character(128) :: rest
    !! Node name suffix after node_ prefix

    layer_id = -1
    is_primary = .false.

    if(index(trim(node_name), 'node_') .ne. 1) return
    rest = trim(node_name(6:))
    if(index(rest, '_') .gt. 0) return

    read(rest, *, iostat=stat) layer_id
    if(stat .eq. 0 .and. layer_id .gt. 0) is_primary = .true.

  end subroutine parse_primary_layer_id
!###############################################################################


!###############################################################################
  integer function find_metadata_for_layer_id(meta_keys, num_meta, layer_id)
    !! Return metadata index for a given layer id, or 0 if absent.
    implicit none

    ! Arguments
    character(256), intent(in) :: meta_keys(:)
    !! Metadata keys list
    integer, intent(in) :: num_meta, layer_id
    !! Number of metadata entries and target layer id

    ! Local variables
    integer :: i, id_tmp
    !! Loop index and parsed id candidate
    logical :: found
    !! Whether a key parsed successfully

    find_metadata_for_layer_id = 0
    do i = 1, num_meta
       call parse_meta_layer_id(meta_keys(i), id_tmp, found)
       if(found .and. id_tmp .eq. layer_id)then
          find_metadata_for_layer_id = i
          return
       end if
    end do

  end function find_metadata_for_layer_id
!###############################################################################


!###############################################################################
  subroutine parse_meta_layer_id(meta_key, layer_id, found)
    !! Parse athena_gnn_node_<id> or athena_nop_node_<id> metadata key layer id.
    implicit none

    ! Arguments
    character(*), intent(in) :: meta_key
    !! Metadata key potentially containing a layer id
    integer, intent(out) :: layer_id
    !! Parsed layer id value
    logical, intent(out) :: found
    !! Whether parsing succeeded

    ! Local variables
    integer :: pos, stat
    !! Prefix position and read status
    character(128) :: rest
    !! Metadata suffix containing the candidate id

    layer_id = -1
    found = .false.

    pos = index(trim(meta_key), 'athena_gnn_node_')
    if(pos .gt. 0)then
       rest = adjustl(trim(meta_key(pos+16:)))
       read(rest, *, iostat=stat) layer_id
       if(stat .eq. 0 .and. layer_id .gt. 0) found = .true.
       return
    end if

    pos = index(trim(meta_key), 'athena_nop_node_')
    if(pos .gt. 0)then
       rest = adjustl(trim(meta_key(pos+16:)))
       read(rest, *, iostat=stat) layer_id
       if(stat .eq. 0 .and. layer_id .gt. 0) found = .true.
       return
    end if

  end subroutine parse_meta_layer_id
!###############################################################################


!###############################################################################
  integer function find_primary_node_for_layer_id(nodes, num_nodes, layer_id)
    !! Return node index for primary node_<id>, or 0 if not found.
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes, layer_id
    !! Number of valid nodes and target layer id

    ! Local variables
    integer :: i, id_tmp
    !! Loop index and parsed node id candidate
    logical :: is_primary
    !! Whether current node matches primary pattern

    find_primary_node_for_layer_id = 0
    do i = 1, num_nodes
       call parse_primary_layer_id(nodes(i)%name, id_tmp, is_primary)
       if(is_primary .and. id_tmp .eq. layer_id)then
          find_primary_node_for_layer_id = i
          return
       end if
    end do

  end function find_primary_node_for_layer_id
!###############################################################################


!###############################################################################
  integer function find_activation_node_for_layer_id(nodes, num_nodes, layer_id)
    !! Return node index for activation attached to node_<id>, or 0.
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes, layer_id
    !! Number of valid nodes and target layer id

    ! Local variables
    integer :: i
    !! Loop index
    character(128) :: prefix
    !! Prefix for activation nodes linked to layer_id

    write(prefix, '("node_",I0,"_")') layer_id
    find_activation_node_for_layer_id = 0

    do i = 1, num_nodes
       if(index(trim(nodes(i)%name), trim(prefix)) .ne. 1) cycle
       if(is_activation_op_type(trim(nodes(i)%op_type)))then
          find_activation_node_for_layer_id = i
          return
       end if
    end do

  end function find_activation_node_for_layer_id
!###############################################################################


!###############################################################################
  logical function is_activation_op_type(op_type)
    !! Return true for ONNX activation nodes emitted by ATHENA export.
    implicit none

    ! Arguments
    character(*), intent(in) :: op_type
    !! ONNX operation type string

    select case(trim(op_type))
    case('Relu', 'LeakyRelu', 'Sigmoid', 'Softmax', 'Tanh', 'Selu', 'Swish')
       is_activation_op_type = .true.
    case default
       is_activation_op_type = .false.
    end select

  end function is_activation_op_type
!###############################################################################


!###############################################################################
  subroutine sort_int_array(values)
    !! Sort an integer array in ascending order.
    implicit none

    ! Arguments
    integer, allocatable, intent(inout) :: values(:)
    !! Integer array sorted in ascending order in-place

    ! Local variables
    integer :: i, j, tmp
    !! Loop indices and swap temporary

    if(size(values) .le. 1) return

    do i = 1, size(values) - 1
       do j = i + 1, size(values)
          if(values(j) .lt. values(i))then
             tmp = values(i)
             values(i) = values(j)
             values(j) = tmp
          end if
       end do
    end do

  end subroutine sort_int_array
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

    ! Arguments
    type(network_type), intent(inout) :: network
    !! Network to populate from parsed ONNX content
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid entries in nodes
    type(onnx_initialiser_type), intent(in) :: inits(:)
    !! Parsed ONNX initialisers
    integer, intent(in) :: num_inits
    !! Number of valid entries in inits
    type(onnx_tensor_type), intent(in) :: inputs(:)
    !! Parsed graph input tensors
    integer, intent(in) :: num_inputs
    !! Number of valid entries in inputs
    integer, intent(in) :: verbose_
    !! Effective verbosity level

    ! Local variables
    type(onnx_tensor_type), allocatable :: value_infos(:)
    !! Synthesised tensor value_info entries
    integer :: i, j, k, num_vi, ndims, n_kernel_dims
    !! Loop indices and temporary dimension counters
    character(128) :: out_name
    !! Current node output tensor name
    character(32) :: op_type_name
    !! Current node ONNX op type

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

       if(index(op_type_name, 'Pool', back=.true.) .eq. &
            len_trim(op_type_name) - 3) then
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

    ! Arguments
    character(*), intent(in) :: line, key
    !! Source line and key token to find
    character(*), intent(out) :: value
    !! Extracted string value

    ! Local variables
    integer :: pos, pos2, pos3
    !! Temporary indices used while slicing quoted text

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

    ! Arguments
    character(*), intent(in) :: line, key
    !! Source line and key token to find
    integer, intent(out) :: value
    !! Extracted integer value

    ! Local variables
    integer :: pos, pos2, stat
    !! Temporary indices and read status
    character(64) :: numstr
    !! Numeric substring buffer

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

    ! Arguments
    character(*), intent(in) :: line, key
    !! Source line and array key token
    character(128), intent(inout) :: values(:)
    !! Destination array for parsed values
    integer, intent(inout) :: n
    !! Number of valid parsed values

    ! Local variables
    integer :: pos, pos2, pos3
    !! Temporary indices while scanning quoted values

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

    ! Arguments
    character(*), intent(in) :: line
    !! Source line containing a JSON array
    integer, allocatable, intent(inout) :: values(:)
    !! Parsed integer values

    ! Local variables
    integer :: pos, pos2, pos3, stat, ival
    !! Temporary indices, read status and parsed integer value
    character(64) :: numstr
    !! Numeric token buffer

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

    ! Arguments
    character(*), intent(in) :: line
    !! Source line containing one or more JSON attribute objects
    type(onnx_attribute_type), allocatable, intent(inout) :: attrs(:)
    !! Destination list of parsed attributes
    integer, intent(inout) :: n_attrs
    !! Number of valid attributes in attrs

    ! Local variables
    integer :: pos, brace_start, brace_end, depth, k
    !! Scan positions and brace depth state

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

    ! Arguments
    character(*), intent(in) :: line
    !! Source line for one JSON attribute object
    type(onnx_attribute_type), allocatable, intent(inout) :: attrs(:)
    !! Destination list of parsed attributes
    integer, intent(inout) :: n_attrs
    !! Number of valid attributes in attrs

    ! Local variables
    type(onnx_attribute_type) :: attr
    !! Parsed attribute record
    character(64) :: attr_type_str
    !! Raw attribute type token
    character(256) :: val_str
    !! Temporary attribute value buffer

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
