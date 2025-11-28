submodule(athena__onnx) athena__onnx_submodule
  !! Submodule containing implementations for ONNX operations
  use athena__base_layer, only: base_layer_type, learnable_layer_type
  use athena__misc_types, only: &
       onnx_attribute_type, onnx_node_type, onnx_initialiser_type
  use coreutils, only: real32, to_lower, to_upper, to_camel_case, icount
  use athena__tools_infile, only: assign_val, assign_vec, allocate_and_assign_vec

contains

!###############################################################################
  module subroutine write_onnx(file, network)
    !! Export the network to ONNX format
    implicit none

    ! Arguments
    class(network_type), intent(in) :: network
    !! Instance of network
    character(*), intent(in) :: file
    !! File to export the network to

    ! Local variables
    integer :: unit, i, j, idx
    !! Unit number and loop indices
    character(256) :: layer_name
    !! Layer name for ONNX
    character(20) :: node_name, input_name, tmp_input_name
    !! Node name
    character(:), allocatable :: suffix
    !!! Suffix for input names

    open(newunit=unit, file=file, status='replace')

    ! Write ONNX header
    write(unit, '(A)') 'ir_version: 8'
    write(unit, '(A)') 'producer_name: "Athena"'
    write(unit, '(A)') 'producer_version: "1.0"'
    write(unit, '(A)') 'domain: "ai.onnx"'
    write(unit, '(A)') 'model_version: 1'
    write(unit, '(A)') 'doc_string: "Athena neural network model"'
    write(unit, '(A)') ''

    ! Write graph definition
    write(unit, '(A)') 'graph {'
    write(unit, '(A)') '  name: "athena_network"'
    write(unit, '(A)') ''

    ! Write nodes (layers)
    write(unit, '(A)') '  # Nodes'
    do i = 1, network%auto_graph%num_vertices
       idx = network%auto_graph%vertex(network%vertex_order(i))%id
       write(node_name, '("node_", I0)') network%model(idx)%layer%id

       select case(trim(network%model(idx)%layer%type))
       case('inpt')
          layer_name = 'Input'
          cycle
       case('full')
          layer_name = 'MatMul'
       case('conv')
          layer_name = 'Conv'
       case('pool')
          layer_name = to_camel_case( &
               trim(adjustl(network%model(idx)%layer%subtype))//"_"//&
               trim(adjustl(network%model(idx)%layer%type)), &
               capitalise_first_letter = .true. &
          )
       case('actv')
          layer_name = to_camel_case( &
               adjustl(network%model(idx)%layer%subtype), &
               capitalise_first_letter = .true. &
          )
       case('flat')
          layer_name = 'Flatten'
       case('batc')
          layer_name = 'BatchNormalization'
       case('drop')
          layer_name = 'Dropout'
       case('msgp')
          layer_name = 'GNNLayer'
       case default
          layer_name = 'Unknown'
       end select

       write(unit, '(A)') '  node {'
       write(unit, '(A,A,A)') '    name: "', trim(node_name), '"'
       write(unit, '(A,A,A)') '    op_type: "', trim(layer_name), '"'

       ! Write input connections
       if(all(network%auto_graph%adjacency(:,network%vertex_order(i)).eq.0))then
          cycle
          ! write(unit, '(A,I0,A)') '    input: "input_',network%model(idx)%layer%id,'"'
       else
          do j = 1, network%auto_graph%num_vertices
             if(network%auto_graph%adjacency(j,network%vertex_order(i)).eq.0) cycle
             if(all(network%auto_graph%adjacency(:,j).eq.0))then
                write(input_name,'("input_",I0)') &
                     network%model(network%auto_graph%vertex(j)%id)%layer%id
                suffix = ''
             else
                write(input_name,'("node_",I0)') &
                     network%model(network%auto_graph%vertex(j)%id)%layer%id
                suffix = '_output'
             end if
             if(network%model(idx)%layer%use_graph_input)then
                write(tmp_input_name,'(A,A,A)') &
                     trim(adjustl(input_name)), '_vertex', suffix
                write(unit,'(4X,"input: """,A,"""")') trim(adjustl(tmp_input_name))
                if(network%model(idx)%layer%input_shape(2) .gt. 0)then
                   write(tmp_input_name,'(A,A,A)') &
                        trim(adjustl(input_name)), '_edge', suffix
                   write(unit,'(4X,"input: """,A,"""")') trim(adjustl(tmp_input_name))
                end if
             else
                write(unit,'(4X,"input: """,A,A,"""")') &
                     trim(adjustl(input_name)), suffix
             end if
          end do
       end if
       select type(layer => network%model(idx)%layer)
       class is(learnable_layer_type)
          if(layer%transfer%name.ne."none")then
             suffix = '_pre_function'
          else
             suffix = ''
          end if
          do j = 1, size(layer%weight_shape, dim=2)
             write(unit, '(4X,"input: ""node_",I0,"_weight",I0,"""")') &
                  network%model(idx)%layer%id, j
             if(layer%has_bias)then
                write(unit, '(4X,"input: ""node_",I0,"_bias",I0,"""")') &
                     network%model(idx)%layer%id, j
             end if
          end do
       class default
          suffix = ''
       end select

       ! Write output
       if(network%model(idx)%layer%use_graph_output)then
          write(unit, '(4X,"output: ""node_",I0,"_vertex_output",A,"""")') &
               network%model(idx)%layer%id, trim(adjustl(suffix))
          write(unit, '(4X,"output: ""node_",I0,"_edge_output",A,"""")') &
               network%model(idx)%layer%id, trim(adjustl(suffix))
       else
          write(unit, '(4X,"output: ""node_",I0,"_output",A,"""")') &
               network%model(idx)%layer%id, trim(adjustl(suffix))
       end if

       call write_onnx_attributes(unit, network%model(idx)%layer)

       write(unit, '(A)') '  }'
       write(unit, '(A)') ''

       select type(layer => network%model(idx)%layer)
       class is(learnable_layer_type)
          call write_onnx_initialisers(unit, layer, prefix = trim(node_name) )
          if(layer%transfer%name.ne."none")then
             if(layer%use_graph_output)then
                call write_onnx_function( &
                     unit, layer%transfer%name, &
                     prefix = trim(node_name)//'_vertex' &
                )
                if(network%model(idx)%layer%input_shape(2) .gt. 0)then
                   call write_onnx_function( &
                        unit, layer%transfer%name, &
                        prefix = trim(node_name)//'_edge' &
                   )
                end if
             else
                call write_onnx_function( &
                     unit, layer%transfer%name, &
                     prefix = trim(node_name) &
                )
             end if
          end if
       end select
    end do


    ! write all layer output shapes
    do i = 1, network%auto_graph%num_vertices
       idx = network%auto_graph%vertex(network%vertex_order(i))%id
       if(.not.allocated(network%model(idx)%layer%output_shape)) cycle
       if(network%model(idx)%layer%use_graph_output)then
          write(node_name, '("node_",I0,"_vertex_output")') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "value_info", &
               trim(adjustl(node_name)), &
               [ network%model(idx)%layer%output_shape(1) ], &
               network%batch_size &
          )
          if(network%model(idx)%layer%output_shape(2) .gt. 0)then
             write(node_name, '("node_",I0,"_edge_output")') network%model(idx)%layer%id
             call write_onnx_tensor( &
                  unit, &
                  "value_info", &
                  trim(adjustl(node_name)), &
                  [ network%model(idx)%layer%output_shape(2) ], &
                  network%batch_size &
             )
          end if
       else
          write(node_name, '("node_",I0,"_output")') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "value_info", &
               trim(adjustl(node_name)), &
               network%model(idx)%layer%output_shape, &
               network%batch_size &
          )
       end if
    end do

    ! Write inputs
    write(unit, '(A)') '  # Inputs'
    do i = 1, size(network%root_vertices, dim=1)
       idx = network%root_vertices(i)
       if(network%model(idx)%layer%use_graph_output)then
          write(node_name, '("input_",I0,"_vertex")') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "input", &
               trim(adjustl(node_name)), &
               [ network%model(idx)%layer%input_shape(1) ], &
               network%batch_size &
          )
          if(network%model(idx)%layer%input_shape(2) .gt. 0)then
             write(node_name, '("input_",I0,"_edge")') network%model(idx)%layer%id
             call write_onnx_tensor( &
                  unit, &
                  "input", &
                  trim(adjustl(node_name)), &
                  [ network%model(idx)%layer%input_shape(2) ], &
                  network%batch_size &
             )
          end if
       else
          write(node_name, '("input_",I0)') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "input", &
               trim(adjustl(node_name)), &
               network%model(idx)%layer%input_shape, &
               network%batch_size &
          )
       end if
    end do

    ! Write outputs
    write(unit, '(A)') '  # Outputs'
    do i = 1, size(network%leaf_vertices, dim=1)
       idx = network%leaf_vertices(i)
       if(network%model(idx)%layer%use_graph_output)then
          write(node_name, '("node_",I0,"_vertex_output")') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "output", &
               trim(adjustl(node_name)), &
               [ network%model(idx)%layer%output_shape(1) ], &
               network%batch_size &
          )
          if(network%model(idx)%layer%output_shape(2) .gt. 0)then
             write(node_name, '("node_",I0,"_edge_output")') network%model(idx)%layer%id
             call write_onnx_tensor( &
                  unit, &
                  "output", &
                  trim(adjustl(node_name)), &
                  [ network%model(idx)%layer%output_shape(2) ], &
                  network%batch_size &
             )
          end if
       else
          write(node_name, '("node_",I0,"_output")') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "output", &
               trim(adjustl(node_name)), &
               network%model(idx)%layer%output_shape, &
               network%batch_size &
          )
       end if
    end do

    write(unit, '(A)') '}'

    ! Write ONNX footer
    write(unit, '(A)') 'opset_import {'
    write(unit, '(A)') '  domain: "ai.onnx"'
    write(unit, '(A,I0)') '  version: ', 13  ! ONNX version
    write(unit, '(A)') '}'

    close(unit)

  end subroutine write_onnx
!###############################################################################


!###############################################################################
  module function read_onnx(file, verbose) result(network)
    !! Import a network from ONNX format
    implicit none

    ! Arguments
    character(*), intent(in) :: file
    !! File to import the network from
    integer, optional, intent(in) :: verbose
    !! Verbosity level (0=quiet, 1=normal, 2=debug)

    ! Return value
    type(network_type) :: network
    !! Network instance

    ! Local variables
    integer :: unit, stat, i, j, k, node_count, itmp1
    character(1024) :: trimmed_line, line
    character(256) :: op_type, node_name, temp_str
    character(20), allocatable, dimension(:) :: input_names, output_names
    integer, allocatable, dimension(:) :: dims
    integer :: num_inputs, num_outputs, batch_size
    real(real32), allocatable, dimension(:) :: float_data
    logical :: in_node, in_initialiser, reading_dims, reading_data
    integer :: node_id
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes

    integer :: verbose_
    character(1024) :: buffer1
    character(20) :: buffer2
    character(20), allocatable :: inputs(:), outputs(:)

    ! Node information storage
    type(onnx_node_type), allocatable, dimension(:) :: nodes

    ! Initializer storage
    type(onnx_initialiser_type), allocatable, dimension(:) :: initializers
    integer :: num_initialisers

    ! Tensor info storage (inputs, outputs, value_info)
    type :: tensor_info_type
       character(20) :: name
       integer, allocatable, dimension(:) :: dims
    end type tensor_info_type

    type(tensor_info_type), allocatable, dimension(:) :: input_tensors, &
         output_tensors
    integer :: num_input_tensors, num_output_tensors

    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    open(newunit=unit, file=file, status='old', action='read', iostat=stat)
    if(stat .ne. 0)then
       write(*,*) "ERROR: Could not open file: ", trim(file)
       return
    end if

    ! Initialise counters
    node_count = 0
    num_initialisers = 0
    num_input_tensors = 0
    num_output_tensors = 0
    in_node = .false.
    in_initialiser = .false.
    reading_dims = .false.
    reading_data = .false.
    batch_size = 1

    ! First pass: count nodes, initialisers, and tensors
    do
       !call read_full_line(unit, line)
       read(unit, '(A)', iostat=stat) line
       if(stat .ne. 0) exit

       trimmed_line = adjustl(trim(line))

       if(index(trimmed_line, 'node {') .gt. 0)then
          node_count = node_count + 1
       elseif(index(trimmed_line, 'initializer {') .gt. 0)then
          num_initialisers = num_initialisers + 1
       elseif(index(trimmed_line, 'input {') .gt. 0)then
          num_input_tensors = num_input_tensors + 1
       elseif(index(trimmed_line, 'output {') .gt. 0)then
          num_output_tensors = num_output_tensors + 1
       end if
    end do

    ! Allocate storage
    allocate(nodes(node_count))
    allocate(initializers(num_initialisers))
    allocate(input_tensors(num_input_tensors))
    allocate(output_tensors(num_output_tensors))

    ! Reset file for second pass
    rewind(unit)

    node_count = 0
    num_initialisers = 0
    num_input_tensors = 0
    num_output_tensors = 0

    ! Initialise node structures
    do i = 1, node_count
       nodes(i)%num_inputs = 0
       nodes(i)%num_outputs = 0
       nodes(i)%op_type = ""
       nodes(i)%name = ""
    end do

    ! Second pass: parse file content
    do
       read(unit, '(A)', iostat=stat) line
       if(stat .ne. 0) exit

       trimmed_line = trim(adjustl(line))
       buffer1 = trimmed_line

       ! Parse nodes
       if(index(trimmed_line, 'node {') .gt. 0)then
          in_node = .true.
          node_count = node_count + 1
          nodes(node_count)%num_inputs = 0
          nodes(node_count)%num_outputs = 0
          allocate(inputs(0))
          allocate(outputs(0))
          allocate(attributes(0))

       elseif(in_node .and. index(trimmed_line, '}') .gt. 0)then
          in_node = .false.
          if(size(attributes) .gt. 0)then
             allocate(nodes(node_count)%attributes(size(attributes)))
             do i = 1, size(attributes)
                nodes(node_count)%attributes(i) = attributes(i)
             end do
          end if
          if(size(inputs) .gt. 0)then
             allocate(nodes(node_count)%inputs(size(inputs)))
             do i = 1, size(inputs)
                nodes(node_count)%inputs(i) = inputs(i)
             end do
          end if
          if(size(outputs) .gt. 0)then
             allocate(nodes(node_count)%outputs(size(outputs)))
             do i = 1, size(outputs)
                nodes(node_count)%outputs(i) = outputs(i)
             end do
          end if
          deallocate(attributes)
          deallocate(inputs)
          deallocate(outputs)

       elseif(in_node)then
          if(index(trimmed_line, 'name:') .gt. 0)then
             call assign_val(buffer1, nodes(node_count)%name, itmp1, fs=":")
          elseif(index(trimmed_line, 'op_type:') .gt. 0)then
             call assign_val(buffer1, &
                  nodes(node_count)%op_type, itmp1, fs=":")
          elseif(index(trimmed_line, 'input:') .gt. 0)then
             nodes(node_count)%num_inputs = &
                  nodes(node_count)%num_inputs + 1
             buffer2 = trim(adjustl(trimmed_line(index(trimmed_line, 'input:') + 6:)))
             inputs = [ inputs, buffer2 ]
          elseif(index(trimmed_line, 'output:') .gt. 0)then
             nodes(node_count)%num_outputs = &
                  nodes(node_count)%num_outputs + 1
             buffer2 = trim(adjustl(trimmed_line(index(trimmed_line, 'output:') + 7:)))
             outputs = [ outputs, buffer2 ]
          elseif(index(trimmed_line, 'attribute {') .gt. 0)then
             attributes = [attributes, read_attribute(unit)]
          end if
       end if

       ! Parse initialisers
       if(index(trimmed_line, 'initializer {') .gt. 0)then
          in_initialiser = .true.
          num_initialisers = num_initialisers + 1
          reading_dims = .false.
          reading_data = .false.

       elseif(in_initialiser .and. index(trimmed_line, '}') .gt. 0)then
          in_initialiser = .false.
          reading_dims = .false.
          reading_data = .false.

       elseif(in_initialiser)then
          if(index(trimmed_line, 'name:') .gt. 0)then
             call assign_val(buffer1, &
                  initializers(num_initialisers)%name, itmp1, fs=":")
          elseif(index(trimmed_line, 'dims:') .gt. 0)then
             if(.not. reading_dims)then
                reading_dims = .true.
                if(allocated(dims)) deallocate(dims)
                allocate(dims(0))
             end if
             call assign_val(buffer1, j, itmp1, fs=":")
             dims = [dims, j]
             initializers(num_initialisers)%dims = dims
          elseif(index(trimmed_line, 'float_data:') .gt. 0)then
             reading_data = .true.
             allocate(initializers(num_initialisers)%data(0))
             do while(reading_data)
                read(unit, '(A)', iostat=stat) line
                if(stat .ne. 0) exit
                trimmed_line = trim(adjustl(line))
                if(index(trimmed_line, 'float_data:') .gt. 0)then
                   trimmed_line = trimmed_line(index(trimmed_line, 'float_data:') + 11:)
                end if
                if(index(trimmed_line, ']') .gt. 0)then
                   reading_data = .false.
                elseif(trim(adjustl(trimmed_line)) .ne. '')then
                   call allocate_and_assign_vec(trimmed_line, float_data, fs=":")
                   initializers(num_initialisers)%data = &
                        [initializers(num_initialisers)%data, float_data]
                   deallocate(float_data)
                end if
             end do

          end if
       end if

       ! Parse input tensors
       if(index(trimmed_line, 'input {') .gt. 0 .and. &
            .not. in_node .and. .not. in_initialiser)then
          num_input_tensors = num_input_tensors + 1
       elseif(num_input_tensors > 0 .and. &
            index(trimmed_line, 'name:') .gt. 0)then
          call assign_val(buffer1, &
               input_tensors(num_input_tensors)%name, itmp1, fs=":")
       elseif(num_input_tensors > 0 .and. &
            index(trimmed_line, 'dim_value:') .gt. 0)then
          if(.not. allocated(input_tensors(num_input_tensors)%dims))then
             allocate(input_tensors(num_input_tensors)%dims(0))
          end if
          call assign_val(buffer1, j, itmp1, fs=":")
          input_tensors(num_input_tensors)%dims = &
               [input_tensors(num_input_tensors)%dims, j]
       end if

       ! Parse output tensors
       if(index(trimmed_line, 'output {') .gt. 0 .and. &
            .not. in_node .and. .not. in_initialiser)then
          num_output_tensors = num_output_tensors + 1
       elseif(num_output_tensors > 0 .and. &
            index(trimmed_line, 'name:') .gt. 0)then
          call assign_val(buffer1, &
               output_tensors(num_output_tensors)%name, itmp1, fs=":")
       end if
    end do

    close(unit)

    ! Now construct the network from parsed information
    call network%build_from_onnx( &
         nodes, initializers, &
         verbose=verbose_ &
    )

  end function read_onnx
!###############################################################################


!###############################################################################
  function read_attribute(unit) result(attr)
    !! Reads an entire attribute block from an ONNX file
    !! Handles multi-line attributes (e.g., multiple ints or floats)
    implicit none
    integer, intent(in) :: unit
    type(onnx_attribute_type) :: attr
    character(1024) :: line, trimmed_line
    character(1024) :: value_buffer
    character(20) :: key, attr_type_key
    character(256) :: value_str
    integer :: stat, colon_pos
    logical :: done

    ! Initialise attribute
    attr%name = ""
    attr%type = ""
    allocate(character(0) :: attr%value)
    value_buffer = ""

    ! Read the opening "attribute {" line (already read by caller, so we're inside)
    done = .false.

    do while(.not. done)
       read(unit, '(A)', iostat=stat) line
       if(stat .ne. 0) exit

       trimmed_line = adjustl(trim(line))
       if(trim(trimmed_line) .eq. 'attribute {') cycle

       ! Check for closing brace
       if(index(trimmed_line, '}') .gt. 0) then
          done = .true.
          exit
       end if

       ! Parse line with colon separator
       colon_pos = index(trimmed_line, ':')
       if(colon_pos .gt. 0) then
          key = adjustl(trim(trimmed_line(1:colon_pos-1)))
          value_str = adjustl(trim(trimmed_line(colon_pos+1:)))
          ! strip all quotes from value_str if present
          if( &
               ( &
                    value_str(1:1) .eq. '"' .and. &
                    value_str(len(trim(value_str)):len(trim(value_str))) .eq. '"' &
               ) .or. ( &
                    value_str(1:1) .eq. '''' .and. &
                    value_str(len(trim(value_str)):len(trim(value_str))) .eq. '''' &
               ) &
          )then
             value_str = value_str(2:len(trim(value_str))-1)
          end if

          select case(trim(key))
          case('name')
             attr%name = trim(value_str)
          case('type')
             if(attr%type .ne. '')then
                write(0,*) "WARNING: Multiple 'type' entries in attribute. &
                     &Using the first one."
                cycle
             end if
             attr_type_key = trim(value_str)
             attr%type = to_lower(trim(attr_type_key))
          case('ints', 'floats', 'strings', 'i', 'f', 's')
             ! Accumulate multiple values with space separator
             if(len_trim(value_buffer) .eq. 0) then
                value_buffer = trim(value_str)
             else
                value_buffer = trim(value_buffer) // ' ' // trim(value_str)
             end if
          end select
       end if
    end do

    ! Store accumulated values
    if(len_trim(value_buffer) .gt. 0) then
       attr%value = trim(value_buffer)
    end if

  end function read_attribute
!###############################################################################


!###############################################################################
  subroutine write_onnx_tensor(unit, tensor_type, name, output_shape, batch_size)
    !! Write ONNX value info for a layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    character(*), intent(in) :: tensor_type
    !! Type of the tensor
    character(*), intent(in) :: name
    !! Name of the layer
    integer, intent(in), dimension(:) :: output_shape
    !! Shape of the layer output
    integer, intent(in) :: batch_size
    !! Batch size for the output

    ! Local variables
    integer :: i
    !! Loop index


    write(unit, '(A,A,A)') '  ',tensor_type,' {'
    write(unit, '(A,A,A)') '    name: "',name,'"'
    write(unit, '(A)') '    type {'
    write(unit, '(A)') '      tensor_type {'
    write(unit, '(A)') '        elem_type: 1'
    write(unit, '(A)') '        shape {'
    write(unit, '(A,I0)') '          dim { dim_value: ', max(1,batch_size)
    write(unit, '(A)') '          }'
    do i = size(output_shape), 1, -1
       write(unit, '(A,I0)') '          dim { dim_value: ', output_shape(i)
       write(unit, '(A)') '          }'
    end do
    write(unit, '(A)') '        }'
    write(unit, '(A)') '      }'
    write(unit, '(A)') '    }'
    write(unit, '(A)') '  }'

  end subroutine write_onnx_tensor
!###############################################################################


!###############################################################################
  subroutine write_onnx_initialisers(unit, layer, prefix)
    !! Write ONNX initialisers (weights and biases)
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    class(learnable_layer_type), intent(in) :: layer
    !! Instance of a layer
    character(*), intent(in) :: prefix
    !! Optional prefix for weight and bias names

    ! Local variables
    integer :: i, j, num_params, num_params_old
    !! Loop indices
    character(20) :: name
    !! Names for weights and biases


    if(allocated(layer%params))then
       num_params_old = 0
       do i = 1, size(layer%weight_shape, 2)
          write(name, '(A,A,I0)') trim(prefix), '_weight', i
          write(unit, '(2X,A)') 'initializer {'
          write(unit, '(4X,"name: """,A,"""")') trim(name)
          write(unit, '(4X,A)') 'data_type: 1'  ! FLOAT
          do j = size(layer%weight_shape, 1), 1, -1
             write(unit, '(4X,A,I0)') 'dims: ', layer%weight_shape(j,i)
          end do
          num_params = product(layer%weight_shape(:, i))

          write(unit, '(4X,"float_data: [ ")')
          write(unit, '(20(F0.6,", "))') layer%params(num_params_old + 1:num_params-1)
          write(unit, '(F0.6)') layer%params(num_params + num_params_old)
          write(unit, '(A)') ' ]'
          write(unit, '(A)') '  }'
          write(unit, '(A)') ''

          num_params_old = num_params_old + num_params

          if(layer%has_bias)then
             write(name, '(A,A,I0)') trim(prefix), '_bias', i
             write(unit, '(2X,A)') 'initializer {'
             write(unit, '(4X,"name: """,A,"""")') trim(name)
             write(unit, '(4X,A)') 'data_type: 1'  ! FLOAT
             write(unit, '(4X,A,I0)') 'dims: ', layer%bias_shape(i)
             num_params = layer%bias_shape(i)

             write(unit, '(4X,"float_data: [ ")')
             write(unit, '(20(F0.6,", "))') &
                  layer%params(num_params_old + 1:num_params_old + num_params - 1)
             write(unit, '(F0.6)') layer%params(num_params_old + num_params)
             write(unit, '(A)') ' ]'
             write(unit, '(A)') '  }'
             write(unit, '(A)') ''

             num_params_old = num_params_old + layer%bias_shape(i)
          end if
       end do
    end if

  end subroutine write_onnx_initialisers
!###############################################################################


!###############################################################################
  subroutine write_onnx_function(unit, function_name, prefix)
    !! Write ONNX function definition
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    character(*), intent(in) :: function_name
    !! Name of the function
    character(*), intent(in) :: prefix
    !! Optional prefix for the function name

    ! Local variables
    character(256) :: full_name
    !! Full name of the function
    character(:), allocatable :: function_name_camel_case
    !! Camel case version of the function name

    function_name_camel_case = &
         to_camel_case(trim(adjustl(function_name)), capitalise_first_letter = .true.)
    if(prefix .eq. "")then
       full_name = trim(adjustl(function_name))
    else
       full_name = trim(prefix) // "_" // trim(adjustl(function_name))
    end if


    write(unit, '(A)') '  node {'
    write(unit, '(A,A,A)') '    name: "', trim(full_name), '"'
    write(unit, '(A,A,A)') '    op_type: "', trim(function_name_camel_case), '"'
    write(unit, '(A,A,A)') '    input: "', trim(prefix), '_output_pre_function"'
    write(unit, '(A,A,A)') '    output: "', trim(prefix), '_output"'
    write(unit, '(A)') '  }'
    write(unit, '(A)') ''

  end subroutine write_onnx_function
!###############################################################################


!###############################################################################
  subroutine write_onnx_attributes(unit, layer)
    !! Write ONNX attributes for a layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    class(base_layer_type), intent(in) :: layer
    !! Instance of a layer

    ! Local variables
    integer :: i, j, itmp1
    !! Loop index
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    character(:), allocatable :: type_lw, type_up
    integer, allocatable, dimension(:) :: ivar_list
    real(real32), allocatable, dimension(:) :: rvar_list


    attributes = layer%get_attributes()
    if(allocated(attributes).and. size(attributes) .gt. 0)then
       do i = 1, size(attributes)
          write(unit, '(4X,A)') 'attribute {'
          write(unit, '(6X,"name: """,A,"""")') trim(attributes(i)%name)
          ! determine whether the attribute is a list or a single value
          type_lw = to_lower(trim(adjustl(attributes(i)%type)))
          type_up = to_upper(trim(adjustl(attributes(i)%type)))
          itmp1 = icount(attributes(i)%value)
          select case(type_lw)
          case('ints','int')
             allocate(ivar_list(itmp1))
             read(attributes(i)%value,*) ivar_list
             do j = 1, size(ivar_list)
                write(unit, '(6X,A,": ",I0)') type_lw, ivar_list(j)
             end do
             deallocate(ivar_list)
          case('floats','float')
             allocate(rvar_list(itmp1))
             read(attributes(i)%value,*) rvar_list
             do j = 1, size(rvar_list), 1
                write(unit, '(6X,A,": ",F0.6)') type_lw, rvar_list(j)
             end do
             deallocate(rvar_list)
          case('strings','string')
          case default
             write(unit, '(6X,A,": ",A)') trim(adjustl(attributes(i)%type)), &
                  trim(adjustl(attributes(i)%value))
          end select
          write(unit,'(6X,"type: ",A)') type_up
          write(unit,'(4X,"}")')
       end do
    end if

  end subroutine write_onnx_attributes
!###############################################################################

end submodule athena__onnx_submodule
