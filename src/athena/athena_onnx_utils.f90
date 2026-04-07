module athena__onnx_utils
  !! Shared utility routines for ONNX JSON export
  !!
  !! Contains base64 encoding, node emission helpers, col→row transpose,
  !! and activation/attribute-building utilities used by both the main
  !! write_onnx procedure and layer-specific emit_onnx_nodes overrides.
  use coreutils, only: real32, to_lower
  use athena__misc_types, only: onnx_node_type, onnx_initialiser_type, &
       onnx_tensor_type, onnx_attribute_type
  use athena__base_layer, only: base_layer_type, learnable_layer_type
  implicit none

  private

  public :: emit_node
  public :: emit_squeeze_node
  public :: emit_constant_int64
  public :: emit_constant_float
  public :: emit_constant_of_shape_float
  public :: emit_activation_node
  public :: emit_initialisers
  public :: build_attributes_json
  public :: col_to_row_major_2d
  public :: write_json_nodes
  public :: write_json_initialisers
  public :: write_json_tensors
  public :: encode_float32_base64
  public :: encode_float32_base64_alloc
  public :: encode_int64_base64
  public :: encode_int64_base64_alloc
  public :: decode_base64_to_float32
  public :: decode_base64_to_int64
  public :: row_to_col_major_2d
  public :: encode_string_base64
  public :: base64_encode_bytes
  public :: base64_encode_bytes_fixed
  public :: parse_space_separated_ints
  public :: onnx_to_athena_activation

contains


!###############################################################################
  subroutine emit_node(op_type, name, out1, attr_json, nodes, num_nodes, &
       in1, in2, in3)
    !! Emit a simple ONNX node (individual string interface)
    !! Avoids gfortran array constructor issues
    implicit none

    ! Arguments
    character(*), intent(in) :: op_type, name, out1, attr_json
    !! ONNX operation type, node name, output name, and attribute JSON
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    !! Node accumulator array
    integer, intent(inout) :: num_nodes
    !! Current number of populated nodes
    character(*), intent(in), optional :: in1, in2, in3
    !! Optional input tensor names

    ! Local variables
    integer :: n_in
    !! Number of connected inputs for the emitted node

    n_in = 0
    if(present(in1)) n_in = 1
    if(present(in2)) n_in = 2
    if(present(in3)) n_in = 3

    num_nodes = num_nodes + 1
    nodes(num_nodes)%name = trim(name)
    nodes(num_nodes)%op_type = trim(op_type)
    allocate(nodes(num_nodes)%inputs(n_in))
    if(present(in1)) nodes(num_nodes)%inputs(1) = trim(in1)
    if(present(in2)) nodes(num_nodes)%inputs(2) = trim(in2)
    if(present(in3)) nodes(num_nodes)%inputs(3) = trim(in3)
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(out1)
    nodes(num_nodes)%attributes_json = attr_json

  end subroutine emit_node
!###############################################################################


!###############################################################################
  subroutine emit_squeeze_node(name, input, axes_input, output, &
       nodes, num_nodes)
    !! Emit a Squeeze node (ONNX opset 13+: axes as input)
    implicit none

    ! Arguments
    character(*), intent(in) :: name, input, axes_input, output
    !! Node name, data input, axes input, and output tensor name
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    !! Node accumulator array
    integer, intent(inout) :: num_nodes
    !! Current number of populated nodes

    call emit_node('Squeeze', name, output, '', nodes, num_nodes, &
         in1=input, in2=axes_input)

  end subroutine emit_squeeze_node
!###############################################################################


!###############################################################################
  subroutine emit_constant_int64(name, values, dims, &
       nodes, num_nodes, inits, num_inits)
    !! Emit a Constant node producing an int64 tensor
    implicit none

    ! Arguments
    character(*), intent(in) :: name
    !! Constant node and output tensor name
    integer, intent(in) :: values(:), dims(:)
    !! Constant values and tensor dimensions
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    !! Node accumulator array
    integer, intent(inout) :: num_nodes
    !! Current number of populated nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    !! Initialiser accumulator array (unused, kept for interface symmetry)
    integer, intent(inout) :: num_inits
    !! Current number of populated initialisers (unused)

    ! Local variables
    character(4096) :: attr_str
    !! Serialized ONNX attribute JSON for the constant payload
    character(256) :: raw_b64
    !! Base64-encoded raw tensor data
    integer :: i
    !! Dimension loop index

    ! Encode int64 values as base64
    call encode_int64_base64(values, raw_b64)

    ! Build dims string
    attr_str = '        "attribute": [{"name": "value", "t": {'
    if(size(dims) .gt. 0)then
       attr_str = trim(attr_str) // '"dims": ['
       do i = 1, size(dims)
          if(i .gt. 1) attr_str = trim(attr_str) // ', '
          write(attr_str, '(A,"""",I0,"""")') trim(attr_str), dims(i)
       end do
       attr_str = trim(attr_str) // '], '
    end if
    attr_str = trim(attr_str) // '"dataType": 7, "rawData": "' // &
         trim(raw_b64) // '"}, "type": "TENSOR"}]'

    num_nodes = num_nodes + 1
    nodes(num_nodes)%name = trim(name)
    nodes(num_nodes)%op_type = 'Constant'
    allocate(nodes(num_nodes)%inputs(0))
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(name)
    nodes(num_nodes)%attributes_json = trim(attr_str)

  end subroutine emit_constant_int64
!###############################################################################


!###############################################################################
  subroutine emit_constant_float(name, values, dims, &
       nodes, num_nodes, inits, num_inits)
    !! Emit a Constant node producing a float32 tensor
    implicit none

    ! Arguments
    character(*), intent(in) :: name
    !! Constant node and output tensor name
    real(real32), intent(in) :: values(:)
    !! Float32 constant values to embed in the node
    integer, intent(in) :: dims(:)
    !! Tensor dimensions for the constant value
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    !! Node accumulator array
    integer, intent(inout) :: num_nodes
    !! Current number of populated nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    !! Initialiser accumulator array (unused, kept for interface symmetry)
    integer, intent(inout) :: num_inits
    !! Current number of populated initialisers (unused)

    ! Local variables
    character(4096) :: attr_str
    !! Serialized ONNX attribute JSON for the constant payload
    character(256) :: raw_b64
    !! Base64-encoded raw tensor data
    integer :: i
    !! Dimension loop index

    call encode_float32_base64(values, size(values), raw_b64)

    attr_str = '        "attribute": [{"name": "value", "t": {'
    if(size(dims) .gt. 0)then
       attr_str = trim(attr_str) // '"dims": ['
       do i = 1, size(dims)
          if(i .gt. 1) attr_str = trim(attr_str) // ', '
          write(attr_str, '(A,"""",I0,"""")') trim(attr_str), dims(i)
       end do
       attr_str = trim(attr_str) // '], '
    end if
    attr_str = trim(attr_str) // '"dataType": 1, "rawData": "' // &
         trim(raw_b64) // '"}, "type": "TENSOR"}]'

    num_nodes = num_nodes + 1
    nodes(num_nodes)%name = trim(name)
    nodes(num_nodes)%op_type = 'Constant'
    allocate(nodes(num_nodes)%inputs(0))
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(name)
    nodes(num_nodes)%attributes_json = trim(attr_str)

  end subroutine emit_constant_float
!###############################################################################


!###############################################################################
  subroutine emit_constant_of_shape_float(name, shape_input, value, output, &
       nodes, num_nodes, inits, num_inits)
    !! Emit a ConstantOfShape node
    implicit none

    ! Arguments
    character(*), intent(in) :: name, shape_input, output
    !! Node name, shape tensor input, and output tensor name
    real(real32), intent(in) :: value
    !! Fill value to use for the generated tensor
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    !! Node accumulator array
    integer, intent(inout) :: num_nodes
    !! Current number of populated nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    !! Initialiser accumulator array (unused, kept for interface symmetry)
    integer, intent(inout) :: num_inits
    !! Current number of populated initialisers (unused)

    ! Local variables
    character(4096) :: attr_str
    !! Serialized ONNX attribute JSON for the fill tensor
    character(256) :: raw_b64
    !! Base64-encoded fill value

    call encode_float32_base64([value], 1, raw_b64)

    attr_str = '        "attribute": [{"name": "value", "t": {' // &
         '"dims": ["1"], "dataType": 1, "rawData": "' // &
         trim(raw_b64) // '"}, "type": "TENSOR"}]'

    call emit_node('ConstantOfShape', name, output, &
         trim(attr_str), nodes, num_nodes, &
         in1=shape_input)

  end subroutine emit_constant_of_shape_float
!###############################################################################


!###############################################################################
  subroutine emit_activation_node(name, prefix, input_override, &
       nodes, num_nodes, max_nodes)
    !! Emit an activation function node
    use coreutils, only: to_camel_case
    implicit none

    ! Arguments
    character(*), intent(in) :: name, prefix, input_override
    !! Activation name, node prefix, and optional override for the input name
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    !! Node accumulator array
    integer, intent(inout) :: num_nodes
    !! Current number of populated nodes
    integer, intent(in) :: max_nodes
    !! Maximum number of nodes available in the accumulator

    ! Local variables
    character(128) :: actv_name, input_n, output_n
    !! Normalised ONNX op name, input tensor name, and output tensor name
    character(4096) :: attr_str
    !! Serialized ONNX attribute JSON for activation-specific options

    actv_name = to_camel_case( &
         trim(adjustl(name)), &
         capitalise_first_letter = .true.)

    if(len_trim(input_override) .gt. 0)then
       input_n = trim(input_override)
    else
       input_n = trim(prefix) // '_output'
    end if
    output_n = trim(prefix) // '_' // trim(adjustl(name)) // '_output'

    attr_str = ''
    ! LeakyRelu needs alpha attribute
    if(trim(name) .eq. 'leaky_relu')then
       actv_name = 'LeakyRelu'
       attr_str = '        "attribute": [{"name": "alpha", ' // &
            '"f": 0.01, "type": "FLOAT"}]'
    end if

    call emit_node(trim(actv_name), &
         trim(prefix)//'_'//trim(adjustl(name)), &
         trim(output_n), trim(attr_str), nodes, num_nodes, &
         in1=trim(input_n))

  end subroutine emit_activation_node
!###############################################################################


!###############################################################################
  subroutine emit_initialisers(layer, prefix, inits, num_inits, max_inits)
    !! Emit initialisers for a learnable layer
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: layer
    !! Learnable layer containing parameter tensors and shape metadata
    character(*), intent(in) :: prefix
    !! Name prefix used to generate exported initialiser names
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    !! Initialiser accumulator array
    integer, intent(inout) :: num_inits
    !! Current number of populated initialisers
    integer, intent(in) :: max_inits
    !! Maximum number of initialisers available in the accumulator

    ! Local variables
    integer :: i, j, n
    !! Parameter index, shape index, and flattened tensor size
    character(128) :: name
    !! Generated ONNX initialiser name for the current parameter tensor

    if(.not.allocated(layer%params)) return

    do i = 1, size(layer%params)
       n = size(layer%params(i)%val, 1)
       num_inits = num_inits + 1
       write(name, '(A,"_param",I0)') trim(prefix), i
       inits(num_inits)%name = trim(name)
       inits(num_inits)%data_type = 1

       ! Set dims from weight_shape
       if(allocated(layer%weight_shape))then
          if(i .le. size(layer%weight_shape, 2))then
             j = 0
             allocate(inits(num_inits)%dims(size(layer%weight_shape, 1)))
             inits(num_inits)%dims = 0
             do j = 1, size(layer%weight_shape, 1)
                if(layer%weight_shape(j,i) .gt. 0)then
                   inits(num_inits)%dims(j) = layer%weight_shape(j,i)
                end if
             end do
             ! Remove zero dims
             inits(num_inits)%dims = pack(inits(num_inits)%dims, &
                  inits(num_inits)%dims .gt. 0)
          else
             allocate(inits(num_inits)%dims(1))
             inits(num_inits)%dims = [n]
          end if
       else
          allocate(inits(num_inits)%dims(1))
          inits(num_inits)%dims = [n]
       end if

       allocate(inits(num_inits)%data(n))
       ! Convert column-major to row-major for 2D weight matrices
       if(allocated(layer%weight_shape))then
          if(i .le. size(layer%weight_shape, 2))then
             if(count(layer%weight_shape(:,i) .gt. 0) .eq. 2)then
                call col_to_row_major_2d( &
                     layer%params(i)%val(:,1), &
                     inits(num_inits)%data, &
                     layer%weight_shape(1,i), &
                     layer%weight_shape(2,i))
             else
                inits(num_inits)%data = layer%params(i)%val(:,1)
             end if
          else
             inits(num_inits)%data = layer%params(i)%val(:,1)
          end if
       else
          inits(num_inits)%data = layer%params(i)%val(:,1)
       end if
    end do

  end subroutine emit_initialisers
!###############################################################################


!###############################################################################
  subroutine build_attributes_json(layer, op_type, attr_json)
    !! Build JSON string for layer attributes
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: layer
    !! Layer supplying ONNX attribute metadata
    character(*), intent(in) :: op_type
    !! ONNX operation type used to handle special cases
    character(4096), intent(out) :: attr_json
    !! Serialized JSON fragment containing the emitted attributes

    ! Local variables
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Attribute list returned by the layer
    integer :: i, j, itmp1
    !! Attribute index, value index, and temporary integer count
    character(256) :: buf
    !! Temporary string buffer for serialised scalar values
    integer, allocatable :: ivar_list(:)
    !! Parsed integer attribute payload for INTS-valued attributes
    real(real32), allocatable :: rvar_list(:)
    !! Placeholder for future real-valued list attributes
    character(10) :: type_lw
    !! Lower-case attribute type string used in the select case

    attr_json = ''

    ! For Gemm, add transB attribute
    if(trim(op_type) .eq. 'Gemm')then
       attr_json = '        "attribute": [' // &
            '{"name": "alpha", "f": 1.0, "type": "FLOAT"}, ' // &
            '{"name": "beta", "f": 1.0, "type": "FLOAT"}, ' // &
            '{"name": "transB", "i": "1", "type": "INT"}]'
       return
    end if

    attributes = layer%get_attributes()
    if(.not.allocated(attributes) .or. size(attributes) .eq. 0) return

    attr_json = '        "attribute": ['
    do i = 1, size(attributes)
       if(i .gt. 1) attr_json = trim(attr_json) // ', '
       attr_json = trim(attr_json) // '{"name": "' // &
            trim(attributes(i)%name) // '"'

       type_lw = trim(adjustl(attributes(i)%type))
       select case(type_lw)
       case('int')
          attr_json = trim(attr_json) // ', "i": "' // &
               trim(adjustl(attributes(i)%val)) // '", "type": "INT"}'
       case('ints')
          ! Parse multiple ints
          itmp1 = 1
          do j = 1, len_trim(attributes(i)%val)
             if(attributes(i)%val(j:j) .eq. ' ') itmp1 = itmp1 + 1
          end do
          allocate(ivar_list(itmp1))
          read(attributes(i)%val, *) ivar_list
          attr_json = trim(attr_json) // ', "ints": ['
          do j = 1, itmp1
             if(j .gt. 1) attr_json = trim(attr_json) // ', '
             write(buf, '("""",I0,"""")') ivar_list(j)
             attr_json = trim(attr_json) // trim(adjustl(buf))
          end do
          attr_json = trim(attr_json) // '], "type": "INTS"}'
          deallocate(ivar_list)
       case('float')
          attr_json = trim(attr_json) // ', "f": ' // &
               trim(adjustl(attributes(i)%val)) // ', "type": "FLOAT"}'
       case('string')
          ! Base64 encode the string value
          call encode_string_base64( &
               trim(adjustl(attributes(i)%val)), buf)
          attr_json = trim(attr_json) // ', "s": "' // &
               trim(buf) // '", "type": "STRING"}'
       case default
          attr_json = trim(attr_json) // '}'
       end select
    end do
    attr_json = trim(attr_json) // ']'

  end subroutine build_attributes_json
!###############################################################################


! =============================================================================
! JSON serialisation utilities
! =============================================================================

!###############################################################################
  subroutine write_json_nodes(unit, nodes, num_nodes)
    !! Write nodes array to JSON
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Output unit receiving the JSON text
    type(onnx_node_type), intent(in), dimension(:) :: nodes
    !! Node collection to serialise
    integer, intent(in) :: num_nodes
    !! Number of populated nodes in the collection

    ! Local variables
    integer :: i, j
    !! Node and tensor index counters

    write(unit, '(A)') '    "node": ['
    do i = 1, num_nodes
       write(unit, '(A)') '      {'
       ! Write inputs
       if(allocated(nodes(i)%inputs) .and. size(nodes(i)%inputs) .gt. 0)then
          write(unit, '(A)', advance='no') '        "input": ['
          do j = 1, size(nodes(i)%inputs)
             if(j .gt. 1) write(unit, '(A)', advance='no') ', '
             write(unit, '(A,A,A)', advance='no') '"', &
                  trim(adjustl(nodes(i)%inputs(j))), '"'
          end do
          write(unit, '(A)') '],'
       end if
       ! Write outputs
       if(allocated(nodes(i)%outputs))then
          write(unit, '(A)', advance='no') '        "output": ['
          do j = 1, size(nodes(i)%outputs)
             if(j .gt. 1) write(unit, '(A)', advance='no') ', '
             write(unit, '(A,A,A)', advance='no') '"', &
                  trim(adjustl(nodes(i)%outputs(j))), '"'
          end do
          write(unit, '(A)') '],'
       end if
       ! Name
       write(unit, '(A,A,A)', advance='no') '        "name": "', &
            trim(adjustl(nodes(i)%name)), '"'
       ! OpType
       write(unit, '(A)') ','
       write(unit, '(A,A,A)', advance='no') '        "opType": "', &
            trim(adjustl(nodes(i)%op_type)), '"'
       ! Attributes
       if(len_trim(nodes(i)%attributes_json) .gt. 0)then
          write(unit, '(A)') ','
          write(unit, '(A)') trim(nodes(i)%attributes_json)
       else
          write(unit, '(A)') ''
       end if
       if(i .lt. num_nodes)then
          write(unit, '(A)') '      },'
       else
          write(unit, '(A)') '      }'
       end if
    end do
    write(unit, '(A)') '    ]'

  end subroutine write_json_nodes
!###############################################################################


!###############################################################################
  subroutine write_json_initialisers(unit, inits, num_inits)
    !! Write initialisers array to JSON with base64-encoded rawData
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Output unit receiving the JSON text
    type(onnx_initialiser_type), intent(in), dimension(:) :: inits
    !! Initialiser collection to serialise
    integer, intent(in) :: num_inits
    !! Number of populated initialisers in the collection

    ! Local variables
    integer :: i, j, n
    !! Initialiser index, dimension index, and raw element count
    character(:), allocatable :: raw_b64
    !! Base64-encoded raw tensor payload

    write(unit, '(A)') '    "initializer": ['
    do i = 1, num_inits
       write(unit, '(A)') '      {'
       ! Dims
       if(allocated(inits(i)%dims))then
          write(unit, '(A)', advance='no') '        "dims": ['
          do j = 1, size(inits(i)%dims)
             if(j .gt. 1) write(unit, '(A)', advance='no') ', '
             write(unit, '("""",I0,"""")' , advance='no') inits(i)%dims(j)
          end do
          write(unit, '(A)') '],'
       end if
       ! Data type
       write(unit, '(A,I0,A)') '        "dataType": ', inits(i)%data_type, ','
       ! Name
       write(unit, '(A,A,A)') '        "name": "', &
            trim(adjustl(inits(i)%name)), '",'
       ! Raw data (base64 encoded)
       if(allocated(inits(i)%data))then
          n = size(inits(i)%data)
          call encode_float32_base64_alloc(inits(i)%data, n, raw_b64)
          write(unit, '(A,A,A)') '        "rawData": "', raw_b64, '"'
       else if(allocated(inits(i)%int_data))then
          n = size(inits(i)%int_data)
          call encode_int64_base64_alloc(inits(i)%int_data, n, raw_b64)
          write(unit, '(A,A,A)') '        "rawData": "', raw_b64, '"'
       else
          write(unit, '(A)') '        "rawData": ""'
       end if
       if(i .lt. num_inits)then
          write(unit, '(A)') '      },'
       else
          write(unit, '(A)') '      }'
       end if
    end do
    write(unit, '(A)') '    ]'

  end subroutine write_json_initialisers
!###############################################################################


!###############################################################################
  subroutine write_json_tensors(unit, section_name, tensors, num_tensors)
    !! Write input/output tensor specifications to JSON
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Output unit receiving the JSON text
    character(*), intent(in) :: section_name
    !! JSON section name, e.g. input or output
    type(onnx_tensor_type), intent(in), dimension(:) :: tensors
    !! Tensor collection to serialise
    integer, intent(in) :: num_tensors
    !! Number of populated tensors in the collection

    ! Local variables
    integer :: i, j
    !! Tensor and dimension index counters

    write(unit, '(A,A,A)') '    "', trim(section_name), '": ['
    do i = 1, num_tensors
       write(unit, '(A)') '      {'
       write(unit, '(A,A,A)') '        "name": "', &
            trim(adjustl(tensors(i)%name)), '",'
       write(unit, '(A)') '        "type": {'
       write(unit, '(A)') '          "tensorType": {'
       write(unit, '(A,I0,A)') '            "elemType": ', &
            tensors(i)%elem_type, ','
       write(unit, '(A)') '            "shape": {'
       write(unit, '(A)') '              "dim": ['
       if(allocated(tensors(i)%dims))then
          do j = 1, size(tensors(i)%dims)
             write(unit, '(A)') '                {'
             if(allocated(tensors(i)%dim_params))then
                if(len_trim(tensors(i)%dim_params(j)) .gt. 0)then
                   write(unit, '(A,A,A)') '                  "dimParam": "', &
                        trim(adjustl(tensors(i)%dim_params(j))), '"'
                else
                   write(unit, '(A,"""",I0,"""")') &
                        '                  "dimValue": ', &
                        tensors(i)%dims(j)
                end if
             else
                write(unit, '(A,"""",I0,"""")') &
                     '                  "dimValue": ', &
                     tensors(i)%dims(j)
             end if
             if(j .lt. size(tensors(i)%dims))then
                write(unit, '(A)') '                },'
             else
                write(unit, '(A)') '                }'
             end if
          end do
       end if
       write(unit, '(A)') '              ]'
       write(unit, '(A)') '            }'
       write(unit, '(A)') '          }'
       write(unit, '(A)') '        }'
       if(i .lt. num_tensors)then
          write(unit, '(A)') '      },'
       else
          write(unit, '(A)') '      }'
       end if
    end do
    write(unit, '(A,A,A)') '    ]'

  end subroutine write_json_tensors
!###############################################################################


! =============================================================================
! Data layout utilities
! =============================================================================

!###############################################################################
  subroutine col_to_row_major_2d(data_in, data_out, m, n)
    !! Convert flat column-major [m,n] to flat row-major [m,n]
    !! Fortran stores arrays column-major; ONNX rawData expects row-major.
    implicit none
    integer, intent(in) :: m, n
    real(real32), intent(in) :: data_in(m * n)
    real(real32), intent(out) :: data_out(m * n)
    integer :: i, j
    do i = 1, m
       do j = 1, n
          data_out((i-1)*n + j) = data_in((j-1)*m + i)
       end do
    end do
  end subroutine col_to_row_major_2d
!###############################################################################


! =============================================================================
! Base64 encoding utilities
! =============================================================================

!###############################################################################
  subroutine encode_float32_base64(values, n, output)
    !! Encode float32 array as base64 string (fixed-length output)
    use iso_fortran_env, only: int8
    implicit none
    real(real32), intent(in) :: values(:)
    integer, intent(in) :: n
    character(256), intent(out) :: output

    character(:), allocatable :: result
    call encode_float32_base64_alloc(values, n, result)
    output = result

  end subroutine encode_float32_base64
!###############################################################################


!###############################################################################
  subroutine encode_float32_base64_alloc(values, n, output)
    !! Encode float32 array as base64 string (allocatable output)
    use iso_fortran_env, only: int8, int32
    implicit none
    real(real32), intent(in) :: values(:)
    integer, intent(in) :: n
    character(:), allocatable, intent(out) :: output

    integer(int8), allocatable :: bytes(:)
    integer(int32) :: ival
    integer :: i, j, nbytes

    nbytes = n * 4
    allocate(bytes(nbytes))

    do i = 1, n
       ival = transfer(values(i), ival)
       bytes((i-1)*4 + 1) = int(iand(ival, 255), int8)
       bytes((i-1)*4 + 2) = int(iand(ishft(ival, -8), 255), int8)
       bytes((i-1)*4 + 3) = int(iand(ishft(ival, -16), 255), int8)
       bytes((i-1)*4 + 4) = int(iand(ishft(ival, -24), 255), int8)
    end do

    call base64_encode_bytes(bytes, nbytes, output)
    deallocate(bytes)

  end subroutine encode_float32_base64_alloc
!###############################################################################


!###############################################################################
  subroutine encode_int64_base64(values, output)
    !! Encode integer array as base64 int64 string (fixed-length output)
    use iso_fortran_env, only: int8, int64
    implicit none
    integer, intent(in) :: values(:)
    character(256), intent(out) :: output

    integer(int8), allocatable :: bytes(:)
    integer(int64) :: ival64
    integer :: i, j, n, nbytes

    n = size(values)
    nbytes = n * 8
    allocate(bytes(nbytes))

    do i = 1, n
       ival64 = int(values(i), int64)
       do j = 0, 7
          bytes((i-1)*8 + j + 1) = &
               int(iand(ishft(ival64, -j*8), int(255, int64)), int8)
       end do
    end do

    call base64_encode_bytes_fixed(bytes, nbytes, output)
    deallocate(bytes)

  end subroutine encode_int64_base64
!###############################################################################


!###############################################################################
  subroutine encode_int64_base64_alloc(values, n, output)
    !! Encode integer array as base64 int64 string (allocatable output)
    use iso_fortran_env, only: int8, int64
    implicit none
    integer, intent(in) :: values(:)
    integer, intent(in) :: n
    character(:), allocatable, intent(out) :: output

    integer(int8), allocatable :: bytes(:)
    integer(int64) :: ival64
    integer :: i, j, nbytes

    nbytes = n * 8
    allocate(bytes(nbytes))

    do i = 1, n
       ival64 = int(values(i), int64)
       do j = 0, 7
          bytes((i-1)*8 + j + 1) = &
               int(iand(ishft(ival64, -j*8), int(255, int64)), int8)
       end do
    end do

    call base64_encode_bytes(bytes, nbytes, output)
    deallocate(bytes)

  end subroutine encode_int64_base64_alloc
!###############################################################################


!###############################################################################
  subroutine encode_string_base64(str, output)
    !! Encode a string as base64
    use iso_fortran_env, only: int8
    implicit none
    character(*), intent(in) :: str
    character(256), intent(out) :: output

    integer(int8), allocatable :: bytes(:)
    integer :: i, n

    n = len_trim(str)
    allocate(bytes(n))
    do i = 1, n
       bytes(i) = int(ichar(str(i:i)), int8)
    end do

    call base64_encode_bytes_fixed(bytes, n, output)
    deallocate(bytes)

  end subroutine encode_string_base64
!###############################################################################


!###############################################################################
  subroutine base64_encode_bytes(bytes, nbytes, output)
    !! Core base64 encoder (allocatable output)
    use iso_fortran_env, only: int8
    implicit none
    integer(int8), intent(in) :: bytes(:)
    integer, intent(in) :: nbytes
    character(:), allocatable, intent(out) :: output

    character(64), parameter :: b64 = &
         'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    integer :: i, j, ngroups, out_len
    integer :: b0, b1, b2, idx

    ngroups = (nbytes + 2) / 3
    out_len = ngroups * 4
    allocate(character(out_len) :: output)

    j = 1
    do i = 1, nbytes, 3
       b0 = iand(int(bytes(i)), 255)
       if(i + 1 .le. nbytes)then
          b1 = iand(int(bytes(i+1)), 255)
       else
          b1 = 0
       end if
       if(i + 2 .le. nbytes)then
          b2 = iand(int(bytes(i+2)), 255)
       else
          b2 = 0
       end if

       idx = ishft(b0, -2) + 1
       output(j:j) = b64(idx:idx)

       idx = ior(ishft(iand(b0, 3), 4), ishft(b1, -4)) + 1
       output(j+1:j+1) = b64(idx:idx)

       if(i + 1 .le. nbytes)then
          idx = ior(ishft(iand(b1, 15), 2), ishft(b2, -6)) + 1
          output(j+2:j+2) = b64(idx:idx)
       else
          output(j+2:j+2) = '='
       end if

       if(i + 2 .le. nbytes)then
          idx = iand(b2, 63) + 1
          output(j+3:j+3) = b64(idx:idx)
       else
          output(j+3:j+3) = '='
       end if

       j = j + 4
    end do

  end subroutine base64_encode_bytes
!###############################################################################


!###############################################################################
  subroutine base64_encode_bytes_fixed(bytes, nbytes, output)
    !! Core base64 encoder (fixed-length output)
    use iso_fortran_env, only: int8
    implicit none
    integer(int8), intent(in) :: bytes(:)
    integer, intent(in) :: nbytes
    character(256), intent(out) :: output

    character(:), allocatable :: tmp
    call base64_encode_bytes(bytes, nbytes, tmp)
    output = tmp

  end subroutine base64_encode_bytes_fixed
!###############################################################################


! =============================================================================
! Base64 decoding utilities
! =============================================================================

!###############################################################################
  subroutine base64_decode_bytes(input, bytes, nbytes)
    !! Core base64 decoder
    use iso_fortran_env, only: int8
    implicit none
    character(*), intent(in) :: input
    integer(int8), allocatable, intent(out) :: bytes(:)
    integer, intent(out) :: nbytes

    character(64), parameter :: b64 = &
         'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    integer :: i, j, in_len, ngroups, pad
    integer :: v0, v1, v2, v3

    in_len = len_trim(input)
    if(in_len .eq. 0)then
       allocate(bytes(0))
       nbytes = 0
       return
    end if

    ! Count padding
    pad = 0
    if(input(in_len:in_len) .eq. '=') pad = pad + 1
    if(in_len .ge. 2 .and. input(in_len-1:in_len-1) .eq. '=') pad = pad + 1

    ngroups = in_len / 4
    nbytes = ngroups * 3 - pad
    allocate(bytes(nbytes))

    j = 1
    do i = 1, in_len, 4
       v0 = index(b64, input(i:i)) - 1
       v1 = index(b64, input(i+1:i+1)) - 1
       if(input(i+2:i+2) .ne. '=')then
          v2 = index(b64, input(i+2:i+2)) - 1
       else
          v2 = 0
       end if
       if(input(i+3:i+3) .ne. '=')then
          v3 = index(b64, input(i+3:i+3)) - 1
       else
          v3 = 0
       end if

       if(j .le. nbytes) &
            bytes(j) = int(ior(ishft(v0, 2), ishft(v1, -4)), int8)
       if(j + 1 .le. nbytes) &
            bytes(j+1) = int(ior(ishft(iand(v1, 15), 4), ishft(v2, -2)), int8)
       if(j + 2 .le. nbytes) &
            bytes(j+2) = int(ior(ishft(iand(v2, 3), 6), v3), int8)
       j = j + 3
    end do

  end subroutine base64_decode_bytes
!###############################################################################


!###############################################################################
  subroutine decode_base64_to_float32(input, values, n)
    !! Decode base64 string to float32 array
    use iso_fortran_env, only: int8, int32
    implicit none
    character(*), intent(in) :: input
    real(real32), allocatable, intent(out) :: values(:)
    integer, intent(out) :: n

    integer(int8), allocatable :: bytes(:)
    integer :: nbytes, i
    integer(int32) :: ival

    call base64_decode_bytes(input, bytes, nbytes)
    n = nbytes / 4
    allocate(values(n))

    do i = 1, n
       ival = ior(ior(ior( &
            iand(int(bytes((i-1)*4 + 1), int32), 255), &
            ishft(iand(int(bytes((i-1)*4 + 2), int32), 255), 8)), &
       ishft(iand(int(bytes((i-1)*4 + 3), int32), 255), 16)), &
  ishft(iand(int(bytes((i-1)*4 + 4), int32), 255), 24))
       values(i) = transfer(ival, values(i))
    end do

    deallocate(bytes)

  end subroutine decode_base64_to_float32
!###############################################################################


!###############################################################################
  subroutine decode_base64_to_int64(input, values, n)
    !! Decode base64 string to integer array (from 8-byte int64 encoding)
    use iso_fortran_env, only: int8, int64
    implicit none
    character(*), intent(in) :: input
    integer, allocatable, intent(out) :: values(:)
    integer, intent(out) :: n

    integer(int8), allocatable :: bytes(:)
    integer :: nbytes, i, j
    integer(int64) :: ival64

    call base64_decode_bytes(input, bytes, nbytes)
    n = nbytes / 8
    allocate(values(n))

    do i = 1, n
       ival64 = 0
       do j = 0, 7
          ival64 = ior(ival64, &
               ishft(iand(int(bytes((i-1)*8 + j + 1), int64), &
                    int(255, int64)), j*8))
       end do
       values(i) = int(ival64)
    end do

    deallocate(bytes)

  end subroutine decode_base64_to_int64
!###############################################################################


! =============================================================================
! Reverse data layout utility
! =============================================================================

!###############################################################################
  subroutine row_to_col_major_2d(data_in, data_out, m, n)
    !! Convert flat row-major [m,n] to flat column-major [m,n]
    !! Inverse of col_to_row_major_2d.
    implicit none
    integer, intent(in) :: m, n
    real(real32), intent(in) :: data_in(m * n)
    real(real32), intent(out) :: data_out(m * n)
    integer :: i, j
    do i = 1, m
       do j = 1, n
          data_out((j-1)*m + i) = data_in((i-1)*n + j)
       end do
    end do
  end subroutine row_to_col_major_2d
!###############################################################################


!###############################################################################
  subroutine parse_space_separated_ints(str, values)
    !! Parse space-separated integers from a string into an allocatable array
    implicit none
    character(*), intent(in) :: str
    integer, allocatable, intent(out) :: values(:)

    integer :: i, stat, ival
    character(256) :: work
    character(32) :: token

    work = trim(adjustl(str))
    allocate(values(0))

    do while(len_trim(work) .gt. 0)
       i = index(trim(work), ' ')
       if(i .eq. 0)then
          token = trim(work)
          work = ''
       else
          token = work(1:i-1)
          work = adjustl(work(i+1:))
       end if
       read(token, *, iostat=stat) ival
       if(stat .eq. 0) values = [values, ival]
    end do

  end subroutine parse_space_separated_ints
!###############################################################################


!###############################################################################
  function onnx_to_athena_activation(optype) result(name)
    !! Convert an ONNX activation op_type string to the Athena activation name
    implicit none
    character(*), intent(in) :: optype
    character(64) :: name

    select case(trim(optype))
    case('LeakyRelu')
       name = 'leaky_relu'
    case('Relu')
       name = 'relu'
    case('Sigmoid')
       name = 'sigmoid'
    case('Softmax')
       name = 'softmax'
    case('Tanh')
       name = 'tanh'
    case('Selu')
       name = 'selu'
    case('Swish')
       name = 'swish'
    case default
       name = to_lower(trim(optype))
    end select

  end function onnx_to_athena_activation
!###############################################################################


end module athena__onnx_utils
