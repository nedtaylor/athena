module athena__onnx_nop_utils
  !! Shared utility routines for NOP ONNX export/import.
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: onnx_attribute_type, onnx_node_type, &
       onnx_initialiser_type
  use athena__onnx_utils, only: emit_node, col_to_row_major_2d, &
       row_to_col_major_2d
  use diffstruc, only: array_type
  implicit none

  private

  public :: emit_nop_input_transpose
  public :: emit_nop_output_tail
  public :: emit_float_initialiser
  public :: emit_matrix_initialiser
  public :: emit_nop_metadata
  public :: parse_nop_metadata
  public :: extract_nop_prefix
  public :: load_nop_param_from_inits
  public :: find_initialiser_by_name
  public :: infer_dynamic_lno_poles
  public :: find_onnx_expanded_node_by_suffix
  public :: find_node_initialiser_index
  public :: detect_onnx_expanded_nop_activation
  public :: load_onnx_expanded_matrix_param

contains

!###############################################################################
  subroutine emit_nop_input_transpose(prefix, input_name, nodes, num_nodes, &
       output_name)
    !! Emit the common NOP input transpose.
    implicit none

    character(*), intent(in) :: prefix, input_name
    type(onnx_node_type), intent(inout) :: nodes(:)
    integer, intent(inout) :: num_nodes
    character(*), intent(in) :: output_name

    character(4096) :: perm_attr

    perm_attr = '        "attribute": [{"name": "perm", "ints": ' // &
         '["1", "0"], "type": "INTS"}]'

    call emit_node('Transpose', '/' // trim(prefix) // '/Transpose', &
         trim(output_name), trim(perm_attr), nodes, num_nodes, &
         in1=trim(input_name))

  end subroutine emit_nop_input_transpose
!###############################################################################


!###############################################################################
  subroutine emit_nop_output_tail(prefix, activation_name, is_last_layer, &
       input_name, nodes, num_nodes, final_output)
    !! Emit the common transpose and optional activation at the end of a NOP.
    implicit none

    character(*), intent(in) :: prefix, activation_name, input_name
    logical, intent(in) :: is_last_layer
    type(onnx_node_type), intent(inout) :: nodes(:)
    integer, intent(inout) :: num_nodes
    character(128), intent(out) :: final_output

    character(4096) :: perm_attr
    character(128) :: transpose_output

    perm_attr = '        "attribute": [{"name": "perm", "ints": ' // &
         '["1", "0"], "type": "INTS"}]'

    if(is_last_layer .and. trim(activation_name) .eq. 'none')then
       transpose_output = 'output'
    else
       write(transpose_output, '("/",A,"/Transpose_1_output_0")') &
            trim(prefix)
    end if

    call emit_node('Transpose', '/' // trim(prefix) // '/Transpose_1', &
         trim(transpose_output), trim(perm_attr), nodes, num_nodes, &
         in1=trim(input_name))

    if(trim(activation_name) .ne. 'none')then
       if(is_last_layer)then
          final_output = 'output'
       else
          write(final_output, '("/",A,"/Relu_output_0")') trim(prefix)
       end if
       call emit_node('Relu', '/' // trim(prefix) // '/Relu', &
            trim(final_output), '', nodes, num_nodes, &
            in1=trim(transpose_output))
    else
       final_output = transpose_output
    end if

  end subroutine emit_nop_output_tail
!###############################################################################


!###############################################################################
  subroutine emit_float_initialiser(name, data, dims, inits, num_inits)
    !! Emit a float32 initialiser with explicit dimensions.
    implicit none

    character(*), intent(in) :: name
    real(real32), intent(in) :: data(:)
    integer, intent(in) :: dims(:)
    type(onnx_initialiser_type), intent(inout) :: inits(:)
    integer, intent(inout) :: num_inits

    num_inits = num_inits + 1
    inits(num_inits)%name = trim(name)
    inits(num_inits)%data_type = 1
    allocate(inits(num_inits)%dims(size(dims)))
    inits(num_inits)%dims = dims
    allocate(inits(num_inits)%data(size(data)))
    inits(num_inits)%data = data

  end subroutine emit_float_initialiser
!###############################################################################


!###############################################################################
  subroutine emit_matrix_initialiser(name, data_col_major, rows, cols, inits, &
       num_inits)
    !! Emit a 2D float32 initialiser after converting to row-major order.
    implicit none

    character(*), intent(in) :: name
    real(real32), intent(in) :: data_col_major(:)
    integer, intent(in) :: rows, cols
    type(onnx_initialiser_type), intent(inout) :: inits(:)
    integer, intent(inout) :: num_inits

    real(real32), allocatable :: row_major(:)

    allocate(row_major(size(data_col_major)))
    call col_to_row_major_2d(data_col_major, row_major, rows, cols)
    call emit_float_initialiser(name, row_major, [rows, cols], inits, &
         num_inits)
    deallocate(row_major)

  end subroutine emit_matrix_initialiser
!###############################################################################


!###############################################################################
  subroutine emit_nop_metadata(layer, prefix, metadata, num_meta)
    !! Build the metadata entry required to reconstruct a NOP layer.
    implicit none

    class(base_layer_type), intent(in) :: layer
    character(*), intent(in) :: prefix
    character(4096), intent(inout) :: metadata(:)
    integer, intent(inout) :: num_meta

    type(onnx_attribute_type), allocatable :: attrs(:)
    integer :: i
    character(2048) :: value_str

    attrs = layer%get_attributes()
    if(.not.allocated(attrs)) return
    if(size(attrs) .eq. 0) return

    value_str = 'subtype=' // trim(adjustl(layer%name))
    do i = 1, size(attrs)
       value_str = trim(value_str) // ';' // trim(attrs(i)%name) // '=' // &
            trim(adjustl(attrs(i)%val))
    end do

    num_meta = num_meta + 1
    write(metadata(num_meta), '(A)') &
         '      {"key": "athena_nop_' // trim(prefix) // &
         '", "value": "' // trim(value_str) // '"}'

  end subroutine emit_nop_metadata
!###############################################################################


!###############################################################################
  subroutine parse_nop_metadata(meta_value, &
       num_inputs, num_outputs, num_modes, use_bias, activation_name)
    !! Parse common NOP hyperparameters from metadata value string.
    implicit none

    character(*), intent(in) :: meta_value
    integer, intent(inout) :: num_inputs, num_outputs, num_modes
    logical, intent(inout) :: use_bias
    character(64), intent(inout) :: activation_name

    integer :: k, pos, pos2, stat
    character(256) :: token, key, val
    logical :: logical_val

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
       val = trim(adjustl(token(k+1:)))
       select case(trim(key))
       case('num_inputs')
          read(val, *) num_inputs
       case('num_outputs')
          read(val, *) num_outputs
       case('num_modes', 'num_basis')
          read(val, *) num_modes
       case('use_bias')
          read(val, *, iostat=stat) logical_val
          if(stat .eq. 0)then
             use_bias = logical_val
          else
             select case(trim(adjustl(val)))
             case('1', 'T', 't', 'true', 'TRUE', 'True')
                use_bias = .true.
             case('0', 'F', 'f', 'false', 'FALSE', 'False')
                use_bias = .false.
             case default
                call stop_program('parse_nop_metadata: invalid use_bias value')
             end select
          end if
       case('activation')
          activation_name = trim(val)
       end select
    end do

  end subroutine parse_nop_metadata
!###############################################################################


!###############################################################################
  function extract_nop_prefix(meta_key) result(prefix)
    !! Extract the node prefix from an athena_nop_node_X metadata key.
    implicit none

    character(*), intent(in) :: meta_key
    character(64) :: prefix

    integer :: pos

    prefix = trim(meta_key)
    pos = index(prefix, 'athena_nop_')
    if(pos .gt. 0) prefix = prefix(pos+11:)

  end function extract_nop_prefix
!###############################################################################


!###############################################################################
  subroutine load_nop_param_from_inits( &
       param, prefix, suffix, inits, num_inits, dims)
    !! Load a parameter from ONNX initialisers into a diffstruc array.
    implicit none

    type(array_type), intent(inout) :: param
    character(*), intent(in) :: prefix, suffix
    type(onnx_initialiser_type), intent(in) :: inits(:)
    integer, intent(in) :: num_inits
    integer, intent(in) :: dims(2)

    integer :: k
    character(128) :: target_name
    real(real32), allocatable :: col_data(:)

    write(target_name, '(A,A)') trim(prefix), suffix

    do k = 1, num_inits
       if(trim(inits(k)%name) .ne. trim(target_name)) cycle
       if(.not.allocated(inits(k)%data)) cycle

       if(dims(2) .gt. 1)then
          allocate(col_data(size(inits(k)%data)))
          call row_to_col_major_2d(inits(k)%data, col_data, dims(1), dims(2))
          param%val(:,1) = col_data
          deallocate(col_data)
       else
          param%val(:,1) = inits(k)%data
       end if
       return
    end do

  end subroutine load_nop_param_from_inits
!###############################################################################


!###############################################################################
  integer function find_initialiser_by_name(name, inits, num_inits)
    !! Return the index of a named initialiser, or zero when not found.
    implicit none

    character(*), intent(in) :: name
    type(onnx_initialiser_type), intent(in) :: inits(:)
    integer, intent(in) :: num_inits

    integer :: i

    find_initialiser_by_name = 0
    do i = 1, num_inits
       if(trim(inits(i)%name) .eq. trim(name))then
          find_initialiser_by_name = i
          return
       end if
    end do

  end function find_initialiser_by_name
!###############################################################################


!###############################################################################
  subroutine infer_dynamic_lno_poles(e_args_init, d_args_init, num_inputs, &
       num_outputs, poles)
    !! Reconstruct dynamic LNO poles from exported encoder/decoder arguments.
    implicit none

    type(onnx_initialiser_type), intent(in) :: e_args_init, d_args_init
    integer, intent(in) :: num_inputs, num_outputs
    real(real32), intent(out) :: poles(:)

    integer :: k, idx, num_modes
    real(real32) :: pi_value

    num_modes = size(poles)

    if(num_inputs .gt. 1 .and. allocated(e_args_init%data))then
       do k = 1, num_modes
          idx = (k - 1) * num_inputs + num_inputs
          poles(k) = -e_args_init%data(idx)
       end do
       return
    end if

    if(num_outputs .gt. 1 .and. allocated(d_args_init%data))then
       do k = 1, num_modes
          idx = (num_outputs - 1) * num_modes + k
          poles(k) = -d_args_init%data(idx)
       end do
       return
    end if

    pi_value = acos(-1.0_real32)
    do k = 1, num_modes
       poles(k) = real(k, real32) * pi_value
    end do

  end subroutine infer_dynamic_lno_poles
!###############################################################################

!###############################################################################
  integer function find_onnx_expanded_node_by_suffix( &
       nodes, num_nodes, prefix, suffix)
    !! Return the node index matching one /layerN/suffix name, or zero.
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid node entries
    character(*), intent(in) :: prefix, suffix
    !! Layer prefix and trailing node name token

    ! Local variables
    integer :: i
    !! Loop index
    character(128) :: target_name
    !! Full node name to match

    write(target_name, '("/",A,"/",A)') trim(prefix), trim(suffix)
    find_onnx_expanded_node_by_suffix = 0

    do i = 1, num_nodes
       if(trim(nodes(i)%name) .eq. trim(target_name))then
          find_onnx_expanded_node_by_suffix = i
          return
       end if
    end do

  end function find_onnx_expanded_node_by_suffix
!###############################################################################


!###############################################################################
  integer function find_node_initialiser_index(node, inits, num_inits)
    !! Return the first initialiser referenced by a node's inputs.
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! Parsed ONNX node whose inputs may reference an initialiser
    type(onnx_initialiser_type), intent(in) :: inits(:)
    !! Parsed ONNX initialisers
    integer, intent(in) :: num_inits
    !! Number of valid initialiser entries

    ! Local variables
    integer :: i, init_idx
    !! Loop index and candidate initialiser index

    find_node_initialiser_index = 0
    if(.not.allocated(node%inputs)) return

    do i = 1, size(node%inputs)
       init_idx = find_initialiser_by_name(node%inputs(i), inits, num_inits)
       if(init_idx .gt. 0)then
          find_node_initialiser_index = init_idx
          return
       end if
    end do

  end function find_node_initialiser_index
!###############################################################################


!###############################################################################
  function detect_onnx_expanded_nop_activation(prefix, nodes, num_nodes) &
       result(name)
    !! Reconstruct the activation name from the tail of an expanded-ONNX NOP
    !! cluster.
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Layer node prefix without leading slash
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid node entries
    character(64) :: name
    !! Reconstructed ATHENA activation name

    if(find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, 'Relu') &
         .gt. 0) &
    then
       name = 'relu'
    else
       name = 'none'
    end if

  end function detect_onnx_expanded_nop_activation
!###############################################################################


!###############################################################################
  subroutine load_onnx_expanded_matrix_param(param, init, rows, cols)
    !! Copy a row-major ONNX matrix initialiser into a diffstruc parameter.
    implicit none

    ! Arguments
    type(array_type), intent(inout) :: param
    !! Destination diffstruc parameter tensor
    type(onnx_initialiser_type), intent(in) :: init
    !! Row-major ONNX initialiser data
    integer, intent(in) :: rows, cols
    !! Matrix shape

    ! Local variables
    real(real32), allocatable :: col_major(:)
    !! Temporary column-major buffer for ATHENA internal storage

    allocate(col_major(rows * cols))
    call row_to_col_major_2d(init%data, col_major, rows, cols)
    param%val(:,1) = col_major
    deallocate(col_major)

  end subroutine load_onnx_expanded_matrix_param
!###############################################################################

end module athena__onnx_nop_utils
