module athena__onnx_binary_c
  !! Module providing ISO C bindings for the ONNX binary protobuf reader/writer.
  !! These interface directly with the C implementation in c/onnx_binary.c
  use, intrinsic :: iso_c_binding
  implicit none

  private

  ! Reading API
  public :: c_onnx_binary_read
  public :: c_onnx_binary_ir_version
  public :: c_onnx_binary_producer_name
  public :: c_onnx_binary_producer_version
  public :: c_onnx_binary_domain
  public :: c_onnx_binary_graph_name
  public :: c_onnx_binary_num_nodes
  public :: c_onnx_binary_num_initializers
  public :: c_onnx_binary_num_inputs
  public :: c_onnx_binary_num_outputs
  public :: c_onnx_binary_num_value_infos
  public :: c_onnx_binary_node_name
  public :: c_onnx_binary_node_op_type
  public :: c_onnx_binary_node_num_inputs
  public :: c_onnx_binary_node_input
  public :: c_onnx_binary_node_num_outputs
  public :: c_onnx_binary_node_output
  public :: c_onnx_binary_node_num_attrs
  public :: c_onnx_binary_attr_name
  public :: c_onnx_binary_attr_type_str
  public :: c_onnx_binary_attr_value_str
  public :: c_onnx_binary_init_name
  public :: c_onnx_binary_init_num_dims
  public :: c_onnx_binary_init_dim
  public :: c_onnx_binary_init_num_floats
  public :: c_onnx_binary_init_float_data
  public :: c_onnx_binary_input_name
  public :: c_onnx_binary_input_elem_type
  public :: c_onnx_binary_input_num_dims
  public :: c_onnx_binary_input_dim
  public :: c_onnx_binary_output_name
  public :: c_onnx_binary_output_elem_type
  public :: c_onnx_binary_output_num_dims
  public :: c_onnx_binary_output_dim
  public :: c_onnx_binary_vi_name
  public :: c_onnx_binary_vi_elem_type
  public :: c_onnx_binary_vi_num_dims
  public :: c_onnx_binary_vi_dim

  ! Writing API
  public :: c_onnx_binary_create
  public :: c_onnx_binary_set_ir_version
  public :: c_onnx_binary_set_producer
  public :: c_onnx_binary_set_domain
  public :: c_onnx_binary_set_graph_name
  public :: c_onnx_binary_add_opset
  public :: c_onnx_binary_add_node
  public :: c_onnx_binary_node_add_input_w
  public :: c_onnx_binary_node_add_output_w
  public :: c_onnx_binary_node_add_attr_ints
  public :: c_onnx_binary_node_add_attr_floats
  public :: c_onnx_binary_node_add_attr_string
  public :: c_onnx_binary_add_initializer
  public :: c_onnx_binary_add_input_w
  public :: c_onnx_binary_add_output_w
  public :: c_onnx_binary_add_value_info
  public :: c_onnx_binary_write
  public :: c_onnx_binary_free


  interface

     ! ---- Reading API ----

     integer(c_int) function c_onnx_binary_read(filename) &
          bind(C, name="onnx_binary_read")
     import :: c_char, c_int
     character(kind=c_char), intent(in) :: filename(*)
     end function

     integer(c_int64_t) function c_onnx_binary_ir_version(h) &
          bind(C, name="onnx_binary_ir_version")
     import :: c_int, c_int64_t
     integer(c_int), value :: h
     end function

     subroutine c_onnx_binary_producer_name(h, buf, buflen) &
          bind(C, name="onnx_binary_producer_name")
       import :: c_int, c_char
       integer(c_int), value :: h
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     subroutine c_onnx_binary_producer_version(h, buf, buflen) &
          bind(C, name="onnx_binary_producer_version")
       import :: c_int, c_char
       integer(c_int), value :: h
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     subroutine c_onnx_binary_domain(h, buf, buflen) &
          bind(C, name="onnx_binary_domain")
       import :: c_int, c_char
       integer(c_int), value :: h
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     subroutine c_onnx_binary_graph_name(h, buf, buflen) &
          bind(C, name="onnx_binary_graph_name")
       import :: c_int, c_char
       integer(c_int), value :: h
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     integer(c_int) function c_onnx_binary_num_nodes(h) &
          bind(C, name="onnx_binary_num_nodes")
     import :: c_int
     integer(c_int), value :: h
     end function

     integer(c_int) function c_onnx_binary_num_initializers(h) &
          bind(C, name="onnx_binary_num_initializers")
     import :: c_int
     integer(c_int), value :: h
     end function

     integer(c_int) function c_onnx_binary_num_inputs(h) &
          bind(C, name="onnx_binary_num_inputs")
     import :: c_int
     integer(c_int), value :: h
     end function

     integer(c_int) function c_onnx_binary_num_outputs(h) &
          bind(C, name="onnx_binary_num_outputs")
     import :: c_int
     integer(c_int), value :: h
     end function

     integer(c_int) function c_onnx_binary_num_value_infos(h) &
          bind(C, name="onnx_binary_num_value_infos")
     import :: c_int
     integer(c_int), value :: h
     end function

     ! Node queries
     subroutine c_onnx_binary_node_name(h, idx, buf, buflen) &
          bind(C, name="onnx_binary_node_name")
       import :: c_int, c_char
       integer(c_int), value :: h, idx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     subroutine c_onnx_binary_node_op_type(h, idx, buf, buflen) &
          bind(C, name="onnx_binary_node_op_type")
       import :: c_int, c_char
       integer(c_int), value :: h, idx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     integer(c_int) function c_onnx_binary_node_num_inputs(h, idx) &
          bind(C, name="onnx_binary_node_num_inputs")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     subroutine c_onnx_binary_node_input(h, nidx, iidx, buf, buflen) &
          bind(C, name="onnx_binary_node_input")
       import :: c_int, c_char
       integer(c_int), value :: h, nidx, iidx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     integer(c_int) function c_onnx_binary_node_num_outputs(h, idx) &
          bind(C, name="onnx_binary_node_num_outputs")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     subroutine c_onnx_binary_node_output(h, nidx, oidx, buf, buflen) &
          bind(C, name="onnx_binary_node_output")
       import :: c_int, c_char
       integer(c_int), value :: h, nidx, oidx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     integer(c_int) function c_onnx_binary_node_num_attrs(h, idx) &
          bind(C, name="onnx_binary_node_num_attrs")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     ! Attribute queries
     subroutine c_onnx_binary_attr_name(h, nidx, aidx, buf, buflen) &
          bind(C, name="onnx_binary_attr_name")
       import :: c_int, c_char
       integer(c_int), value :: h, nidx, aidx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     subroutine c_onnx_binary_attr_type_str(h, nidx, aidx, buf, buflen) &
          bind(C, name="onnx_binary_attr_type_str")
       import :: c_int, c_char
       integer(c_int), value :: h, nidx, aidx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     subroutine c_onnx_binary_attr_value_str(h, nidx, aidx, buf, buflen) &
          bind(C, name="onnx_binary_attr_value_str")
       import :: c_int, c_char
       integer(c_int), value :: h, nidx, aidx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     ! Initializer queries
     subroutine c_onnx_binary_init_name(h, idx, buf, buflen) &
          bind(C, name="onnx_binary_init_name")
       import :: c_int, c_char
       integer(c_int), value :: h, idx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     integer(c_int) function c_onnx_binary_init_num_dims(h, idx) &
          bind(C, name="onnx_binary_init_num_dims")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     integer(c_int64_t) function c_onnx_binary_init_dim(h, idx, didx) &
          bind(C, name="onnx_binary_init_dim")
     import :: c_int, c_int64_t
     integer(c_int), value :: h, idx, didx
     end function

     integer(c_int) function c_onnx_binary_init_num_floats(h, idx) &
          bind(C, name="onnx_binary_init_num_floats")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     subroutine c_onnx_binary_init_float_data(h, idx, buf, buflen) &
          bind(C, name="onnx_binary_init_float_data")
       import :: c_int, c_float
       integer(c_int), value :: h, idx
       real(c_float), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     ! Input queries
     subroutine c_onnx_binary_input_name(h, idx, buf, buflen) &
          bind(C, name="onnx_binary_input_name")
       import :: c_int, c_char
       integer(c_int), value :: h, idx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     integer(c_int) function c_onnx_binary_input_elem_type(h, idx) &
          bind(C, name="onnx_binary_input_elem_type")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     integer(c_int) function c_onnx_binary_input_num_dims(h, idx) &
          bind(C, name="onnx_binary_input_num_dims")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     integer(c_int64_t) function c_onnx_binary_input_dim(h, idx, didx) &
          bind(C, name="onnx_binary_input_dim")
     import :: c_int, c_int64_t
     integer(c_int), value :: h, idx, didx
     end function

     ! Output queries
     subroutine c_onnx_binary_output_name(h, idx, buf, buflen) &
          bind(C, name="onnx_binary_output_name")
       import :: c_int, c_char
       integer(c_int), value :: h, idx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     integer(c_int) function c_onnx_binary_output_elem_type(h, idx) &
          bind(C, name="onnx_binary_output_elem_type")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     integer(c_int) function c_onnx_binary_output_num_dims(h, idx) &
          bind(C, name="onnx_binary_output_num_dims")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     integer(c_int64_t) function c_onnx_binary_output_dim(h, idx, didx) &
          bind(C, name="onnx_binary_output_dim")
     import :: c_int, c_int64_t
     integer(c_int), value :: h, idx, didx
     end function

     ! Value-info queries
     subroutine c_onnx_binary_vi_name(h, idx, buf, buflen) &
          bind(C, name="onnx_binary_vi_name")
       import :: c_int, c_char
       integer(c_int), value :: h, idx
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value :: buflen
     end subroutine

     integer(c_int) function c_onnx_binary_vi_elem_type(h, idx) &
          bind(C, name="onnx_binary_vi_elem_type")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     integer(c_int) function c_onnx_binary_vi_num_dims(h, idx) &
          bind(C, name="onnx_binary_vi_num_dims")
     import :: c_int
     integer(c_int), value :: h, idx
     end function

     integer(c_int64_t) function c_onnx_binary_vi_dim(h, idx, didx) &
          bind(C, name="onnx_binary_vi_dim")
     import :: c_int, c_int64_t
     integer(c_int), value :: h, idx, didx
     end function


     ! ---- Writing API ----

     integer(c_int) function c_onnx_binary_create() &
          bind(C, name="onnx_binary_create")
     import :: c_int
     end function

     subroutine c_onnx_binary_set_ir_version(h, v) &
          bind(C, name="onnx_binary_set_ir_version")
       import :: c_int, c_int64_t
       integer(c_int), value :: h
       integer(c_int64_t), value :: v
     end subroutine

     subroutine c_onnx_binary_set_producer(h, name, version) &
          bind(C, name="onnx_binary_set_producer")
       import :: c_int, c_char
       integer(c_int), value :: h
       character(kind=c_char), intent(in) :: name(*)
       character(kind=c_char), intent(in) :: version(*)
     end subroutine

     subroutine c_onnx_binary_set_domain(h, domain) &
          bind(C, name="onnx_binary_set_domain")
       import :: c_int, c_char
       integer(c_int), value :: h
       character(kind=c_char), intent(in) :: domain(*)
     end subroutine

     subroutine c_onnx_binary_set_graph_name(h, name) &
          bind(C, name="onnx_binary_set_graph_name")
       import :: c_int, c_char
       integer(c_int), value :: h
       character(kind=c_char), intent(in) :: name(*)
     end subroutine

     subroutine c_onnx_binary_add_opset(h, domain, version) &
          bind(C, name="onnx_binary_add_opset")
       import :: c_int, c_int64_t, c_char
       integer(c_int), value :: h
       character(kind=c_char), intent(in) :: domain(*)
       integer(c_int64_t), value :: version
     end subroutine

     integer(c_int) function c_onnx_binary_add_node(h, name, op_type) &
          bind(C, name="onnx_binary_add_node")
     import :: c_int, c_char
     integer(c_int), value :: h
     character(kind=c_char), intent(in) :: name(*)
     character(kind=c_char), intent(in) :: op_type(*)
     end function

     subroutine c_onnx_binary_node_add_input_w(h, nidx, name) &
          bind(C, name="onnx_binary_node_add_input_w")
       import :: c_int, c_char
       integer(c_int), value :: h, nidx
       character(kind=c_char), intent(in) :: name(*)
     end subroutine

     subroutine c_onnx_binary_node_add_output_w(h, nidx, name) &
          bind(C, name="onnx_binary_node_add_output_w")
       import :: c_int, c_char
       integer(c_int), value :: h, nidx
       character(kind=c_char), intent(in) :: name(*)
     end subroutine

     subroutine c_onnx_binary_node_add_attr_ints(h, nidx, name, v, n) &
          bind(C, name="onnx_binary_node_add_attr_ints")
       import :: c_int, c_int64_t, c_char
       integer(c_int), value :: h, nidx
       character(kind=c_char), intent(in) :: name(*)
       integer(c_int64_t), intent(in) :: v(*)
       integer(c_int), value :: n
     end subroutine

     subroutine c_onnx_binary_node_add_attr_floats(h, nidx, name, v, n) &
          bind(C, name="onnx_binary_node_add_attr_floats")
       import :: c_int, c_float, c_char
       integer(c_int), value :: h, nidx
       character(kind=c_char), intent(in) :: name(*)
       real(c_float), intent(in) :: v(*)
       integer(c_int), value :: n
     end subroutine

     subroutine c_onnx_binary_node_add_attr_string(h, nidx, name, val) &
          bind(C, name="onnx_binary_node_add_attr_string")
       import :: c_int, c_char
       integer(c_int), value :: h, nidx
       character(kind=c_char), intent(in) :: name(*)
       character(kind=c_char), intent(in) :: val(*)
     end subroutine

     integer(c_int) function c_onnx_binary_add_initializer( &
          h, name, dims, ndims, data, nfloats) &
     bind(C, name="onnx_binary_add_initializer")
     import :: c_int, c_int64_t, c_float, c_char
     integer(c_int), value :: h
     character(kind=c_char), intent(in) :: name(*)
     integer(c_int64_t), intent(in) :: dims(*)
     integer(c_int), value :: ndims
     real(c_float), intent(in) :: data(*)
     integer(c_int), value :: nfloats
     end function

     integer(c_int) function c_onnx_binary_add_input_w( &
          h, name, elem_type, dims, ndims) &
     bind(C, name="onnx_binary_add_input_w")
     import :: c_int, c_int64_t, c_char
     integer(c_int), value :: h
     character(kind=c_char), intent(in) :: name(*)
     integer(c_int), value :: elem_type
     integer(c_int64_t), intent(in) :: dims(*)
     integer(c_int), value :: ndims
     end function

     integer(c_int) function c_onnx_binary_add_output_w( &
          h, name, elem_type, dims, ndims) &
     bind(C, name="onnx_binary_add_output_w")
     import :: c_int, c_int64_t, c_char
     integer(c_int), value :: h
     character(kind=c_char), intent(in) :: name(*)
     integer(c_int), value :: elem_type
     integer(c_int64_t), intent(in) :: dims(*)
     integer(c_int), value :: ndims
     end function

     integer(c_int) function c_onnx_binary_add_value_info( &
          h, name, elem_type, dims, ndims) &
     bind(C, name="onnx_binary_add_value_info")
     import :: c_int, c_int64_t, c_char
     integer(c_int), value :: h
     character(kind=c_char), intent(in) :: name(*)
     integer(c_int), value :: elem_type
     integer(c_int64_t), intent(in) :: dims(*)
     integer(c_int), value :: ndims
     end function

     integer(c_int) function c_onnx_binary_write(h, filename) &
          bind(C, name="onnx_binary_write")
     import :: c_int, c_char
     integer(c_int), value :: h
     character(kind=c_char), intent(in) :: filename(*)
     end function

     subroutine c_onnx_binary_free(h) &
          bind(C, name="onnx_binary_free")
       import :: c_int
       integer(c_int), value :: h
     end subroutine

  end interface


contains


  !> Convert a Fortran string to a C null-terminated string
  pure function f_to_c_string(fstr) result(cstr)
    character(len=*), intent(in) :: fstr
    character(len=:, kind=c_char), allocatable :: cstr
    cstr = trim(fstr) // c_null_char
  end function f_to_c_string


  !> Convert a C null-terminated buffer to a Fortran string
  function c_to_f_string(cbuf, buflen) result(fstr)
    character(kind=c_char), intent(in) :: cbuf(*)
    integer, intent(in) :: buflen
    character(len=:), allocatable :: fstr
    integer :: i, slen

    slen = 0
    do i = 1, buflen
       if (cbuf(i) == c_null_char) exit
       slen = slen + 1
    end do
    allocate(character(len=slen) :: fstr)
    do i = 1, slen
       fstr(i:i) = cbuf(i)
    end do
  end function c_to_f_string


end module athena__onnx_binary_c
