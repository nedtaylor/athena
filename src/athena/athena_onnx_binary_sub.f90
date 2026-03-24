submodule(athena__onnx_binary) athena__onnx_binary_submodule
  !! Submodule containing implementations for binary ONNX I/O
  use athena__onnx_binary_c
  use athena__base_layer, only: base_layer_type, learnable_layer_type
  use athena__misc_types, only: &
       onnx_attribute_type, onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use coreutils, only: real32, to_lower, to_upper, to_camel_case, icount
  use, intrinsic :: iso_c_binding
  implicit none

  integer, parameter :: CBUF_LEN = 256
  !! C buffer length for strings

contains


!###############################################################################
!  Helper: Fortran string -> C null-terminated string
!###############################################################################
  function fc(fstr) result(cstr)
    character(len=*), intent(in) :: fstr
    character(len=:, kind=c_char), allocatable :: cstr
    cstr = trim(fstr) // c_null_char
  end function fc


!###############################################################################
!  Helper: Read C string buffer into Fortran string
!###############################################################################
  function read_c_buf(cbuf) result(fstr)
    character(kind=c_char), intent(in) :: cbuf(CBUF_LEN)
    character(len=:), allocatable :: fstr
    integer :: i, slen

    slen = 0
    do i = 1, CBUF_LEN
       if (cbuf(i) == c_null_char) exit
       slen = slen + 1
    end do
    allocate(character(len=slen) :: fstr)
    do i = 1, slen
       fstr(i:i) = cbuf(i)
    end do
  end function read_c_buf


!###############################################################################
!  READ BINARY ONNX
!###############################################################################
  module function read_onnx_binary(file, verbose) result(network)
    implicit none

    ! Arguments
    character(*), intent(in) :: file
    integer, optional, intent(in) :: verbose
    type(network_type) :: network

    ! Local variables
    integer(c_int) :: h, i, j, k, nd
    integer :: verbose_
    integer :: num_nodes, num_inits, num_inputs, num_outputs, num_vis
    character(kind=c_char) :: cbuf(CBUF_LEN)
    character(kind=c_char) :: vbuf(4096)

    type(onnx_node_type), allocatable, dimension(:) :: nodes
    type(onnx_initialiser_type), allocatable, dimension(:) :: initialisers
    type(onnx_tensor_type), allocatable, dimension(:) :: input_tensors
    type(onnx_tensor_type), allocatable, dimension(:) :: output_tensors
    type(onnx_tensor_type), allocatable, dimension(:) :: value_infos

    verbose_ = 0
    if (present(verbose)) verbose_ = verbose

    ! Read binary file via C
    h = c_onnx_binary_read(fc(file))
    if (h < 0) then
       write(*,*) "ERROR: Could not open binary ONNX file: ", trim(file)
       return
    end if

    ! Get counts
    num_nodes   = c_onnx_binary_num_nodes(h)
    num_inits   = c_onnx_binary_num_initializers(h)
    num_inputs  = c_onnx_binary_num_inputs(h)
    num_outputs = c_onnx_binary_num_outputs(h)
    num_vis     = c_onnx_binary_num_value_infos(h)

    if (verbose_ .gt. 0) then
       write(*,'(A,I0,A)') " Binary ONNX: ", num_nodes, " nodes"
       write(*,'(A,I0,A)') " Binary ONNX: ", num_inits, " initialisers"
       write(*,'(A,I0,A)') " Binary ONNX: ", num_inputs, " inputs"
       write(*,'(A,I0,A)') " Binary ONNX: ", num_outputs, " outputs"
       write(*,'(A,I0,A)') " Binary ONNX: ", num_vis, " value_infos"
    end if

    ! Allocate Fortran structures
    allocate(nodes(num_nodes))
    allocate(initialisers(num_inits))
    allocate(input_tensors(num_inputs))
    allocate(output_tensors(num_outputs))
    allocate(value_infos(num_vis))

    ! --- Nodes ---
    do i = 0, num_nodes - 1
       call c_onnx_binary_node_name(h, i, cbuf, CBUF_LEN)
       nodes(i+1)%name = read_c_buf(cbuf)

       call c_onnx_binary_node_op_type(h, i, cbuf, CBUF_LEN)
       nodes(i+1)%op_type = read_c_buf(cbuf)

       if (verbose_ .gt. 1) then
          write(*,'(A,I0,A,A,A,A)') "  Node ", i, ": ", &
               trim(nodes(i+1)%name), " (", trim(nodes(i+1)%op_type)//")"
       end if

       nodes(i+1)%num_inputs = c_onnx_binary_node_num_inputs(h, i)
       allocate(nodes(i+1)%inputs(nodes(i+1)%num_inputs))
       do j = 0, nodes(i+1)%num_inputs - 1
          call c_onnx_binary_node_input(h, i, j, cbuf, CBUF_LEN)
          nodes(i+1)%inputs(j+1) = read_c_buf(cbuf)
          if (verbose_ .gt. 1) write(*,'(A,A)') "    input: ", &
               trim(nodes(i+1)%inputs(j+1))
       end do

       nodes(i+1)%num_outputs = c_onnx_binary_node_num_outputs(h, i)
       allocate(nodes(i+1)%outputs(nodes(i+1)%num_outputs))
       do j = 0, nodes(i+1)%num_outputs - 1
          call c_onnx_binary_node_output(h, i, j, cbuf, CBUF_LEN)
          nodes(i+1)%outputs(j+1) = read_c_buf(cbuf)
          if (verbose_ .gt. 1) write(*,'(A,A)') "    output: ", &
               trim(nodes(i+1)%outputs(j+1))
       end do

       ! Attributes
       k = c_onnx_binary_node_num_attrs(h, i)
       if (k .gt. 0) then
          allocate(nodes(i+1)%attributes(k))
          do j = 0, k - 1
             call c_onnx_binary_attr_name(h, i, j, cbuf, CBUF_LEN)
             nodes(i+1)%attributes(j+1)%name = read_c_buf(cbuf)

             call c_onnx_binary_attr_type_str(h, i, j, cbuf, CBUF_LEN)
             nodes(i+1)%attributes(j+1)%type = read_c_buf(cbuf)

             call c_onnx_binary_attr_value_str(h, i, j, vbuf, 4096)
             nodes(i+1)%attributes(j+1)%val = read_vbuf(vbuf, 4096)
             if (verbose_ .gt. 1) then
                write(*,'(A,A,A,A,A,A)') "    attr: ", &
                     trim(nodes(i+1)%attributes(j+1)%name), " = [", &
                     trim(nodes(i+1)%attributes(j+1)%val), &
                     "] type=", trim(nodes(i+1)%attributes(j+1)%type)
             end if
          end do
       end if
    end do

    ! --- Initialisers ---
    do i = 0, num_inits - 1
       call c_onnx_binary_init_name(h, i, cbuf, CBUF_LEN)
       initialisers(i+1)%name = read_c_buf(cbuf)

       nd = c_onnx_binary_init_num_dims(h, i)
       allocate(initialisers(i+1)%dims(nd))
       do j = 0, nd - 1
          initialisers(i+1)%dims(j+1) = &
               int(c_onnx_binary_init_dim(h, i, j))
       end do

       k = c_onnx_binary_init_num_floats(h, i)
       allocate(initialisers(i+1)%data(k))
       if (k .gt. 0) then
          call c_onnx_binary_init_float_data(h, i, &
               initialisers(i+1)%data, int(k, c_int))
       end if

       if (verbose_ .gt. 1) then
          write(*,'(A,A,A,I0,A,I0)') "  Init: ", &
               trim(initialisers(i+1)%name), &
               " dims=", nd, " floats=", k
          if (nd .gt. 0) write(*,'(A,*(I0,1X))') "    dims: ", initialisers(i+1)%dims
       end if
    end do

    ! --- Input tensors ---
    do i = 0, num_inputs - 1
       call c_onnx_binary_input_name(h, i, cbuf, CBUF_LEN)
       input_tensors(i+1)%name = read_c_buf(cbuf)

       input_tensors(i+1)%elem_type = c_onnx_binary_input_elem_type(h, i)

       nd = c_onnx_binary_input_num_dims(h, i)
       allocate(input_tensors(i+1)%dims(nd))
       do j = 0, nd - 1
          input_tensors(i+1)%dims(j+1) = &
               int(c_onnx_binary_input_dim(h, i, j))
       end do
       if (verbose_ .gt. 1) then
          write(*,'(A,A,A,I0)') "  Input: ", trim(input_tensors(i+1)%name), &
               " ndims=", nd
          if (nd .gt. 0) write(*,'(A,*(I0,1X))') "    dims: ", input_tensors(i+1)%dims
       end if
    end do

    ! --- Output tensors ---
    do i = 0, num_outputs - 1
       call c_onnx_binary_output_name(h, i, cbuf, CBUF_LEN)
       output_tensors(i+1)%name = read_c_buf(cbuf)

       output_tensors(i+1)%elem_type = c_onnx_binary_output_elem_type(h, i)

       nd = c_onnx_binary_output_num_dims(h, i)
       allocate(output_tensors(i+1)%dims(nd))
       do j = 0, nd - 1
          output_tensors(i+1)%dims(j+1) = &
               int(c_onnx_binary_output_dim(h, i, j))
       end do
    end do

    ! --- Value infos ---
    do i = 0, num_vis - 1
       call c_onnx_binary_vi_name(h, i, cbuf, CBUF_LEN)
       value_infos(i+1)%name = read_c_buf(cbuf)

       value_infos(i+1)%elem_type = c_onnx_binary_vi_elem_type(h, i)

       nd = c_onnx_binary_vi_num_dims(h, i)
       allocate(value_infos(i+1)%dims(nd))
       do j = 0, nd - 1
          value_infos(i+1)%dims(j+1) = &
               int(c_onnx_binary_vi_dim(h, i, j))
       end do
    end do

    ! Free C model
    call c_onnx_binary_free(h)

    ! Build network using existing infrastructure
    call network%build_from_onnx( &
         nodes, initialisers, input_tensors, value_infos, &
         verbose=verbose_ &
    )

  contains

    function read_vbuf(buf, blen) result(fstr)
      character(kind=c_char), intent(in) :: buf(*)
      integer, intent(in) :: blen
      character(len=:), allocatable :: fstr
      integer :: ii, sl
      sl = 0
      do ii = 1, blen
         if (buf(ii) == c_null_char) exit
         sl = sl + 1
      end do
      allocate(character(len=sl) :: fstr)
      do ii = 1, sl
         fstr(ii:ii) = buf(ii)
      end do
    end function read_vbuf

  end function read_onnx_binary
!###############################################################################


!###############################################################################
!  WRITE BINARY ONNX
!###############################################################################
  module subroutine write_onnx_binary(file, network)
    use athena__tools_infile, only: assign_val, assign_vec, allocate_and_assign_vec
    implicit none

    ! Arguments
    class(network_type), intent(in) :: network
    character(*), intent(in) :: file

    ! Local variables
    integer(c_int) :: h, nidx, rc
    integer :: i, j, layer_id, input_layer_id
    character(256) :: layer_name
    character(64) :: node_name, input_name, tmp_input_name
    character(:), allocatable :: suffix
    integer(c_int64_t), allocatable :: dims_c(:)
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    character(:), allocatable :: type_lw
    integer, allocatable, dimension(:) :: ivar_list
    real(real32), allocatable, dimension(:) :: rvar_list
    integer :: itmp1

    ! Create C model
    h = c_onnx_binary_create()
    if (h < 0) then
       write(*,*) "ERROR: Could not create ONNX binary model handle"
       return
    end if

    ! Set model metadata
    call c_onnx_binary_set_ir_version(h, 8_c_int64_t)
    call c_onnx_binary_set_producer(h, fc("Athena"), fc("1.0"))
    call c_onnx_binary_set_domain(h, fc("ai.onnx"))
    call c_onnx_binary_set_graph_name(h, fc("athena_network"))
    call c_onnx_binary_add_opset(h, fc(""), 13_c_int64_t)

    ! Write nodes (layers) - same traversal order as text writer
    do i = 1, network%auto_graph%num_vertices
       layer_id = network%auto_graph%vertex(network%vertex_order(i))%id
       write(node_name, '("node_", I0)') network%model(layer_id)%layer%id

       select case(trim(network%model(layer_id)%layer%type))
       case('inpt')
          layer_name = 'Input'
          cycle
       case('full')
          layer_name = 'Gemm'
       case('conv')
          layer_name = 'Conv'
       case('pool')
          layer_name = to_camel_case( &
               trim(adjustl(network%model(layer_id)%layer%subtype))//"_"//&
               trim(adjustl(network%model(layer_id)%layer%type)), &
               capitalise_first_letter = .true. &
          )
       case('actv')
          layer_name = to_camel_case( &
               adjustl(network%model(layer_id)%layer%subtype), &
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

       ! Add node
       nidx = c_onnx_binary_add_node(h, fc(trim(node_name)), &
            fc(trim(layer_name)))

       ! --- Write input connections ---
       if (.not. all(network%auto_graph%adjacency( &
            :,network%vertex_order(i)).eq.0)) then
          do j = 1, network%auto_graph%num_vertices
             input_layer_id = network%auto_graph%vertex(j)%id
             if (network%auto_graph%adjacency(j,network%vertex_order(i)) &
                  .eq.0) cycle
             if (all(network%auto_graph%adjacency(:,j).eq.0)) then
                write(input_name,'("input_",I0)') &
                     network%model(input_layer_id)%layer%id
                suffix = ''
             else
                write(input_name,'("node_",I0)') &
                     network%model(input_layer_id)%layer%id
                suffix = '_output'
                select type(prev_layer => network%model(input_layer_id)%layer)
                class is(learnable_layer_type)
                   if (prev_layer%activation%name.ne."none") then
                      suffix = '_' // trim(adjustl(prev_layer%activation%name)) &
                           // '_output'
                   end if
                end select
             end if
             if (network%model(layer_id)%layer%use_graph_input) then
                write(tmp_input_name,'(A,A,A)') &
                     trim(adjustl(input_name)), '_vertex', suffix
                call c_onnx_binary_node_add_input_w(h, nidx, &
                     fc(trim(adjustl(tmp_input_name))))
                if (network%model(layer_id)%layer%input_shape(2) .gt. 0) then
                   write(tmp_input_name,'(A,A,A)') &
                        trim(adjustl(input_name)), '_edge', suffix
                   call c_onnx_binary_node_add_input_w(h, nidx, &
                        fc(trim(adjustl(tmp_input_name))))
                end if
             else
                call c_onnx_binary_node_add_input_w(h, nidx, &
                     fc(trim(adjustl(input_name)) // suffix))
             end if
          end do
       end if

       ! Add parameter inputs (weights, biases)
       select type(layer => network%model(layer_id)%layer)
       class is(learnable_layer_type)
          do j = 1, size(layer%params)
             write(input_name, '("node_",I0,"_param",I0)') &
                  network%model(layer_id)%layer%id, j
             call c_onnx_binary_node_add_input_w(h, nidx, fc(trim(input_name)))
          end do
       end select
       suffix = ''

       ! --- Output ---
       if (network%model(layer_id)%layer%use_graph_output) then
          write(input_name, '("node_",I0,"_vertex_output")') &
               network%model(layer_id)%layer%id
          call c_onnx_binary_node_add_output_w(h, nidx, &
               fc(trim(adjustl(input_name))))
          write(input_name, '("node_",I0,"_edge_output")') &
               network%model(layer_id)%layer%id
          call c_onnx_binary_node_add_output_w(h, nidx, &
               fc(trim(adjustl(input_name))))
       else
          write(input_name, '("node_",I0,"_output")') &
               network%model(layer_id)%layer%id
          call c_onnx_binary_node_add_output_w(h, nidx, &
               fc(trim(adjustl(input_name))))
       end if

       ! --- Attributes ---
       call write_binary_attributes(h, nidx, network%model(layer_id)%layer)

       ! --- Initialisers and activation function nodes ---
       select type(layer => network%model(layer_id)%layer)
       class is(learnable_layer_type)
          call write_binary_initialisers(h, layer, trim(node_name))
          if (layer%activation%name .ne. "none") then
             if (layer%use_graph_output) then
                call write_binary_function(h, layer%activation%name, &
                     trim(node_name)//'_vertex')
                if (network%model(layer_id)%layer%input_shape(2) .gt. 0) then
                   call write_binary_function(h, layer%activation%name, &
                        trim(node_name)//'_edge')
                end if
             else
                call write_binary_function(h, layer%activation%name, &
                     trim(node_name))
             end if
          end if
       end select
    end do

    ! --- Value infos (intermediate layer outputs) ---
    do i = 1, network%auto_graph%num_vertices
       layer_id = network%auto_graph%vertex(network%vertex_order(i))%id
       if (.not. allocated(network%model(layer_id)%layer%output_shape)) cycle
       if (network%model(layer_id)%layer%use_graph_output) then
          write(node_name, '("node_",I0,"_vertex_output")') &
               network%model(layer_id)%layer%id
          call add_tensor_info(h, 'vi', trim(adjustl(node_name)), &
               [ network%model(layer_id)%layer%output_shape(1) ], &
               network%batch_size)
          if (network%model(layer_id)%layer%output_shape(2) .gt. 0) then
             write(node_name, '("node_",I0,"_edge_output")') &
                  network%model(layer_id)%layer%id
             call add_tensor_info(h, 'vi', trim(adjustl(node_name)), &
                  [ network%model(layer_id)%layer%output_shape(2) ], &
                  network%batch_size)
          end if
       else
          write(node_name, '("node_",I0,"_output")') &
               network%model(layer_id)%layer%id
          call add_tensor_info(h, 'vi', trim(adjustl(node_name)), &
               network%model(layer_id)%layer%output_shape, &
               network%batch_size)
       end if
    end do

    ! --- Inputs ---
    do i = 1, size(network%root_vertices, dim=1)
       layer_id = network%auto_graph%vertex(network%root_vertices(i))%id
       if (network%model(layer_id)%layer%use_graph_output) then
          write(node_name, '("input_",I0,"_vertex")') &
               network%model(layer_id)%layer%id
          call add_tensor_info(h, 'input', trim(adjustl(node_name)), &
               [ network%model(layer_id)%layer%input_shape(1) ], &
               network%batch_size)
          if (network%model(layer_id)%layer%input_shape(2) .gt. 0) then
             write(node_name, '("input_",I0,"_edge")') &
                  network%model(layer_id)%layer%id
             call add_tensor_info(h, 'input', trim(adjustl(node_name)), &
                  [ network%model(layer_id)%layer%input_shape(2) ], &
                  network%batch_size)
          end if
       else
          write(node_name, '("input_",I0)') network%model(layer_id)%layer%id
          call add_tensor_info(h, 'input', trim(adjustl(node_name)), &
               network%model(layer_id)%layer%input_shape, &
               network%batch_size)
       end if
    end do

    ! --- Outputs ---
    do i = 1, size(network%leaf_vertices, dim=1)
       layer_id = network%auto_graph%vertex(network%leaf_vertices(i))%id
       if (network%model(layer_id)%layer%use_graph_output) then
          write(node_name, '("node_",I0,"_vertex_output")') &
               network%model(layer_id)%layer%id
          call add_tensor_info(h, 'output', trim(adjustl(node_name)), &
               [ network%model(layer_id)%layer%output_shape(1) ], &
               network%batch_size)
          if (network%model(layer_id)%layer%output_shape(2) .gt. 0) then
             write(node_name, '("node_",I0,"_edge_output")') &
                  network%model(layer_id)%layer%id
             call add_tensor_info(h, 'output', trim(adjustl(node_name)), &
                  [ network%model(layer_id)%layer%output_shape(2) ], &
                  network%batch_size)
          end if
       else
          select type(layer => network%model(layer_id)%layer)
          class is(learnable_layer_type)
             if (layer%activation%name.eq."none") then
                suffix = ''
             else
                suffix = '_' // trim(adjustl(layer%activation%name))
             end if
          class default
             suffix = ''
          end select
          write(node_name, '("node_",I0,A,"_output")') &
               network%model(layer_id)%layer%id, trim(adjustl(suffix))
          call add_tensor_info(h, 'output', trim(adjustl(node_name)), &
               network%model(layer_id)%layer%output_shape, &
               network%batch_size)
       end if
    end do

    ! Write to file
    rc = c_onnx_binary_write(h, fc(file))
    if (rc /= 0) then
       write(*,*) "ERROR: Failed to write binary ONNX file: ", trim(file)
    end if

    ! Free C handle
    call c_onnx_binary_free(h)

  contains

    !> Add a tensor info (input, output, or value_info)
    subroutine add_tensor_info(handle, kind, name, output_shape, batch_size)
      integer(c_int), intent(in) :: handle
      character(*), intent(in) :: kind, name
      integer, intent(in), dimension(:) :: output_shape
      integer, intent(in) :: batch_size
      integer(c_int64_t), allocatable :: dims(:)
      integer :: k, rc_

      allocate(dims(size(output_shape) + 1))
      dims(1) = int(max(1, batch_size), c_int64_t)
      do k = 1, size(output_shape)
         dims(k+1) = int(output_shape(size(output_shape)+1-k), c_int64_t)
      end do

      select case(kind)
      case('input')
         rc_ = c_onnx_binary_add_input_w(handle, fc(name), &
              1, dims, int(size(dims), c_int))
      case('output')
         rc_ = c_onnx_binary_add_output_w(handle, fc(name), &
              1, dims, int(size(dims), c_int))
      case('vi')
         rc_ = c_onnx_binary_add_value_info(handle, fc(name), &
              1, dims, int(size(dims), c_int))
      end select
      deallocate(dims)
    end subroutine add_tensor_info

    !> Write initialisers (weights/biases) for a learnable layer
    subroutine write_binary_initialisers(handle, layer, prefix)
      integer(c_int), intent(in) :: handle
      class(learnable_layer_type), intent(in) :: layer
      character(*), intent(in) :: prefix
      integer :: p, np, k, nd, rc_
      character(64) :: pname
      integer(c_int64_t) :: dims_buf(8)

      if (.not. allocated(layer%params)) return
      do p = 1, size(layer%params)
         np = size(layer%params(p)%val, 1)
         write(pname, '(A,A,I0)') trim(prefix), '_param', p

         if (p == 1 .and. allocated(layer%weight_shape)) then
            ! Weight tensor: reverse Fortran col-major shape to ONNX row-major
            nd = size(layer%weight_shape, 1)
            do k = 1, nd
               dims_buf(k) = int(layer%weight_shape(nd+1-k, 1), c_int64_t)
            end do
            rc_ = c_onnx_binary_add_initializer(handle, fc(trim(pname)), &
                 dims_buf, nd, layer%params(p)%val(:,1), int(np, c_int))
         else if (p == 2 .and. allocated(layer%bias_shape)) then
            ! Bias tensor: use bias_shape directly
            nd = size(layer%bias_shape)
            do k = 1, nd
               dims_buf(k) = int(layer%bias_shape(k), c_int64_t)
            end do
            rc_ = c_onnx_binary_add_initializer(handle, fc(trim(pname)), &
                 dims_buf, nd, layer%params(p)%val(:,1), int(np, c_int))
         else
            ! Fallback: 1D flat tensor
            dims_buf(1) = int(np, c_int64_t)
            rc_ = c_onnx_binary_add_initializer(handle, fc(trim(pname)), &
                 dims_buf, 1, layer%params(p)%val(:,1), int(np, c_int))
         end if
      end do
    end subroutine write_binary_initialisers

    !> Write an activation function as a separate ONNX node
    subroutine write_binary_function(handle, function_name, prefix)
      integer(c_int), intent(in) :: handle
      character(*), intent(in) :: function_name, prefix
      character(256) :: full_name
      character(:), allocatable :: camel_name
      integer(c_int) :: fn_idx

      camel_name = to_camel_case(trim(adjustl(function_name)), &
           capitalise_first_letter = .true.)
      if (len_trim(prefix) == 0) then
         full_name = trim(adjustl(function_name))
      else
         full_name = trim(prefix) // "_" // trim(adjustl(function_name))
      end if

      fn_idx = c_onnx_binary_add_node(handle, fc(trim(full_name)), &
           fc(trim(camel_name)))
      call c_onnx_binary_node_add_input_w(handle, fn_idx, &
           fc(trim(prefix) // '_output'))
      call c_onnx_binary_node_add_output_w(handle, fn_idx, &
           fc(trim(full_name) // '_output'))
    end subroutine write_binary_function

    !> Write layer attributes to C model
    subroutine write_binary_attributes(handle, node_idx, layer)
      integer(c_int), intent(in) :: handle, node_idx
      class(base_layer_type), intent(in) :: layer
      type(onnx_attribute_type), allocatable, dimension(:) :: attrs
      integer :: a, n_vals
      character(:), allocatable :: type_lw_
      integer, allocatable, dimension(:) :: ivals
      real(real32), allocatable, dimension(:) :: rvals
      integer(c_int64_t), allocatable :: ivals_c(:)

      attrs = layer%get_attributes()
      if (.not. allocated(attrs)) return
      if (size(attrs) == 0) return

      do a = 1, size(attrs)
         type_lw_ = to_lower(trim(adjustl(attrs(a)%type)))
         n_vals = icount(attrs(a)%val)

         select case(type_lw_)
         case('ints', 'int')
            allocate(ivals(n_vals))
            read(attrs(a)%val, *) ivals
            allocate(ivals_c(n_vals))
            ivals_c = int(ivals, c_int64_t)
            call c_onnx_binary_node_add_attr_ints(handle, node_idx, &
                 fc(trim(attrs(a)%name)), ivals_c, int(n_vals, c_int))
            deallocate(ivals, ivals_c)
         case('floats', 'float')
            allocate(rvals(n_vals))
            read(attrs(a)%val, *) rvals
            call c_onnx_binary_node_add_attr_floats(handle, node_idx, &
                 fc(trim(attrs(a)%name)), rvals, int(n_vals, c_int))
            deallocate(rvals)
         case('strings', 'string')
            call c_onnx_binary_node_add_attr_string(handle, node_idx, &
                 fc(trim(attrs(a)%name)), fc(trim(attrs(a)%val)))
         case default
            call c_onnx_binary_node_add_attr_string(handle, node_idx, &
                 fc(trim(attrs(a)%name)), fc(trim(attrs(a)%val)))
         end select
      end do
    end subroutine write_binary_attributes

  end subroutine write_onnx_binary
!###############################################################################

end submodule athena__onnx_binary_submodule
