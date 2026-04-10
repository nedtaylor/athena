program test_container_layer_registry
  use athena, only: input_layer_type
  use athena__base_layer, only: base_layer_type
  use athena__container_layer, only: &
       read_layer_container, &
       list_of_layer_types, &
       allocate_list_of_layer_types, &
       onnx_create_layer_container, &
       list_of_onnx_layer_creators, &
       allocate_list_of_onnx_layer_creators, &
       onnx_meta_create_layer_container, &
       list_of_onnx_meta_layer_creators, &
       allocate_list_of_onnx_meta_layer_creators, &
       onnx_expanded_nop_create_layer_container, &
       list_of_onnx_expanded_nop_layer_creators, &
       allocate_list_of_onnx_expanded_nop_layer_creators, &
       onnx_expanded_gnn_create_layer_container, &
       list_of_onnx_expanded_gnn_layer_creators, &
       allocate_list_of_onnx_expanded_gnn_layer_creators
  use athena__misc_types, only: onnx_node_type, onnx_initialiser_type, &
       onnx_tensor_type
  implicit none

  logical :: success

  success = .true.

  call test_layer_reader_registry(success)
  call test_onnx_layer_creator_registry(success)
  call test_onnx_meta_creator_registry(success)
  call test_onnx_expanded_nop_registry(success)
  call test_onnx_expanded_gnn_registry(success)

  if (success) then
     write(*,*) 'test_container_layer_registry passed all tests'
  else
     write(0,*) 'test_container_layer_registry failed one or more tests'
     stop 1
  end if

contains

  subroutine test_layer_reader_registry(success)
    logical, intent(inout) :: success
    integer :: baseline_size

    write(*,'("Testing layer reader registry")')

    if (allocated(list_of_layer_types)) deallocate(list_of_layer_types)
    call allocate_list_of_layer_types()

    if (.not. allocated(list_of_layer_types)) then
       success = .false.
       write(0,*) 'layer reader registry was not allocated'
       return
    end if

    if (.not. has_reader_name('reshape') .or. &
         .not. has_reader_name('flatten')) then
       success = .false.
       write(0,*) 'layer reader registry is missing expected built-in entries'
    end if

    baseline_size = size(list_of_layer_types)
    call allocate_list_of_layer_types([ &
         read_layer_container(name='dummy', read_ptr=read_dummy_layer) &
    ])

    if (size(list_of_layer_types) /= 2 * baseline_size + 1) then
       success = .false.
       write(0,*) 'layer reader registry appended the wrong number of entries'
    end if
    if (trim(list_of_layer_types(size(list_of_layer_types))%name) /= &
         'dummy') then
       success = .false.
       write(0,*) 'layer reader registry did not append the custom entry'
    end if
  end subroutine test_layer_reader_registry

  subroutine test_onnx_layer_creator_registry(success)
    logical, intent(inout) :: success
    integer :: baseline_size

    write(*,'("Testing ONNX layer creator registry")')

    if (allocated(list_of_onnx_layer_creators)) then
       deallocate(list_of_onnx_layer_creators)
    end if
    call allocate_list_of_onnx_layer_creators()

    if (.not. has_onnx_op_type('Flatten') .or. &
         .not. has_onnx_op_type('Selu')) then
       success = .false.
       write(0,*) 'ONNX layer creator registry is missing expected '
       write(0,*) 'built-in entries'
    end if

    baseline_size = size(list_of_onnx_layer_creators)
    call allocate_list_of_onnx_layer_creators([ &
         onnx_create_layer_container( &
              op_type='Dummy', create_ptr=create_dummy_onnx_layer) &
    ])

    if (size(list_of_onnx_layer_creators) /= 2 * baseline_size + 1) then
       success = .false.
       write(0,*) 'ONNX layer creator registry appended the wrong '
       write(0,*) 'number of entries'
    end if
    if (trim(list_of_onnx_layer_creators( &
         size(list_of_onnx_layer_creators))%op_type) /= 'Dummy') then
       success = .false.
       write(0,*) 'ONNX layer creator registry did not append the custom entry'
    end if
  end subroutine test_onnx_layer_creator_registry

  subroutine test_onnx_meta_creator_registry(success)
    logical, intent(inout) :: success
    integer :: baseline_size

    write(*,'("Testing ONNX meta-layer registry")')

    if (allocated(list_of_onnx_meta_layer_creators)) then
       deallocate(list_of_onnx_meta_layer_creators)
    end if
    call allocate_list_of_onnx_meta_layer_creators()

    if (.not. has_meta_subtype('kipf') .or. &
         .not. has_meta_subtype('dynamic_lno')) then
       success = .false.
       write(0,*) 'ONNX meta-layer registry is missing expected built-in entries'
    end if

    baseline_size = size(list_of_onnx_meta_layer_creators)
    call allocate_list_of_onnx_meta_layer_creators([ &
         onnx_meta_create_layer_container( &
              layer_subtype='dummy_meta', &
              create_ptr=create_dummy_meta_layer) &
    ])

    if (size(list_of_onnx_meta_layer_creators) /= 2 * baseline_size + 1) then
       success = .false.
       write(0,*) 'ONNX meta-layer registry appended the wrong number of entries'
    end if
    if (trim(list_of_onnx_meta_layer_creators( &
         size(list_of_onnx_meta_layer_creators))%layer_subtype) /= &
         'dummy_meta') then
       success = .false.
       write(0,*) 'ONNX meta-layer registry did not append the custom entry'
    end if
  end subroutine test_onnx_meta_creator_registry

  subroutine test_onnx_expanded_nop_registry(success)
    logical, intent(inout) :: success
    integer :: baseline_size

    write(*,'("Testing ONNX expanded NOP registry")')

    if (allocated(list_of_onnx_expanded_nop_layer_creators)) then
       deallocate(list_of_onnx_expanded_nop_layer_creators)
    end if
    call allocate_list_of_onnx_expanded_nop_layer_creators()

    if (.not. has_nop_subtype('dynamic_lno') .or. &
         .not. has_nop_subtype('spectral_filter')) then
       success = .false.
       write(0,*) 'ONNX expanded NOP registry is missing expected '
       write(0,*) 'built-in entries'
    end if

    baseline_size = size(list_of_onnx_expanded_nop_layer_creators)
    call allocate_list_of_onnx_expanded_nop_layer_creators([ &
         onnx_expanded_nop_create_layer_container( &
              nop_subtype='dummy_nop', &
              classify_ptr=classify_dummy_expanded_nop, &
              build_ptr=build_dummy_expanded_nop) &
    ])

    if (size(list_of_onnx_expanded_nop_layer_creators) /= &
         2 * baseline_size + 1) then
       success = .false.
       write(0,*) 'ONNX expanded NOP registry appended the wrong '
       write(0,*) 'number of entries'
    end if
    if (trim(list_of_onnx_expanded_nop_layer_creators( &
         size(list_of_onnx_expanded_nop_layer_creators))%nop_subtype) /= &
         'dummy_nop') then
       success = .false.
       write(0,*) 'ONNX expanded NOP registry did not append the custom entry'
    end if
  end subroutine test_onnx_expanded_nop_registry

  subroutine test_onnx_expanded_gnn_registry(success)
    logical, intent(inout) :: success
    integer :: baseline_size

    write(*,'("Testing ONNX expanded GNN registry")')

    if (allocated(list_of_onnx_expanded_gnn_layer_creators)) then
       deallocate(list_of_onnx_expanded_gnn_layer_creators)
    end if
    call allocate_list_of_onnx_expanded_gnn_layer_creators()

    if (.not. has_gnn_subtype('kipf') .or. &
         .not. has_gnn_subtype('duvenaud')) then
       success = .false.
       write(0,*) 'ONNX expanded GNN registry is missing expected '
       write(0,*) 'built-in entries'
    end if

    baseline_size = size(list_of_onnx_expanded_gnn_layer_creators)
    call allocate_list_of_onnx_expanded_gnn_layer_creators([ &
         onnx_expanded_gnn_create_layer_container( &
              gnn_subtype='dummy_gnn', &
              classify_ptr=classify_dummy_expanded_gnn, &
              build_ptr=build_dummy_expanded_gnn) &
    ])

    if (size(list_of_onnx_expanded_gnn_layer_creators) /= &
         2 * baseline_size + 1) then
       success = .false.
       write(0,*) 'ONNX expanded GNN registry appended the wrong '
       write(0,*) 'number of entries'
    end if
    if (trim(list_of_onnx_expanded_gnn_layer_creators( &
         size(list_of_onnx_expanded_gnn_layer_creators))%gnn_subtype) /= &
         'dummy_gnn') then
       success = .false.
       write(0,*) 'ONNX expanded GNN registry did not append the custom entry'
    end if
  end subroutine test_onnx_expanded_gnn_registry

  logical function has_reader_name(name)
    character(*), intent(in) :: name
    integer :: i

    has_reader_name = .false.
    do i = 1, size(list_of_layer_types)
       if (trim(list_of_layer_types(i)%name) == trim(name)) then
          has_reader_name = .true.
          return
       end if
    end do
  end function has_reader_name

  logical function has_onnx_op_type(op_type)
    character(*), intent(in) :: op_type
    integer :: i

    has_onnx_op_type = .false.
    do i = 1, size(list_of_onnx_layer_creators)
       if (trim(list_of_onnx_layer_creators(i)%op_type) == trim(op_type)) then
          has_onnx_op_type = .true.
          return
       end if
    end do
  end function has_onnx_op_type

  logical function has_meta_subtype(subtype)
    character(*), intent(in) :: subtype
    integer :: i

    has_meta_subtype = .false.
    do i = 1, size(list_of_onnx_meta_layer_creators)
       if (trim(list_of_onnx_meta_layer_creators(i)%layer_subtype) == &
            trim(subtype)) then
          has_meta_subtype = .true.
          return
       end if
    end do
  end function has_meta_subtype

  logical function has_nop_subtype(subtype)
    character(*), intent(in) :: subtype
    integer :: i

    has_nop_subtype = .false.
    do i = 1, size(list_of_onnx_expanded_nop_layer_creators)
       if (trim(list_of_onnx_expanded_nop_layer_creators(i)%nop_subtype) == &
            trim(subtype)) then
          has_nop_subtype = .true.
          return
       end if
    end do
  end function has_nop_subtype

  logical function has_gnn_subtype(subtype)
    character(*), intent(in) :: subtype
    integer :: i

    has_gnn_subtype = .false.
    do i = 1, size(list_of_onnx_expanded_gnn_layer_creators)
       if (trim(list_of_onnx_expanded_gnn_layer_creators(i)%gnn_subtype) == &
            trim(subtype)) then
          has_gnn_subtype = .true.
          return
       end if
    end do
  end function has_gnn_subtype

  function read_dummy_layer(unit, verbose) result(layer)
    integer, intent(in) :: unit
    integer, intent(in), optional :: verbose
    class(base_layer_type), allocatable :: layer

    allocate(layer, source=input_layer_type(input_shape=[1]))
  end function read_dummy_layer

  function create_dummy_onnx_layer( &
       nodes, initialisers, value_info, verbose) result(layer)
    type(onnx_node_type), intent(in) :: nodes
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    integer, intent(in), optional :: verbose
    class(base_layer_type), allocatable :: layer

    allocate(layer, source=input_layer_type(input_shape=[1]))
  end function create_dummy_onnx_layer

  function create_dummy_meta_layer( &
       meta_key, meta_value, inits, verbose) result(layer)
    character(*), intent(in) :: meta_key
    character(*), intent(in) :: meta_value
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    integer, intent(in), optional :: verbose
    class(base_layer_type), allocatable :: layer

    allocate(layer, source=input_layer_type(input_shape=[1]))
  end function create_dummy_meta_layer

  logical function classify_dummy_expanded_nop(prefix, nodes, num_nodes)
    character(*), intent(in) :: prefix
    type(onnx_node_type), intent(in) :: nodes(:)
    integer, intent(in) :: num_nodes

    classify_dummy_expanded_nop = trim(prefix) == 'dummy'
  end function classify_dummy_expanded_nop

  function build_dummy_expanded_nop( &
       prefix, nodes, num_nodes, inits, num_inits) result(layer)
    character(*), intent(in) :: prefix
    type(onnx_node_type), intent(in) :: nodes(:)
    integer, intent(in) :: num_nodes
    type(onnx_initialiser_type), intent(in) :: inits(:)
    integer, intent(in) :: num_inits
    class(base_layer_type), allocatable :: layer

    allocate(layer, source=input_layer_type(input_shape=[1]))
  end function build_dummy_expanded_nop

  logical function classify_dummy_expanded_gnn(prefix, nodes, num_nodes)
    character(*), intent(in) :: prefix
    type(onnx_node_type), intent(in) :: nodes(:)
    integer, intent(in) :: num_nodes

    classify_dummy_expanded_gnn = trim(prefix) == 'dummy'
  end function classify_dummy_expanded_gnn

  function build_dummy_expanded_gnn( &
       prefix, nodes, num_nodes, inits, num_inits, inputs, &
       num_inputs) result(layer)
    character(*), intent(in) :: prefix
    type(onnx_node_type), intent(in) :: nodes(:)
    integer, intent(in) :: num_nodes
    type(onnx_initialiser_type), intent(in) :: inits(:)
    integer, intent(in) :: num_inits
    type(onnx_tensor_type), intent(in) :: inputs(:)
    integer, intent(in) :: num_inputs
    class(base_layer_type), allocatable :: layer

    allocate(layer, source=input_layer_type(input_shape=[1]))
  end function build_dummy_expanded_gnn

end program test_container_layer_registry
