module nid_helpers
  use coreutils, only: real32
  use athena, only: &
       network_type, &
       full_layer_type, &
       kipf_msgpass_layer_type, &
       sgd_optimiser_type, &
       graph_type
  implicit none

  real(real32), parameter :: network_input_dispatch_tol = 1.e-6_real32

contains

  subroutine compile_dense_network(network, num_inputs)
    type(network_type), intent(out) :: network
    integer, intent(in) :: num_inputs

    call network%add(full_layer_type( &
         num_inputs = num_inputs, &
         num_outputs = 1, &
         activation = 'linear', &
         kernel_initialiser = 'ones', &
         use_bias = .false. &
    ))
    call network%compile( &
         optimiser = sgd_optimiser_type(learning_rate=1.e-2_real32), &
         loss_method = 'mse', &
         metrics = ['loss'], &
         batch_size = 1, &
         verbose = 0 &
    )
  end subroutine compile_dense_network

  subroutine compile_graph_network(network, num_vertex_features)
    type(network_type), intent(out) :: network
    integer, intent(in) :: num_vertex_features

    call network%add(kipf_msgpass_layer_type( &
         num_vertex_features = [num_vertex_features, num_vertex_features], &
         num_time_steps = 1 &
    ))
    call network%compile( &
         optimiser = sgd_optimiser_type(learning_rate=1.e-2_real32), &
         loss_method = 'mse', &
         metrics = ['loss'], &
         batch_size = 1, &
         verbose = 0 &
    )
  end subroutine compile_graph_network

  subroutine setup_graph(graph)
    type(graph_type), intent(out) :: graph
    integer, allocatable :: edge_index(:,:)

    graph%is_sparse = .true.
    call graph%set_num_vertices(3, 2)
    call graph%set_num_edges(2)

    graph%vertex_features(1,:) = [1._real32, 2._real32, 3._real32]
    graph%vertex_features(2,:) = [0.5_real32, 1.5_real32, 2.5_real32]

    allocate(edge_index(2,2))
    edge_index(:,1) = [1, 2]
    edge_index(:,2) = [2, 3]
    call graph%generate_adjacency(edge_index)
    deallocate(edge_index)

    graph%edge_weights = [1._real32, 1._real32]
  end subroutine setup_graph

  subroutine require(condition, message, success)
    logical, intent(in) :: condition
    character(*), intent(in) :: message
    logical, intent(inout) :: success

    if(.not.condition)then
       success = .false.
       write(0,*) trim(message)
    end if
  end subroutine require

  subroutine require_close_2d(actual, expected, message, success)
    real(real32), intent(in) :: actual(:,:), expected(:,:)
    character(*), intent(in) :: message
    logical, intent(inout) :: success

    call require(all(shape(actual).eq.shape(expected)), &
         trim(message)//' (shape mismatch)', success)
    if(all(shape(actual).eq.shape(expected)))then
       call require(all(abs(actual - expected) .lt. network_input_dispatch_tol), &
            message, success)
    end if
  end subroutine require_close_2d

end module nid_helpers

program test_network_input_dispatch
  use coreutils, only: real32, test_error_handling
  use athena, only: network_type, array_ptr_type, graph_type
  use diffstruc, only: array_type
  use nid_helpers, only: compile_dense_network, compile_graph_network, &
       setup_graph, require, require_close_2d
  implicit none

  logical :: success
  type(network_type) :: network
  type(array_type) :: scalar_input
  type(array_ptr_type) :: scalar_ptr_input
  type(array_type), target :: scalar_ptr_cells(1,1)
  type(array_type) :: rank1_input(1)
  type(array_type) :: vector_input(1), matrix_input(1,2)
  type(graph_type) :: graph_vector(2), graph_matrix(1,2)
  type(array_ptr_type) :: ptr_input(1)
  type(array_type), target :: ptr_cells(1,1)
  type(array_type) :: graph_mismatch_input(1)
  real(real32) :: data(3,2)
  real(real32) :: sample_data(2,1)
  real(real32) :: rank3_input(2,2,2), rank3_expected(4,2)
  real(real32) :: rank4_input(2,2,2,2), rank4_expected(8,2)
  real(real32) :: rank5_input(2,2,2,2,2), rank5_expected(16,2)
  real(real32) :: rank1_real_input(3)
  real(real32) :: ptr_data(2,1)
  real(real32) :: rank6_real_input(1,1,1,1,1,1)
  real(real32) :: graph_data(2,2)
  integer :: rank0_integer_input
  integer :: rank1_integer_input(2)
  integer :: rank2_integer_input(1,1)
  integer :: rank3_integer_input(1,1,1)
  integer :: rank4_integer_input(1,1,1,1)
  integer :: rank5_integer_input(1,1,1,1,1)
  integer :: index
  integer :: num_samples

  success = .true.

  data = reshape([1._real32, 2._real32, 3._real32, 4._real32, 5._real32, &
       6._real32], shape(data))
  call compile_dense_network(network, num_inputs=3)
  call scalar_input%allocate(array_shape=shape(data))
  call scalar_input%set(data)
  num_samples = network%save_input(scalar_input)
  call require(num_samples.eq.2, &
       'scalar array_type input returned wrong sample count', success)
  call require(allocated(network%input_array), &
       'scalar array_type input did not allocate input_array', success)
  call require(size(network%input_array,1).eq.1 .and. &
       size(network%input_array,2).eq.1, &
       'scalar array_type input created unexpected input_array shape', &
       success)
  call require_close_2d(network%input_array(1,1)%val, data, &
       'scalar array_type input did not preserve values', success)

  data = reshape([2._real32, 4._real32, 6._real32, 1._real32, 3._real32, &
       5._real32], shape(data))
  call compile_dense_network(network, num_inputs=3)
  call scalar_ptr_cells(1,1)%allocate(array_shape=shape(data))
  call scalar_ptr_cells(1,1)%set(data)
  scalar_ptr_input%array => scalar_ptr_cells
  num_samples = network%save_input(scalar_ptr_input)
  call require(num_samples.eq.2, &
       'scalar array_ptr_type input returned wrong sample count', success)
  call require(allocated(network%input_array), &
       'scalar array_ptr_type input did not allocate input_array', success)
  call require(size(network%input_array,1).eq.1 .and. &
       size(network%input_array,2).eq.1, &
       'scalar array_ptr_type input created unexpected input_array shape', &
       success)
  call require_close_2d(network%input_array(1,1)%val, data, &
       'scalar array_ptr_type input did not preserve values', success)

  data = reshape([-1._real32, 0._real32, 1._real32, 2._real32, 3._real32, &
       4._real32], shape(data))
  call compile_dense_network(network, num_inputs=3)
  call rank1_input(1)%allocate(array_shape=shape(data))
  call rank1_input(1)%set(data)
  num_samples = network%save_input(rank1_input)
  call require(num_samples.eq.2, &
       'rank-1 array_type input returned wrong sample count', success)
  call require(allocated(network%input_array), &
       'rank-1 array_type input did not allocate input_array', success)
  call require(size(network%input_array,1).eq.1 .and. &
       size(network%input_array,2).eq.1, &
       'rank-1 array_type input created unexpected input_array shape', &
       success)
  call require_close_2d(network%input_array(1,1)%val, data, &
       'rank-1 array_type input did not preserve values', success)

  sample_data = reshape([1._real32, 2._real32], shape(sample_data))
  call compile_graph_network(network, num_vertex_features=2)
  call vector_input(1)%allocate(array_shape=shape(sample_data))
  call vector_input(1)%set(sample_data)
  num_samples = network%save_input(vector_input)
  call require(num_samples.eq.1, &
       'graph-enabled rank-1 array_type input returned wrong sample count', &
       success)
  call matrix_input(1,1)%allocate(array_shape=shape(sample_data))
  call matrix_input(1,1)%set(sample_data)
  call matrix_input(1,2)%allocate(array_shape=shape(sample_data))
  call matrix_input(1,2)%set(2._real32 * sample_data)
  num_samples = network%save_input(matrix_input)
  call require(num_samples.eq.2, &
       'graph-enabled rank-2 array_type input returned wrong sample count', &
       success)
  call require(allocated(network%input_array), &
       'graph-enabled rank-2 array_type input did not allocate input_array', &
       success)
  call require(size(network%input_array,1).eq.1 .and. &
       size(network%input_array,2).eq.2, &
       'graph-enabled rank-2 array_type input created unexpected shape', &
       success)

  call compile_graph_network(network, num_vertex_features=2)
  call setup_graph(graph_vector(1))
  call setup_graph(graph_vector(2))
  call setup_graph(graph_matrix(1,1))
  call setup_graph(graph_matrix(1,2))
  num_samples = network%save_input(graph_vector)
  call require(num_samples.eq.2, &
       'rank-1 graph input returned wrong sample count', success)
  call require(allocated(network%input_graph), &
       'rank-1 graph input did not allocate input_graph', success)
  call require(size(network%input_graph,1).eq.1 .and. &
       size(network%input_graph,2).eq.2, &
       'rank-1 graph input created unexpected input_graph shape', success)
  num_samples = network%save_input(graph_matrix)
  call require(num_samples.eq.2, &
       'rank-2 graph input returned wrong sample count', success)
  call require(allocated(network%input_graph), &
       'rank-2 graph input did not preserve input_graph allocation', success)
  call require(size(network%input_graph,1).eq.1 .and. &
       size(network%input_graph,2).eq.2, &
       'rank-2 graph input created unexpected input_graph shape', success)

  rank3_input = reshape([(real(index, real32), index=1, size(rank3_input))], &
       shape(rank3_input))
  rank3_expected = reshape(rank3_input, shape(rank3_expected))
  call compile_dense_network(network, num_inputs=4)
  num_samples = network%save_input(rank3_input)
  call require(num_samples.eq.2, &
       'rank-3 real input returned wrong sample count', success)
  call require_close_2d(network%input_array(1,1)%val, rank3_expected, &
       'rank-3 real input did not preserve flattened values', success)

  rank4_input = reshape([(real(index, real32), index=1, size(rank4_input))], &
       shape(rank4_input))
  rank4_expected = reshape(rank4_input, shape(rank4_expected))
  call compile_dense_network(network, num_inputs=8)
  num_samples = network%save_input(rank4_input)
  call require(num_samples.eq.2, &
       'rank-4 real input returned wrong sample count', success)
  call require_close_2d(network%input_array(1,1)%val, rank4_expected, &
       'rank-4 real input did not preserve flattened values', success)

  rank5_input = reshape([(real(index, real32), index=1, size(rank5_input))], &
       shape(rank5_input))
  rank5_expected = reshape(rank5_input, shape(rank5_expected))
  call compile_dense_network(network, num_inputs=16)
  num_samples = network%save_input(rank5_input)
  call require(num_samples.eq.2, &
       'rank-5 real input returned wrong sample count', success)
  call require_close_2d(network%input_array(1,1)%val, rank5_expected, &
       'rank-5 real input did not preserve flattened values', success)

  test_error_handling = .true.

  rank1_real_input = [1._real32, 2._real32, 3._real32]
  call compile_dense_network(network, num_inputs=1)
  num_samples = network%save_input(rank1_real_input)

  ptr_data = reshape([1._real32, 2._real32], shape(ptr_data))
  call compile_dense_network(network, num_inputs=2)
  call ptr_cells(1,1)%allocate(array_shape=shape(ptr_data))
  call ptr_cells(1,1)%set(ptr_data)
  ptr_input(1)%array => ptr_cells
  num_samples = network%save_input(ptr_input)

  call compile_graph_network(network, num_vertex_features=2)
  num_samples = network%save_input(ptr_input)

  rank0_integer_input = 1
  call compile_dense_network(network, num_inputs=1)
  num_samples = network%save_input(rank0_integer_input)

  rank1_integer_input = [1, 2]
  call compile_dense_network(network, num_inputs=1)
  num_samples = network%save_input(rank1_integer_input)

  rank2_integer_input = 1
  call compile_dense_network(network, num_inputs=1)
  num_samples = network%save_input(rank2_integer_input)

  rank3_integer_input = 1
  call compile_dense_network(network, num_inputs=1)
  num_samples = network%save_input(rank3_integer_input)

  rank4_integer_input = 1
  call compile_dense_network(network, num_inputs=1)
  num_samples = network%save_input(rank4_integer_input)

  rank5_integer_input = 1
  call compile_dense_network(network, num_inputs=1)
  num_samples = network%save_input(rank5_integer_input)

  graph_data = reshape([1._real32, 2._real32, 3._real32, 4._real32], &
       shape(graph_data))
  call compile_graph_network(network, num_vertex_features=2)
  call graph_mismatch_input(1)%allocate(array_shape=shape(graph_data))
  call graph_mismatch_input(1)%set(graph_data)
  num_samples = network%save_input(graph_mismatch_input)

  rank6_real_input = 1._real32
  call compile_dense_network(network, num_inputs=1)
  num_samples = network%save_input(rank6_real_input)

  test_error_handling = .false.

  if(success)then
     write(*,*) 'test_network_input_dispatch passed all tests'
  else
     write(0,*) 'test_network_input_dispatch failed one or more tests'
     stop 1
  end if

end program test_network_input_dispatch
