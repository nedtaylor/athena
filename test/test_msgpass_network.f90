program test_msgpass_network
  use athena, only: &
       network_type, &
       kipf_msgpass_layer_type, &
       duvenaud_msgpass_layer_type, &
       full_layer_type, &
       base_optimiser_type, &
       sgd_optimiser_type
  use athena__misc_types, only: array2d_type
  use graphstruc, only: graph_type
  implicit none

  type(network_type) :: kipf_network, duvenaud_network
  type(graph_type), dimension(1,1) :: graph_data_kipf, graph_data_duvenaud

  integer, parameter :: num_vertex_features = 8
  integer, parameter :: num_edge_features = 2
  integer, parameter :: num_vertices = 5
  integer, parameter :: num_edges = 6
  integer, parameter :: num_time_steps = 1
  integer, parameter :: num_outputs = 3
  integer, parameter :: batch_size = 1

  logical :: success = .true.


!-------------------------------------------------------------------------------
! Set up test graph data (same for both networks)
!-------------------------------------------------------------------------------
  call setup_test_graph(graph_data_kipf(1,1), .false.)
  call setup_test_graph(graph_data_duvenaud(1,1), .true.)


!-------------------------------------------------------------------------------
! Test Kipf msgpass network
!-------------------------------------------------------------------------------
  kipf_network_test: block
    type(array2d_type), dimension(1,1) :: target_array

    call kipf_network%add(kipf_msgpass_layer_type( &
         num_vertex_features = [num_vertex_features, num_vertex_features], &
         num_time_steps = num_time_steps, &
         batch_size = batch_size &
    ))

    call kipf_network%compile( &
         optimiser = sgd_optimiser_type(learning_rate=0.1), &
         loss_method = 'mse', &
         accuracy_method = 'mse', &
         metrics = ['loss'], &
         batch_size = batch_size, &
         verbose = 1 &
    )

    ! Check network has correct number of layers (input + msgpass)
    if(kipf_network%num_layers .ne. 2) then
       success = .false.
       write(0,*) 'Kipf network has wrong number of layers:', &
            kipf_network%num_layers, 'expected: 2'
    end if

    ! Set up target output for testing
    call target_array(1,1)%allocate([num_outputs, 1])
    target_array(1,1)%val = reshape([0.5, 0.3, 0.2], [num_outputs, 1])

    ! Test training for a few epochs
    call kipf_network%train( &
         graph_data_kipf, &
         graph_data_kipf, &
         num_epochs = 5, &
         shuffle_batches = .false. &
    )

    ! Test prediction
    call kipf_network%test(graph_data_kipf, graph_data_kipf)

    ! Check that network produces reasonable outputs
    if(kipf_network%accuracy .lt. 0.0 .or. kipf_network%accuracy .gt. 1.0) then
       success = .false.
       write(0,*) 'Kipf network accuracy out of range:', kipf_network%accuracy
    end if

    if(kipf_network%loss .lt. 0.0) then
       success = .false.
       write(0,*) 'Kipf network loss is negative:', kipf_network%loss
    end if

    deallocate(target_array(1,1)%val)
  end block kipf_network_test


!-------------------------------------------------------------------------------
! Test Duvenaud msgpass network
!-------------------------------------------------------------------------------
  duvenaud_network_test: block
    type(array2d_type), dimension(1,1) :: target_array

    call duvenaud_network%add(duvenaud_msgpass_layer_type( &
         num_vertex_features = [num_vertex_features], &
         num_edge_features = [num_edge_features], &
         num_time_steps = num_time_steps, &
         max_vertex_degree = 4, &
         num_outputs = num_outputs, &
         kernel_initialiser = 'ones', &
         readout_activation_function = 'linear', &
         batch_size = batch_size &
    ))

    call duvenaud_network%compile( &
         optimiser = sgd_optimiser_type(learning_rate=0.1), &
         loss_method = 'mse', &
         accuracy_method = 'mse', &
         metrics = ['loss'], &
         batch_size = batch_size, &
         verbose = 1 &
    )

    ! Check network has correct number of layers (input + msgpass)
    if(duvenaud_network%num_layers .ne. 2) then
       success = .false.
       write(0,*) 'Duvenaud network has wrong number of layers:', &
            duvenaud_network%num_layers, 'expected: 2'
    end if

    ! Set up target output for testing
    allocate(target_array(1,1)%val(num_outputs, 1))
    target_array(1,1)%val = reshape([0.4, 0.4, 0.2], [num_outputs, 1])

    ! Test training for a few epochs
    call duvenaud_network%train( &
         graph_data_duvenaud, &
         target_array, &
         num_epochs = 5, &
         shuffle_batches = .false. &
    )

    ! Test prediction
    call duvenaud_network%test(graph_data_duvenaud, target_array)

    ! Check that network produces reasonable outputs
    if(duvenaud_network%accuracy < 0.0 .or. &
         duvenaud_network%accuracy > 1.0) then
       success = .false.
       write(0,*) 'Duvenaud network accuracy out of range:', &
            duvenaud_network%accuracy
    end if

    if(duvenaud_network%loss < 0.0) then
       success = .false.
       write(0,*) 'Duvenaud network loss is negative:', duvenaud_network%loss
    end if

    deallocate(target_array(1,1)%val)
  end block duvenaud_network_test


!-------------------------------------------------------------------------------
! Test graph network parameter management
!-------------------------------------------------------------------------------
  parameter_test: block
    integer :: num_params_kipf, num_params_duvenaud
    real, allocatable, dimension(:) :: params

    ! Test parameter counting
    num_params_kipf = kipf_network%get_num_params()
    if(num_params_kipf <= 0) then
       success = .false.
       write(0,*) 'Kipf network parameter count invalid:', num_params_kipf
    end if

    num_params_duvenaud = duvenaud_network%get_num_params()
    if(num_params_duvenaud <= 0) then
       success = .false.
       write(0,*) 'Duvenaud network parameter count invalid:', &
            num_params_duvenaud
    end if

    ! Test parameter retrieval
    params = kipf_network%get_params()
    if(.not. allocated(params) .or. size(params) .ne. num_params_kipf) then
       success = .false.
       write(0,*) 'Kipf network parameter retrieval failed'
    end if

    if(allocated(params)) deallocate(params)
    params = duvenaud_network%get_params()
    if(.not. allocated(params) .or. size(params) .ne. num_params_duvenaud) then
       success = .false.
       write(0,*) 'Duvenaud network parameter retrieval failed'
    end if
  end block parameter_test


!-------------------------------------------------------------------------------
! Test network reset functionality
!-------------------------------------------------------------------------------
  reset_test: block
    type(network_type) :: test_network

    call test_network%add(kipf_msgpass_layer_type( &
         num_vertex_features = [num_vertex_features, 8], &
         num_time_steps = 1, &
         batch_size = batch_size &
    ))
    call test_network%compile( &
         optimiser = base_optimiser_type(learning_rate=0.01), &
         loss_method = 'mse' &
    )

    ! Reset and check
    call test_network%reset()
    if(test_network%num_layers .ne. 0) then
       success = .false.
       write(0,*) 'Network reset failed - layers not cleared'
    end if
  end block reset_test


!-------------------------------------------------------------------------------
! Check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success) then
     write(*,*) 'test_msgpass_network passed all tests'
  else
     write(0,*) 'test_msgpass_network failed one or more tests'
     stop 1
  end if


contains

!-------------------------------------------------------------------------------
! Set up test graph with simple connected structure
!-------------------------------------------------------------------------------
  subroutine setup_test_graph(graph, allocate_edge_weights)
    type(graph_type), intent(inout) :: graph
    integer, parameter :: num_vertex_features_local = num_vertex_features
    integer, parameter :: num_edge_features_local = num_edge_features
    integer, allocatable :: index_list(:,:)
    logical, intent(in) :: allocate_edge_weights

    ! Initialize graph structure
    graph%is_sparse = .true.
    call graph%set_num_vertices(num_vertices, num_vertex_features_local)
    if(allocate_edge_weights)then
       call graph%set_num_edges(num_edges, num_edge_features_local)
    else
       call graph%set_num_edges(num_edges)  ! No edge features
    end if

    ! Set up vertex features (simple test pattern)
    graph%vertex_features(1,:) = [1.0, 2.0, 3.0, 4.0, 5.0]  ! Node IDs
    graph%vertex_features(2,:) = [0.1, 0.2, 0.3, 0.4, 0.5]  ! Feature 1
    graph%vertex_features(3,:) = [1.1, 1.2, 1.3, 1.4, 1.5]  ! Feature 2
    graph%vertex_features(4,:) = [0.5, 0.4, 0.3, 0.2, 0.1]  ! Feature 3
    graph%vertex_features(5,:) = [2.0, 1.8, 1.6, 1.4, 1.2]  ! Feature 4
    graph%vertex_features(6,:) = [0.0, 0.1, 0.2, 0.3, 0.4]  ! Feature 5
    graph%vertex_features(7,:) = [1.0, 0.9, 0.8, 0.7, 0.6]  ! Feature 6
    graph%vertex_features(8,:) = [0.2, 0.4, 0.6, 0.8, 1.0]  ! Feature 7

    ! Set up edge connectivity (creating a connected graph)
    allocate(index_list(2, num_edges))
    index_list(:, 1) = [1, 2]  ! Edge 1-2
    index_list(:, 2) = [1, 3]  ! Edge 1-3
    index_list(:, 3) = [2, 3]  ! Edge 2-3
    index_list(:, 4) = [2, 4]  ! Edge 2-4
    index_list(:, 5) = [3, 5]  ! Edge 3-5
    index_list(:, 6) = [4, 5]  ! Edge 4-5

    ! Generate adjacency matrix
    call graph%generate_adjacency(index_list)
    deallocate(index_list)

    ! Set edge weights and features
    graph%edge_weights = [1.0, 1.2, 0.8, 1.1, 0.9, 1.3]

    if(allocate_edge_weights) then
       graph%edge_features(1,:) = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  ! Edge feature 1
       graph%edge_features(2,:) = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]  ! Edge feature 2
    end if
  end subroutine setup_test_graph


end program test_msgpass_network
