program onnx_gnn_example
  !! ONNX round-trip test for a Kipf-based GNN architecture.
  !!
  !! This example builds a multi-layer Kipf message passing network followed by
  !! dense heads, exports it to ONNX JSON, reads it back with read_onnx, and
  !! verifies that both networks produce matching outputs on the same graph.
  use athena
  use coreutils, only: real32
  use read_chemical_graphs, only: read_extxyz_db

  implicit none

  integer :: i
  !! Loop index
  integer :: seed
  !! Random seed for reproducible initialisation
  type(network_type) :: network, reloaded_network
  !! Original network and reloaded ONNX network
  type(metric_dict_type), dimension(2) :: metric_dict
  !! Metric configuration passed to compile
  class(clip_type), allocatable :: clip
  !! Gradient clipping configuration

  type(graph_type), allocatable, dimension(:,:) :: graphs_in
  !! Input graph batch
  type(array_type), dimension(1,1) :: output
  !! Target output array expected by read_extxyz_db

  real(real32) :: orig_output, imported_output, rel_diff
  !! Output values and relative difference for round-trip validation
  character(1024) :: train_file, onnx_file
  !! Dataset path and ONNX JSON path


  !-----------------------------------------------------------------------------
  ! Read graph data
  !-----------------------------------------------------------------------------
  train_file = "example/msgpass_chemical/database.xyz"
  onnx_file = "example/onnx_gnn/model.json"
  write(*,*) "Reading dataset..."
  call read_extxyz_db(train_file, graphs_in, output)
  do i = 1, size(graphs_in)
     call graphs_in(1,i)%add_self_loops()
     if(.not.graphs_in(1,i)%is_sparse) call graphs_in(1,i)%convert_to_sparse()
  end do


  !-----------------------------------------------------------------------------
  ! Build a different GNN architecture (Kipf + dense heads)
  !-----------------------------------------------------------------------------
  seed = 123
  call random_setup(seed, restart=.false.)

  call network%add(kipf_msgpass_layer_type( &
       num_time_steps = 1, &
       num_vertex_features = [graphs_in(1,1)%num_vertex_features, 16], &
       activation = 'relu', &
       kernel_initialiser = 'he_normal' &
  ))

  call network%add(kipf_msgpass_layer_type( &
       num_time_steps = 1, &
       num_vertex_features = [16, 8], &
       activation = 'swish', &
       kernel_initialiser = 'he_normal' &
  ))

  call network%add(kipf_msgpass_layer_type( &
       num_time_steps = 1, &
       num_vertex_features = [8, 4], &
       activation = 'tanh', &
       kernel_initialiser = 'he_normal' &
  ))


  !-----------------------------------------------------------------------------
  ! Compile and run reference forward pass
  !-----------------------------------------------------------------------------
  allocate(clip, source=clip_type(clip_norm = 1.E-1_real32))
  metric_dict%active = .false.
  metric_dict(1)%key = "loss"
  metric_dict(2)%key = "accuracy"
  metric_dict%threshold = 1.E-1_real32

  call network%compile( &
       optimiser = adam_optimiser_type( &
            clip_dict = clip, &
            learning_rate = 1.E-2_real32 &
       ), &
       loss_method = "mse", &
       metrics = metric_dict, &
       batch_size = 1, verbose = 1, &
       accuracy_method = "mse" &
  )

  call network%set_batch_size(1)
  call network%set_inference_mode()

  call network%forward(graphs_in(:,1:1))
  orig_output = network%model( &
       network%auto_graph%vertex(network%leaf_vertices(1))%id &
  )%layer%output(1,1)%val(1,1)


  !-----------------------------------------------------------------------------
  ! ONNX export / import round-trip
  !-----------------------------------------------------------------------------
  write(*,*) "Writing ONNX file: ", trim(onnx_file)
  call write_onnx(onnx_file, network)

  write(*,*) "Reading ONNX file..."
  reloaded_network = read_onnx(onnx_file, verbose=2)

  call reloaded_network%compile( &
       optimiser = adam_optimiser_type(learning_rate = 1.E-2_real32), &
       loss_method = "mse", &
       metrics = metric_dict, &
       batch_size = 1, verbose = 1, &
       accuracy_method = "mse" &
  )

  call reloaded_network%set_batch_size(1)
  call reloaded_network%set_inference_mode()

  call reloaded_network%forward(graphs_in(:,1:1))
  imported_output = reloaded_network%model( &
       reloaded_network%auto_graph%vertex( &
            reloaded_network%leaf_vertices(1))%id &
  )%layer%output(1,1)%val(1,1)

  rel_diff = abs(orig_output - imported_output) / &
       (abs(orig_output) + 1.E-30_real32)

  write(*,'(A,ES20.12)') "Original output:  ", orig_output
  write(*,'(A,ES20.12)') "Imported output:  ", imported_output
  write(*,'(A,ES20.12)') "Relative diff:    ", rel_diff

  if(rel_diff .lt. 1.E-5_real32)then
     write(*,*) "Kipf ONNX round-trip test PASSED"
  else
     write(*,*) "Kipf ONNX round-trip test FAILED"
     error stop 1
  end if

end program onnx_gnn_example
