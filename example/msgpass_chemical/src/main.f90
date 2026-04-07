program msgpass_chemical_example
  !! Message Passing Neural Network for molecular property prediction
  !!
  !! This example demonstrates using graph neural networks (GNNs) to predict
  !! molecular properties from chemical structure graphs.
  !!
  !! ## Problem Description
  !!
  !! Given a molecular graph \( G = (V, E) \) where:
  !! - \( V \) represents atoms with features (element type, coordinates, etc.)
  !! - \( E \) represents chemical bonds
  !!
  !! Predict scalar molecular properties (e.g., energy, formation enthalpy).
  !!
  !! ## Message Passing Framework
  !!
  !! The network iteratively updates node (atom) representations by aggregating
  !! information from neighboring nodes:
  !!
  !! 1. **Message function**: Compute messages from neighbors
  !!    $$\mathbf{m}_{ij}^{(t)} = M(\mathbf{h}_i^{(t)}, \mathbf{h}_j^{(t)}, \mathbf{e}_{ij})$$
  !!
  !! 2. **Aggregation**: Combine messages from all neighbors
  !!    $$\mathbf{a}_i^{(t)} = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(t)}$$
  !!
  !! 3. **Update**: Compute new node representation
  !!    $$\mathbf{h}_i^{(t+1)} = U(\mathbf{h}_i^{(t)}, \mathbf{a}_i^{(t)})$$
  !!
  !! 4. **Readout**: Aggregate node features to graph-level prediction
  !!    $$\mathbf{y} = R\left(\{\mathbf{h}_i^{(T)} | i \in V\}\right)$$
  !!
  !! ## Duvenaud Message Passing
  !!
  !! This example uses the Duvenaud et al. architecture, which computes:
  !! $$\mathbf{h}_i^{(t+1)} = \sigma\left(\mathbf{W}^{(t)}_{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(t)} + \mathbf{b}^{(t)}\right)$$
  !!
  !! where different weight matrices are learned for different vertex degrees.
  !!
  !! ## Data Format
  !!
  !! Input: Extended XYZ files containing molecular structures and energies
  use athena
  use coreutils, only: real32
  use read_chemical_graphs, only: read_extxyz_db

  implicit none

  integer :: seed = 42
  type(network_type) :: network
  class(base_layer_type), allocatable :: layer
  type(metric_dict_type), dimension(2) :: metric_dict
  class(clip_type), allocatable :: clip

  logical :: restart = .false.
  logical :: do_training = .true.

  ! data loading and preprocessing
  type(graph_type), allocatable, dimension(:,:) :: graphs_in
  real(real32), allocatable, dimension(:,:) :: labels
  character(1024) :: file, train_file, onnx_file
  integer :: unit

  ! training loop variables
  integer :: num_tests = 10, num_epochs = 20, batch_size = 8
  integer :: num_time_steps = 4
  integer :: i, j, s

  integer :: num_dense_inputs = 10, num_outputs = 1
  integer :: num_params
  integer, dimension(:), allocatable :: sample_list
  real(real32), dimension(:), allocatable :: feature_in_norm
  type(array_type), dimension(1,1) :: output
  real(real32) :: output_min, output_max

  class(*), allocatable, dimension(:,:) :: data_poly


  !-----------------------------------------------------------------------------
  ! read training dataset
  !-----------------------------------------------------------------------------
  train_file = "example/msgpass_chemical/database.xyz"
  onnx_file = "example/msgpass_chemical/model.json"
  write(*,*) "Reading training dataset..."
  call read_extxyz_db(train_file, graphs_in, output)!labels)
  write(*,*) "Reading finished"
  do s = 1, size(graphs_in)
     call graphs_in(1,s)%add_self_loops()
     if(.not.graphs_in(1,s)%is_sparse) call graphs_in(1,s)%convert_to_sparse()
  end do


  !-----------------------------------------------------------------------------
  ! initialise random seed
  !-----------------------------------------------------------------------------
  call random_setup(seed, restart=.false.)


  !-----------------------------------------------------------------------------
  ! initialise convolutional and pooling layers
  !-----------------------------------------------------------------------------
  if(restart)then
     write(*,*) "Reading network from file..."
     call network%read(file="network.txt")
     write(*,*) "Reading finished"
  else
     write(6,*) "Initialising MSGPASS..."

     call network%add(duvenaud_msgpass_layer_type( &
          num_time_steps = num_time_steps, &
          num_vertex_features = [ graphs_in(1,1)%num_vertex_features ], &
          num_edge_features =   [ graphs_in(1,1)%num_edge_features ], &
          num_outputs = num_dense_inputs, &
          kernel_initialiser = 'glorot_normal', &
          readout_activation = 'softmax', &
          min_vertex_degree = 1, &
          max_vertex_degree = 10 &
     ))
     call network%add(full_layer_type( &
          num_inputs  = num_dense_inputs, &
          num_outputs = 128, &
          activation = 'leaky_relu', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
     call network%add(full_layer_type( &
          num_outputs = 64, &
          activation = 'leaky_relu', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
     call network%add(full_layer_type( &
          num_outputs = num_outputs, &
          activation = 'leaky_relu', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
  end if

  ! normalise the input features
  allocate(feature_in_norm(graphs_in(1,1)%num_vertex_features))
  feature_in_norm = 0._real32
  do i = 1, graphs_in(1,1)%num_vertex_features
     do s = 1, size(graphs_in,2)
        feature_in_norm(i) = &
             max(feature_in_norm(i),maxval(graphs_in(1,s)%vertex_features(i,:))) - &
             min(feature_in_norm(i),minval(graphs_in(1,s)%vertex_features(i,:)))
     end do
  end do


  !-----------------------------------------------------------------------------
  ! compile network
  !-----------------------------------------------------------------------------
  ! allocate(clip, source=clip_type(-1.E0_real32, 1.E0_real32))
  allocate(clip, source=clip_type(clip_norm = 1.E-1_real32))
  metric_dict%active = .false.
  metric_dict(1)%key = "loss"
  metric_dict(2)%key = "accuracy"
  metric_dict%threshold = 1.E-1_real32
  call network%compile( &
       optimiser = adam_optimiser_type( &
            clip_dict = clip, &
            learning_rate = 1.E-2_real32 &
            ! lr_decay = exp_lr_decay_type(1.E-2_real32) &
            ! lr_decay = step_lr_decay_type(0.5_real32, 5) &
       ), &
       loss_method = "mse", &
       metrics = metric_dict, &
       batch_size = batch_size, verbose = 1, &
       accuracy_method = "mse" &
  )


  !-----------------------------------------------------------------------------
  ! print network and dataset summary
  !-----------------------------------------------------------------------------
  num_params = network%get_num_params()
  write(*,*) "Number of layers:", network%num_layers
  write(*,*) "Number of parameters:", num_params
  write(*,*) "Number of samples:", size(output(1,1)%val,2)
  write(*,*) "Number of tests:", num_tests


  !-----------------------------------------------------------------------------
  ! export initialised network parameters to ONNX for cross-language validation
  !-----------------------------------------------------------------------------
  write(*,*) "Writing ONNX file: ", trim(onnx_file)
  call write_onnx(onnx_file, network)
  write(*,*) "ONNX export complete"


  !-----------------------------------------------------------------------------
  ! forward pass on first sample with initial weights (matches ONNX export)
  !-----------------------------------------------------------------------------
  call network%set_batch_size(1)
  call network%set_inference_mode()
  call network%forward(graphs_in(:,1:1))
  ! Write output of leaf layer for cross-validation
  open(newunit=unit, file="example/msgpass_chemical/fortran_output.txt", &
       status='replace')
  write(unit, '(*(ES20.12,1X))') network%model( &
       network%auto_graph%vertex(network%leaf_vertices(1))%id &
  )%layer%output(1,1)%val
  close(unit)
  write(*,*) "Fortran forward pass output written"


  !-----------------------------------------------------------------------------
  ! ONNX round-trip test: read back and compare forward pass results
  !-----------------------------------------------------------------------------
  call onnx_import_test(onnx_file)
  call network%set_batch_size(batch_size)


!-----------------------------------------------------------------------------
! export ALL graphs for Python cross-validation
!-----------------------------------------------------------------------------
  call export_all_graphs("example/msgpass_chemical/all_graphs.txt")


  !-----------------------------------------------------------------------------
  ! forward pass on first sample for cross-language comparison
  !-----------------------------------------------------------------------------
  call network%set_batch_size(1)
  output_min = minval(output(1,1)%val)
  output_max = maxval(output(1,1)%val)
  write(*,'(A,ES20.12)') "output_min = ", output_min
  write(*,'(A,ES20.12)') "output_max = ", output_max
  write(*,'(A,ES20.12)') "output_range = ", output_max - output_min
  output(1,1)%val = ( output(1,1)%val - output_min ) / &
       ( output_max - output_min )


  !-----------------------------------------------------------------------------
  ! training loop
  !-----------------------------------------------------------------------------
  call network%set_batch_size(batch_size)
  call network%train( &
       graphs_in, &
       output, &
       num_epochs = num_epochs, &
       shuffle_batches = .true. &
  )


  !--------------------------------------------------------------------------
  ! testing loop
  !--------------------------------------------------------------------------
  write(*,*) "Starting testing..."
  call network%test( &
       graphs_in, &
       output &
  )
  write(*,*) "Testing finished"
  write(6,'("Overall accuracy=",F0.5)') network%accuracy_val
  write(6,'("Overall loss=",F0.5)')     network%loss_val

  if(.not.restart)then
     call network%print(file="network.txt")
  else
     call network%print(file="tmp.txt")
  end if

contains

!###############################################################################
  subroutine onnx_import_test(file)
    !! Test ONNX import by writing the network to an ONNX file, reading it back,
    !! and comparing the forward pass results on the same input.
    implicit none

    ! Arguments
    character(*), intent(in) :: file
    !! File to import the network from

    ! Local variables
    type(network_type) :: imported_network
    !! Imported network reconstructed from the ONNX file
    real(real32) :: orig_output, imported_output, rel_diff
    !! Original output, imported output, and relative difference

    write(*,*) "Testing ONNX round-trip: reading back ", trim(file)
    imported_network = read_onnx(file, verbose=1)

    ! Compile the imported network
    call imported_network%compile( &
         optimiser = adam_optimiser_type( &
              learning_rate = 1.E-2_real32 &
         ), &
         loss_method = "mse", &
         metrics = metric_dict, &
         batch_size = 1, verbose = 1, &
         accuracy_method = "mse" &
    )

    ! Set up for inference
    call imported_network%set_batch_size(1)
    call imported_network%set_inference_mode()

    ! Forward pass on same input
    call imported_network%forward(graphs_in(:,1:1))
    call network%print_summary()
    call imported_network%print_summary()
    write(*,*) imported_network%leaf_vertices(1)
    write(*,*) imported_network%auto_graph%vertex(imported_network%leaf_vertices(1))%id

    ! Compare outputs
    orig_output = network%model( &
         network%auto_graph%vertex(network%leaf_vertices(1))%id &
    )%layer%output(1,1)%val(1,1)
    imported_output = imported_network%model( &
         imported_network%auto_graph%vertex( &
              imported_network%leaf_vertices(1))%id &
    )%layer%output(1,1)%val(1,1)

    write(*,'(A,ES20.12)') " Original output:  ", orig_output
    write(*,'(A,ES20.12)') " Imported output:  ", imported_output
    rel_diff = abs(orig_output - imported_output) / &
         (abs(orig_output) + 1.E-30_real32)
    write(*,'(A,ES20.12)') " Relative diff:    ", rel_diff

    if(rel_diff .lt. 1.E-5_real32)then
       write(*,*) "ONNX round-trip test PASSED"
    else
       write(*,*) "ONNX round-trip test FAILED"
    end if
  end subroutine onnx_import_test
!###############################################################################


!###############################################################################
  subroutine export_all_graphs(file)
    !! Export all input graphs to a text file for cross-language validation
    implicit none

    ! Arguments
    character(*), intent(in) :: file
    !! File to write the graph data to

    ! Local variables
    integer :: j, s, unit
    !! Loop indices and file unit

    write(*,*) "Exporting all graph data for cross-validation..."
    open(newunit=unit, file=file, status='replace')
    write(unit, '(I0)') size(graphs_in,2)
    do s = 1, size(graphs_in,2)
       ! Header: num_vertices num_edges num_csr_entries num_vertex_features num_edge_features
       write(unit, '(I0,1X,I0,1X,I0,1X,I0,1X,I0)') &
            graphs_in(1,s)%num_vertices, &
            graphs_in(1,s)%num_edges, &
            size(graphs_in(1,s)%adj_ja, 2), &
            graphs_in(1,s)%num_vertex_features, &
            graphs_in(1,s)%num_edge_features
       ! Vertex features: one row per vertex
       do j = 1, graphs_in(1,s)%num_vertices
          write(unit, '(*(ES16.8E2,1X))') graphs_in(1,s)%vertex_features(:,j)
       end do
       ! Edge features: one row per edge
       do j = 1, graphs_in(1,s)%num_edges
          write(unit, '(*(ES16.8E2,1X))') graphs_in(1,s)%edge_features(:,j)
       end do
       ! CSR adj_ia (num_vertices + 1 values)
       write(unit, '(*(I0,1X))') graphs_in(1,s)%adj_ia
       ! CSR adj_ja (2 x num_csr_entries)
       do j = 1, size(graphs_in(1,s)%adj_ja, 2)
          write(unit, '(I0,1X,I0)') graphs_in(1,s)%adj_ja(1,j), &
               graphs_in(1,s)%adj_ja(2,j)
       end do
       ! Write energy label for this sample
       write(unit, '(ES16.8E2)') output(1,1)%val(1,s)
    end do
    close(unit)
    write(*,*) "Graph export complete"
  end subroutine export_all_graphs
!###############################################################################

end program msgpass_chemical_example
