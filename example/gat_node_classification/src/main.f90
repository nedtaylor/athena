program gat_node_classification
  !! Graph Attention Network (GAT) for node classification
  !!
  !! This example demonstrates using GAT layers to classify nodes in a
  !! synthetic graph. It creates a graph with 3 communities and trains a
  !! GAT network to predict the community label of each node.
  !!
  !! ## Problem Description
  !!
  !! Classify nodes in a graph into 3 classes based on graph structure
  !! and node features. Each node has 4 input features and the output
  !! is a 3-class one-hot encoding.
  !!
  !! ## GAT Architecture
  !!
  !! The network uses multi-head attention to learn which neighbors are
  !! most important for each node's classification:
  !!
  !! $$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_k]))}$$
  !!
  !! $$\mathbf{h}_i' = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j \right)$$
  use athena
  use coreutils, only: real32

  implicit none

  integer, parameter :: num_nodes = 30
  integer, parameter :: num_features_in = 4
  integer, parameter :: num_classes = 3
  integer, parameter :: num_heads = 2
  integer, parameter :: hidden_dim = 8   ! per head, so total = 16

  integer :: seed = 42
  type(network_type) :: network
  type(metric_dict_type), dimension(2) :: metric_dict
  class(clip_type), allocatable :: clip

  type(graph_type), allocatable, dimension(:,:) :: graphs_in, graphs_out
  type(graph_type), allocatable, dimension(:,:) :: graphs_predicted

  integer :: i, j, v, s, e_idx, num_edges_total
  integer :: num_epochs, batch_size, num_params
  integer :: class_id, num_intra, num_inter
  integer :: nodes_per_class
  real(real32) :: r
  real(real32), dimension(num_features_in) :: feat

  ! Edge tracking
  integer, parameter :: max_edges = 300
  integer :: edge_pairs(2, max_edges)
  real(real32) :: edge_feats(1, max_edges)

  type(vertex_type), dimension(num_nodes) :: vertices
  type(edge_type), allocatable, dimension(:) :: edges


  write(*,*) "=== GAT Node Classification Example ==="
  write(*,*) ""
  write(*,*) "Creating synthetic graph with", num_nodes, "nodes and", &
       num_classes, "classes"

  !-----------------------------------------------------------------------------
  ! Create synthetic graph with community structure
  !-----------------------------------------------------------------------------
  ! Nodes 1-10: class 1, 11-20: class 2, 21-30: class 3
  nodes_per_class = num_nodes / num_classes

  ! Create node features correlated with class
  call random_setup(seed, restart=.false.)
  do v = 1, num_nodes
     class_id = (v - 1) / nodes_per_class + 1
     ! Features are noisy class indicators
     feat = 0.1_real32
     feat(class_id) = 0.8_real32
     ! Add small random feature
     call random_number(r)
     feat(num_features_in) = r * 0.5_real32
     allocate(vertices(v)%feature(num_features_in))
     vertices(v)%feature = feat
     vertices(v)%id = v
  end do

  ! Create edges: dense intra-community, sparse inter-community
  num_edges_total = 0

  ! Intra-community edges (high connectivity within communities)
  do class_id = 1, num_classes
     do i = (class_id-1)*nodes_per_class + 1, class_id*nodes_per_class
        do j = i+1, class_id*nodes_per_class
           call random_number(r)
           if(r < 0.6_real32) then  ! 60% intra-community connectivity
              num_edges_total = num_edges_total + 1
              if(num_edges_total > max_edges) then
                 write(*,*) "Warning: too many edges, truncating"
                 num_edges_total = num_edges_total - 1
                 goto 100
              end if
              edge_pairs(1, num_edges_total) = i
              edge_pairs(2, num_edges_total) = j
              edge_feats(1, num_edges_total) = 1.0_real32
           end if
        end do
     end do
  end do

  ! Inter-community edges (sparse)
  do i = 1, num_nodes
     do j = i+1, num_nodes
        if((i-1)/nodes_per_class == (j-1)/nodes_per_class) cycle ! skip same class
        call random_number(r)
        if(r < 0.05_real32) then  ! 5% inter-community connectivity
           num_edges_total = num_edges_total + 1
           if(num_edges_total > max_edges) then
              num_edges_total = num_edges_total - 1
              goto 100
           end if
           edge_pairs(1, num_edges_total) = i
           edge_pairs(2, num_edges_total) = j
           edge_feats(1, num_edges_total) = 1.0_real32
        end if
     end do
  end do

100 continue

  write(*,'("  Created ",I0," edges")') num_edges_total

  ! Build edge array
  allocate(edges(num_edges_total))
  do e_idx = 1, num_edges_total
     edges(e_idx)%index = edge_pairs(:, e_idx)
     edges(e_idx)%weight = 1.0_real32
     allocate(edges(e_idx)%feature(1))
     edges(e_idx)%feature(1) = edge_feats(1, e_idx)
  end do


  !-----------------------------------------------------------------------------
  ! Create input and output graphs
  !-----------------------------------------------------------------------------
  ! We train on a single graph (transductive setting)
  allocate(graphs_in(1, 1))
  allocate(graphs_out(1, 1))

  ! Build input graph with community structure
  graphs_in(1,1) = graph_type( &
       vertex = vertices, &
       edge = edges, &
       is_sparse = .false. &
  )
  ! Add self-loops (standard for GAT)
  call graphs_in(1,1)%add_self_loops()
  ! Convert to sparse CSR format
  call graphs_in(1,1)%convert_to_sparse()

  write(*,'("  Input graph: ",I0," vertices, ",I0," edges (with self-loops)")') &
       graphs_in(1,1)%num_vertices, graphs_in(1,1)%num_edges

  ! Build output graph with class labels as vertex features
  ! Output: one-hot encoding of class labels
  ! Copy graph structure from input
  graphs_out(1,1)%is_sparse = graphs_in(1,1)%is_sparse
  graphs_out(1,1)%directed = graphs_in(1,1)%directed
  graphs_out(1,1)%has_self_loops = graphs_in(1,1)%has_self_loops
  graphs_out(1,1)%num_vertices = graphs_in(1,1)%num_vertices
  graphs_out(1,1)%num_edges = graphs_in(1,1)%num_edges
  graphs_out(1,1)%num_edge_features = graphs_in(1,1)%num_edge_features
  graphs_out(1,1)%num_vertex_features = num_classes
  if(allocated(graphs_in(1,1)%adj_ia)) then
     graphs_out(1,1)%adj_ia = graphs_in(1,1)%adj_ia
  end if
  if(allocated(graphs_in(1,1)%adj_ja)) then
     graphs_out(1,1)%adj_ja = graphs_in(1,1)%adj_ja
  end if
  if(allocated(graphs_in(1,1)%edge_features)) then
     graphs_out(1,1)%edge_features = graphs_in(1,1)%edge_features
  end if
  if(allocated(graphs_in(1,1)%edge_weights)) then
     graphs_out(1,1)%edge_weights = graphs_in(1,1)%edge_weights
  end if
  allocate(graphs_out(1,1)%vertex_features(num_classes, num_nodes))
  graphs_out(1,1)%vertex_features = 0._real32
  do v = 1, num_nodes
     class_id = (v - 1) / nodes_per_class + 1
     graphs_out(1,1)%vertex_features(class_id, v) = 1._real32
  end do


  !-----------------------------------------------------------------------------
  ! Build GAT Network
  !-----------------------------------------------------------------------------
  write(*,*) ""
  write(*,*) "Building GAT network..."
  write(*,'("  Architecture: ",I0," -> ",I0," (x",I0," heads) -> ",I0)') &
       num_features_in, hidden_dim, num_heads, num_classes

  ! Layer 1: GAT with multi-head attention (concatenate heads)
  ! Input: 4 features, Output: hidden_dim * num_heads = 16 features
  call network%add( &
       gat_msgpass_layer_type( &
            num_time_steps = 1, &
            num_vertex_features = [ num_features_in, hidden_dim * num_heads ], &
            num_heads = num_heads, &
            concat_heads = .true., &
            negative_slope = 0.2, &
            activation = 'relu', &
            kernel_initialiser = 'glorot_uniform' &
       ) &
  )

  ! Layer 2: GAT with single head (averaging for final output)
  ! Input: hidden_dim * num_heads = 16, Output: num_classes = 3
  call network%add( &
       gat_msgpass_layer_type( &
            num_time_steps = 1, &
            num_vertex_features = [ hidden_dim * num_heads, num_classes ], &
            num_heads = 1, &
            concat_heads = .false., &
            negative_slope = 0.2, &
            activation = 'softmax', &
            kernel_initialiser = 'glorot_uniform' &
       ) &
  )


  !-----------------------------------------------------------------------------
  ! Compile and train
  !-----------------------------------------------------------------------------
  allocate(clip, source=clip_type(-1.0_real32, 1.0_real32))
  metric_dict%active = .false.
  metric_dict(1)%key = "loss"
  metric_dict(2)%key = "accuracy"
  metric_dict%threshold = 1.E-1_real32
  num_epochs = 100
  batch_size = 1

  call network%compile( &
       optimiser = adam_optimiser_type( &
            clip_dict = clip, &
            learning_rate = 5.E-3_real32 &
       ), &
       loss_method = "mse", &
       accuracy_method = "mse", &
       metrics = metric_dict, &
       batch_size = batch_size, &
       verbose = 1 &
  )

  num_params = network%get_num_params()
  write(*,'("  Number of layers: ",I0)') network%num_layers
  write(*,'("  Number of parameters: ",I0)') num_params
  write(*,*) ""

  !-----------------------------------------------------------------------------
  ! Training
  !-----------------------------------------------------------------------------
  write(*,*) "Training..."
  call network%set_batch_size(batch_size)
  call network%train( &
       graphs_in, &
       graphs_out, &
       num_epochs = num_epochs &
  )


  !-----------------------------------------------------------------------------
  ! Testing
  !-----------------------------------------------------------------------------
  write(*,*) ""
  write(*,*) "Testing..."
  call network%test(graphs_in, graphs_out)
  write(*,'("  Final loss: ",F0.6)') network%loss_val
  write(*,'("  Final accuracy: ",F0.6)') network%accuracy_val


  !-----------------------------------------------------------------------------
  ! Prediction and evaluation
  !-----------------------------------------------------------------------------
  graphs_predicted = network%predict(graphs_in)

  write(*,*) ""
  write(*,*) "Node classification results:"
  write(*,'("  Node | True | Pred | Correct")')
  write(*,'("  -----+------+------+--------")')

  j = 0  ! correct count
  do v = 1, num_nodes
     class_id = (v - 1) / nodes_per_class + 1
     i = maxloc(graphs_predicted(1,1)%vertex_features(:,v), dim=1)
     if(i == class_id) j = j + 1
     if(v <= 10 .or. mod(v, 10) == 0) then
        write(*,'("  ",I4," | ",I4," | ",I4," | ",L1)') &
             v, class_id, i, (i == class_id)
     end if
  end do

  write(*,*) ""
  write(*,'("  Classification accuracy: ",I0,"/",I0," = ",F0.1,"%")') &
       j, num_nodes, 100.0 * real(j) / real(num_nodes)
  write(*,*) ""
  write(*,*) "=== Done ==="

end program gat_node_classification
