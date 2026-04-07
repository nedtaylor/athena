program gno_regression
  !! Graph Neural Operator regression example
  !!
  !! Demonstrates the Graph Neural Operator (GNO) layer by learning
  !! a smooth function on a synthetic 1D graph.
  !!
  !! ## Problem
  !!
  !! Given a 1D chain graph with node coordinates \( x_i \in [0, 2\pi] \),
  !! learn the nonlinear mapping:
  !! $$f(h_i, \{x_j\}_{j \in \mathcal{N}(i)}) \;\to\;
  !!   \begin{bmatrix} \sin(x_i) \\ \cos(x_i) \end{bmatrix}$$
  !!
  !! The input feature at each node is simply \( h_i = 1 \) (constant),
  !! so the GNO must learn to infer the output purely from the
  !! coordinate differences between neighbours.
  !!
  !! ## Architecture
  !!
  !! - GNO layer: 1 input feature -> 8 output features (relu)
  !! - GNO layer: 8 input features -> 2 output features (none)
  !!
  !! The kernel MLP inside each GNO layer learns to produce a weight
  !! matrix from the relative coordinate \( x_i - x_j \).
  use athena
  use coreutils, only: real32
  implicit none

  integer, parameter :: num_vertices = 20
  integer, parameter :: coord_dim = 1
  integer, parameter :: F_in = 1
  integer, parameter :: F_hidden = 8
  integer, parameter :: F_out = 2
  integer, parameter :: num_edges = num_vertices - 1
  integer, parameter :: num_epochs = 200
  integer, parameter :: batch_size = 1

  real(real32), parameter :: pi = 4.0_real32 * atan(1.0_real32)
  real(real32), parameter :: lr = 0.01_real32

  type(network_type) :: network
  type(graph_type), dimension(1) :: graph
  type(graph_type), allocatable, dimension(:,:) :: &
       graphs_in, graphs_out

  real(real32) :: coords(coord_dim, num_vertices)
  real(real32) :: edge_coords(coord_dim, num_edges)
  real(real32) :: features(F_in, num_vertices)
  real(real32) :: targets(F_out, num_vertices)
  integer :: i
  integer, allocatable :: index_list(:,:)


  !-----------------------------------------------------------------------------
  ! Set random seed for reproducibility
  !-----------------------------------------------------------------------------
  call random_setup(42, restart=.false.)


  !-----------------------------------------------------------------------------
  ! Build a 1D chain graph
  !-----------------------------------------------------------------------------
  allocate(index_list(2, num_edges))
  do i = 1, num_edges
     index_list(:, i) = [i, i + 1]
  end do

  call graph(1)%set_num_vertices(num_vertices, F_in)
  call graph(1)%set_num_edges(num_edges)
  graph(1)%is_sparse = .true.

  allocate(graph(1)%vertex_features(F_in, num_vertices))
  graph(1)%vertex_features = 1.0_real32

  call graph(1)%generate_adjacency(index_list)
  deallocate(index_list)

  allocate(graph(1)%edge_weights(num_edges))
  graph(1)%edge_weights = 1.0_real32
  allocate(graph(1)%edge_features(1, num_edges), source=0.0_real32)


  !-----------------------------------------------------------------------------
  ! Generate coordinates and target values
  !   coords:  evenly spaced in [0, 2*pi]
  !   targets: [sin(x), cos(x)]
  !-----------------------------------------------------------------------------
  do i = 1, num_vertices
     coords(1, i) = real(i - 1, real32) / real(num_vertices - 1, real32) &
          * 2.0_real32 * pi
     features(:, i) = 1.0_real32
     targets(1, i) = sin(coords(1, i))
     targets(2, i) = cos(coords(1, i))
  end do


  !-----------------------------------------------------------------------------
  ! Prepare graph input/output for training
  !   The GNO layer expects input(1,:) = vertex features and
  !   input(2,:) = per-edge geometric features.
  !-----------------------------------------------------------------------------
  do i = 1, num_edges
     edge_coords(:, i) = coords(:, i) - coords(:, i + 1)
  end do

  allocate(graphs_in(1, batch_size))
  graphs_in(1, 1) = graph(1)
  graphs_in(1, 1)%num_vertex_features = F_in
  graphs_in(1, 1)%vertex_features = features
  graphs_in(1,1)%edge_features = edge_coords
  graphs_in(1, 1)%num_edge_features = coord_dim

  allocate(graphs_out(1, batch_size))
  graphs_out(1, 1) = graph(1)
  graphs_out(1, 1)%num_vertex_features = F_out
  graphs_out(1, 1)%vertex_features = targets


  !-----------------------------------------------------------------------------
  ! Build network
  !-----------------------------------------------------------------------------
  write(*,*) "Building GNO graph regression network..."
  call network%add(graph_nop_layer_type( &
       num_inputs=F_in, num_outputs=F_hidden, coord_dim=coord_dim, &
       kernel_hidden=8, activation="relu"))
  call network%add(graph_nop_layer_type( &
       num_outputs=F_out, coord_dim=coord_dim, &
       kernel_hidden=8))
  call network%compile( &
       optimiser=base_optimiser_type(learning_rate=lr), &
       loss_method="mse", accuracy_method="mse", &
       metrics=["loss"], &
       verbose=1)
  call network%set_batch_size(batch_size)

  write(*,*) "Number of parameters:", network%get_num_params()


  !-----------------------------------------------------------------------------
  ! Training loop
  !-----------------------------------------------------------------------------
  write(*,*) "Training..."
  call network%train( &
       graphs_in, &
       graphs_out, &
       num_epochs=num_epochs, &
       verbose=1 &
  )


  !-----------------------------------------------------------------------------
  ! Final results
  !-----------------------------------------------------------------------------
  write(*,*)
  write(*,'(A, F12.6)') " Final loss: ", network%loss_val
  write(*,*)
  write(*,*) "GNO graph regression example completed."

end program gno_regression
