program msgpass_euler_example
  !! Graph Neural Network for solving the Euler equations of fluid dynamics
  !!
  !! This example demonstrates using message passing neural networks to predict
  !! steady-state solutions of compressible inviscid flow over a 2D bump.
  !! This example has been developed with the help of Artan Qerushi, with them
  !! providing the dataset and physical problem description.
  !!
  !! ## Problem Description
  !!
  !! Solve the 2D Euler equations for compressible inviscid flow:
  !!
  !! $$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}}{\partial x} + \frac{\partial \mathbf{G}}{\partial y} = 0$$
  !!
  !! where the conserved variables are:
  !! $$\mathbf{U} = \begin{bmatrix} \rho \\ \rho u \\ \rho v \\ \rho E \end{bmatrix}$$
  !!
  !! and the flux terms are:
  !! $$\mathbf{F} = \begin{bmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ (\rho E + p)u \end{bmatrix}, \quad
  !!   \mathbf{G} = \begin{bmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ (\rho E + p)v \end{bmatrix}$$
  !!
  !! Here:
  !! - \( \rho \) is density
  !! - \( u, v \) are velocity components
  !! - \( p \) is pressure
  !! - \( E \) is total energy per unit mass
  !!
  !! The equation of state for an ideal gas relates these quantities:
  !! $$p = (\gamma - 1)\rho\left(E - \frac{1}{2}(u^2 + v^2)\right)$$
  !!
  !! where \( \gamma \) is the ratio of specific heats (typically 1.4 for air).
  !!
  !! ## Graph Neural Network Approach
  !!
  !! The computational domain is represented as a graph where:
  !! - **Nodes**: Spatial points in the flow field with features \( [x, y, \rho, u, v, p] \)
  !! - **Edges**: Connect neighboring points in the mesh
  !!
  !! The Kipf message passing layer aggregates information:
  !! $$\mathbf{h}_i^{(t+1)} = \sigma\left(\mathbf{W}\left(\mathbf{h}_i^{(t)} + \sum_{j \in \mathcal{N}(i)} \frac{\mathbf{h}_j^{(t)}}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}}\right)\right)$$
  !!
  !! ## Training Objective
  !!
  !! Learn the mapping from initial flow conditions to steady-state solution:
  !! $$\mathbf{U}_{\text{steady}} = f_{\theta}(\mathbf{U}_{\text{init}}, \text{geometry})$$
  !!
  !! This bypasses expensive iterative CFD solvers for prediction.
  use athena
  use coreutils, only: real32
  use read_euler, only: read_graph

  implicit none

  integer :: seed = 1
  !! Random seed for the initialisation of the network weights
  type(network_type) :: network
  class(base_layer_type), allocatable :: layer
  type(metric_dict_type), dimension(2) :: metric_dict
  class(clip_type), allocatable :: clip

  logical :: restart = .false.
  !! Boolean whether to restart the training from a saved network or not.
  logical :: normalise = .false.
  !! Boolean whether to normalise the input and output features or not.

  ! data loading and preprocessing
  type(graph_type), allocatable, dimension(:,:) :: &
       graphs_in, graphs_out, &
       graphs_in_expected, graphs_out_expected, &
       graphs_predicted
  character(1024) :: file, train_file

  ! training loop variables
  integer :: num_epochs = 200, batch_size = 2
  integer :: num_time_steps = 5
  integer :: num_samples
  integer :: i, n

  integer :: unit
  integer :: num_params
  integer :: v, s
  integer, dimension(:), allocatable :: sample_list
  real(real32), dimension(:), allocatable :: feature_in_norm, feature_out_norm

  character(len=1024) :: vertex_file, edge_file



  !-----------------------------------------------------------------------------
  ! read training dataset
  !-----------------------------------------------------------------------------
  write(edge_file, '(A,I0,A)') "example/msgpass_euler/data/bump_edgeData_1.txt"
  n = 2
  allocate(graphs_in(1,n))
  allocate(graphs_out(1,n))
  do i = 1, n
     write(vertex_file, '(A,I0,A)') &
          "example/msgpass_euler/data/bump_nodeData_in_", i, ".txt"
     write(*,*) "Reading training dataset ", i
     call read_graph(vertex_file, edge_file, graphs_in(1,i))
     write(vertex_file, '(A,I0,A)') &
          "example/msgpass_euler/data/bump_nodeData_out_", i, ".txt"
     call read_graph(vertex_file, edge_file, graphs_out(1,i))
  end do
  allocate(graphs_in_expected(1,1))
  allocate(graphs_out_expected(1,1))
  write(vertex_file, '(A)') &
       "example/msgpass_euler/data/bump_nodeData_in_expected.txt"
  write(*,*) "Reading training dataset expected"
  call read_graph(vertex_file, edge_file, graphs_in_expected(1,1))
  write(vertex_file, '(A)') &
       "example/msgpass_euler/data/bump_nodeData_out_expected.txt"
  call read_graph(vertex_file, edge_file, graphs_out_expected(1,1))
  write(*,*) "Reading finished"


  !-----------------------------------------------------------------------------
  ! normalise the input and output features
  !-----------------------------------------------------------------------------
  if(normalise)then
     write(*,*) "Normalising input features..."
     allocate(feature_in_norm(graphs_in(1,1)%num_vertex_features))
     feature_in_norm = 0._real32
     do i = 1, graphs_in(1,1)%num_vertex_features
        feature_in_norm(i) = &
             maxval(graphs_in(1,1)%vertex_features(i,:)) - &
             minval(graphs_in(1,1)%vertex_features(i,:))
        do s = 2, size(graphs_in,2), 1
           feature_in_norm(i) = &
                max( &
                     feature_in_norm(i), &
                     maxval(graphs_in(1,s)%vertex_features(i,:)) &
                ) - &
                min(feature_in_norm(i),minval(graphs_in(1,s)%vertex_features(i,:)))
        end do
        do s = 1, size(graphs_in,2)
           graphs_in(1,s)%vertex_features(i,:) = &
                graphs_in(1,s)%vertex_features(i,:) / feature_in_norm(i)
        end do
     end do

     write(*,*) "Normalising output features..."
     allocate(feature_out_norm(graphs_out(1,1)%num_vertex_features))
     feature_out_norm = 0._real32
     do i = 1, graphs_out(1,1)%num_vertex_features
        feature_out_norm(i) = &
             maxval(graphs_out(1,1)%vertex_features(i,:)) - &
             minval(graphs_out(1,1)%vertex_features(i,:))
        do s = 2, size(graphs_out,2), 1
           feature_out_norm(i) = &
                max( &
                     feature_out_norm(i), &
                     maxval(graphs_out(1,s)%vertex_features(i,:)) &
                ) - &
                min(feature_out_norm(i),minval(graphs_out(1,s)%vertex_features(i,:)))
        end do
        do s = 1, size(graphs_out,1)
           graphs_out(1,s)%vertex_features(i,:) = &
                graphs_out(1,s)%vertex_features(i,:) / feature_out_norm(i)
        end do
     end do
  else
     write(*,*) "Not normalising input and output features"
  end if

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
     write(6,*) "Initialising graph neural network..."
     ! add the initial message passing layer, this takes in the input features
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, & ! number of time steps for the message passing
               num_vertex_features = [ 3, 6 ], & ! input and output vertex feature sizes
               activation = 'softmax', & ! activation function of the layer
               kernel_initialiser = 'he_normal' & ! initialiser for the weights
          ) &
     )
     ! add the second message passing layer, this takes in the output of the first layer
     ! and the input features, and concatenates them
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 9, 14 ], &
               activation = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], & ! input list for the layer
          !  0 is the first layer
          ! -1 is the output of the previously added layer
          operator = 'concatenate' & ! operator to use for combining the inputs
     )
     ! all subsequent layers repeat the same process of the second layer,
     ! taking in the output of the previous layer and the input features
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 17, 32 ], &
               activation = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 35, 64 ], &
               activation = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 67, 32 ], &
               activation = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 35, 14 ], &
               activation = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 17, 7 ], &
               activation = 'swish', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
  end if


  !-----------------------------------------------------------------------------
  ! compile network
  !-----------------------------------------------------------------------------
  ! gradient clipping can be used to avoid exploding gradients
  allocate(clip, source=clip_type(-1.E0_real32, 1.E0_real32)) ! clip the gradients
  metric_dict%active = .false.
  metric_dict(1)%key = "loss"
  metric_dict(2)%key = "accuracy"
  metric_dict%threshold = 1.E-1_real32
  call network%compile( &
       optimiser = adam_optimiser_type( &                 ! use the Adam optimiser
            clip_dict = clip, &                           ! gradient clipping
            learning_rate = 2.E-2_real32, &               ! initial learning rate
            lr_decay = exp_lr_decay_type(1.E-3_real32) &  ! apply learning rate decay
       ), &
       loss_method = "mse", & ! mean squared error loss
       accuracy_method = "mse", & ! use mean squared error for accuracy calculation
       metrics = metric_dict, &
       batch_size = min(batch_size, size(graphs_in,2)), &
       verbose = 1 &
  )


  !-----------------------------------------------------------------------------
  ! print network and dataset summary
  !-----------------------------------------------------------------------------
  num_params = network%get_num_params()
  write(*,*) "NUMBER OF LAYERS",network%num_layers
  write(*,*) "Number of parameters", num_params
  write(*,*) "Number of samples",size(graphs_in,2)


  !-----------------------------------------------------------------------------
  ! training loop
  !-----------------------------------------------------------------------------
  call network%set_batch_size(min(batch_size, size(graphs_in,2)))
  call network%train( &
       graphs_in, &
       graphs_out, &
       num_epochs = num_epochs &
  )


  !-----------------------------------------------------------------------------
  ! testing loop
  !-----------------------------------------------------------------------------
  write(*,*) "Starting testing..."
  call network%test( &
       graphs_in, &
       graphs_out &
  )
  write(*,*) "Testing finished"

  write(*,'("Overall accuracy=",F0.5)') network%accuracy_val
  write(*,'("Overall loss=",F0.5)')     network%loss_val

  !-----------------------------------------------------------------------------
  ! predicting
  !-----------------------------------------------------------------------------
  graphs_predicted = network%predict( graphs_in_expected )

  open(newunit=unit, file="predicted_vs_expected.txt")
  do v = 1, graphs_out_expected(1,1)%num_vertices
     write(unit, *) v, graphs_out_expected(1,1)%vertex_features(:,v), &
          graphs_predicted(1,1)%vertex_features(:,v)
  end do
  close(unit)


  !-----------------------------------------------------------------------------
  ! print the learned network
  !-----------------------------------------------------------------------------
  if(.not.restart)then
     call network%print(file="network.txt")
  else
     call network%print(file="tmp.txt")
  end if

end program msgpass_euler_example
