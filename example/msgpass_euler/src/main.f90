program msgpass_euler_example
  !! Program to demonstrate the use of a message passing neural network
  !!
  !! This program reads a dataset of graphs and trains a message passing neural
  !! network to predict the output features of the graphs.
  !! The dataset is read from text files that give the initial setup of a graph
  !! of points and the neural network is trained to predict the steady state
  !! solution of the flow of a fluid over a bump.
  use athena
  use constants_mnist, only: real32
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

  ! data loading and preoprocessing
  type(graph_type), allocatable, dimension(:,:) :: &
       graphs_in, graphs_out, graphs_predicted
  character(1024) :: file, train_file

  ! training loop variables
  integer :: num_tests = 10, num_epochs = 200, batch_size = 2
  integer :: num_time_steps = 5
  integer :: num_samples
  integer :: i, n

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
               activation_function = 'softmax', & ! activation function of the layer
               kernel_initialiser = 'he_normal' & ! initialiser for the weights
          ) &
     )
     ! add the second message passing layer, this takes in the output of the first layer
     ! and the input features, and concatenates them
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 9, 14 ], &
               activation_function = 'softmax', &
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
               activation_function = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 35, 64 ], &
               activation_function = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 67, 32 ], &
               activation_function = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 35, 14 ], &
               activation_function = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 17, 7 ], &
               activation_function = 'swish', &
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
       batch_size = batch_size, &
       verbose = 1 &
  )


  !-----------------------------------------------------------------------------
  ! print network and dataset summary
  !-----------------------------------------------------------------------------
  num_params = network%get_num_params()
  write(*,*) "NUMBER OF LAYERS",network%num_layers
  write(*,*) "Number of parameters", num_params
  write(*,*) "Number of samples",size(graphs_in,2)
  write(*,*) "Number of tests",num_tests


  !-----------------------------------------------------------------------------
  ! training loop
  !-----------------------------------------------------------------------------
  call network%set_batch_size(batch_size)
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

  write(*,'("Overall accuracy=",F0.5)') network%accuracy
  write(*,'("Overall loss=",F0.5)')     network%loss

  !-----------------------------------------------------------------------------
  ! predicting
  !-----------------------------------------------------------------------------
  graphs_predicted = network%predict( graphs_in )


  !-----------------------------------------------------------------------------
  ! print the learned network
  !-----------------------------------------------------------------------------
  if(.not.restart)then
     call network%print(file="network.txt")
  else
     call network%print(file="tmp.txt")
  end if

end program msgpass_euler_example
