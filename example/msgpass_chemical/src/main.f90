!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
program mnist_example
  use athena
  use constants_mnist, only: real32
  use read_chemical_graphs, only: read_extxyz_db

  implicit none

  integer :: seed = 1
  type(network_type) :: network
  class(base_layer_type), allocatable :: layer
  type(metric_dict_type), dimension(2) :: metric_dict
  class(clip_type), allocatable :: clip

  logical :: restart = .false.

  !! data loading and preoprocessing
  type(graph_type), allocatable, dimension(:,:) :: graphs_in
  real(real32), allocatable, dimension(:,:) :: labels
  character(1024) :: file, train_file

  !! training loop variables
  integer :: num_tests = 10, num_epochs = 100, batch_size = 4
  integer :: num_time_steps = 4
  integer :: i, itmp1, n, num_iterations

  integer :: num_dense_inputs = 10, num_outputs = 1
  integer :: num_params
  integer :: v, s
  integer, dimension(:), allocatable :: sample_list
  real(real32), dimension(:), allocatable :: feature_in_norm
  type(array2d_type), dimension(1,1) :: output



!!!-----------------------------------------------------------------------------
!!! read training dataset
!!!-----------------------------------------------------------------------------
  train_file = "example/msgpass_chemical/database.xyz"
  write(*,*) "Reading training dataset..."
  call read_extxyz_db(train_file, graphs_in, output)!labels)
  write(*,*) "Reading finished"
  do s = 1, size(graphs_in)
     if(.not.graphs_in(1,s)%is_sparse) call graphs_in(1,s)%convert_to_sparse()
  end do


!!!-----------------------------------------------------------------------------
!!! initialise random seed
!!!-----------------------------------------------------------------------------
  call random_setup(seed, restart=.false.)


!!!-----------------------------------------------------------------------------
!!! initialise convolutional and pooling layers
!!!-----------------------------------------------------------------------------
  if(restart)then
     ! write(*,*) "Reading network from file..."
     ! call network%read(file=input_file)
     ! write(*,*) "Reading finished"
  else
     write(6,*) "Initialising MSGPASS..."

     call network%add(duvenaud_msgpass_layer_type( &
          num_time_steps = num_time_steps, &
          num_vertex_features = [ graphs_in(1,1)%num_vertex_features ], &
          num_edge_features =   [ graphs_in(1,1)%num_edge_features ], &
          num_outputs = num_dense_inputs, &
          kernel_initialiser = 'he_normal', &
          max_vertex_degree = 6, &
          batch_size = batch_size &
     ))
     call network%add(full_layer_type( &
          num_inputs  = num_dense_inputs, &
          num_outputs = 128, &
          batch_size  = batch_size, &
          activation_function = 'leaky_relu', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
     call network%add(full_layer_type( &
          num_outputs = 64, &
          batch_size  = batch_size, &
          activation_function = 'leaky_relu', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
     call network%add(full_layer_type( &
          num_outputs = 1, &
          batch_size  = batch_size, &
          activation_function = 'leaky_relu', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
  end if

  ! normalise the input features
  allocate(feature_in_norm(graphs_in(1,1)%num_vertex_features))
  feature_in_norm = 0._real32
  do i = 1, graphs_in(1,1)%num_vertex_features
     do s = 1, size(graphs_in,1)
        feature_in_norm(i) = &
             max(feature_in_norm(i),maxval(graphs_in(s,1)%vertex_features(i,:))) - &
             min(feature_in_norm(i),minval(graphs_in(s,1)%vertex_features(i,:)))
     end do
     do s = 1, size(graphs_in,1)
        graphs_in(s,1)%vertex_features(i,:) = &
             graphs_in(s,1)%vertex_features(i,:) / feature_in_norm(i)
     end do
  end do


!!!-----------------------------------------------------------------------------
!!! compile network
!!!-----------------------------------------------------------------------------
  allocate(clip, source=clip_type(clip_norm = 1.E1_real32))
  metric_dict%active = .false.
  metric_dict(1)%key = "loss"
  metric_dict(2)%key = "accuracy"
  metric_dict%threshold = 1.E-1_real32
  call network%compile( &
       optimiser = sgd_optimiser_type( &
            clip_dict = clip, &
            learning_rate = 1.E-2_real32 &
       ), &
       loss_method = "mse", metrics = metric_dict, &
       batch_size = batch_size, verbose = 1, &
       accuracy_method = "mse" &
  )


!!!-----------------------------------------------------------------------------
!!! print network and dataset summary
!!!-----------------------------------------------------------------------------
  num_params = network%get_num_params()
  write(*,*) "NUMBER OF LAYERS",network%num_layers
  write(*,*) "Number of parameters", num_params
  write(*,*) "Number of samples",size(output(1,1)%val,2)
  write(*,*) "Number of tests",num_tests


!!!-----------------------------------------------------------------------------
!!! training loop
!!! ... loops over num_epoch number of epochs
!!! ... i.e. it trains on the same datapoints num_epoch times
!!!-----------------------------------------------------------------------------
  call network%set_batch_size(batch_size)
  output(1,1)%val = -1._real32 * output(1,1)%val
  output(1,1)%val = output(1,1)%val / maxval(output(1,1)%val)
  call network%train( &
       graphs_in, &
       output, &
       num_epochs = num_epochs &
  )


!!!-----------------------------------------------------------------------------
!!! testing loop
!!!-----------------------------------------------------------------------------
  write(*,*) "Starting testing..."
  call network%test( &
       graphs_in, &
       labels &
  )
  write(*,*) "Testing finished"


  write(6,'("Overall accuracy=",F0.5)') network%accuracy
  write(6,'("Overall loss=",F0.5)')     network%loss

  if(.not.restart)then
     call network%print(file="network.txt")
  else
     call network%print(file="tmp.txt")
  end if

end program mnist_example
!!!#############################################################################
