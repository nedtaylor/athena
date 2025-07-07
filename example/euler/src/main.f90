!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
program mnist_example
  use athena
  use constants_mnist, only: real32
  use read_euler, only: read_graph

  implicit none

  integer :: seed = 1
  type(network_type) :: network
  class(base_layer_type), allocatable :: layer
  type(metric_dict_type), dimension(2) :: metric_dict
  class(clip_type), allocatable :: clip

  logical :: restart = .false.

  !! data loading and preoprocessing
  type(graph_type), allocatable, dimension(:,:) :: &
       graphs_in, graphs_out, graphs_predicted
  type(array2d_type), allocatable, dimension(:,:) :: features_out
  character(1024) :: file, train_file

  !! training loop variables
  integer :: num_tests = 10, num_epochs = 200, batch_size = 4
  integer :: num_time_steps = 5
  integer :: num_samples
  integer :: i, itmp1, n, num_iterations
  real(real32), dimension(:,:), allocatable :: output_tmp, output

  integer :: num_params
  integer :: v, s
  integer, dimension(:), allocatable :: sample_list
  real(real32), dimension(:), allocatable :: feature_in_norm, feature_out_norm

  character(len=1024) :: vertex_file, edge_file



!!!-----------------------------------------------------------------------------
!!! read training dataset
!!!-----------------------------------------------------------------------------
  write(edge_file, '(A,I0,A)') "example/euler/data/bump_edgeData_1.txt"
  n = 12
  allocate(graphs_in(1,n))
  allocate(graphs_out(1,n))
  do i = 1, n
     write(vertex_file, '(A,I0,A)') "example/euler/data/bump_nodeData_in_", i, ".txt"
     write(*,*) "Reading training dataset ", i
     call read_graph(vertex_file, edge_file, graphs_in(1,i))
     write(vertex_file, '(A,I0,A)') "example/euler/data/bump_nodeData_out_", i, ".txt"
     call read_graph(vertex_file, edge_file, graphs_out(1,i))
  end do
  write(*,*) "Reading finished"


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
     write(6,*) "Initialising MPNN..."
     ! call network%add(kipf_msgpass_layer_type( &
     !      num_time_steps = num_time_steps, &
     !      num_vertex_features = [ 3, 6, 12, 24, 14, 7 ], &
     !      num_edge_features = [ 0 ], &
     !      activation_function = 'softmax', &
     !      kernel_initialiser = 'he_normal' &
     ! ))

     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 3, 6 ], &
               num_edge_features = [ 0 ], &
               activation_function = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ) &
     )
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 8, 14 ], &
               num_edge_features = [ 0 ], &
               activation_function = 'softmax', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
     call network%add( &
          kipf_msgpass_layer_type( &
               num_time_steps = 1, &
               num_vertex_features = [ 17, 32 ], &
               num_edge_features = [ 0 ], &
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
               num_edge_features = [ 0 ], &
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
               num_edge_features = [ 0 ], &
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
               num_edge_features = [ 0 ], &
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
               num_edge_features = [ 0 ], &
               activation_function = 'swish', &
               kernel_initialiser = 'he_normal' &
          ), &
          input_list = [ 0, -1 ], &
          operator = 'concatenate' &
     )
     !  call network%add(full_layer_type( &
     !       num_inputs  = 10, &
     !       num_outputs = 1, &
     !       batch_size  = 1, &
     !       activation_function='leaky_relu', &
     !       kernel_initialiser='he_normal', &
     !       bias_initialiser='ones' &
     !  ))
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

  ! normalise the output features
  allocate(feature_out_norm(graphs_out(1,1)%num_vertex_features))
  feature_out_norm = 0._real32
  do i = 1, graphs_out(1,1)%num_vertex_features
     do s = 1, size(graphs_out,1)
        feature_out_norm(i) = &
             max(feature_out_norm(i),maxval(graphs_out(s,1)%vertex_features(i,:))) - &
             min(feature_out_norm(i),minval(graphs_out(s,1)%vertex_features(i,:)))
     end do
     do s = 1, size(graphs_out,1)
        graphs_out(s,1)%vertex_features(i,:) = &
             graphs_out(s,1)%vertex_features(i,:) / feature_out_norm(i)
     end do
  end do
  open(14, file="fort.14", status="replace")
  do i = 1, size(graphs_out(1,1)%vertex_features,dim=2)
     write(14,*) graphs_out(1,1)%vertex_features(:,i)
  end do
  close(14)


!!!-----------------------------------------------------------------------------
!!! compile network
!!!-----------------------------------------------------------------------------
  allocate(clip, source=clip_type(-1.E0_real32, 1.E0_real32))
  metric_dict%active = .false.
  metric_dict(1)%key = "loss"
  metric_dict(2)%key = "accuracy"
  metric_dict%threshold = 1.E-1_real32
  call network%compile( &
       optimiser = adam_optimiser_type( &
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
  write(*,*) "Number of samples",size(graphs_in,2)
  write(*,*) "Number of tests",num_tests


!!!-----------------------------------------------------------------------------
!!! training loop
!!! ... loops over num_epoch number of epochs
!!! ... i.e. it trains on the same datapoints num_epoch times
!!!-----------------------------------------------------------------------------
  call network%set_batch_size(batch_size)
  !labels = -1.E0 * labels
  !labels = labels / maxval(labels)
  ! allocate(output_tmp(1,1))
  ! allocate(output(1,size(labels)))
  ! output(1,:) = labels
  call network%train( &
       graphs_in, &
       graphs_out, &
       num_epochs = num_epochs &
  )


!!!-----------------------------------------------------------------------------
!!! testing loop
!!!-----------------------------------------------------------------------------
  write(*,*) "Starting testing..."
  call network%test( &
       graphs_in, &
       graphs_out &
  )
  write(*,*) "Testing finished"

  write(*,'("Overall accuracy=",F0.5)') network%accuracy
  write(*,'("Overall loss=",F0.5)')     network%loss

!!!-----------------------------------------------------------------------------
!!! predicting
!!!-----------------------------------------------------------------------------
  graphs_predicted = network%predict( graphs_in )
  open(15, file="fort.15", status="replace")
  do i = 1, size(graphs_predicted(1,1)%vertex_features,dim=2)
     write(15,*) graphs_predicted(1,1)%vertex_features(:,i)
  end do
  close(15)

end program mnist_example
!!!#############################################################################
