program pinn_chemical_example
  !! Program to demonstrate the use of a message passing neural network
  !!
  !! This program reads a dataset of chemical graphs in the XYZ format
  !! and trains a message passing neural network to predict the energy of the
  !! chemical graph.
  use athena
  use coreutils, only: real32
  use forces_loss, only: forces_loss_type
  use read_chemical_graphs_extd, only: read_extxyz_db

  implicit none

  integer :: seed = 42
  type(network_type) :: network
  class(base_layer_type), allocatable :: layer
  type(metric_dict_type), dimension(2) :: metric_dict
  class(clip_type), allocatable :: clip

  logical :: restart = .false.

  ! data loading and preprocessing
  type(graph_type), allocatable, dimension(:,:) :: graphs_in
  real(real32), allocatable, dimension(:,:) :: labels
  character(1024) :: file, train_file

  ! training loop variables
  integer :: num_tests = 10, num_epochs = 100, batch_size = 8
  integer :: num_time_steps = 4
  integer :: i, n, s

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
          readout_activation_function = 'softmax', &
          min_vertex_degree = 1, &
          max_vertex_degree = 10, &
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
          num_outputs = num_outputs, &
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
     do s = 1, size(graphs_in,2)
        feature_in_norm(i) = &
             max(feature_in_norm(i),maxval(graphs_in(1,s)%vertex_features(i,:))) - &
             min(feature_in_norm(i),minval(graphs_in(1,s)%vertex_features(i,:)))
     end do
     do s = 1, size(graphs_in,2)
        ! graphs_in(1,s)%vertex_features(i,:) = &
        !      graphs_in(1,s)%vertex_features(i,:) / feature_in_norm(i)
        write(14,*) graphs_in(1,s)%edge_features(:,:)
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
       loss_method = forces_loss_type(), &
       metrics = metric_dict, &
       batch_size = batch_size, verbose = 1, &
       accuracy_method = "mse" &
  )


  !-----------------------------------------------------------------------------
  ! print network and dataset summary
  !-----------------------------------------------------------------------------
  num_params = network%get_num_params()
  write(*,*) "NUMBER OF LAYERS",network%num_layers
  write(*,*) "Number of parameters", num_params
  write(*,*) "Number of samples",size(output(1,1)%val,2)
  write(*,*) "Number of tests",num_tests


  !-----------------------------------------------------------------------------
  ! training loop
  !-----------------------------------------------------------------------------
  call network%set_batch_size(batch_size)
  output_min = minval(output(1,1)%val)
  output_max = maxval(output(1,1)%val)
  output(1,1)%val = ( output(1,1)%val - output_min ) / &
       ( output_max - output_min )
  call network%train( &
       graphs_in, &
       output, &
       num_epochs = num_epochs, &
       shuffle_batches = .true. &
  )
  write(*,*) "autodifferentiation"
  write(*,*) network%model(network%root_vertices(1))%layer%output(1,1)%grad%val(:,1)


  !-----------------------------------------------------------------------------
  ! testing loop
  !-----------------------------------------------------------------------------
  write(*,*) "Starting testing..."
  call network%test( &
       graphs_in, &
       output &
  )
  write(*,*) "Testing finished"

  data_poly = network%predict_generic( graphs_in, output_as_graph = .false.)
  select type(data_poly)
  type is(array_type)
     write(*,*) "Predicted output:"
     write(*,*) data_poly(1,1)%val * ( output_max - output_min ) + output_min
     write(*,*) output(1,1)%val * ( output_max - output_min ) + output_min
  end select

  write(6,'("Overall accuracy=",F0.5)') network%accuracy_val
  write(6,'("Overall loss=",F0.5)')     network%loss_val

  if(.not.restart)then
     call network%print(file="network.txt")
  else
     call network%print(file="tmp.txt")
  end if

end program pinn_chemical_example
