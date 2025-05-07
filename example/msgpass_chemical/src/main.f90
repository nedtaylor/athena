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
  type(graph_type), allocatable, dimension(:,:) :: graphs
  real(real32), allocatable, dimension(:) :: labels
  character(1024) :: file, train_file

  !! training loop variables
  integer :: num_tests = 10, num_epochs = 500, batch_size = 4
  integer :: i, itmp1, n, num_iterations
  real(real32), dimension(:,:), allocatable :: output_tmp, output

  integer :: num_dense_inputs = 10, num_outputs = 1
  integer :: num_params
  integer :: v, s
  integer, dimension(:), allocatable :: sample_list
  type(array2d_type), dimension(1,1) :: labels_tmp



!!!-----------------------------------------------------------------------------
!!! read training dataset
!!!-----------------------------------------------------------------------------
  train_file = "example/msgpass_chemical/database.xyz"
  write(*,*) "Reading training dataset..."
  call read_extxyz_db(train_file, graphs, labels)
  write(*,*) "Reading finished"
  do s = 1, size(graphs)
     if(.not.graphs(s,1)%is_sparse) call graphs(s,1)%convert_to_sparse()
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
          num_time_steps = 4, &
          num_features = [ &
               graphs(1,1)%num_vertex_features, &
               graphs(1,1)%num_edge_features &
          ], &
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
            learning_rate = 1.E-3_real32 &
       ), &
       loss_method = "mse", metrics = metric_dict, &
       batch_size = batch_size, verbose = 1 &
  )


!!!-----------------------------------------------------------------------------
!!! print network and dataset summary
!!!-----------------------------------------------------------------------------
  num_params = network%get_num_params()
  write(*,*) "NUMBER OF LAYERS",network%num_layers
  write(*,*) "Number of parameters", num_params
  write(*,*) "Number of samples",size(labels)
  write(*,*) "Number of tests",num_tests


!!!-----------------------------------------------------------------------------
!!! training loop
!!! ... loops over num_epoch number of epochs
!!! ... i.e. it trains on the same datapoints num_epoch times
!!!-----------------------------------------------------------------------------
  call network%set_batch_size(batch_size)
  labels = -1.E0 * labels
  labels = labels / maxval(labels)
  ! allocate(output(1,size(labels)))
  do n = 1, num_epochs
     if(mod(n,10) == 0) write(*,*) "Epoch", n
     sample_list = [ &
          (i, i = 1, size(labels) - num_tests - batch_size + 1, batch_size) &
     ]
     call shuffle(sample_list)
     do s = 1, size(sample_list), 1
        call network%forward_graph( &
             reshape( &
                  graphs(sample_list(s):sample_list(s)+batch_size-1,1), &
                  [batch_size,1] &
             ) &
        )
        call labels_tmp(1,1)%allocate(array_shape=[num_outputs,batch_size])
        call labels_tmp(1,1)%set( &
             reshape( &
                  labels(sample_list(s):sample_list(s)+batch_size-1), &
                  [num_outputs,batch_size] &
             ) &
        )
        call network%backward_mixed(labels_tmp)
        if(labels_tmp(1,1)%allocated) call labels_tmp(1,1)%deallocate()
        call network%update()
     end do
  end do


!!!-----------------------------------------------------------------------------
!!! testing loop
!!!-----------------------------------------------------------------------------
  allocate(output_tmp(num_outputs,batch_size))
  write(*,*) "Starting testing..."
  write(*,*) "Testing on", num_tests, "samples", size(labels), size(graphs)
  do s = 1, size(labels) - batch_size + 1, batch_size
     call network%forward_graph( &
          reshape( &
               graphs(s:s+batch_size-1,1), [batch_size,1] &
          ) &
     )
     call network%model(network%output_vertices(1))%layer%get_output(output_tmp)
     write(*,*) "imputed", output_tmp(1,:)
     write(*,*) "expected", labels(s:s+batch_size-1)
     write(*,*)
  end do
  write(*,*) "Testing finished"


  ! write(6,'("Overall accuracy=",F0.5)') network%accuracy
  ! write(6,'("Overall loss=",F0.5)')     network%loss

end program mnist_example
!!!#############################################################################
