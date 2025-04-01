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
  integer :: num_samples, num_tests
  integer :: i, itmp1, n, num_iterations
  real(real32), dimension(:,:), allocatable :: output_tmp, output

  integer :: v, s
  integer, dimension(:), allocatable :: sample_list



!!!-----------------------------------------------------------------------------
!!! read training dataset
!!!-----------------------------------------------------------------------------
  train_file = "example/mpnn_chemical/database.xyz"
  write(*,*) "Reading training dataset..."
  call read_extxyz_db(train_file, graphs, labels)
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

     call network%add(kipf_msgpass_layer_type( &
          num_time_steps=4, &
          num_features=[2,1] &
     ))
     !  call network%add(full_layer_type( &
     !       num_inputs  = 10, &
     !       num_outputs = 1, &
     !       batch_size  = 1, &
     !       activation_function='leaky_relu', &
     !       kernel_initialiser='he_normal', &
     !       bias_initialiser='ones' &
     !  ))
  end if


!!!-----------------------------------------------------------------------------
!!! compile network
!!!-----------------------------------------------------------------------------
  allocate(clip, source=clip_type(-1.E1_real32, 1.E1_real32))
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
       batch_size = 1, verbose = 1 &
  )


!!!-----------------------------------------------------------------------------
!!! print network and dataset summary
!!!-----------------------------------------------------------------------------
  num_tests = 10
  write(*,*) "NUMBER OF LAYERS",network%num_layers
  write(*,*) "Number of samples",size(labels)
  write(*,*) "Number of tests",num_tests


!!!-----------------------------------------------------------------------------
!!! training loop
!!! ... loops over num_epoch number of epochs
!!! ... i.e. it trains on the same datapoints num_epoch times
!!!-----------------------------------------------------------------------------
  call network%set_batch_size(1)
  !labels = -1.E0 * labels
  !labels = labels / maxval(labels)
  allocate(output_tmp(1,1))
  allocate(output(1,size(labels)))
  do n = 1, 100
     sample_list = [(i, i = 1, size(labels) - num_tests)]
     call shuffle(sample_list)
     do s = 1, size(sample_list)

        ! write(*,*) n, s
        call network%forward_graph(graphs(s:s,:))
        call network%backward_graph(graphs(s:s,:)) ! reshape([labels(sample_list(s))], [1,1]))
        ! write(*,*) "predicted",network%model(2)%layer%output%val, labels(sample_list(s))

        call network%update()

     end do
  end do


! !!!-----------------------------------------------------------------------------
! !!! testing loop
! !!!-----------------------------------------------------------------------------
!   write(*,*) "Starting testing..."
!   do s = size(labels) - num_tests + 1, size(labels)
!      call network%forward(graph)
!      call network%model(2)%layer%get_output(output_tmp)
!      write(*,*) "predicted",output_tmp(1,1), labels(s)
!   end do
!   write(*,*) "Testing finished"


  ! write(6,'("Overall accuracy=",F0.5)') network%accuracy
  ! write(6,'("Overall loss=",F0.5)')     network%loss

end program mnist_example
!!!#############################################################################
