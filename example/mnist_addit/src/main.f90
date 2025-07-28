program mnist_example
  !! Program to demonstrate the use of multi-input convolutional neural networks
  !!
  !! This program reads the MNIST dataset of handwritten digits and trains a
  !! convolutional neural network to classify the digits. An additional set of
  !! inputs is used to demonstrate how to use multiple inputs in a neural network.
#ifdef _OPENMP
  use omp_lib
#endif
  use athena
  use constants_mnist, only: real32
  use read_mnist, only: read_mnist_db
  use inputs

  implicit none

  type(network_type) :: network

  ! data loading and preoprocessing
  real(real32), allocatable, dimension(:,:,:,:) :: input_images, test_images
  class(array_container_type), allocatable, dimension(:) :: &
       input_container, test_container
  integer, allocatable, dimension(:) :: labels, test_labels
  integer, allocatable, dimension(:,:) :: input_labels
  character(1024) :: train_file, test_file

  ! neural network size and shape variables
  integer, parameter :: num_classes = 10    ! Number of output classes
  integer :: image_size
  integer :: input_channels

  ! training loop variables
  integer :: num_samples, num_samples_test


  integer :: i, itmp1

#ifdef _OPENMP
  integer, allocatable, dimension(:) :: label_slice
  real(real32), allocatable, dimension(:,:,:,:) :: image_slice
#endif


  !-----------------------------------------------------------------------------
  ! initialise global variables
  !-----------------------------------------------------------------------------
  call set_global_vars(param_file="example/mnist/test_job.in")
#ifdef _OPENMP
  write(*,*) "number of threads:", num_threads
  call omp_set_num_threads(num_threads)
#endif


  !-----------------------------------------------------------------------------
  ! read training dataset
  !-----------------------------------------------------------------------------
  train_file = trim(data_dir)//'/MNIST_train.txt'
  call read_mnist_db(train_file,input_images, labels, &
       maxval(cv_kernel_size), image_size, padding_method)
  input_channels = size(input_images, 3)
  num_samples = size(input_images, 4)


  !-----------------------------------------------------------------------------
  ! read testing dataset
  !-----------------------------------------------------------------------------
  test_file = trim(data_dir)//'/MNIST_test.txt'
  call read_mnist_db(test_file,test_images, test_labels, &
       maxval(cv_kernel_size), itmp1, padding_method)
  num_samples_test = size(test_images, 4)


  !-----------------------------------------------------------------------------
  ! initialise random seed
  !-----------------------------------------------------------------------------
  call random_setup(seed, restart=.false.)


  !-----------------------------------------------------------------------------
  ! shuffle dataset
  !-----------------------------------------------------------------------------
  if(shuffle_dataset)then
     write(6,*) "Shuffling training dataset..."
     call shuffle(input_images, labels, 4, seed)
     write(6,*) "Training dataset shuffled"
     if(verbosity.eq.-1)then
        write(6,*) "Check fort.11 and fort.12 to ensure data shuffling &
             &executed properly"
        do i=1,batch_size*2
           write(11,*) input_images(:,:,:,i)
        end do
        write(12,*) labels
     end if
  end if


  !-----------------------------------------------------------------------------
  ! initialise convolutional and pooling layers
  !-----------------------------------------------------------------------------
  if(restart)then
     write(*,*) "Reading network from file..."
     call network%read(file=input_file)
     write(*,*) "Reading finished"
  else
     write(6,*) "Initialising CNN..."

     call network%add(conv2d_layer_type( &
          input_shape = [image_size,image_size,input_channels], &
          num_filters = cv_num_filters, kernel_size = 3, stride = 1, &
          padding=padding_method, &
          calc_input_gradients = .false., &
          activation_function = "relu" &
     ))
     call network%add(maxpool2d_layer_type(&
          pool_size = 2, stride = 2))
     call network%add(input_layer_type(input_shape=[2], index=2))
     call network%add(full_layer_type( &
          num_outputs = 100, &
          activation_function = "relu", &
          kernel_initialiser = "he_uniform", &
          bias_initialiser = "he_uniform" &
     ), input_list = [2, 3], operator="||")
     call network%add(input_layer_type(input_shape=[100], index=3))
     call network%add(full_layer_type( &
          num_outputs = 10,&
          activation_function = "softmax", &
          kernel_initialiser = "glorot_uniform", &
          bias_initialiser = "glorot_uniform" &
     ), input_list = [4, 5], operator="+")
  end if

  call network%compile(optimiser=optimiser, &
       loss_method=loss_method, accuracy_method=accuracy_method, &
       metrics=metric_dict, &
       batch_size = batch_size, verbose = verbosity)

  write(*,*) "NUMBER OF LAYERS",network%num_layers


  !-----------------------------------------------------------------------------
  ! training loop
  ! ... loops over num_epoch number of epochs
  ! ... i.e. it trains on the same datapoints num_epoch times
  !-----------------------------------------------------------------------------
  allocate(input_labels(num_classes,num_samples))
  input_labels = 0
  do i=1,num_samples
     input_labels(labels(i),i) = 1
  end do

  allocate(input_container(3))
  allocate(input_container(1)%array, source = array4d_type())
  allocate(input_container(2)%array, source = array2d_type())
  allocate(input_container(3)%array, source = array2d_type())
  call input_container(1)%array%allocate(source = input_images)
  call input_container(2)%array%allocate(array_shape=[2,num_samples], &
       source = 1._real32)
  call input_container(3)%array%allocate(array_shape=[100,num_samples], &
       source = 1._real32)

  write(6,*) "Starting training..."
  call network%train(input_container, input_labels, num_epochs, batch_size, &
       plateau_threshold = plateau_threshold, &
       shuffle_batches = shuffle_dataset, &
       batch_print_step = batch_print_step, verbose = verbosity)


  !-----------------------------------------------------------------------------
  ! print weights and biases of CNN to file
  !-----------------------------------------------------------------------------
  write(*,*) "Writing network to file..."
  call network%print(file=output_file)
  write(*,*) "Writing finished"


  !-----------------------------------------------------------------------------
  ! testing loop
  !-----------------------------------------------------------------------------
  deallocate(input_labels)
  allocate(input_labels(num_classes,num_samples_test))
  input_labels = 0
  do i=1,num_samples_test
     input_labels(test_labels(i),i) = 1
  end do
  allocate(test_container(3))
  allocate(test_container(1)%array, source = array4d_type())
  allocate(test_container(2)%array, source = array2d_type())
  allocate(test_container(3)%array, source = array2d_type())
  call test_container(1)%array%allocate(source = test_images)
  call test_container(2)%array%allocate(array_shape=[2,num_samples_test], &
       source = 1._real32)
  call test_container(3)%array%allocate(array_shape=[100,num_samples_test], &
       source = 1._real32)

  write(*,*) "Starting testing..."
  call network%test(test_container,input_labels)
  write(*,*) "Testing finished"
  write(6,'("Overall accuracy=",F0.5)') network%accuracy_val
  write(6,'("Overall loss=",F0.5)')     network%loss_val

end program mnist_example
  !#############################################################################
