program mnist_example
  !! DropBlock regularization demonstration on MNIST digit classification
  !!
  !! This example demonstrates DropBlock, a structured form of dropout designed
  !! for convolutional neural networks. Unlike standard dropout which drops
  !! individual activations, DropBlock drops contiguous regions, which is more
  !! effective for CNNs where nearby activations are correlated.
  !!
  !! ## DropBlock Algorithm
  !!
  !! For a feature map \( \mathbf{X} \in \mathbb{R}^{H \times W \times C} \):
  !!
  !! 1. **Sample drop mask**: For each spatial location, randomly decide whether
  !!    to be a block center with probability related to drop rate
  !!
  !! 2. **Create blocks**: Expand each drop center to a block of size \( b \times b \)
  !!    $$M_{i,j,c} = 0 \text{ if } (i,j) \text{ is within any block}$$
  !!
  !! 3. **Apply mask**: Zero out blocked regions
  !!    $$\mathbf{Y} = \frac{\mathbf{X} \odot M}{\text{keep_prob}}$$
  !!
  !! The normalization factor ensures the expected sum is preserved.
  !!
  !! ## Advantages over Dropout
  !!
  !! - **Spatial correlation**: Drops connected regions, forcing the network
  !!   to learn from less spatially complete information
  !! - **Better for CNNs**: More effective than random dropout for convolutional layers
  !! - **Improved generalization**: Reduces overfitting in image classification
  !!
  !! ## Reference
  !!
  !! Ghiasi et al., "DropBlock: A regularization method for convolutional networks," NeurIPS 2018
#ifdef _OPENMP
  use omp_lib
#endif
  use athena
  use constants_mnist, only: real32
  use mnist_example_utils, only: limit_mnist_dataset
  use read_mnist, only: read_mnist_db
  use inputs
  use mnist_drop_runtime_config, only: initialise_training_state

  implicit none

  type(network_type) :: network
  class(base_optimiser_type), allocatable :: optimiser
  type(metric_dict_type), dimension(2) :: metric_dict

  ! data loading and preoprocessing
  real(real32), allocatable, dimension(:,:,:,:) :: input_images, test_images
  integer, allocatable, dimension(:) :: labels, test_labels
  real(real32), allocatable, dimension(:,:) :: input_labels
  character(1024) :: train_file, test_file

  ! neural network size and shape variables
  integer, parameter :: num_classes = 10    ! Number of output classes
  logical :: limit_dataset = .false.
  integer, parameter :: max_train_samples = 512
  integer, parameter :: max_test_samples = 128
  integer, parameter :: max_epochs = 1
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
  call initialise_training_state( &
       "example/mnist_drop/test_job.in", optimiser, metric_dict)
#ifdef _OPENMP
  write(*,*) "number of threads:", num_threads
  call omp_set_num_threads(num_threads)
#endif


  !-----------------------------------------------------------------------------
  ! read training dataset
  !-----------------------------------------------------------------------------
  train_file = trim(data_dir)//'/MNIST_train.txt'
  call read_mnist_db(train_file,input_images, labels, &
       maxval(cv_kernel_size), image_size, "none") !padding_method)
  input_channels = size(input_images, 3)
  num_samples = size(input_images, 4)


  !-----------------------------------------------------------------------------
  ! read testing dataset
  !-----------------------------------------------------------------------------
  test_file = trim(data_dir)//'/MNIST_test.txt'
  call read_mnist_db(test_file,test_images, test_labels, &
       maxval(cv_kernel_size), itmp1, "none") !padding_method)
  num_samples_test = size(test_images, 4)

  if(limit_dataset)then
     call limit_mnist_dataset(input_images, labels, max_train_samples)
     num_samples = size(input_images, 4)
     call limit_mnist_dataset(test_images, test_labels, max_test_samples)
     num_samples_test = size(test_images, 4)
     num_epochs = min(num_epochs, max_epochs)
  end if


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
          activation = "relu" &
     ))
     call network%add(dropblock2d_layer_type( &
          rate = 0.25, block_size = 5))
     call network%add(maxpool2d_layer_type(&
          pool_size = 2, stride = 2))
     call network%add(full_layer_type( &
          num_outputs = 100, &
          activation = "relu", &
          kernel_initialiser = "he_uniform", &
          bias_initialiser = "he_uniform" &
     ))
     call network%add(full_layer_type( &
          num_outputs = 10,&
          activation = "softmax", &
          kernel_initialiser = "glorot_uniform", &
          bias_initialiser = "glorot_uniform" &
     ))
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
  allocate(input_labels(num_classes,num_samples), source = 0._real32)
  do i=1,num_samples
     input_labels(labels(i),i) = 1._real32
  end do

  write(6,*) "Starting training..."
  call network%train(input_images, input_labels, num_epochs, batch_size, &
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

  write(*,*) "Starting testing..."
  call network%test(test_images,input_labels)
  write(*,*) "Testing finished"
  write(6,'("Overall accuracy=",F0.5)') network%accuracy_val
  write(6,'("Overall loss=",F0.5)')     network%loss_val

end program mnist_example
