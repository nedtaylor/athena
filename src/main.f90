!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
program ConvolutionalNeuralNetwork
#ifdef _OPENMP
  use omp_lib
#endif
  use constants, only: real12
  use random, only: random_setup
  use misc, only: shuffle
  use infile_tools, only: stop_check

  use loss_categorical
  use inputs

  use network, only: network_type

  use container_layer, only: container_layer_type
  use input3d_layer,   only: input3d_layer_type
  use input4d_layer,   only: input4d_layer_type
  use full_layer,      only: full_layer_type
  use conv2d_layer,    only: conv2d_layer_type
  use conv3d_layer,    only: conv3d_layer_type
  use maxpool2d_layer, only: maxpool2d_layer_type
  use maxpool3d_layer, only: maxpool3d_layer_type
  use flatten2d_layer, only: flatten2d_layer_type
  use flatten3d_layer, only: flatten3d_layer_type


  implicit none

  type(network_type) :: network

  !! seed variables
  integer :: nseed=1
  integer, allocatable, dimension(:) :: seed_arr

  !! data loading and preoprocessing
  real(real12), allocatable, dimension(:,:,:,:) :: input_images, test_images
  real(real12), allocatable, dimension(:,:,:,:,:) :: input_spread
  integer, allocatable, dimension(:) :: labels, test_labels
  integer, allocatable, dimension(:,:) :: input_labels
  character(1024) :: train_file, test_file

  !! neural network size and shape variables
  integer, parameter :: num_classes = 10    ! Number of output classes
  integer :: image_size
  integer :: input_channels

  !! training loop variables
  integer :: num_batches, num_samples, num_samples_test
  integer :: epoch, batch, sample, start_index, end_index
  integer, allocatable, dimension(:) :: batch_order
  !real(real12), allocatable, dimension(:,:,:) :: bn_output, bn_gradients


  integer :: i, l, time, time_old, clock_rate, itmp1, cv_mask_size
  real(real12) :: rtmp1
  !  real(real12) :: drop_gamma
  !  real(real12), allocatable, dimension(:) :: mean, variance
  logical, allocatable, dimension(:,:) :: cv_mask

#ifdef _OPENMP
  integer, allocatable, dimension(:) :: label_slice
  real(real12), allocatable, dimension(:,:,:,:) :: image_slice
#endif


!!!-----------------------------------------------------------------------------
!!! initialise global variables
!!!-----------------------------------------------------------------------------
  call set_global_vars()
#ifdef _OPENMP
  call omp_set_num_threads(num_threads)
#endif


!!!-----------------------------------------------------------------------------
!!! read training dataset
!!!-----------------------------------------------------------------------------
  train_file = '/nutanix/gpshome/ntt203/DCoding/DTest_dir/DMNIST/MNIST_train.txt'
  call read_mnist(train_file,input_images, labels, &
       maxval(cv_kernel_size), image_size, padding_method)
  input_channels = size(input_images, 3)
  num_samples = size(input_images, 4)


!!!-----------------------------------------------------------------------------
!!! read testing dataset
!!!-----------------------------------------------------------------------------
  test_file = '/nutanix/gpshome/ntt203/DCoding/DTest_dir/DMNIST/MNIST_test.txt'
  call read_mnist(test_file,test_images, test_labels, &
       maxval(cv_kernel_size), itmp1, padding_method)
  num_samples_test = size(test_images, 4)


!!!-----------------------------------------------------------------------------
!!! initialise random seed
!!!-----------------------------------------------------------------------------
  call random_setup(seed, num_seed=1, restart=.false.)


!!!-----------------------------------------------------------------------------
!!! shuffle dataset
!!!-----------------------------------------------------------------------------
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


!!!-----------------------------------------------------------------------------
!!! initialise convolutional and pooling layers
!!!-----------------------------------------------------------------------------
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
           activation_function = "relu"))
     call network%add(maxpool2d_layer_type(&
           pool_size = 2, stride = 2))
     call network%add(full_layer_type( &
           num_outputs = 100, &
           activation_function = "relu", &
           kernel_initialiser = "he_uniform", &
           bias_initialiser = "he_uniform" &
           ))
     call network%add(full_layer_type( &
           num_outputs = 10,&
           activation_function = "softmax", &
           kernel_initialiser = "glorot_uniform", &
           bias_initialiser = "glorot_uniform" &
           ))

     !!call network%add(conv3d_layer_type( &
     !!     input_shape = [image_size,image_size,1,input_channels], &
     !!     num_filters = cv_num_filters, kernel_size = [3,3,1], stride = 1, &
     !!     padding = padding_method, &
     !!     calc_input_gradients = .false., &
     !!     activation_function = "relu"))
     !!call network%add(maxpool3d_layer_type(&
     !!     pool_size = [2,2,1], stride = [2,2,1]))
     !!call network%add(full_layer_type( &
     !!     num_outputs = 100, &
     !!     activation_function = "relu", &
     !!     kernel_initialiser = "he_uniform", &
     !!     bias_initialiser = "he_uniform" &
     !!     ))
     !!call network%add(full_layer_type( &
     !!     num_outputs = 10,&
     !!     activation_function = "softmax", &
     !!     kernel_initialiser = "glorot_uniform", &
     !!     bias_initialiser = "glorot_uniform" &
     !!     ))
  end if

  call network%compile(optimiser=optimiser, loss=loss_method, metrics=metric_dict)
  input_spread = spread(input_images,3,1)

  write(*,*) "NUMBER OF LAYERS",network%num_layers


!!!-----------------------------------------------------------------------------
!!! training loop
!!! ... loops over num_epoch number of epochs
!!! ... i.e. it trains on the same datapoints num_epoch times
!!!-----------------------------------------------------------------------------
  allocate(input_labels(num_classes,num_samples))
  input_labels = 0
  do i=1,num_samples
     input_labels(labels(i),i) = 1
  end do

  write(6,*) "Starting training..."
  call network%train(input_images, input_labels, num_epochs, batch_size, &
       plateau_threshold, shuffle_dataset, 20, verbosity)
  !!call network%train(input_spread, input_labels, num_epochs, batch_size, &
  !!     plateau_threshold, shuffle_dataset, 20, verbosity)
  write(*,*) "Training finished"


!!!-----------------------------------------------------------------------------
!!! print weights and biases of CNN to file
!!!-----------------------------------------------------------------------------
    write(*,*) "Writing network to file..."
    call network%print(file=output_file)
    write(*,*) "Writing finished"


!!!-----------------------------------------------------------------------------
!!! testing loop
!!!-----------------------------------------------------------------------------
  deallocate(input_labels)
  allocate(input_labels(num_classes,num_samples_test))
  input_labels = 0
  do i=1,num_samples_test
     input_labels(test_labels(i),i) = 1
  end do

  write(*,*) "Starting testing..."
  call network%test(test_images,input_labels)
  write(*,*) "Testing finished"
  write(6,'("Overall accuracy=",F0.5)') network%accuracy
  write(6,'("Overall loss=",F0.5)')     network%loss

  
!!!#############################################################################
!!!#############################################################################
!!! * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  !!!
!!!#############################################################################
!!!#############################################################################
  
contains

!!!#############################################################################
!!! compare two positive input values
!!! ... -1 = initial, ignore
!!! ... -2 = different
!!! +ve    = same and value
!!!#############################################################################
  subroutine compare_val(output, input)
    implicit none
    integer, intent(in) :: input
    integer, intent(out) :: output

    if(output.eq.-1)then
       output = input
    elseif(output.eq.-2)then
       return
    elseif(output.ne.input)then
       output = -2
    end if
    
  end subroutine compare_val
!!!#############################################################################


!!!#############################################################################
!!! compute accuracy
!!! this only works (and is only valid for?) categorisation problems
!!!#############################################################################
  function compute_accuracy(output, expected) result(accuracy)
    implicit none
    real(real12), dimension(:), intent(in) :: output
    integer, intent(in) :: expected
    real(real12) :: accuracy

    !! Compute the accuracy
    if (output(expected).eq.maxval(output)) then
       accuracy = 1._real12
    else
       accuracy = 0._real12
    end if

  end function compute_accuracy
!!!#############################################################################


!!!#############################################################################
!!! read mnist dataset
!!!#############################################################################
  subroutine read_mnist(file,images,labels,kernel_size,image_size,padding_method)
    use misc, only: icount
    use misc_ml, only: set_padding
    implicit none
    integer :: i, j, k, Reason, unit
    integer :: num_samples, num_pixels, padding
    character(2048) :: buffer
    character(:), allocatable :: t_padding_method

    integer, intent(out) :: image_size
    integer, optional, intent(in) :: kernel_size
    character(*), optional, intent(in) :: padding_method
    character(1024), intent(in) :: file
    real(real12), allocatable, dimension(:,:,:,:), intent(out) :: images
    integer, allocatable, dimension(:), intent(out) :: labels


!!!-----------------------------------------------------------------------------
!!! open file
!!!-----------------------------------------------------------------------------
    unit = 10
    open(unit=unit,file=file)


!!!-----------------------------------------------------------------------------
!!! count number of samples
!!!-----------------------------------------------------------------------------
    i = 0
    num_pixels = 0
    line_count: do
       i = i + 1
       read(unit,'(A)',iostat=Reason) buffer
       if(Reason.ne.0)then
          num_samples = i - 1
          exit line_count
       elseif(i.gt.90000)then
          write(0,*) "Too many lines to read in file provided (over 90000)"
          write(0,*) "Exiting..."
          stop 0
       elseif(i.eq.1)then
          num_pixels = icount(buffer,",") - 1
       end if
    end do line_count
    if(num_pixels.eq.0)then
       stop "Could not determine number of pixels"
    end if


!!!-----------------------------------------------------------------------------
!!! calculate size of image
!!!-----------------------------------------------------------------------------
    image_size = nint(sqrt(real(num_pixels,real12)))


!!!-----------------------------------------------------------------------------
!!! rewind file and allocate labels
!!!-----------------------------------------------------------------------------
    rewind(unit)
    if(allocated(labels)) deallocate(labels)
    allocate(labels(num_samples), source=0)


!!!-----------------------------------------------------------------------------
!!! handle padding type name
!!!-----------------------------------------------------------------------------
    !! none  = alt. name for 'valid'
    !! zero  = alt. name for 'same'
    !! symmetric = alt.name for 'replication'
    !! valid = no padding
    !! same  = maintain spatial dimensions
    !!         ... (i.e. padding added = (kernel_size - 1)/2)
    !!         ... defaults to zeros in the padding
    !! full  = enough padding for filter to slide over every possible position
    !!         ... (i.e. padding added = (kernel_size - 1)
    !! circular = maintain spatial dimensions
    !!            ... wraps data around for padding (periodic)
    !! reflection = maintains spatial dimensions
    !!              ... reflect data (about boundary index)
    !! replication = maintains spatial dimensions
    !!               ... reflect data (boundary included)
    if(present(padding_method))then
       t_padding_method = trim(padding_method)
    else
       t_padding_method = "valid"
    end if
    call set_padding(padding, kernel_size, t_padding_method)
    

!!!-----------------------------------------------------------------------------
!!! allocate data set
!!! ... if appropriate, add padding
!!!-----------------------------------------------------------------------------
    !! dim=1: image width in pixels
    !! dim=2: image height in pixels
    !! dim=3: image number of channels (1 due to black-white images)
    !! dim=4: number of images
    if(padding.eq.0)then
       if(allocated(images)) deallocate(images)
       allocate(images(image_size, image_size, 1, num_samples))
    elseif(present(kernel_size))then

       if(allocated(images)) deallocate(images)
       allocate(images(&
            -padding+1:image_size+padding + (1-mod(kernel_size,2)),&
            -padding+1:image_size+padding + (1-mod(kernel_size,2)),&
            1, num_samples), source=0._real12)

       !! initialise padding for constant padding types (i.e. zeros)
       !!-----------------------------------------------------------------------
!!! LATER MAKE THE CONSTANT AN OPTIONAL VALUE
       select case(t_padding_method)
       case ("same")
          images = 0._real12
       case("full")
          images = 0._real12
       end select
       
    else
       stop "ERROR: kernel_size not provided to read_mnist for padding &
            &method "//t_padding_method
    end if


!!!-----------------------------------------------------------------------------
!!! read in dataset
!!!-----------------------------------------------------------------------------
    do i=1,num_samples
       read(unit,*) labels(i), ((images(j,k,1,i),k=1,image_size),j=1,image_size)
    end do

    close(unit)


!!!-----------------------------------------------------------------------------
!!! populate padding
!!!-----------------------------------------------------------------------------
    select case(t_padding_method)
    case ("circular")
       images(-padding+1:0:1, 1:image_size, :, :) = &
            images(image_size-padding+1:image_size:1, 1:image_size, :, :)
       images(image_size+1:image_size+padding:1, 1:image_size, :, :) = &
            images(1:1+padding-1:1, 1:image_size, :, :)

       images(:, -padding+1:0:1, :, :) = &
            images(:, image_size-padding+1:image_size:1, :, :)
       images(:, image_size+1:image_size+padding:1, :, :) = &
            images(:, 1:1+padding-1:1, :, :)
    case("reflection")
       images(0:-padding+1:-1, 1:image_size, :, :) = &
            images(2:2+padding-1:1, 1:image_size, :, :)
       images(image_size+1:image_size+padding:1, 1:image_size, :, :) = &
            images(image_size-1:image_size-padding:-1, 1:image_size, :, :)

       images(:, 0:-padding+1:-1, :, :) = &
            images(:, 2:2+padding-1:1, :, :)
       images(:, image_size+1:image_size+padding:1, :, :) = &
            images(:, image_size-1:image_size-padding:-1, :, :)
    case("replication")
       images(0:-padding+1:-1, 1:image_size, :, :) = &
            images(1:1+padding-1:1, 1:image_size, :, :)
       images(image_size+1:image_size+padding:1, 1:image_size, :, :) = &
            images(image_size:image_size-padding+1:-1, 1:image_size, :, :)

       images(:, 0:-padding+1:-1, :, :) = &
            images(:, 1:1+padding-1:1, :, :)
       images(:, image_size+1:image_size+padding:1, :, :) = &
            images(:, image_size:image_size-padding+1:-1, :, :)
    end select
    

!!!-----------------------------------------------------------------------------
!!! increase label values to match fortran indices
!!!-----------------------------------------------------------------------------
    images = images/255._real12
    labels = labels + 1
    write(6,*) "Data read"

    return
  end subroutine read_mnist
!!!#############################################################################

end program ConvolutionalNeuralNetwork
!!!###################################################################################
