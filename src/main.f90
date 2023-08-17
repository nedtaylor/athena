!!!#############################################################################
!!! 
!!!#############################################################################
program ConvolutionalNeuralNetwork
#ifdef _OPENMP
  use omp_lib
#endif
  use constants, only: real12
  use random, only: random_setup
  use misc, only: shuffle
  !use misc_maths, only: mean
  use infile_tools, only: stop_check

  use loss_categorical!, only: loss_mse, loss_nll, loss_cce, loss_type
  use inputs

  use network, only: network_type

  use container_layer, only: container_layer_type
  use input3d_layer,   only: input3d_layer_type
  use full_layer,      only: full_layer_type
  use conv2d_layer,    only: conv2d_layer_type
  use maxpool2d_layer, only: maxpool2d_layer_type
  use flatten2d_layer, only: flatten2d_layer_type
  use flatten3d_layer, only: flatten3d_layer_type


  implicit none

  type(network_type) :: network

  !! seed variables
  integer :: nseed=1
  integer, allocatable, dimension(:) :: seed_arr

  !! data loading and preoprocessing
  real(real12), allocatable, dimension(:,:,:,:) :: input_images, test_images
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
!!! set up reduction for gradient custom type
!!! ...
!!! https://www.openmp.org/spec-html/5.0/openmpsu107.html
!!! https://stackoverflow.com/questions/61141297/openmp-reduction-on-user-defined-fortran-type-containing-allocatable-array
!!! https://fortran-lang.discourse.group/t/openmp-reduction-on-operator/5887
!!!-----------------------------------------------------------------------------
  !  !$omp declare reduction(cv_grad_sum:cv_gradient_type:omp_out = omp_out + omp_in) &
  !  !$omp& initializer(cv_gradient_alloc(omp_priv, omp_orig, .false.))
  !  !$omp declare reduction(fc_grad_sum:fc_gradient_type:omp_out = omp_out + omp_in) &
  !  !$omp& initializer(fc_gradient_alloc(omp_priv, omp_orig, .false.))
  !  !$omp declare reduction(compare_val:integer:compare_val(omp_out,omp_in)) &
  !  !$omp& initializer(omp_priv = omp_orig)
  !!  !$omp declare reduction(+:metric_dict_type:omp_out = omp_out + omp_in) &
  !!  !$omp& initializer(metric_dict_alloc(omp_priv, omp_orig))


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
  write(6,*) "Initialising CNN..."
  !! Initialise the convolution layer
  network%num_layers = 6
  allocate(network%model(network%num_layers))
  allocate(network%model(1)%layer, source = input3d_layer_type(&
       input_shape = [size(input_images,1),size(input_images,2),input_channels]))
  allocate(network%model(2)%layer, source = conv2d_layer_type( &
       input_shape = [28,28,input_channels], &
       num_filters = cv_num_filters, kernel_size = 3, stride = 1, &
       padding=padding_method, &
       calc_input_gradients = .false., &
       activation_function = "relu", clip_dict = cv_clip))
  allocate(network%model(3)%layer, source = maxpool2d_layer_type(&
       input_shape=[28,28,cv_num_filters], &
       pool_size=2, stride=2))
  allocate(network%model(4)%layer, source = flatten2d_layer_type(&
       input_shape=[14,14,cv_num_filters]))
  allocate(network%model(5)%layer, source = full_layer_type( &
       num_inputs=product([14,14,cv_num_filters]), &
       num_outputs=100, clip_dict = fc_clip, &
       activation_function = "relu", &
       kernel_initialiser="he_uniform", &
       bias_initialiser="he_uniform" &
       ))
  allocate(network%model(6)%layer, source = full_layer_type( &
       num_inputs=100, &
       num_outputs=10, clip_dict = fc_clip,&
       activation_function="softmax", &
       kernel_initialiser="glorot_uniform", &
       bias_initialiser="glorot_uniform" &
       ))
  network%num_outputs = 10
  network%optimiser = optimiser
  network%metrics = metric_dict


  !  if(restart)then
  !     call cv_init(file = input_file, &
  !          learning_parameters=learning_parameters)
  !  else
  !     call cv_init(seed, num_layers = cv_num_filters, &
  !          kernel_size = cv_kernel_size, stride = cv_stride, &
  !          full_padding = trim(padding_method).eq."full",&
  !          learning_parameters=learning_parameters,&
  !          kernel_initialiser=cv_kernel_initialiser,&
  !          bias_initialiser=cv_bias_initialiser,&
  !          activation_scale=cv_activation_scale,&
  !          activation_function=cv_activation_function)
  !  end if


!!!!-----------------------------------------------------------------------------
!!!! allocate and initialise layer outputs and gradients
!!!!-----------------------------------------------------------------------------
  !!  allocate(bn_output, source=cv_output)
  !!  allocate(mean(output_channels))
  !!  allocate(variance(output_channels))


!!!!-----------------------------------------------------------------------------
!!!! initialise non-fully connected layer gradients
!!!!-----------------------------------------------------------------------------
  !  select case(learning_parameters%method)
  !  case("adam")
  !    call cv_gradient_init(cv_gradients, image_size, adam_learning = .true.)
  !     if(batch_learning) &
  !          call cv_gradient_init(comb_cv_gradients, image_size, adam_learning = .true.)
  !     update_iteration = 1
  !  case default
  !     call cv_gradient_init(cv_gradients, image_size)
  !     if(batch_learning) call cv_gradient_init(comb_cv_gradients, image_size)     
  !  end select
  !!  allocate(bn_gradients, source=cv_output)
  !
  !
!!!!-----------------------------------------------------------------------------
!!!! initialise fully connected layer inputs and gradients
!!!!-----------------------------------------------------------------------------
  !  allocate(fc_input(input_size))
  !  fc_input = 0._real12
  !
  !  select case(learning_parameters%method)
  !  case("adam")
  !    call fc_gradient_init(fc_gradients, input_size, adam_learning = .true.)
  !     if(batch_learning) &
  !          call fc_gradient_init(comb_fc_gradients, input_size, adam_learning = .true.)
  !     update_iteration = 1
  !  case default
  !     call fc_gradient_init(fc_gradients, input_size)
  !     if(batch_learning) call fc_gradient_init(comb_fc_gradients, input_size)     
  !  end select
  !
  !
!!!-----------------------------------------------------------------------------
!!! initialise loss method
!!!-----------------------------------------------------------------------------
  select case(loss_method)
  case("cce")
     network%get_loss => compute_loss_cce
     write(*,*) "Loss method: Categorical Cross Entropy"
  case("mse")
     network%get_loss => compute_loss_mse
     write(*,*) "Loss method: Mean Squared Error"
  case("nll")
     network%get_loss => compute_loss_nll
     write(*,*) "Loss method: Negative log likelihood"
  case default
     write(*,*) "Failed loss method: "//trim(loss_method)
     stop "ERROR: No loss method provided"
  end select
  network%get_loss_deriv => compute_loss_derivative


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
  write(*,*) "Training finished"


!!!-----------------------------------------------------------------------------
!!! print weights and biases of CNN to file
!!!-----------------------------------------------------------------------------
!!!  call network%write()


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
