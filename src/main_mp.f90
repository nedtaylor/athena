!!!#############################################################################
!!! 
!!!#############################################################################
program ConvolutionalNeuralNetwork
  use omp_lib
  use constants, only: real12
  use misc, only: shuffle
  !use misc_maths, only: mean
  use infile_tools, only: stop_check
  use normalisation, only: linear_renormalise
  use inputs
  use ConvolutionLayer, cv_init => initialise, cv_forward => forward, &
       cv_backward => backward, &
       cv_update => update_weights_and_biases, &
       cv_write => write_file
  use PoolingLayer, pl_init => initialise, pl_forward => forward, &
       pl_backward => backward
  use FullyConnectedLayer, fc_init => initialise, fc_forward => forward, &
       fc_backward => backward, &
       fc_update => update_weights_and_biases, &
       fc_write => write_file, &
       fc_norm_delta => normalise_delta_batch, &
       fc_reset_delta => reset_delta_batch
  use SoftmaxLayer, sm_init => initialise, sm_forward => forward, &
       sm_backward => backward

  implicit none

  !! seed variables
  integer :: nseed=1
  integer, allocatable, dimension(:) :: seed_arr

  !! training and testing monitoring
  real(real12) :: loss, accuracy, sum_loss, sum_accuracy, overall_loss
  real(real12) :: exploding_check, exploding_check_old
  real(real12), allocatable, dimension(:) :: loss_history

  !! data loading and preoprocessing
  real(real12), allocatable, dimension(:,:,:,:) :: input_images, test_images
  integer, allocatable, dimension(:) :: labels, test_labels
  character(1024) :: train_file, test_file, cnn_file

  !! neural network size and shape variables
  integer, parameter :: num_classes = 10    ! Number of output classes
  integer :: input_channels  ! Number of input channels (i.e. RGB)
  integer :: image_size
  integer :: input_size
  integer :: num_pool
  integer :: fc_num_layers  
  integer, allocatable, dimension(:) :: tmp_num_hidden

  !! training loop variables
  integer :: num_batches, num_samples, num_samples_test
  integer :: epoch, batch, sample, start_index, end_index
  integer :: expected
  real(real12), allocatable, dimension(:) :: fc_input, fc_output, sm_output, pl_output_rs
  real(real12), allocatable, dimension(:) :: fc_gradients, sm_gradients
  real(real12), allocatable, dimension(:,:,:) :: cv_output, pl_output
  real(real12), allocatable, dimension(:,:,:) :: cv_gradients, pl_gradients, fc_gradients_rs
  real(real12), allocatable, dimension(:) :: comb_fc_gradients
  real(real12), allocatable, dimension(:,:,:) :: comb_cv_gradients

  integer :: i
  real(real12) :: rtmp1
  integer, allocatable, dimension(:) :: label_slice
  real(real12), allocatable, dimension(:,:,:,:) :: image_slice


!!!-----------------------------------------------------------------------------
!!! initialise global variables
!!!-----------------------------------------------------------------------------
  call set_global_vars()


!!!-----------------------------------------------------------------------------
!!! read training dataset
!!!-----------------------------------------------------------------------------
  train_file = '/nutanix/gpshome/ntt203/DCoding/DTest_dir/DMNIST/MNIST_train.txt'
  call read_mnist(train_file,input_images, labels)
  image_size = size(input_images, 1)
  num_samples = size(input_images, 4)
  input_channels = size(input_images, 3)
  input_images = input_images/255.0
  num_pool = (image_size - pool_kernel_size) / pool_stride + 1
  input_size = num_pool**2 * cv_num_filters * input_channels
  num_batches = num_samples / batch_size


!!!-----------------------------------------------------------------------------
!!! read testing dataset
!!!-----------------------------------------------------------------------------
  test_file = '/nutanix/gpshome/ntt203/DCoding/DTest_dir/DMNIST/MNIST_test.txt'
  call read_mnist(test_file,test_images, test_labels)
  num_samples_test = size(test_images, 4)
  test_images = test_images/255.0


!!!-----------------------------------------------------------------------------
!!! initialise monitoring variables
!!!-----------------------------------------------------------------------------
  allocate(loss_history(10))
  loss_history = -huge(1._real12)
  loss_threshold = 2._real12
  if(batch_learning)then
     exploding_check = 0._real12
  else
     exploding_check = 1._real12
  end if


!!!-----------------------------------------------------------------------------
!!! initialise random seed
!!!-----------------------------------------------------------------------------
  call random_seed(size=nseed)
  allocate(seed_arr(nseed))
  seed_arr = seed
  call random_seed(put=seed_arr)

!!!-----------------------------------------------------------------------------
!!! shuffle dataset
!!!-----------------------------------------------------------------------------
  if(shuffle_dataset)then
     write(6,*) "Shuffling training dataset..."
     call shuffle(input_images, labels, 4, seed)
     write(6,*) "Training dataset shuffled"
     if(verbosity.gt.0)then
        write(6,*) "Check fort.11 and fort.12 to ensure data shuffling &
             &executed properly"
        do i=1,batch_size*2
           write(11,*) input_images(:,:,:,i) 
        end do
        write(12,*) labels
     end if
  end if


!!!-----------------------------------------------------------------------------
!!! reformulate fully connected layers to include input and output layers
!!! ... user provides only hidden layers
!!!-----------------------------------------------------------------------------
  fc_num_layers = size(fc_num_hidden,dim=1) + 2
  allocate(tmp_num_hidden(fc_num_layers))
  tmp_num_hidden(1) = input_size
  tmp_num_hidden(2:fc_num_layers-1) = fc_num_hidden
  tmp_num_hidden(fc_num_layers) = num_classes
  call move_alloc(tmp_num_hidden, fc_num_hidden)


!!!-----------------------------------------------------------------------------
!!! initialise convolutional neural network layers
!!!-----------------------------------------------------------------------------
  write(6,*) "Initialising CNN..."
  !! Initialise the convolution layer
  call cv_init(image_size, seed, num_layers = cv_num_filters, &
       kernel_size = cv_kernel_size, stride = cv_stride)
  !! Initialise the pooling layer
  call pl_init(pool_kernel_size, pool_stride)
  !! Initialise the fully connected layer
  call fc_init(seed, num_layers=fc_num_layers, &
       num_inputs=input_size, num_hidden=fc_num_hidden, &
       activation_function=activation_function)
  !! Initialise the softmax layer
  call sm_init(num_classes)
  write(6,*) "CNN initialised"


!!!-----------------------------------------------------------------------------
!!! allocate and initialise layer outputs and gradients
!!!-----------------------------------------------------------------------------
  allocate(cv_output(image_size, image_size, cv_num_filters*input_channels))
  allocate(pl_output(num_pool, num_pool, cv_num_filters*input_channels))
  allocate(fc_output(num_classes))
  allocate(sm_output(num_classes))
  cv_output = 0._real12
  pl_output = 0._real12
  fc_output = 0._real12
  sm_output = 0._real12

  allocate(fc_input(input_size))
  allocate(pl_output_rs(input_size))
  allocate(fc_gradients(input_size))
  fc_input = 0._real12
  pl_output_rs = 0._real12
  fc_gradients = 0._real12

  allocate(cv_gradients(image_size, image_size, cv_num_filters*input_channels))
  allocate(pl_gradients,mold=cv_output)
  allocate(fc_gradients_rs,mold=pl_output)
  allocate(sm_gradients(num_classes))
  cv_gradients = 0._real12
  pl_gradients = 0._real12
  fc_gradients_rs = 0._real12
  sm_gradients = 0._real12

  allocate(comb_cv_gradients(image_size, image_size, cv_num_filters*input_channels))
  allocate(comb_fc_gradients(input_size))
  comb_cv_gradients = 0._real12
  comb_fc_gradients = 0._real12


  allocate(image_slice(image_size,image_size,size(input_images,dim=3),batch_size))
  allocate(label_slice(batch_size))


!!!-----------------------------------------------------------------------------
!!! training loop
!!! ... loops over num_epoch number of epochs
!!! ... i.e. it trains on the same datapoints num_epoch times
!!!-----------------------------------------------------------------------------
  write(6,*) "Starting training..."
  epoch_loop: do epoch = 1, num_epochs
     !!-------------------------------------------------------------------------
     !! batch loop
     !! ... split data up into minibatches for training
     !!-------------------------------------------------------------------------
     batch_loop: do batch = 1, num_batches
        start_index = (batch - 1) * batch_size + 1
        end_index = batch * batch_size

        loss = 0._real12
        accuracy = 0._real12

        !! reset (zero) summed gradients
        !! ... UNCOMMENT if running mini-batch training 
        !!----------------------------------------------------------------------
        call fc_reset_delta()
        comb_cv_gradients = 0._real12
        comb_fc_gradients = 0._real12


        !!-------------!!
        !! parallelise !!
        !!-------------!!

        image_slice(:,:,:,:) = input_images(:,:,:,start_index:end_index)
        label_slice(:) = labels(start_index:end_index)

        !!----------------------------------------------------------------------
        !! sample loop
        !! ... test each sample and get gradients and losses from each
        !!----------------------------------------------------------------------
        !$OMP PARALLEL DO &
        !$OMP DEFAULT(NONE) &
        !$OMP SHARED(image_slice,label_slice,input_size) &
        !$OMP SHARED(fc_clip, cv_clip,batch_learning) &
        !$OMP PRIVATE(sample,cv_output,pl_output,fc_input,fc_output,sm_output) &
        !$OMP PRIVATE(rtmp1,expected) &
        !$OMP PRIVATE(sm_gradients,fc_gradients,fc_gradients_rs,pl_gradients,cv_gradients) &
        !$OMP REDUCTION(+:comb_cv_gradients) &
        !$OMP REDUCTION(+:comb_fc_gradients) &
        !$OMP REDUCTION(+:exploding_check) &
        !$OMP REDUCTION(+:loss,accuracy)
        train_loop: do sample = 1,batch_size !start_index, end_index
           !write(0,*) "HERE", OMP_GET_THREAD_NUM()

           !! Forward pass
           !!-------------------------------------------------------------------
           !call cv_forward(input_images(:,:,:,sample), cv_output)
           call cv_forward(image_slice(:,:,:,sample), cv_output)
           call pl_forward(cv_output, pl_output)
           fc_input = reshape(pl_output, [input_size])
           call linear_renormalise(fc_input)
           call fc_forward(fc_input, fc_output)
           call sm_forward(fc_output, sm_output)

  
           !! check for NaN and infinity
           !!----------------------------------------------------------------------
           if(any(isnan(sm_output)))then
              write(0,*) "ERROR: Softmax outputs are NaN"
              stop
           end if
           if(batch_learning)then
              exploding_check = sum(fc_output)
              stop "ERROR: non-batch learning not yet parallelised"
           end if

           
           !! compute loss and accuracy (for monitoring)
           !!----------------------------------------------------------------------
           expected = label_slice(sample)!labels(sample)
           loss = categorical_cross_entropy(sm_output, expected)
           accuracy = compute_accuracy(sm_output, expected)


           !! Backward pass
           !!----------------------------------------------------------------------
           call sm_backward(sm_output, expected, sm_gradients)
           call fc_backward(fc_input, sm_gradients, fc_gradients, fc_clip)
           fc_gradients_rs = reshape(fc_gradients, shape(fc_gradients_rs))
           call pl_backward(cv_output, fc_gradients_rs, pl_gradients)
           call cv_backward(image_slice(:,:,:,sample), pl_gradients, cv_gradients, cv_clip)
           !call cv_backward(input_images(:,:,:,sample), pl_gradients, cv_gradients, cv_clip)


           !! if mini-batch ...
           !! ... sum gradients for mini-batch training
           !! if not mini-batch
           !! ... update weights and biases using optimization algorithm
           !! ... (gradient descent)
           !!----------------------------------------------------------------------
           if(batch_learning)then
              comb_cv_gradients = cv_gradients
              comb_fc_gradients = fc_gradients
           end if
           

        end do train_loop
        !$OMP END PARALLEL DO
        !$OMP BARRIER


        !! Error checking and handling
        !!-------------------------------------------------------------------------
        if(sum(abs(comb_fc_gradients)).lt.1.D-8)then
           write(0,*) "ERROR: FullyConnected gradients are zero"
           write(0,*) "Exiting..."
           stop
        end if
        loss_history = cshift(loss_history, shift=-1, dim=1)
        loss_history(1) = loss
        
        if(abs(sum(loss_history)).lt.loss_threshold)then
           write(6,*) "Convergence achieved, accuracy threshold reached"
           write(6,*) "Exiting training loop"
           exit epoch_loop
        elseif(all(abs(loss_history-loss).lt.plateau_threshold))then
           write(0,*) "sm_output", sm_output
           write(0,*) "sm_grad", sm_gradients
           write(0,*) "fc_gradients", fc_gradients
           write(0,*) "ERROR: accuracy has remained constant for 10 runs"
           write(0,*) "Exiting..."
           stop
           exit epoch_loop
        end if


        !! if mini-batch ...
        !! ... update weights and biases using optimization algorithm
        !! ... (gradient descent)
        !!-------------------------------------------------------------------------
        if(batch_learning)then
           exploding_check = (exploding_check/batch_size)
           if(epoch.gt.1.or.batch.gt.1)then
              rtmp1 = abs(exploding_check/exploding_check_old)
              if(rtmp1.gt.1.E3_real12)then
                 write(0,*) "WARNING: FC outputs are expanding too quickly!"
                 write(0,*) "check:", sample, exploding_check, exploding_check_old
              elseif(rtmp1.lt.1.E-3_real12)then
                 write(0,*) "WARNING: FC outputs are vanishing too quickly!"
                 write(0,*) "check:", exploding_check, exploding_check_old
              end if
           end if
           exploding_check_old = exploding_check
           exploding_check = 0._real12
           
           comb_cv_gradients = comb_cv_gradients/batch_size
           comb_fc_gradients = comb_fc_gradients/batch_size
           call fc_norm_delta(batch_size)        
           call cv_update(learning_rate, input_images(:,:,:,sample), comb_cv_gradients, &
                l1_lambda, l2_lambda, momentum)
           call fc_update(learning_rate, fc_input, comb_fc_gradients, &
                l1_lambda, l2_lambda, momentum, l_batch=batch_learning)
        end if

!!! NOTE:
!!! FC DOESN'T ACTUALLY USE THE GRADIENTS SUPPLIED !!!
!!! comb_fc_gradients ALSO ONLY HAS THE GRADIENTS FOR THE FINAL LAYER !!!


        !! print batch results
        !!-------------------------------------------------------------------------
        write(6,'("epoch=",I0,", batch=",I0", learning_rate=",F0.3,", loss=",F0.3)') epoch, batch, learning_rate, loss


        !! check for user-name stop file
        !!-------------------------------------------------------------------------
        if(stop_check())then
           write(0,*) "STOPCAR ENCOUNTERED"
           write(0,*) "Exiting training loop..."
           exit epoch_loop
        end if

     end do batch_loop


     !! print epoch summary results
     !!----------------------------------------------------------------------------
     if(mod(epoch,20).eq.0.E0) &
          write(6,'("epoch=",I0,", batch=",I0", learning_rate=",F0.3,", loss=",F0.3)') epoch, batch, learning_rate, loss

  end do epoch_loop


!!!-----------------------------------------------------------------------------
!!! print weights and biases of CNN to file
!!!-----------------------------------------------------------------------------
  cnn_file = '../cnn_layers.txt'
  open(unit=10,file=cnn_file,status='replace')
  close(10)
  call cv_write(cnn_file)
  call fc_write(cnn_file)


!!!-----------------------------------------------------------------------------
!!! testing loop
!!!-----------------------------------------------------------------------------
  test_loop: do sample = 1, num_samples_test

     call cv_forward(test_images(:,:,:,sample), cv_output)
     call pl_forward(cv_output, pl_output)
     fc_input = reshape(pl_output, [input_size])
     call linear_renormalise(fc_input)
     call fc_forward(fc_input, fc_output)
     call sm_forward(fc_output, sm_output)


     !! compute loss and accuracy (for monitoring)
     !!-------------------------------------------------------------------------
     expected = test_labels(sample)
     loss = categorical_cross_entropy(sm_output, expected)
     accuracy = compute_accuracy(sm_output, expected)
     sum_loss = sum_loss + loss
     sum_accuracy = sum_accuracy + accuracy


     !! print testing results
     !!-------------------------------------------------------------------------
     write(6,'(I4," Expected=",I3,", Got=",I3,", Accuracy=",F0.3)') sample,expected-1, maxloc(sm_output,dim=1)-1, accuracy
     write(0,*) sm_output
     write(0,*)

  end do test_loop


  overall_loss = real(sum_loss)/real(num_samples_test)
  write(6,'("Overall accuracy=",F0.5)') overall_loss



!!!#############################################################################
!!!#############################################################################
!!! * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  !!!
!!!#############################################################################
!!!#############################################################################

contains

!!!#############################################################################
!!! compute losses
!!! method: custom: RMSE-like???
!!!#############################################################################
  function compute_loss(output, expected) result(loss)
    implicit none
    real(real12), dimension(:), intent(in) :: output
    integer, intent(in) :: expected
    integer :: i
    real(real12) :: loss, total

    ! Compute the cross-entropy loss
    total = 0._real12
    do i=1,size(output)
       if(i.eq.expected)then
          total = total + (output(i) - 1._real12)**2.E0
       else
          total = total + output(i)**2.E0
       end if
    end do
    !loss = -log(output(expected))
    !! ERROR: total = 0 means no loss, but log(0) = INF
    !loss = -log(total)
    loss = total

  end function compute_loss
!!!#############################################################################


!!!#############################################################################
!!! compute losses
!!! method: categorical cross entropy
!!!#############################################################################
  function categorical_cross_entropy(output, expected) result(loss)
    implicit none
    real(real12), dimension(:), intent(in) :: output
    integer, intent(in) :: expected
    real(real12) :: loss, epsilon

    epsilon = 1.E-10_real12
    loss = -1._real12/real(size(output, dim=1),real12) * log(output(expected)+epsilon)

  end function categorical_cross_entropy
!!!#############################################################################


!!!#############################################################################
!!! compute loss derivative
!!! method: categorical cross entropy
!!! this is handled by the softmax backward subroutine
!!!#############################################################################
  subroutine categorical_cross_entropy_derivative(output, expected, gradient)
    implicit none
    integer, intent(in) :: expected
    real(real12), dimension(:), intent(in) :: output
    real(real12), dimension(:), intent(out) :: gradient
    
    gradient = output
    gradient(expected) = output(expected) - 1._real12

  end subroutine categorical_cross_entropy_derivative
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

    !if(any(isnan(output)))then
    !   write(0,*) "Output is NaN"
    !   stop
    !end if

    ! Compute the accuracy
    if (output(expected).eq.maxval(output)) then
       accuracy = 1._real12
    else
       accuracy = 0._real12
    end if

    if(isnan(accuracy))then
       write(0,*) "ERROR: Accuracy is NaN"
       stop
    end if

  end function compute_accuracy
!!!#############################################################################


!!!#############################################################################
!!! read mnist dataset
!!!#############################################################################
  subroutine read_mnist(file,images,labels)
    use misc, only: icount
    implicit none
    integer :: i, j, k, Reason, unit
    integer :: num_samples, num_pixels, image_size
    character(2048) :: buffer

    character(1024) :: file
    real(real12), dimension(:,:,:,:), allocatable, intent(out) :: images
    integer, dimension(:), allocatable, intent(out) :: labels

    unit = 10
    open(unit=unit,file=file)

    i = 0
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

    image_size = nint(sqrt(real(num_pixels,real12)))

    rewind(unit)
    allocate(labels(num_samples))
    !! dim=1: image width in pixels
    !! dim=2: image height in pixels
    !! dim=3: image number of channels (1 due to black-white images)
    !! dim=4: number of images
    allocate(images(image_size, image_size, 1, num_samples))
    do i=1,num_samples
       read(unit,*) labels(i), ((images(j,k,1,i),k=1,image_size),j=1,image_size)
    end do
    close(unit)

    labels = labels + 1
    write(0,*) "READING DONE"

  end subroutine read_mnist
!!!#############################################################################

end program ConvolutionalNeuralNetwork
!!!###################################################################################
