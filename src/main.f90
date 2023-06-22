!!!#############################################################################
!!! 
!!!#############################################################################
!!! NEED TO INCLUDE:
!!! ... gradient clipping
!!! ... L1 and L2 normalisation
!!! ... adaptive learning rate using a momentum factor
program ConvolutionalNeuralNetwork
  use constants, only: real12
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

  integer :: nseed=1
  integer, allocatable, dimension(:) :: seed_arr

  real(real12), dimension(:), allocatable :: loss_history

  real(real12) :: loss, accuracy, sum_loss, sum_accuracy, overall_loss

  ! ... data loading and preprocessing ...

  real(real12), dimension(:,:,:,:), allocatable :: input_images, test_images
  integer, dimension(:), allocatable :: labels, test_labels
  character(1024) :: train_file, test_file, cnn_file

  integer, parameter :: num_classes = 10    ! Number of output classes
  integer :: input_channels  ! Number of input channels (i.e. RGB)
  integer :: image_size
  integer :: input_size
  integer :: num_pool
  integer :: fc_num_layers
  
  integer, allocatable, dimension(:) :: tmp_num_hidden

  ! ... training loop ...

  integer :: num_batches, num_samples, num_samples_test
  integer :: epoch, batch, sample, start_index, end_index
  integer :: expected
  real(real12), dimension(:), allocatable :: fc_input, fc_output, sm_output, pl_output_rs
  real(real12), dimension(:), allocatable :: fc_gradients, sm_gradients
  real(real12), dimension(:,:,:), allocatable :: cv_output, pl_output
  real(real12), dimension(:,:,:), allocatable :: cv_gradients, pl_gradients, fc_gradients_rs
  real(real12), dimension(:), allocatable :: comb_fc_gradients
  real(real12), dimension(:,:,:), allocatable :: comb_cv_gradients

  integer :: i



  call set_global_vars()


  train_file = '/nutanix/gpshome/ntt203/DCoding/DTest_dir/DMNIST/MNIST_train.txt'
  call read_mnist(train_file,input_images, labels)
  image_size = size(input_images, 1)
  num_samples = size(input_images, 4)
  input_channels = size(input_images, 3)
  input_images = input_images/255.0


  test_file = '/nutanix/gpshome/ntt203/DCoding/DTest_dir/DMNIST/MNIST_test.txt'
  call read_mnist(test_file,test_images, test_labels)
  num_samples_test = size(test_images, 4)
  test_images = test_images/255.0

  allocate(loss_history(10))
  loss_history = -huge(1._real12)
  loss_threshold = 2._real12


  num_pool = (image_size - pool_kernel_size) / pool_stride + 1
  input_size = num_pool**2 * cv_num_filters * input_channels


!!!-----------------------------------------------------------------------------
!!! initialise random seed
!!!-----------------------------------------------------------------------------
  call random_seed(size=nseed)
  allocate(seed_arr(nseed))
  seed_arr = seed
  call random_seed(put=seed_arr)


  fc_num_layers = size(fc_num_hidden,dim=1) + 2
  allocate(tmp_num_hidden(fc_num_layers))
  tmp_num_hidden(1) = input_size
  tmp_num_hidden(2:fc_num_layers-1) = fc_num_hidden
  tmp_num_hidden(fc_num_layers) = num_classes

  call move_alloc(tmp_num_hidden, fc_num_hidden)
  !allocate(fc_num_hidden(fc_num_layers))
  !fc_num_hidden = [input_size,40,num_classes]

  write(6,*) "Initialising CNN..."
  !! Initialise the convolution layer
  call cv_init(image_size, seed, num_layers = cv_num_filters, &
       kernel_size = cv_kernel_size, stride = cv_stride)
  !! Initialise the pooling layer
  call pl_init(pool_kernel_size, pool_stride)
  !! Initialise the fully connected layer
  call fc_init(seed, num_layers=fc_num_layers, &
       num_inputs=input_size, num_hidden=fc_num_hidden)
  !! Initialise the softmax layer
  call sm_init(num_classes)
  write(6,*) "CNN initialised"


  allocate(cv_output(image_size, image_size, cv_num_filters*input_channels))
  allocate(pl_output(num_pool, num_pool, cv_num_filters*input_channels))
  allocate(fc_output(num_classes))
  allocate(sm_output(num_classes))
  cv_output = 0.0
  pl_output = 0.0
  fc_output = 0.0
  sm_output = 0.0

  allocate(fc_input(input_size))
  allocate(pl_output_rs(input_size))
  allocate(fc_gradients(input_size))
  fc_input = 0.0
  pl_output_rs = 0.0
  fc_gradients = 0.0

  allocate(cv_gradients(image_size, image_size, cv_num_filters*input_channels))
  allocate(pl_gradients,mold=cv_output)
  allocate(fc_gradients_rs,mold=pl_output)
  allocate(sm_gradients(num_classes))
  cv_gradients = 0.0
  pl_gradients = 0.0
  fc_gradients_rs = 0.0
  sm_gradients = 0.0

  allocate(comb_cv_gradients(image_size, image_size, cv_num_filters*input_channels))
  allocate(comb_fc_gradients(input_size))
  comb_cv_gradients = 0.0
  comb_fc_gradients = 0.0


  num_batches = num_samples / batch_size

  write(6,*) "Starting training..."
  !! Training loop
  !! ... loops over num_epoch number of epochs
  !! ... i.e. it trains on the same datapoints num_epoch times
  epoch_loop: do epoch = 1, num_epochs
     !! Batch loop
     !! ... split data up into minibatches for training
     batch_loop: do batch = 1, num_batches
        start_index = (batch - 1) * batch_size + 1
        end_index = batch * batch_size

        sum_loss = 0.0
        sum_accuracy = 0.0

        comb_cv_gradients = 0._real12
        comb_fc_gradients = 0._real12

        call fc_reset_delta()

        !!-------------!!
        !! parallelise !!
        !!-------------!!
        train_loop: do sample = start_index, end_index

           !! Forward pass
           cv_output = 0._real12
           pl_output = 0._real12
           fc_input  = 0._real12
           fc_output = 0._real12
           call cv_forward(input_images(:,:,:,sample), cv_output)
           call pl_forward(cv_output, pl_output)
           fc_input = reshape(pl_output, [input_size])
           call linear_renormalise(fc_input)
           !write(0,*) "MIN", minval(fc_input)
           !write(0,*) "MAX", maxval(fc_input)
           call fc_forward(fc_input, fc_output)
           call sm_forward(fc_output, sm_output)

           !write(0,*) "CV output NaN?", any(isnan(cv_output))
           !write(0,*) "PL output NaN?", any(isnan(pl_output))
           !write(0,*) "FC input NaN?", any(isnan(fc_input))
           !write(0,*) "FC output NaN?", any(isnan(fc_output))
           !write(0,*) fc_output
           !write(0,*) sm_output
           !write(0,*) labels(sample)-1


           if(any(isnan(sm_output)))then
              write(0,*) "ERROR: Softmax outputs are NaN"
              stop
           elseif(any(fc_output.gt.1.E9))then
              write(0,*) "WARNING: FC outputs growing beyond 1E9"
              write(0,*) fc_output
           end if

           expected = labels(sample)

           !! Compute loss and accuracy (for monitoring purposes)
           loss = categorical_cross_entropy(sm_output, expected)
           accuracy = compute_accuracy(sm_output, expected)

           sum_loss = sum_loss + loss
           sum_accuracy = sum_accuracy + accuracy

           !write(0,*) fc_output
           !write(0,*) sm_output
           !write(0,*) "loss",sample,loss,sum_loss

           !! Backward pass
           sm_gradients = 0._real12
           fc_gradients = 0._real12
           fc_gradients_rs = 0._real12
           pl_gradients = 0._real12
           cv_gradients = 0._real12
           call sm_backward(sm_output, expected, sm_gradients)
           call fc_backward(fc_input, sm_gradients, fc_gradients, fc_clip)
           fc_gradients_rs = reshape(fc_gradients, shape(fc_gradients_rs))
           call pl_backward(cv_output, fc_gradients_rs, pl_gradients)
           call cv_backward(input_images(:,:,:,sample), pl_gradients, cv_gradients, cv_clip)

           !! Update weights and biases using optimization algorithm (e.g., gradient descent)
           call cv_update(learning_rate, input_images(:,:,:,sample), cv_gradients, &
                l1_lambda, l2_lambda, momentum)
           call fc_update(learning_rate, fc_input, fc_gradients, &
                l1_lambda, l2_lambda, momentum, l_batch=.false.)
           
           comb_cv_gradients = comb_cv_gradients + cv_gradients
           comb_fc_gradients = comb_fc_gradients + fc_gradients

        end do train_loop

        !! Error checking and handling

        if(sum(abs(comb_fc_gradients)).lt.1.D-8)then
           write(0,*) "ERROR: FullyConnected gradients are zero"
           write(0,*) "Exiting..."
           stop
        end if
        loss_history = cshift(loss_history, shift=-1, dim=1)
        loss_history(1) = sum_loss
        
        if(abs(sum(loss_history)).lt.loss_threshold)then
           write(6,*) "Convergence achieved, accuracy threshold reached"
           write(6,*) "Exiting training loop"
           exit epoch_loop
        elseif(all(abs(loss_history-sum_loss).lt.1.E-1))then
           !write(0,*) "fc_input", fc_input
           !write(0,*) "fc_output", fc_output
           write(0,*) "sm_output", sm_output
           write(0,*) "sm_grad", sm_gradients
           write(0,*) "fc_gradients", fc_gradients
           write(0,*) "ERROR: accuracy has remained constant for 10 runs"
           write(0,*) "Exiting..."
           stop
           exit epoch_loop
        end if

        !! Update weights and biases using optimization algorithm (e.g., gradient descent)
        comb_cv_gradients = comb_cv_gradients/batch_size
        comb_fc_gradients = comb_fc_gradients/batch_size
        call fc_norm_delta(batch_size)        

        !call cv_update(learning_rate, input_images(:,:,:,sample), comb_cv_gradients, &
        !     l1_lambda, l2_lambda, momentum)
        !call fc_update(learning_rate, fc_input, comb_fc_gradients, &
        !     l1_lambda, l2_lambda, momentum, l_batch=.true.)

!!! FC DOESN'T ACTUALLY USE THE GRADIENTS SUPPLIED !!!
!!! comb_fc_gradients ALSO ONLY HAS THE GRADIENTS FOR THE FINAL LAYER !!!

        !write(6,'("epoch=",I0,", batch=",I0", lrate=",F0.3,", error=",F0.3)') epoch, batch, learning_rate, sum_accuracy
        write(6,'("epoch=",I0,", batch=",I0", lrate=",F0.3,", error=",F0.3)') epoch, batch, learning_rate, sum_loss

        if(stop_check())then
           write(0,*) "STOPCAR ENCOUNTERED"
           write(0,*) "Exiting training loop..."
           exit epoch_loop
        end if

     end do batch_loop
     if(mod(epoch,20).eq.0.E0)then
        write(6,'("epoch=",I0,", batch=",I0", lrate=",F0.3,", error=",F0.3)') epoch, batch, learning_rate, sum_loss
        !write(6,'("epoch=",I0,", lrate=",F0.3,", error=",F0.3)') epoch, learning_rate, sum_accuracy
     end if

  end do epoch_loop


  !! Print weights and biases of CNN to file
  cnn_file = '../cnn_layers.txt'
  open(unit=10,file=cnn_file,status='replace')
  close(10)
  call cv_write(cnn_file)
  call fc_write(cnn_file)


  !! Testing loop
  test_loop: do sample = 1, num_samples_test

     call cv_forward(test_images(:,:,:,sample), cv_output)
     call pl_forward(cv_output, pl_output)
     fc_input = reshape(pl_output, [input_size])
     call fc_forward(fc_input, fc_output)
     call sm_forward(fc_output, sm_output)

     expected = test_labels(sample)

     ! Compute loss and accuracy (for monitoring purposes)
     loss = categorical_cross_entropy(sm_output, expected)
     accuracy = compute_accuracy(sm_output, expected)

     sum_loss = sum_loss + loss
     sum_accuracy = sum_accuracy + accuracy
     write(6,'(I4," Expected=",I3,", Got=",I3,", Accuracy=",F0.3)') sample,expected-1, maxloc(sm_output,dim=1)-1, accuracy
     write(0,*) sm_output
     write(0,*)

  end do test_loop


  overall_loss = real(sum_loss)/real(num_samples_test)
  write(6,'("Overall accuracy=",F0.5)') overall_loss



!!!###################################################################################

contains


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

  function categorical_cross_entropy(output, expected) result(loss)
    implicit none
    real(real12), dimension(:), intent(in) :: output
    integer, intent(in) :: expected
    real(real12) :: loss, epsilon

    epsilon = 1.E-10_real12
    loss = -1._real12/real(size(output, dim=1),real12) * log(output(expected)+epsilon)

  end function categorical_cross_entropy

  !! this is handled by the softmax backward subroutine
  subroutine categorical_cross_entropy_derivative(output, expected, gradient)
    implicit none
    integer, intent(in) :: expected
    real(real12), dimension(:), intent(in) :: output
    real(real12), dimension(:), intent(out) :: gradient
    
    gradient = output
    gradient(expected) = output(expected) - 1._real12

  end subroutine categorical_cross_entropy_derivative

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
       accuracy = 1.0
    else
       accuracy = 0.0
    end if

    if(isnan(accuracy))then
       write(0,*) "ERROR: Accuracy is NaN"
       stop
    end if

  end function compute_accuracy

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
    allocate(images(image_size, image_size, 1, num_samples))
    do i=1,num_samples
       read(unit,*) labels(i), ((images(j,k,1,i),k=1,image_size),j=1,image_size)
    end do
    close(unit)

    labels = labels + 1
    write(0,*) "READING DONE"

  end subroutine read_mnist


  logical function stop_check()
    implicit none
    integer :: Reason
    integer :: unit=201
    logical :: lfound
    character(7) :: file="STOPCAR"
    character(128) :: buffer

    stop_check = .false.
    inquire(file=trim(file),exist=stop_check)
    file_if: if(lfound)then
       open(unit=unit, file=trim(file))
       file_loop: do
          read(unit,'(A)',iostat=Reason) buffer
          if(Reason.ne.0) exit file_loop
          if(index(buffer,"LSTOP=.TRUE.").ne.0)then
             stop_check = .true.
             exit file_loop
          end if
       end do file_loop
       close(unit,status='delete')
    end if file_if
    
  end function stop_check

end program ConvolutionalNeuralNetwork
!!!###################################################################################
