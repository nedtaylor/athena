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
  use misc_ml, only: step_decay, reduce_lr_on_plateau, &
       generate_bernoulli_mask, drop_block
  !use misc_maths, only: mean
  use infile_tools, only: stop_check

  use normalisation, only: linear_renormalise, &
       renormalise_norm, renormalise_sum
  use loss_categorical!, only: loss_mse, loss_nll, loss_cce, loss_type
  use inputs

  use container_layer, only: container_layer_type
  use input3d_layer,   only: input3d_layer_type
  use full_layer,      only: full_layer_type
  use conv2d_layer,    only: conv2d_layer_type
  use maxpool2d_layer, only: maxpool2d_layer_type
  use flatten2d_layer, only: flatten2d_layer_type
  use flatten3d_layer, only: flatten3d_layer_type


  implicit none

  type(container_layer_type), allocatable, dimension(:) :: model
  character(20) :: fmt

  !! seed variables
  integer :: nseed=1
  integer, allocatable, dimension(:) :: seed_arr

  !! training and testing monitoring
  real(real12) :: batch_loss, batch_accuracy, accuracy, avg_loss, avg_accuracy
  real(real12) :: exploding_check, exploding_check_old
  real(real12), allocatable, dimension(:,:) :: y_pred, y_true
  !class(loss_type), pointer :: get_loss
  procedure(compute_loss_function), pointer :: compute_loss
  
  !! learning parameters
  integer :: converged
  integer :: history_length
  integer :: update_iteration

  !! data loading and preoprocessing
  real(real12), allocatable, dimension(:,:,:,:) :: input_images, test_images
  integer, allocatable, dimension(:) :: labels, test_labels
  character(1024) :: train_file, test_file

  !! neural network size and shape variables
  integer, parameter :: num_classes = 10    ! Number of output classes
  integer :: input_channels  ! Number of input channels (i.e. RGB)
  integer :: image_size, lw_image_size, up_image_size
  integer :: output_size
  integer :: input_size
  integer :: output_channels
  integer :: num_pool
  integer :: fc_num_layers
  integer, allocatable, dimension(:) :: tmp_num_hidden
  integer :: num_layers

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

  real(real12), allocatable, dimension(:,:,:) :: image_sample
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
  output_channels = cv_num_filters * input_channels
  num_samples = size(input_images, 4)
  num_batches = num_samples / batch_size


!!!-----------------------------------------------------------------------------
!!! read testing dataset
!!!-----------------------------------------------------------------------------
  test_file = '/nutanix/gpshome/ntt203/DCoding/DTest_dir/DMNIST/MNIST_test.txt'
  call read_mnist(test_file,test_images, test_labels, &
       maxval(cv_kernel_size), itmp1, padding_method)
  num_samples_test = size(test_images, 4)


!!!-----------------------------------------------------------------------------
!!! initialise monitoring variables
!!!-----------------------------------------------------------------------------
  history_length = max(ceiling(500._real12/batch_size),1)
  do i=1,size(metric_dict,dim=1)
     allocate(metric_dict(i)%history(history_length))
     metric_dict(i)%history = -huge(1._real12)
  end do
  if(batch_learning)then
     exploding_check = 0._real12
     exploding_check_old = 0._real12
  else
     exploding_check = 1._real12
     exploding_check_old = 1._real12
  end if


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
  allocate(batch_order(num_batches))
  do batch = 1, num_batches
     batch_order(batch) = batch
  end do


!!!-----------------------------------------------------------------------------
!!! initialise convolutional and pooling layers
!!!-----------------------------------------------------------------------------
  write(6,*) "Initialising CNN..."
  !! Initialise the convolution layer
  num_layers = 6
  allocate(model(num_layers))
  allocate(model(1)%layer, source = input3d_layer_type(&
       input_shape = [28,28,1]))
  allocate(model(2)%layer, source = conv2d_layer_type( &
       input_shape = [28,28,1], &
       num_filters = cv_num_filters, kernel_size = 3, stride = 1, &
       padding=padding_method, &
       calc_input_gradients = .false., &
       activation_function = "relu"))
  allocate(model(3)%layer, source = maxpool2d_layer_type(&
       input_shape=[28,28,cv_num_filters], &
       pool_size=2, stride=2))
  allocate(model(4)%layer, source = flatten2d_layer_type(&
       input_shape=[14,14,cv_num_filters]))
  allocate(model(5)%layer, source = full_layer_type( &
       num_inputs=product([14,14,cv_num_filters]), &
       num_outputs=100, &
       activation_function = "relu", &
       kernel_initialiser="he_uniform", &
       bias_initialiser="he_uniform" &
       ))
  allocate(model(6)%layer, source = full_layer_type( &
       num_inputs=100, &
       num_outputs=10,&
       activation_function="softmax", &
       kernel_initialiser="glorot_uniform", &
       bias_initialiser="glorot_uniform" &
       ))

  allocate(y_pred(batch_size,num_classes), source=0._real12)
  allocate(y_true(batch_size,num_classes), source=0._real12)


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
  lw_image_size = lbound(input_images,dim=1)
  up_image_size = ubound(input_images,dim=1)


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
     compute_loss => compute_loss_cce
     write(*,*) "Loss method: Categorical Cross Entropy"
  case("mse")
     compute_loss => compute_loss_mse
     write(*,*) "Loss method: Mean Squared Error"
  case("nll")
     compute_loss => compute_loss_nll
     write(*,*) "Loss method: Negative log likelihood"
  case default
     write(*,*) "Failed loss method: "//trim(loss_method)
     stop "ERROR: No loss method provided"
  end select


!!!-----------------------------------------------------------------------------
!!! if parallel, initialise slices
!!!-----------------------------------------------------------------------------
#ifdef _OPENMP
  allocate(image_slice(&
       lw_image_size:up_image_size,&
       lw_image_size:up_image_size,&
       size(input_images,dim=3),batch_size))
  allocate(label_slice(batch_size))
#endif
  allocate(image_sample(&
       lw_image_size:up_image_size,&
       lw_image_size:up_image_size,&
       size(input_images,dim=3)&
       ))
  !drop_gamma = (1 - keep_prob)/block_size**2 * image_size**2/(image_size - block_size + 1)**2

!!!-----------------------------------------------------------------------------
!!! query system clock
!!!-----------------------------------------------------------------------------
  call system_clock(time, count_rate = clock_rate)


!!!-----------------------------------------------------------------------------
!!! training loop
!!! ... loops over num_epoch number of epochs
!!! ... i.e. it trains on the same datapoints num_epoch times
!!!-----------------------------------------------------------------------------
  write(6,*) "Starting training..."
  epoch_loop: do epoch = 1, num_epochs
     !!-------------------------------------------------------------------------
     !! shuffle batch order at the start of each epoch
     !!-------------------------------------------------------------------------
     if(shuffle_dataset)then
        call shuffle(batch_order)
     end if

     avg_loss     = 0._real12
     avg_accuracy = 0._real12

     !!-------------------------------------------------------------------------
     !! batch loop
     !! ... split data up into minibatches for training
     !!-------------------------------------------------------------------------
     batch_loop: do batch = 1, num_batches


        !! set batch start and end index
        !!----------------------------------------------------------------------
        start_index = (batch_order(batch) - 1) * batch_size + 1
        end_index = batch_order(batch) * batch_size
#ifdef _OPENMP
        image_slice(:,:,:,:) = input_images(:,:,:,start_index:end_index)
        label_slice(:) = labels(start_index:end_index)
        start_index = 1
        end_index = batch_size
#endif


        !! reinitialise variables
        !!----------------------------------------------------------------------
        y_true = 0._real12
        y_pred = 0._real12
!!!drop_gamma = (1 - drop_gamma)/block_size**2 * image_size**2/(image_size - block_size + 1)**2
        
        
        !!----------------------------------------------------------------------
        !! sample loop
        !! ... test each sample and get gradients and losses from each
        !!----------------------------------------------------------------------
        train_loop: do concurrent(sample:start_index:end_index:1)

#ifdef _OPENMP
           associate(input => image_slice(:,:,:,sample))
           y_true(sample-start_index+1,label_slice(sample)) = 1._real12
#else
           associate(input => input_images(:,:,:,sample))
           y_true(sample-start_index+1,label(sample)) = 1._real12
#endif
             select type(current => model(1)%layer)
             type is(input3d_layer_type)
                call current%init(input)
             end select
           end associate
           
           
           !! Forward pass
           !!-------------------------------------------------------------------
           do i=2,num_layers,1
              call model(i)%forward(model(i-1))
           end do


           !! Backward pass (final layer)
           !!-------------------------------------------------------------------
           select type(current => model(num_layers)%layer)
           type is(full_layer_type)
!!! SET UP LOSS TO APPLY A NORMALISER BY DEFAULT IF SOFTMAX NOT PREVIOUS
!!! (this is what keras does)
!!! ... USE current%transfer%name TO DETERMINE
!!! https://www.v7labs.com/blog/cross-entropy-loss-guide
!!! https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function
!!! https://math.stackexchange.com/questions/4367458/derivate-of-the-the-negative-log-likelihood-with-composition
              y_pred(sample-start_index+1,:) = current%output
              call model(num_layers)%backward(&
                   model(num_layers-1),&
                   compute_loss_derivative(&
                   current%output,y_true(sample-start_index+1,:)))
           end select

           
           !! Backward pass
           !!-------------------------------------------------------------------
           do i=num_layers-1,2,-1
              select type(next => model(i+1)%layer)
              type is(conv2d_layer_type)
                 call model(i)%backward(model(i-1),next%di)
              type is(maxpool2d_layer_type)
                 call model(i)%backward(model(i-1),next%di)
              type is(full_layer_type)
                 call model(i)%backward(model(i-1),next%di)
              type is(flatten2d_layer_type)
                 call model(i)%backward(model(i-1),next%di)
              type is(flatten3d_layer_type)
                 call model(i)%backward(model(i-1),next%di)
              end select
           end do
           
        end do train_loop

        
        !! compute loss and accuracy (for monitoring)
        !!-------------------------------------------------------------------
        batch_loss = 0._real12
        batch_accuracy = 0._real12
        do sample = 1, end_index-start_index+1, 1
           batch_loss = batch_loss + sum(compute_loss(&
                y_pred(sample,:),y_true(sample,:)))
           batch_accuracy = batch_accuracy + compute_accuracy(&
                y_pred(sample,:),maxloc(y_true(sample,:),dim=1))
        end do


        !! Average metric over batch size and store
        !!----------------------------------------------------------------------
        avg_loss = avg_loss + batch_loss
        avg_accuracy = avg_accuracy + batch_accuracy
        metric_dict(1)%val = batch_loss / batch_size
        metric_dict(2)%val = batch_accuracy / batch_size


        !! Check metric convergence
        !!----------------------------------------------------------------------
        do i=1,size(metric_dict,dim=1)
           call metric_dict(i)%check(plateau_threshold, converged)
           if(converged.ne.0)then
              exit epoch_loop
           end if
        end do
        
        
        !! if mini-batch ...
        !! ... update weights and biases using optimization algorithm
        !! ... (gradient descent)
        !!----------------------------------------------------------------------
        if(batch_learning)then
           exploding_check = (exploding_check/batch_size)
           if(epoch.gt.1.or.batch.gt.1)then
              rtmp1 = abs(exploding_check/exploding_check_old)
              if(rtmp1.gt.1.E3_real12)then
                 write(0,*) "WARNING: FC outputs are expanding too quickly!"
                 write(0,*) "check:", sample,exploding_check,exploding_check_old
              elseif(rtmp1.lt.1.E-3_real12)then
                 write(0,*) "WARNING: FC outputs are vanishing too quickly!"
                 write(0,*) "check:", exploding_check, exploding_check_old
              end if
           end if
           exploding_check_old = exploding_check
           exploding_check = 0._real12

           !! STORE ADAM VALUES IN OPTIMISER
           do i=2, num_layers,1
              select type(current => model(i)%layer)
              type is(conv2d_layer_type)
                 current%dw = current%dw/batch_size
                 current%db = current%db/batch_size
                 call current%update(optimiser,cv_clip)
              type is(full_layer_type)
                 current%dw = current%dw/batch_size
                 call current%update(optimiser,fc_clip)                 
              end select
           end do
           optimiser%iter = optimiser%iter + 1
        end if


        !! print batch results
        !!----------------------------------------------------------------------
101     if(abs(verbosity).gt.0.and.&
             (batch.eq.1.or.mod(batch,batch_print_step).eq.0.E0))then
           write(6,'("epoch=",I0,", batch=",I0,&
                &", learning_rate=",F0.3,", loss=",F0.3,", accuracy=",F0.3)') &
                epoch, batch, optimiser%learning_rate, &
                avg_loss/(batch*batch_size),  avg_accuracy/(batch*batch_size)
           !metric_dict(1)%val, metric_dict(2)%val
        end if

!!! TESTING
        if(batch.gt.200)then
           time_old = time
           call system_clock(time)
           !write(*,'("time check: ",I0," seconds")') (time-time_old)/clock_rate
           write(*,'("time check: ",F8.3," seconds")') real(time-time_old)/clock_rate
           stop "THIS IS FOR TESTING PURPOSES"
        end if
!!!
        
        !! time check
        !!----------------------------------------------------------------------
        if(verbosity.eq.-2)then
           time_old = time
           call system_clock(time)
           !write(*,'("time check: ",I0," seconds")') (time-time_old)/clock_rate
           write(*,'("time check: ",F5.3," seconds")') real(time-time_old)/clock_rate
           time_old = time
        end if
        
        
        !! check for user-name stop file
        !!----------------------------------------------------------------------
        if(stop_check())then
           write(0,*) "STOPCAR ENCOUNTERED"
           write(0,*) "Exiting training loop..."
           exit epoch_loop
        end if

     end do batch_loop


     !! print epoch summary results
     !!-------------------------------------------------------------------------
     if(verbosity.eq.0)then
        !!if(mod(epoch,20).eq.0.E0) &
        write(6,'("epoch=",I0,", batch=",I0,&
             &", learning_rate=",F0.3,", val_loss=",F0.3,", val_accuracy=",F0.3)') &
             epoch, batch, optimiser%learning_rate, &
             metric_dict(1)%val, metric_dict(2)%val
        !     avg_loss/(batch*batch_size),  avg_accuracy/(batch*batch_size)
     end if


  end do epoch_loop
  write(*,*) "Training finished"


!!!!-----------------------------------------------------------------------------
!!!! print weights and biases of CNN to file
!!!!-----------------------------------------------------------------------------
!  write(*,*) "Writing CNN learned parameters to output file"
!  open(unit=10,file=output_file,status='replace')
!  close(10)
!  call cv_write(output_file)
!  call fc_write(output_file)
!
!  if(verbosity.gt.1) open(unit=15,file="results_test.out")
!

!!!-----------------------------------------------------------------------------
!!! testing loop
!!!-----------------------------------------------------------------------------
!!! CAN PARALLELISE THIS SECTION AS THEY ARE INDEPENDENT
  write(*,*) "Starting testing..."
  metric_dict%val = 0._real12
  test_loop: do sample = 1, num_samples_test
     y_true(1,:) = 0._real12
     y_true(1,test_labels(sample)) = 1._real12

     associate(input => test_images(:,:,:,sample))
       select type(current => model(1)%layer)
       type is(input3d_layer_type)
          call current%init(input)
       end select
     end associate

     
     !! Forward pass
     !!-------------------------------------------------------------------------
     do i=2,num_layers,1
        call model(i)%forward(model(i-1))
     end do


     !! compute loss and accuracy (for monitoring)
     !!-------------------------------------------------------------------------
     select type(current => model(num_layers)%layer)
     type is(full_layer_type)
        current%output = exp(current%output - maxval(current%output))
        current%output = current%output/sum(current%output)
        accuracy = compute_accuracy(current%output, test_labels(sample))
        metric_dict(1)%val = metric_dict(1)%val + sum(&
             compute_loss(&
             predicted=current%output, expected=y_true(1,:)))
        metric_dict(2)%val = metric_dict(2)%val + accuracy
        !! print testing results
        !!----------------------------------------------------------------------
        if(abs(verbosity).gt.1)then
           write(15,'(I4," Expected=",I3,", Got=",I3,", Accuracy=",F0.3)') &
                sample,test_labels(sample), maxloc(current%output,dim=1)-1, accuracy
        end if
     end select

  end do test_loop
  if(verbosity.gt.1) close(15)
  write(*,*) "Testing finished"

  write(6,'("Overall accuracy=",F0.5)') metric_dict(2)%val/real(num_samples_test)
  write(6,'("Overall loss=",F0.5)')     metric_dict(1)%val/real(num_samples_test)
  
  
  
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
