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
  use batch_norm, only: bn_init => initialise, &
       bn_forward => forward, &
       bn_backward => backward, &
       bn_update => update
  use normalisation, only: linear_renormalise, &
       renormalise_norm, renormalise_sum
  use loss_categorical!, only: loss_mse, loss_nll, loss_cce, loss_type
  use inputs
  use ConvolutionLayer, only: cv_init => initialise, cv_forward => forward, &
       cv_backward => backward, &
       cv_update => update_weights_and_biases, &
       cv_write => write_file, &
       cv_gradient_type => gradient_type, &
       cv_gradient_alloc => allocate_gradients, &
       cv_gradient_init => initialise_gradients, &
       convolution
  use PoolingLayer, only: pl_init => initialise, pl_forward => forward, &
       pl_backward => backward
  use FullyConnectedLayer, only: fc_init => initialise, fc_forward => forward, &
       fc_backward => backward, &
       fc_update => update_weights_and_biases, &
       fc_write => write_file, &
       fc_gradient_type => gradient_type, &
       fc_gradient_alloc => allocate_gradients, &
       fc_gradient_init => initialise_gradients, &
       fc_hidden_output_type => hidden_output_type, &
       network
  use SoftmaxLayer, only: sm_init => initialise, sm_forward => forward, &
       sm_backward => backward

  implicit none

  !! seed variables
  integer :: nseed=1
  integer, allocatable, dimension(:) :: seed_arr

  !! training and testing monitoring
  integer :: predicted_old, predicted_new
  real(real12) :: accuracy, sum_accuracy, sum_loss, overall_loss
  real(real12) :: exploding_check, exploding_check_old
  logical :: repetitive_predicting
  real(real12), allocatable, dimension(:) :: loss_history
  !class(loss_type), pointer :: get_loss
  procedure(compute_loss_function), pointer :: compute_loss

  !! learning parameters
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

  !! training loop variables
  integer :: num_batches, num_samples, num_samples_test
  integer :: epoch, batch, sample, start_index, end_index
  integer :: expected
  integer, allocatable, dimension(:) :: batch_order
  type(fc_hidden_output_type), allocatable, dimension(:) :: fc_output
  real(real12), allocatable, dimension(:) :: fc_input, &!fc_output, &
       sm_output, sm_gradients
  real(real12), allocatable, dimension(:,:,:) :: cv_output, pl_output, &
       pl_gradients, fc_gradients_rs

  type(cv_gradient_type), allocatable, dimension(:) :: cv_gradients, comb_cv_gradients
  type(fc_gradient_type), allocatable, dimension(:) :: fc_gradients, &
       comb_fc_gradients


  integer :: i, l, time, time_old, clock_rate, itmp1, cv_mask_size
  real(real12) :: rtmp1, drop_gamma
  logical, allocatable, dimension(:,:) :: cv_mask

  real(real12), allocatable, dimension(:,:,:) :: image_sample
#ifdef _OPENMP
  integer, allocatable, dimension(:) :: label_slice
  real(real12), allocatable, dimension(:,:,:,:) :: image_slice
#endif


!!!-----------------------------------------------------------------------------
!!! set up reduction for gradient custom type
!!! ...
!! https://www.openmp.org/spec-html/5.0/openmpsu107.html
!! https://stackoverflow.com/questions/61141297/openmp-reduction-on-user-defined-fortran-type-containing-allocatable-array
!! https://fortran-lang.discourse.group/t/openmp-reduction-on-operator/5887
!!!-----------------------------------------------------------------------------
  !$omp declare reduction(cv_grad_sum:cv_gradient_type:omp_out = omp_out + omp_in) &
  !$omp& initializer(cv_gradient_alloc(omp_priv, omp_orig, .false.))
  !$omp declare reduction(fc_grad_sum:fc_gradient_type:omp_out = omp_out + omp_in) &
  !$omp& initializer(fc_gradient_alloc(omp_priv, omp_orig, .false.))
  !$omp declare reduction(compare_val:integer:compare_val(omp_out,omp_in)) &
  !$omp& initializer(omp_priv = omp_orig)


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
  if(batch_size.lt.50)then
     allocate(loss_history(100))
  elseif(batch_size.lt.num_samples.and.batch_size.lt.500)then
     allocate(loss_history(2*batch_size))
  else
     allocate(loss_history(min(num_samples,100)))
  end if
  loss_history = -huge(1._real12)
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
  if(restart)then
     call cv_init(file = input_file, &
          learning_parameters=learning_parameters)
  else
     call cv_init(seed, num_layers = cv_num_filters, &
          kernel_size = cv_kernel_size, stride = cv_stride, &
          full_padding = trim(padding_method).eq."full",&
          learning_parameters=learning_parameters,&
          kernel_initialiser=cv_kernel_initialiser,&
          bias_initialiser=cv_bias_initialiser,&
          activation_scale=cv_activation_scale,&
          activation_function=cv_activation_function)
  end if

  output_size = floor( (&
       image_size + 2.0 * maxval(convolution(:)%pad) - maxval(cv_kernel_size)&
       )/minval(cv_stride) ) + 1

  !! Initialise the pooling layer
  call pl_init(pool_kernel_size, pool_stride)
  num_pool = (output_size - pool_kernel_size) / pool_stride + 1
  input_size = num_pool**2 * output_channels
  lw_image_size = lbound(input_images,dim=1)
  up_image_size = ubound(input_images,dim=1)


!!!-----------------------------------------------------------------------------
!!! reformulate fully connected layers to include input and output layers
!!! ... user provides only hidden layers
!!!-----------------------------------------------------------------------------
  fc_num_layers = size(fc_num_hidden,dim=1) + 1
  allocate(tmp_num_hidden(fc_num_layers))
  tmp_num_hidden(1:fc_num_layers-1) = fc_num_hidden
  tmp_num_hidden(fc_num_layers) = num_classes
  call move_alloc(tmp_num_hidden, fc_num_hidden)


!!!-----------------------------------------------------------------------------
!!! initialise fully connected and softmax layers
!!!-----------------------------------------------------------------------------
  !! Initialise the fully connected layer
  if(restart)then
     call fc_init(file = input_file, &
          learning_parameters=learning_parameters)
  else
     call fc_init(seed, num_layers=fc_num_layers, &
          num_inputs=input_size, num_hidden=fc_num_hidden, &
          activation_function=fc_activation_function, &
          activation_scale=fc_activation_scale,&
          learning_parameters=learning_parameters,&
          weight_initialiser=fc_weight_initialiser)
  end if

  !! Initialise the softmax layer
  call sm_init(num_classes)
  write(6,*) "CNN initialised"


!!!-----------------------------------------------------------------------------
!!! allocate and initialise layer outputs and gradients
!!!-----------------------------------------------------------------------------
  allocate(cv_output(output_size, output_size, output_channels))
  allocate(pl_output(num_pool, num_pool, output_channels))
  allocate(sm_output(num_classes))
  cv_output = 0._real12
  pl_output = 0._real12
  sm_output = 0._real12
  allocate(fc_output(fc_num_layers))
  do l=1,fc_num_layers
     allocate(fc_output(l)%val(fc_num_hidden(l)))
     fc_output(l)%val = 0._real12
  end do
  !allocate(cv_output_norm(output_size, output_size, output_channels))
  !cv_output_norm = 0._real12
  !allocate(mean(output_channels))
  !allocate(variance(output_channels))


!!!-----------------------------------------------------------------------------
!!! initialise non-fully connected layer gradients
!!!-----------------------------------------------------------------------------
  select case(learning_parameters%method)
  case("adam")
    call cv_gradient_init(cv_gradients, image_size, adam_learning = .true.)
     if(batch_learning) &
          call cv_gradient_init(comb_cv_gradients, image_size, adam_learning = .true.)
     update_iteration = 1
  case default
     call cv_gradient_init(cv_gradients, image_size)
     if(batch_learning) call cv_gradient_init(comb_cv_gradients, image_size)     
  end select
  allocate(pl_gradients,mold=cv_output)
  allocate(sm_gradients(num_classes))
  pl_gradients = 0._real12
  sm_gradients = 0._real12


!!!-----------------------------------------------------------------------------
!!! initialise fully connected layer inputs and gradients
!!!-----------------------------------------------------------------------------
  allocate(fc_input(input_size))
  fc_input = 0._real12
  allocate(fc_gradients_rs,mold=pl_output)
  fc_gradients_rs = 0._real12

  select case(learning_parameters%method)
  case("adam")
    call fc_gradient_init(fc_gradients, input_size, adam_learning = .true.)
     if(batch_learning) &
          call fc_gradient_init(comb_fc_gradients, input_size, adam_learning = .true.)
     update_iteration = 1
  case default
     call fc_gradient_init(fc_gradients, input_size)
     if(batch_learning) call fc_gradient_init(comb_fc_gradients, input_size)     
  end select


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


        !! reset (zero) summed gradients
        !! ... UNCOMMENT if running mini-batch training 
        !!----------------------------------------------------------------------
        if(batch_learning)then
           do l=1,cv_num_filters
              comb_cv_gradients(l)%weight = 0._real12
              comb_cv_gradients(l)%bias = 0._real12
           end do
           do l=1,fc_num_layers
              comb_fc_gradients(l)%weight = 0._real12
           end do
        end if


        !! reinitialise variables
        !!----------------------------------------------------------------------
        sum_loss = 0._real12
        sum_accuracy = 0._real12
        predicted_old = -1
        predicted_new = -1
        repetitive_predicting = .true.
        
        !!!drop_gamma = (1 - drop_gamma)/block_size**2 * image_size**2/(image_size - block_size + 1)**2

        
        !!----------------------------------------------------------------------
        !! sample loop
        !! ... test each sample and get gradients and losses from each
        !!----------------------------------------------------------------------
        !$OMP PARALLEL DO & !! ORDERED
        !$OMP& DEFAULT(NONE) &
        !$OMP& SHARED(network, convolution) &
        !$OMP& SHARED(start_index, end_index) &
        !$OMP& SHARED(image_slice, label_slice) &
        !$OMP& SHARED(input_size, batch_size, pool_normalisation) &
        !$OMP& SHARED(fc_clip, cv_clip, batch_learning) &
        !$OMP& SHARED(fc_num_layers,cv_num_filters) &
        !$OMP& SHARED(cv_keep_prob, seed, cv_block_size, output_channels) &
        !$OMP& SHARED(cv_dropout_method) &
        !$OMP& SHARED(compute_loss) &
!!        !$OMP& FIRSTPRIVATE(network) &
        !$OMP& FIRSTPRIVATE(predicted_old) &
!!        !$OMP& PRIVATE(cv_mask, cv_mask_size) &
        !$OMP& PRIVATE(sample) &
        !$OMP& PRIVATE(fc_gradients, cv_gradients) &
        !$OMP& PRIVATE(cv_output) &
        !$OMP& PRIVATE(pl_output, pl_gradients) &
        !$OMP& PRIVATE(fc_input, fc_output, fc_gradients_rs) &
        !$OMP& PRIVATE(sm_output, sm_gradients) &
        !$OMP& PRIVATE(rtmp1, expected, exploding_check_old) &
        !$OMP& REDUCTION(compare_val:predicted_new) &
        !$OMP& REDUCTION(.and.: repetitive_predicting) &
        !$OMP& REDUCTION(+:sum_loss,sum_accuracy,exploding_check) &
        !$OMP& REDUCTION(cv_grad_sum:comb_cv_gradients) &
        !$OMP& REDUCTION(fc_grad_sum:comb_fc_gradients)
       train_loop: do sample = start_index, end_index

          !image_sample(:,:,:) = image_slice(:,:,:,sample)

          !! Forward pass
          !!-------------------------------------------------------------------
#ifdef _OPENMP
           !image_sample = image_slice(:,:,:,sample)
           call cv_forward(image_slice(:,:,:,sample), cv_output)
#else
           !image_sample = input_images(:,:,:,sample)
           call cv_forward(input_images(:,:,:,sample), cv_output)
#endif

           !! apply a form of dropout regularisation
           !if(cv_dropout_method.eq."dropblock")then
           !   !call generate_bernoulli_mask(cv_mask, drop_gamma, seed)
           !   if(allocated(cv_mask)) deallocate(cv_mask)
           !
           !   cv_mask_size = size(cv_output,dim=1) - &
           !        ( 2*int((cv_block_size -1)/2) + (1 - mod(cv_block_size,2)))
           !   allocate(cv_mask(cv_mask_size, cv_mask_size))
           !
           !   call generate_bernoulli_mask(cv_mask, cv_keep_prob, seed)
           !
           !   do i=1,output_channels
           !      !! need to make cv_output a custom type
           !      !! then set the mask to different size based on filter
           !      !! or just have a bounding box inside the drop_block
           !      call drop_block(cv_output(:,:,i), cv_mask, cv_block_size)
           !   end do
           !end if

           !call bn_forward(cv_output, mean, variance, &
           !     gamma=bn_gamma, beta=bn_beta, input_norm=cv_output_norm)

           call pl_forward(cv_output, pl_output)
           fc_input = reshape(pl_output, [input_size])
           select case(pool_normalisation)
           case("linear")
              call linear_renormalise(fc_input)
           case("norm")
              call renormalise_norm(fc_input, norm=1._real12, mirror=.true.)
           case("sum")
              call renormalise_sum(fc_input, norm=1._real12, mirror=.true., magnitude=.true.)
           end select
           call fc_forward(fc_input, fc_output)
           call sm_forward(fc_output(fc_num_layers)%val, sm_output)
           !write(*,*) sm_output
 
  
           !! check for NaN and infinity
           !!-------------------------------------------------------------------
           if(any(isnan(sm_output)))then
              write(0,*) fc_input
              write(0,*) fc_output(fc_num_layers)%val
              write(0,*) sm_output
              stop "ERROR: Softmax outputs are NaN"
           end if
           if(batch_learning)then
              exploding_check = exploding_check + sum(fc_output(fc_num_layers)%val)
           else
#ifdef _OPENMP
              stop "ERROR: non-batch learning not yet parallelised"
#endif
              exploding_check_old = exploding_check
              exploding_check = sum(fc_output(fc_num_layers)%val)
              !exploding_check=mean(fc_output)/exploding_check
              rtmp1 = abs(exploding_check/exploding_check_old)
              if(rtmp1.gt.1.E3_real12)then
                 write(0,*) "WARNING: FC outputs are expanding too quickly!"
                 write(0,*) "check:", sample,exploding_check,exploding_check_old      
                 write(0,*) "outputs:", fc_output(fc_num_layers)%val
              elseif(rtmp1.lt.1.E-3_real12)then
                 write(0,*) "WARNING: FC outputs are vanishing too quickly!"
                 write(0,*) "check:", sample,exploding_check,exploding_check_old
                 write(0,*) "outputs:", fc_output(fc_num_layers)%val
              end if
           end if

           
           !! compute loss and accuracy (for monitoring)
           !!-------------------------------------------------------------------
#ifdef _OPENMP
           expected = label_slice(sample)
#else
           expected = labels(sample)
#endif
           sum_loss = sum_loss + compute_loss(predicted=sm_output, expected=expected)
           sum_accuracy = sum_accuracy + compute_accuracy(sm_output, expected)


           !! check that isn't just predicting same value every time
           !!-------------------------------------------------------------------
           predicted_new = maxloc(sm_output,dim=1)-1
           if(repetitive_predicting.and.predicted_old.gt.-1)then
              repetitive_predicting = predicted_old.eq.predicted_new
           end if
           predicted_old = predicted_new
           !write(0,'(I0,2X,I0,2X,10(1X,F5.3))') expected, predicted_new, sm_output


           !! Backward pass
           !!-------------------------------------------------------------------
           call sm_backward(sm_output, expected, sm_gradients)
           call fc_backward(fc_input, fc_output, sm_gradients, fc_gradients, fc_clip)
           fc_gradients_rs = reshape(fc_gradients(0)%delta,&
                shape(fc_gradients_rs))
           call pl_backward(cv_output, fc_gradients_rs, pl_gradients)
#ifdef _OPENMP
           call cv_backward(image_slice(:,:,:,sample), pl_gradients, &
                cv_gradients, cv_clip)
#else
           call cv_backward(input_images(:,:,:,sample), pl_gradients, &
                cv_gradients, cv_clip)
#endif
           

           !! if mini-batch ...
           !! ... sum gradients for mini-batch training
           !! if not mini-batch
           !! ... update weights and biases using optimization algorithm
           !! ... (gradient descent)
           !!-------------------------------------------------------------------
           if(batch_learning)then
              comb_cv_gradients = comb_cv_gradients + cv_gradients
              comb_fc_gradients = comb_fc_gradients + fc_gradients
#ifndef _OPENMP
           else
              !call fc_gradient_check(fc_gradients, fc_input)
              !write(*,*)
              !write(*,*)
              !call cv_gradient_check(cv_gradients, input_images(:,:,:,sample))
              !stop
              call cv_update(learning_rate, cv_gradients, update_iteration)
              call fc_update(learning_rate, fc_gradients, update_iteration)
#endif
           end if


        end do train_loop
        !$OMP END PARALLEL DO


        !! Check if categorical predicting is stuck on same value
        !!----------------------------------------------------------------------
#ifdef _OPENMP
        if(repetitive_predicting.and.&
             all(label_slice(:).eq.label_slice(1))) &
             repetitive_predicting = .false.
#else
        if(repetitive_predicting.and.&
             all(labels(start_index:end_index).eq.labels(start_index))) &
             repetitive_predicting = .false.
#endif
        if(repetitive_predicting.and.predicted_new.gt.0)then
           write(0,'("WARNING: all predictions in batch ",I0," &
                &are the same: ", I0)') batch, predicted_new-1
           write(0,*) "WE SHOULD REALL DO SOMETHING TO KICK IT OUT OF THIS"
           !do l=1,fc_num_layers
           !   do i=1,size(comb_fc_gradients(l)%val,dim=1)
           !      call random_number(comb_fc_gradients(l)%val(i))
           !   end do
           !end do
        end if



        !! Average accuracy and loss over batch size and store
        !!----------------------------------------------------------------------
        sum_loss = sum_loss / batch_size
        loss_history = cshift(loss_history, shift=-1, dim=1)
        loss_history(1) = sum_loss
        sum_accuracy = sum_accuracy / batch_size


        !! Check loss convergence
        !!----------------------------------------------------------------------
        if(abs(sum(loss_history)).lt.loss_threshold)then
           write(6,*) "Convergence achieved, accuracy threshold reached"
           write(6,*) "Exiting training loop"
           exit epoch_loop
        elseif(all(abs(loss_history-sum_loss).lt.plateau_threshold))then
           !write(0,*) "sm_output", sm_output
           !write(0,*) "sm_grad", sm_gradients
           !write(0,*) "comv_fc_gradients", comb_fc_gradients(1)%val          
           write(0,*) "ERROR: loss has remained constant for 10 runs"
           write(0,*) loss_history
           write(0,*) "Exiting..."
           stop
           exit epoch_loop
        end if


        !! Error checking and handling
        !!----------------------------------------------------------------------
        if(abs(1._real12-sum_accuracy).gt.1.E-3_real12)then !!! HAVE THIS TIED TO BATCH SIZE
           do l=1,fc_num_layers
              if(batch_learning)then
                 if(all(abs(comb_fc_gradients(l)%weight).lt.1.E-8_real12))then
                    write(0,*) "WARNING: FullyConnected gradients are zero"
                    !write(0,*) "Exiting..."
                    !stop
                 end if
              else
                 if(all(abs(fc_gradients(l)%weight).lt.1.E-8_real12))then
                    write(0,*) "WARNING: FullyConnected gradients are zero"
                    !write(0,*) "Exiting..."
                    !stop
                 end if
              end if
           end do
        else
           goto 101
        end if


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
           
           do l=1,cv_num_filters
              comb_cv_gradients(l)%weight = comb_cv_gradients(l)%weight/batch_size
              comb_cv_gradients(l)%bias   = comb_cv_gradients(l)%bias/batch_size
           end do
           do l=1,fc_num_layers
              comb_fc_gradients(l)%weight = comb_fc_gradients(l)%weight/batch_size
           end do
           call cv_update(learning_rate, comb_cv_gradients, update_iteration)
           call fc_update(learning_rate, comb_fc_gradients, update_iteration)
        end if


        !! print batch results
        !!----------------------------------------------------------------------
101     if(abs(verbosity).gt.0.and.&
             (batch.eq.1.or.mod(batch,batch_print_step).eq.0.E0))then
           write(6,'("epoch=",I0,", batch=",I0,&
                &", learning_rate=",F0.3,", loss=",F0.3,", accuracy=",F0.3)') &
                epoch, batch, learning_rate, sum_loss, sum_accuracy
        end if


        !! time check
        !!----------------------------------------------------------------------
        if(verbosity.eq.-2)then
           time_old = time
           call system_clock(time)
           write(6,'("time check: ",I0," seconds")') (time-time_old)/clock_rate
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
             &", learning_rate=",F0.3,", loss=",F0.3,", accuracy=",F0.3)') &
             epoch, batch, learning_rate, sum_loss, sum_accuracy
     end if


  end do epoch_loop
  write(*,*) "Training finished"


!!!-----------------------------------------------------------------------------
!!! print weights and biases of CNN to file
!!!-----------------------------------------------------------------------------
  write(*,*) "Writing CNN learned parameters to output file"
  open(unit=10,file=output_file,status='replace')
  close(10)
  call cv_write(output_file)
  call fc_write(output_file)

  if(verbosity.gt.1) open(unit=15,file="results_test.out")


!!!-----------------------------------------------------------------------------
!!! testing loop
!!!-----------------------------------------------------------------------------
!!! CAN PARALLELISE THIS SECTION AS THEY ARE INDEPENDENT
  write(*,*) "Starting testing..."
  sum_accuracy = 0._real12
  sum_loss = 0._real12
  test_loop: do sample = 1, num_samples_test

     call cv_forward(test_images(:,:,:,sample), cv_output)
     call pl_forward(cv_output, pl_output)
     fc_input = reshape(pl_output, [input_size])
     select case(pool_normalisation)
     case("linear")
        call linear_renormalise(fc_input)
     case("norm")
        call renormalise_norm(fc_input, norm=1._real12, mirror=.true.)
     case("sum")
        call renormalise_sum(fc_input, norm=1._real12, mirror=.true., magnitude=.true.)
     end select
     call fc_forward(fc_input, fc_output)
     call sm_forward(fc_output(fc_num_layers)%val, sm_output)


     !! compute loss and accuracy (for monitoring)
     !!-------------------------------------------------------------------------
     expected = test_labels(sample)
     sum_loss = sum_loss + compute_loss(predicted=sm_output, expected=expected)     
     accuracy = compute_accuracy(sm_output, expected)
     sum_accuracy = sum_accuracy + accuracy


     !! print testing results
     !!-------------------------------------------------------------------------
     if(abs(verbosity).gt.1)then
        write(15,'(I4," Expected=",I3,", Got=",I3,", Accuracy=",F0.3)') &
             sample,expected-1, maxloc(sm_output,dim=1)-1, accuracy
     end if
     if(verbosity.lt.1)then
        write(0,*) sm_output
        write(0,*)
     end if

  end do test_loop
  if(verbosity.gt.1) close(15)
  write(*,*) "Testing finished"

  overall_loss = real(sum_loss)/real(num_samples_test)
  write(6,'("Overall accuracy=",F0.5)') sum_accuracy/real(num_samples_test)
  write(6,'("Overall loss=",F0.5)') sum_loss/real(num_samples_test)



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

    !if(any(isnan(output)))then
    !   write(0,*) "Output is NaN"
    !   stop
    !end if

    !! Compute the accuracy
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
  subroutine read_mnist(file,images,labels,kernel_size,image_size,padding_method)
    use misc, only: icount
    use misc_ml, only: get_padding_half
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
100    select case(t_padding_method)
       case("none")
          t_padding_method = "valid"
          goto 100
       case("zero")
          t_padding_method = "same"
          goto 100
       case("half")
          t_padding_method = "same"
          goto 100
       case("symmetric")
          t_padding_method = "replication"
          goto 100
       case("valid")
          write(6,*) "Padding type: 'valid' (all possible positions)"
       case("same")
          write(6,*) "Padding type: 'same' (pad with zeros)"
       case("circular")
          write(6,*) "Padding type: 'same' (circular padding)"
       case("full")
          write(6,*) "Padding type: 'full' (all possible positions)"
       case("reflection")
          write(6,*) "Padding type: 'reflection' (reflect on boundary)"
       case("replication")
          write(6,*) "Padding type: 'replication' (reflect after boundary)"
       case default
          stop "ERROR: padding type '"//t_padding_method//"' not known"
       end select
    else
       t_padding_method = "same"
    end if
    

!!!-----------------------------------------------------------------------------
!!! allocate data set
!!! ... if appropriate, add padding
!!!-----------------------------------------------------------------------------
    !! dim=1: image width in pixels
    !! dim=2: image height in pixels
    !! dim=3: image number of channels (1 due to black-white images)
    !! dim=4: number of images
    if(t_padding_method.eq."valid")then
       padding = 0
       if(allocated(images)) deallocate(images)
       allocate(images(image_size, image_size, 1, num_samples))
    elseif(present(kernel_size))then

       !! calculate padding width
       !!-----------------------------------------------------------------------
       select case(t_padding_method)
       case("full")
          padding = kernel_size - 1
       case default
          padding = get_padding_half(kernel_size)
       end select

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




!!!#############################################################################
!!!
!!!#############################################################################
  subroutine cv_gradient_check(gradients, image, epsilon)
    implicit none
    type(cv_gradient_type), dimension(:) :: gradients
    real(real12), dimension(:,:,:) :: image
    real(real12), optional, intent(in) :: epsilon

    integer :: l,i,j
    real(real12) :: t_epsilon
    real(real12) :: loss, lossPlus, lossMinus, numericalGradient
    real(real12) :: weight_store
    
    if(present(epsilon))then
       t_epsilon = epsilon
    else
       t_epsilon = 1.E-1_real12
    end if
    
    
!!! MAKE UP AN IMAGE WE WANT TO ACHIEVE, i.e. 0's everywhere
!!! TRY TO LEARN THAT, THEN NO NEED FOR ANYTHING ELSE PAST CV, maybe also pooling


    ! Compute the numerical gradients and compare with computed gradients
    ! for each weight and bias parameter

    ! with the perturbed weight parameter
    call cv_forward(image, cv_output)
    loss = sum(cv_output - 1._real12)

    !write(*,*) "image check", image
    do l=1,size(gradients,1)
       

       ! Loop over each weight and bias parameter
       do i = lbound(gradients(l)%weight,1),ubound(gradients(l)%weight,1)
          do j = lbound(gradients(l)%weight,2),ubound(gradients(l)%weight,2)
             weight_store = convolution(l)%weight(i,j)


             ! Perturb the weight parameter slightly
             convolution(l)%weight(i,j) = weight_store + t_epsilon

             ! Perform a forward pass and compute the loss
             ! with the perturbed weight parameter
             call cv_forward(image, cv_output)
             !call pl_forward(cv_output, pl_output)
             !fc_input = reshape(pl_output, [size(fc_input,1)])
             !call fc_forward(fc_input, fc_output)
             !call sm_forward(fc_output, sm_output)
             !lossPlus = compute_loss(predicted=sm_output, expected=expected)
             lossPlus = sum(cv_output - 1._real12)
             !write(*,*) cv_output
             

             ! Perturb the weight parameter in the opposite direction
             convolution(l)%weight(i,j) = weight_store - t_epsilon

             ! Perform a forward pass and compute the loss
             ! with the perturbed weight parameter
             call cv_forward(image, cv_output)
             lossMinus = sum(cv_output - 1._real12)


             numericalGradient = (loss - lossMinus) / t_epsilon
             numericalGradient = numericalGradient + (lossPlus - loss)/t_epsilon
             numericalGradient = numericalGradient / 2._real12

             ! Compute the numerical gradient
             !numericalGradient = (lossPlus - lossMinus) / (2._real12 * t_epsilon)

             ! Restore the original weight parameter value
             convolution(l)%weight(i,j) = weight_store


             ! Compare the numerical gradient with the computed gradient
             if (abs(numericalGradient - gradients(l)%weight(i,j)).gt.t_epsilon) then
                write(*,*) "Gradient check failed for parameter ", i,j,l
                write(*,*) numericalGradient, gradients(l)%weight(i,j)
             else
                write(*,*) "Gradient check passed for parameter ", i,j,l
                write(*,*) numericalGradient, gradients(l)%weight(i,j)
             end if

          end do
       end do


       weight_store = convolution(l)%bias
       
       ! Perturb the weight parameter slightly
       convolution(l)%bias = weight_store + t_epsilon
       
       ! Perform a forward pass and compute the loss
       ! with the perturbed weight parameter
       call cv_forward(image, cv_output)
       lossPlus = sum(cv_output - 1._real12)

       ! Perturb the weight parameter in the opposite direction
       convolution(l)%bias = weight_store - t_epsilon

       ! Perform a forward pass and compute the loss
       ! with the perturbed weight parameter
       call cv_forward(image, cv_output)
       lossMinus = sum(cv_output - 1._real12)

       
       numericalGradient = (loss - lossMinus) / t_epsilon
       numericalGradient = numericalGradient + (lossPlus - loss)/t_epsilon
       numericalGradient = numericalGradient / 2._real12

       ! Restore the original weight parameter value
       convolution(l)%bias = weight_store

       
       ! Compare the numerical gradient with the computed gradient
       if (abs(numericalGradient - gradients(l)%bias).gt.t_epsilon) then
          write(*,*) "Gradient check failed for parameter bias", l
          write(*,*) numericalGradient, gradients(l)%bias
       else
          write(*,*) "Gradient check passed for parameter bias", l
          write(*,*) numericalGradient, gradients(l)%bias
       end if


    end do

  end subroutine cv_gradient_check
!!!#############################################################################


!!!#############################################################################
!!!
!!!#############################################################################
  subroutine fc_gradient_check(gradients, input, epsilon)
    implicit none
    type(fc_gradient_type), dimension(0:), intent(in) :: gradients
    real(real12), dimension(:), intent(in) :: input
    real(real12), optional, intent(in) :: epsilon

    integer :: l,i,j
    real(real12) :: t_epsilon
    real(real12) :: loss, lossPlus, lossMinus, numericalGradient
    real(real12) :: weight_store
    
    if(present(epsilon))then
       t_epsilon = epsilon
    else
       t_epsilon = 1.E-3_real12
    end if

    
    ! Compute the numerical gradients and compare with computed gradients
    ! for each weight and bias parameter
    
    do l=1,ubound(gradients,1)

       write(*,*) "Layer",l
       ! Loop over each weight and bias parameter
       do i = lbound(gradients(l)%weight,2),ubound(gradients(l)%weight,2)
          do j = lbound(gradients(l)%weight,1),ubound(gradients(l)%weight,1)
             weight_store = network(l)%neuron(i)%weight(j)

             ! Perturb the weight parameter slightly
             network(l)%neuron(i)%weight(j) = weight_store + t_epsilon

             ! Perform a forward pass and compute the loss
             ! with the perturbed weight parameter
             call fc_forward(input, fc_output)
             call sm_forward(fc_output(size(fc_output,dim=1))%val, sm_output)
             lossPlus = compute_loss(predicted=sm_output, expected=expected)


             ! Perturb the weight parameter in the opposite direction
             network(l)%neuron(i)%weight(j) = weight_store - t_epsilon

             ! Perform a forward pass and compute the loss
             ! with the perturbed weight parameter
             call fc_forward(input, fc_output)
             call sm_forward(fc_output(size(fc_output,dim=1))%val, sm_output)
             lossMinus = compute_loss(predicted=sm_output, expected=expected)

             ! Compute the numerical gradient
             numericalGradient = (lossPlus - lossMinus) / (2._real12 * t_epsilon)

             ! Restore the original weight parameter value
             network(l)%neuron(i)%weight(j) = weight_store


             ! Compare the numerical gradient with the computed gradient
             if (abs(numericalGradient - gradients(l)%weight(j,i)).gt.10*t_epsilon) then
                write(*,*) "Gradient check failed for parameter ", i,j,l
                write(*,*) numericalGradient, gradients(l)%weight(j,i)
             !else
             !   write(*,*) "Gradient check passed for parameter ", i,j,l
             !   write(*,*) numericalGradient, gradients(l)%weight(j,i)
             end if


          end do
       end do
    end do

  end subroutine fc_gradient_check
!!!#############################################################################


end program ConvolutionalNeuralNetwork
!!!###################################################################################
