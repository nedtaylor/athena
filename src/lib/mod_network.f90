!!!#############################################################################
!!! 
!!!#############################################################################
module network
#ifdef _OPENMP
  use omp_lib
#endif
  use constants, only: real12
  use misc, only: shuffle

  use metrics, only: metric_dict_type
  use optimiser, only: optimiser_type
  use loss_categorical, only: &
       comp_loss_func => compute_loss_function, &
       comp_loss_deriv => compute_loss_derivative

  use container_layer, only: container_layer_type
  use input3d_layer,   only: input3d_layer_type
  use full_layer,      only: full_layer_type
  use conv2d_layer,    only: conv2d_layer_type
  use maxpool2d_layer, only: maxpool2d_layer_type
  use flatten2d_layer, only: flatten2d_layer_type
  use flatten3d_layer, only: flatten3d_layer_type

  implicit none

  type :: network_type
     integer :: num_layers
     integer :: num_outputs
     type(optimiser_type) :: optimiser
     type(metric_dict_type), dimension(2) :: metrics
     type(container_layer_type), allocatable, dimension(:) :: model
     procedure(comp_loss_func), nopass, pointer :: get_loss => null()
     procedure(comp_loss_deriv), nopass, pointer :: get_loss_deriv => null()
   contains
     procedure, pass(this) :: train
     procedure, pass(this) :: test

     procedure, pass(this) :: forward => forward_3d    !! TEMPORARY
     procedure, pass(this) :: backward => backward_1d  !! TEMPORARY
  end type network_type

  
  interface compute_accuracy
     procedure compute_accuracy_int, compute_accuracy_real
  end interface compute_accuracy

  interface get_sample
     procedure get_sample_1d, get_sample_3d, get_sample_4d
  end interface get_sample


  private

  public :: network_type


contains

!!!#############################################################################
!!! return sample from any rank
!!!#############################################################################
  pure function get_sample_1d(input, index) result(output)
    implicit none
    integer, intent(in) :: index
    real(real12), dimension(:,:), intent(in) :: input
    real(real12), dimension(size(input,dim=1)) :: output

    output = input(:,index)
  end function get_sample_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function get_sample_3d(input, index) result(output)
    implicit none
    integer, intent(in) :: index
    real(real12), dimension(:,:,:,:), intent(in) :: input
    real(real12), dimension(size(input,1),size(input,2),size(input,3)) :: output

    output = input(:,:,:,index)
  end function get_sample_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function get_sample_4d(input, index) result(output)
    implicit none
    integer, intent(in) :: index
    real(real12), dimension(:,:,:,:,:), intent(in) :: input
    real(real12), dimension(&
         size(input,1),size(input,2),size(input,3),size(input,4)) :: output

    output = input(:,:,:,:,index)
  end function get_sample_4d
!!!#############################################################################


!!!#############################################################################
!!! forward pass
!!!#############################################################################
  pure subroutine forward_3d(this, input)
    implicit none
    class(network_type), intent(inout) :: this
    real(real12), dimension(:,:,:), intent(in) :: input
    
    integer :: i


    select type(current => this%model(1)%layer)
    type is(input3d_layer_type)
       call current%init(input)
    end select

    !! Forward pass
    !!--------------------------------------------------------------------------
    do i=2,this%num_layers,1
       call this%model(i)%forward(this%model(i-1))
    end do

  end subroutine forward_3d
!!!#############################################################################


!!!#############################################################################
!!! backward pass
!!!#############################################################################
  pure subroutine backward_1d(this, output)
    implicit none
    class(network_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: output

    integer :: i


    !! Backward pass (final layer)
    !!-------------------------------------------------------------------
    select type(current => this%model(this%num_layers)%layer)
    type is(full_layer_type)
       call this%model(this%num_layers)%backward(&
            this%model(this%num_layers-1),&
            this%get_loss_deriv(&
            current%output,output))
    end select

    !! Backward pass
    !!-------------------------------------------------------------------
    do i=this%num_layers-1,2,-1
       select type(next => this%model(i+1)%layer)
       type is(conv2d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       type is(maxpool2d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       type is(full_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       type is(flatten2d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       type is(flatten3d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       end select
    end do

  end subroutine backward_1d
!!!#############################################################################


!!!#############################################################################
!!! training loop
!!! ... loops over num_epoch number of epochs
!!! ... i.e. it trains on the same datapoints num_epoch times
!!!#############################################################################
  subroutine train(this, input, output, num_epochs, batch_size, &
       plateau_threshold, shuffle_batches, batch_print_step, verbosity)
    use infile_tools, only: stop_check
    implicit none
    class(network_type), intent(inout) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: input !!!dimension(..), intent(in) :: input
    integer, dimension(:,:), intent(in) :: output !! CONVER THIS LATER TO ANY TYPE AND ANY RANK
    integer, intent(in) :: num_epochs, batch_size

    real(real12), optional, intent(in) :: plateau_threshold
    logical, optional, intent(in) :: shuffle_batches
    integer, optional, intent(in) :: batch_print_step
    integer, optional, intent(in) :: verbosity
    
    !! training and testing monitoring
    real(real12) :: batch_loss, batch_accuracy, avg_loss, avg_accuracy
    real(real12), allocatable, dimension(:,:) :: y_pred
    integer, allocatable, dimension(:,:) ::  y_true

    !! learning parameters
    integer :: num_batches
    integer :: converged
    integer :: history_length
    integer :: t_verb
    integer :: t_batch_print
    real(real12) :: t_plateau
    logical :: t_shuffle

    !! training loop variables
    integer :: epoch, batch, sample, start_index, end_index
    integer, allocatable, dimension(:) :: batch_order

    integer :: i, l, time, time_old, clock_rate


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(plateau_threshold))then
       t_plateau = plateau_threshold
    else
       t_plateau = 1.E-2_real12
    end if
    if(present(shuffle_batches))then
       t_shuffle = shuffle_batches
    else
       t_shuffle = .true.
    end if
    if(present(batch_print_step))then
       t_batch_print = batch_print_step
    else
       t_batch_print = 20
    end if
    if(present(verbosity))then
       t_verb = verbosity
    else
       t_verb = 0
    end if



!!!-----------------------------------------------------------------------------
!!! initialise monitoring variables
!!!-----------------------------------------------------------------------------
    history_length = max(ceiling(500._real12/batch_size),1)
    do i=1,size(this%metrics,dim=1)
       if(allocated(this%metrics(i)%history)) &
            deallocate(this%metrics(i)%history)
       allocate(this%metrics(i)%history(history_length))
       this%metrics(i)%history = -huge(1._real12)
    end do


!!!-----------------------------------------------------------------------------
!!! allocate predicted and true label sets
!!!-----------------------------------------------------------------------------
    allocate(y_pred(this%num_outputs,batch_size), source=0._real12)
    allocate(y_true(this%num_outputs,batch_size), source=0)


!!!-----------------------------------------------------------------------------
!!! if parallel, initialise slices
!!!-----------------------------------------------------------------------------
  num_batches = size(output,dim=2) / batch_size
  allocate(batch_order(num_batches))
  do batch = 1, num_batches
     batch_order(batch) = batch
  end do

    
!!!-----------------------------------------------------------------------------
!!! query system clock
!!!-----------------------------------------------------------------------------
    call system_clock(time, count_rate = clock_rate)


    write(6,*) "Starting training..."
    epoch_loop: do epoch = 1, num_epochs
       !!-----------------------------------------------------------------------
       !! shuffle batch order at the start of each epoch
       !!-----------------------------------------------------------------------
       if(t_shuffle)then
          call shuffle(batch_order)
       end if

       avg_loss     = 0._real12
       avg_accuracy = 0._real12

       !!-----------------------------------------------------------------------
       !! batch loop
       !! ... split data up into minibatches for training
       !!-----------------------------------------------------------------------
       batch_loop: do batch = 1, num_batches


          !! set batch start and end index
          !!--------------------------------------------------------------------
          start_index = (batch_order(batch) - 1) * batch_size + 1
          end_index = batch_order(batch) * batch_size
          
          
          !! reinitialise variables
          !!--------------------------------------------------------------------
          y_true = 0
          y_pred = 0._real12


          y_true(:,1:this%num_outputs) = output(:,start_index:end_index:1)

          
          !!--------------------------------------------------------------------
          !! sample loop
          !! ... test each sample and get gradients and losses from each
          !!--------------------------------------------------------------------
          train_loop: do concurrent(sample=start_index:end_index:1)
             
             
             !! Forward pass
             !!-----------------------------------------------------------------
             call this%forward(get_sample(input,sample))

!!! SET UP LOSS TO APPLY A NORMALISER BY DEFAULT IF SOFTMAX NOT PREVIOUS
!!! (this is what keras does)
!!! ... USE current%transfer%name TO DETERMINE
!!! https://www.v7labs.com/blog/cross-entropy-loss-guide
!!! https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function
!!! https://math.stackexchange.com/questions/4367458/derivate-of-the-the-negative-log-likelihood-with-composition
             

             !! store predicted output
             !!-----------------------------------------------------------------
             select type(current => this%model(this%num_layers)%layer)
             type is(full_layer_type)
                y_pred(:,sample-start_index+1) = current%output
             end select

             !! Backward pass
             !!-----------------------------------------------------------------
             call this%backward(real(y_true(:,sample),real12))

          end do train_loop


          !! compute loss and accuracy (for monitoring)
          !!-------------------------------------------------------------------
          batch_loss = 0._real12
          batch_accuracy = 0._real12
          do sample = 1, end_index-start_index+1, 1
             batch_loss = batch_loss + sum(this%get_loss(&
                  y_pred(sample,:),real(y_true(sample,:),real12)))
             batch_accuracy = batch_accuracy + compute_accuracy(&
                  y_pred(sample,:),y_true(sample,:))
          end do


          !! Average metric over batch size and store
          !!--------------------------------------------------------------------
          avg_loss = avg_loss + batch_loss
          avg_accuracy = avg_accuracy + batch_accuracy
          this%metrics(1)%val = batch_loss / batch_size
          this%metrics(2)%val = batch_accuracy / batch_size


          !! Check metric convergence
          !!--------------------------------------------------------------------
          do i=1,size(this%metrics,dim=1)
             call this%metrics(i)%check(t_plateau, converged)
             if(converged.ne.0)then
                exit epoch_loop
             end if
          end do


          !! if mini-batch ...
          !! ... update weights and biases using optimization algorithm
          !! ... (gradient descent)
          !!--------------------------------------------------------------------
          !! STORE ADAM VALUES IN OPTIMISER
          do i=2, this%num_layers,1
             select type(current => this%model(i)%layer)
             type is(conv2d_layer_type)
                current%dw = current%dw/batch_size
                current%db = current%db/batch_size
                call current%update(this%optimiser)!,cv_clip) !!! CONVERT CLIPS TO LAYER VARIABLES
             type is(full_layer_type)
                current%dw = current%dw/batch_size
                call current%update(this%optimiser)!,fc_clip) !!! CONVERT CLIPS TO LAYER VARIABLES
             end select
          end do
          this%optimiser%iter = this%optimiser%iter + 1


          !! print batch results
          !!----------------------------------------------------------------------
101       if(abs(t_verb).gt.0.and.&
               (batch.eq.1.or.mod(batch,t_batch_print).eq.0.E0))then
             write(6,'("epoch=",I0,", batch=",I0,&
                  &", learning_rate=",F0.3,", loss=",F0.3,", accuracy=",F0.3)') &
                  epoch, batch, this%optimiser%learning_rate, &
                  avg_loss/(batch*batch_size),  avg_accuracy/(batch*batch_size)
          end if

          !! time check
          !!----------------------------------------------------------------------
          if(t_verb.eq.-2)then
             time_old = time
             call system_clock(time)
             write(*,'("time check: ",F5.3," seconds")') &
                  real(time-time_old)/clock_rate
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
       if(t_verb.eq.0)then
          write(6,'("epoch=",I0,", batch=",I0,&
               &", learning_rate=",F0.3,", val_loss=",F0.3,&
               &", val_accuracy=",F0.3)') &
               epoch, batch, this%optimiser%learning_rate, &
               this%metrics(1)%val, this%metrics(2)%val
       end if


    end do epoch_loop
    write(*,*) "Training finished"

  end subroutine train
!!!#############################################################################


!!!#############################################################################
!!! print weights and biases of CNN to file
!!!#############################################################################
!  subroutine write()!(this)
!    implicit none
!    !class()
!!  write(*,*) "Writing CNN learned parameters to output file"
!!  open(unit=10,file=output_file,status='replace')
!!  close(10)
!!  call cv_write(output_file)
!!  call fc_write(output_file)
!!
!!  if(verbosity.gt.1) open(unit=15,file="results_test.out")
!!
!  end subroutine write
!!!#############################################################################


!!!#############################################################################
!!! testing loop
!!!#############################################################################
  subroutine test(this, input, output, verbosity)
    implicit none
    class(network_type), intent(inout) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: input
    integer, dimension(:,:), intent(in) :: output !! CONVER THIS LATER TO ANY TYPE AND ANY RANK
    integer, optional, intent(in) :: verbosity

    integer :: sample, num_samples
    integer :: t_verb
    real(real12) :: accuracy


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(verbosity))then
       t_verb = verbosity
    else
       t_verb = 0
    end if
    this%metrics%val = 0._real12
    num_samples = size(output, dim=2)

!!!! OPEN UNIT 15 !!!! (and use the newunit functionality)


!!!-----------------------------------------------------------------------------
!!! testing loop
!!!-----------------------------------------------------------------------------
    write(*,*) "Starting testing..."
    test_loop: do sample = 1, num_samples

       !! Forward pass
       !!-----------------------------------------------------------------------
       call this%forward(get_sample(input,sample))


       !! compute loss and accuracy (for monitoring)
       !!-----------------------------------------------------------------------
       select type(current => this%model(this%num_layers)%layer)
       type is(full_layer_type)
          accuracy = compute_accuracy(current%output, output(:,sample))
          this%metrics(1)%val = this%metrics(1)%val + sum(&
               this%get_loss(&
               predicted=current%output,expected=real(output(:,sample),real12)))
          this%metrics(2)%val = this%metrics(2)%val + accuracy
          !! print testing results
          !!--------------------------------------------------------------------
          if(abs(t_verb).gt.1)then
             write(15,'(I4," Expected=",I3,", Got=",I3,", Accuracy=",F0.3)') &
                  sample, &
                  maxloc(output(:,sample)), maxloc(current%output,dim=1)-1, &
                  accuracy
          end if
       end select

    end do test_loop
    if(t_verb.gt.1) close(15)
    write(*,*) "Testing finished"

    write(6,'("Overall accuracy=",F0.5)') this%metrics(2)%val/real(num_samples)
    write(6,'("Overall loss=",F0.5)')     this%metrics(1)%val/real(num_samples)


  end subroutine test
!!!#############################################################################


!!!#############################################################################
!!! compute accuracy
!!! this only works (and is only valid for?) categorisation problems
!!!#############################################################################
  function compute_accuracy_int(output, expected) result(accuracy)
    implicit none
    real(real12), dimension(:), intent(in) :: output
    integer, dimension(:) :: expected
    real(real12) :: accuracy

    !! Compute the accuracy
    if (maxval(output).eq.maxval(output)) then
       accuracy = 1._real12
    else
       accuracy = 0._real12
    end if

  end function compute_accuracy_int
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  function compute_accuracy_real(output, expected) result(accuracy)
    implicit none
    real(real12), dimension(:), intent(in) :: output, expected
    real(real12) :: accuracy

    !! Compute the accuracy
    accuracy = sum(expected * output)

  end function compute_accuracy_real
!!!#############################################################################


end module network
!!!###################################################################################