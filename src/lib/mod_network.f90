!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module network
#ifdef _OPENMP
  use omp_lib
#endif
  use constants, only: real12
  use misc, only: shuffle

  use metrics, only: metric_dict_type
  use optimiser, only: optimiser_type
  use loss, only: &
       comp_loss_func => compute_loss_function, &
       comp_loss_deriv => compute_loss_derivative

  use base_layer,      only: base_layer_type, &
       input_layer_type, drop_layer_type, learnable_layer_type
  use container_layer, only: container_layer_type, container_reduction

  !! input layer types
  use input1d_layer,   only: input1d_layer_type
  use input3d_layer,   only: input3d_layer_type
  use input4d_layer,   only: input4d_layer_type

  !! convolution layer types
  use conv2d_layer,    only: conv2d_layer_type, read_conv2d_layer
  use conv3d_layer,    only: conv3d_layer_type, read_conv3d_layer

  !! dropout layer types
  use dropout_layer, only: dropout_layer_type, read_dropout_layer
  use dropblock2d_layer, only: dropblock2d_layer_type, read_dropblock2d_layer
  use dropblock3d_layer, only: dropblock3d_layer_type, read_dropblock3d_layer

  !! pooling layer types
  use maxpool2d_layer, only: maxpool2d_layer_type, read_maxpool2d_layer
  use maxpool3d_layer, only: maxpool3d_layer_type, read_maxpool3d_layer

  !! flatten layer types
  use flatten2d_layer, only: flatten2d_layer_type
  use flatten3d_layer, only: flatten3d_layer_type

  !! fully connected (dense) layer types
  use full_layer,      only: full_layer_type, read_full_layer

  implicit none

  type :: network_type
     real(real12) :: accuracy, loss
     integer :: num_layers
     integer :: num_outputs
     type(optimiser_type) :: optimiser
     type(metric_dict_type), dimension(2) :: metrics
     type(container_layer_type), allocatable, dimension(:) :: model
     procedure(comp_loss_func), nopass, pointer :: get_loss => null()
     procedure(comp_loss_deriv), nopass, pointer :: get_loss_deriv => null()
   contains
     procedure, pass(this) :: print
     procedure, pass(this) :: read
     procedure, pass(this) :: add
     procedure, pass(this) :: compile
     procedure, pass(this) :: train
     procedure, pass(this) :: test
     procedure, pass(this) :: update

     procedure, pass(this) :: forward => forward_1d
     procedure, pass(this) :: backward => backward_1d
  end type network_type
#ifdef _OPENMP
  !$omp declare reduction(network_reduction:network_type:network_reduction(omp_out, omp_in)) &
  !$omp& initializer(omp_priv = omp_orig)
#endif
  
  interface compute_accuracy
     procedure compute_accuracy_int, compute_accuracy_real
  end interface compute_accuracy



  private

  public :: network_type

contains

!!!#############################################################################
!!! network addition
!!!#############################################################################
  subroutine network_reduction(lhs, rhs)
    implicit none
    type(network_type), intent(inout) :: lhs
    type(network_type), intent(in) :: rhs

    integer :: i
    
    lhs%metrics(1)%val = lhs%metrics(1)%val + rhs%metrics(1)%val
    lhs%metrics(2)%val = lhs%metrics(2)%val + rhs%metrics(2)%val
    do i=1,size(lhs%model)
       select type(layer_lhs => lhs%model(i)%layer)
       class is(learnable_layer_type)
          select type(layer_rhs => rhs%model(i)%layer)
          class is(learnable_layer_type)
             call layer_lhs%merge(layer_rhs)
          end select
       end select
    end do

  end subroutine network_reduction
!!!#############################################################################


!!!#############################################################################
!!! network addition
!!!#############################################################################
  subroutine network_copy(lhs, rhs)
    implicit none
    type(network_type), intent(out) :: lhs
    type(network_type), intent(in) :: rhs

    lhs%metrics = rhs%metrics
    lhs%model   = rhs%model
  end subroutine network_copy
!!!#############################################################################


!!!#############################################################################
!!! print network to file
!!!#############################################################################
  subroutine print(this, file)
    implicit none
    class(network_type), intent(in) :: this
    character(*), intent(in) :: file
    
    integer :: l, unit

    open(newunit=unit,file=file,status='replace')
    close(unit)
    
    do l=1,this%num_layers
       call this%model(l)%layer%print(file)
    end do

  end subroutine print
!!!#############################################################################


!!!#############################################################################
!!! read network from file
!!!#############################################################################
  subroutine read(this, file)
   implicit none
   class(network_type), intent(inout) :: this
   character(*), intent(in) :: file
   
   integer :: i, unit, stat
   character(256) :: buffer
   open(newunit=unit,file=file,action='read')
   i = 0
   card_loop: do
      i = i + 1
      read(unit,'(A)',iostat=stat) buffer
      if(stat.lt.0)then
         exit card_loop
      elseif(stat.gt.0)then
         write(0,*) "ERROR: error encountered in network read"
         stop "Exiting..."
      end if
      if(trim(adjustl(buffer)).eq."") cycle card_loop

      !! check if a tag line
      if(scan(buffer,'=').ne.0)then
         write(0,*) "WARNING: unexpected line in read file"
         write(0,*) trim(buffer)
         write(0,*) " skipping..."
         cycle card_loop
      end if

      !! check for card
      select case(trim(adjustl(buffer)))
      case("CONV2D")
         call this%add(read_conv2d_layer(unit))
      case("CONV3D")
         call this%add(read_conv3d_layer(unit))
      case("DROPOUT")
         call this%add(read_dropout_layer(unit))
      case("DROPBLOCK2D")
         call this%add(read_dropblock2d_layer(unit))
      case("DROPBLOCK3D")
         call this%add(read_dropblock3d_layer(unit))
      case("MAXPOOL2D")
         call this%add(read_maxpool2d_layer(unit))
      case("MAXPOOL3D")
         call this%add(read_maxpool3d_layer(unit))
      case("FULL")
         call this%add(read_full_layer(unit))
      case default
         write(0,*) "ERROR: unrecognised card '"//&
              &trim(adjustl(buffer))//"'"
         stop "Exiting..."
      end select
   end do card_loop
   close(unit)

 end subroutine read
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! append layer to network
!!!#############################################################################
  subroutine add(this, layer)
    implicit none
    class(network_type), intent(inout) :: this
    class(base_layer_type), intent(in) :: layer
    
    character(4) :: name
    
    select type(layer)
    class is(input_layer_type)
       name = "inpt"
    type is(conv2d_layer_type)
       name = "conv"
    type is(conv3d_layer_type)
       name = "conv"
    type is(flatten2d_layer_type)
       name = "flat"
    type is(flatten3d_layer_type)
       name = "flat"
    class is(drop_layer_type)
       name = "drop"
    type is(maxpool2d_layer_type)
       name = "pool"
    type is(maxpool3d_layer_type)
       name = "pool"
    type is(full_layer_type)
       name = "full"
    class default
       name = "unkw"
    end select
    
    if(.not.allocated(this%model))then
       this%model = [container_layer_type(name=name)]
    else
       this%model = [this%model(1:), container_layer_type(name=name)]
    end if
    allocate(this%model(size(this%model,dim=1))%layer, source=layer)
       
  end subroutine add
!!!#############################################################################


!!!#############################################################################
!!! set up network
!!!#############################################################################
  subroutine compile(this, optimiser, loss, metrics, verbose)
    use misc, only: to_lower
    use loss, only: &
         compute_loss_bce, compute_loss_cce, &
         compute_loss_mae, compute_loss_mse, &
         compute_loss_nll
    implicit none
    class(network_type), intent(inout) :: this
    type(optimiser_type), intent(in) :: optimiser
    character(*), intent(in) :: loss
    class(*), dimension(..), intent(in) :: metrics
    integer, optional, intent(in) :: verbose
    
    integer :: i
    integer :: t_verb, num_addit_inputs
    character(len=:), allocatable :: loss_method


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(verbose))then
       t_verb = verbose
    else
       t_verb = 0
    end if

    
!!!-----------------------------------------------------------------------------
!!! initialise metrics
!!!-----------------------------------------------------------------------------
    this%metrics%active = .false.
    this%metrics(1)%key = "loss"
    this%metrics(2)%key = "accuracy"
    this%metrics%threshold = 1.E-1_real12
    select rank(metrics)
    rank(0)
       select type(metrics)
       type is(character(*))
          where(to_lower(trim(metrics)).eq.this%metrics%key)
             this%metrics%active = .true.
          end where
       end select
    rank(1)
       select type(metrics)
       type is(character(*))
          do i=1,size(metrics,1)
             where(to_lower(trim(metrics(i))).eq.this%metrics%key)
                this%metrics%active = .true.
             end where
          end do
       type is(metric_dict_type)
          if(size(metrics,1).eq.2)then
             this%metrics(:2) = metrics(:2)
          else
             stop "ERROR: invalid length array for metric_dict_type"
          end if
       end select
    rank default
       stop "ERROR: provided metrics rank in compile invalid"
    end select


!!!-----------------------------------------------------------------------------
!!! initialise optimiser
!!!-----------------------------------------------------------------------------
    this%optimiser = optimiser


!!!-----------------------------------------------------------------------------
!!! initialise loss method
!!!-----------------------------------------------------------------------------
    loss_method = to_lower(loss)
    select case(loss_method)
    case("binary_crossentropy")
       loss_method = "bce"
    case("categorical_crossentropy")
       loss_method = "cce"
    case("mean_absolute_error")
       loss_method = "mae"
    case("mean_squared_error")
       loss_method = "mse"
    case("negative_loss_likelihood")
       loss_method = "nll"
    end select

    select case(loss_method)
    case("bce")
       this%get_loss => compute_loss_bce
       if(t_verb.gt.0) write(*,*) "Loss method: Categorical Cross Entropy"
    case("cce")
       this%get_loss => compute_loss_cce
       if(t_verb.gt.0) write(*,*) "Loss method: Categorical Cross Entropy"
    case("mae")
       this%get_loss => compute_loss_mae
       if(t_verb.gt.0) write(*,*) "Loss method: Mean Absolute Error"
    case("mse")
       this%get_loss => compute_loss_mse
       if(t_verb.gt.0) write(*,*) "Loss method: Mean Squared Error"
    case("nll")
       this%get_loss => compute_loss_nll
       if(t_verb.gt.0) write(*,*) "Loss method: Negative log likelihood"
    case default
       write(0,*) "Failed loss method: "//trim(loss_method)
       stop "ERROR: No loss method provided"
    end select
    this%get_loss_deriv => comp_loss_deriv


!!!-----------------------------------------------------------------------------
!!! check for input layer
!!!-----------------------------------------------------------------------------
    if(.not.allocated(this%model(1)%layer%input_shape))then
       stop "ERROR: input_shape of first layer not defined"
    end if
    select type(first => this%model(1)%layer)
    class is(input_layer_type)
    class default
       this%model = [&
            container_layer_type(name="inpt"),&
            this%model(1:)&
            ]
       associate(next => this%model(2)%layer)
         select case(size(next%input_shape,dim=1))
         case(1)
            allocate(this%model(1)%layer, source=&
                 input1d_layer_type(input_shape=next%input_shape))
         case(3)
            select type(next)
            type is(conv2d_layer_type)
               allocate(this%model(1)%layer, source=&
                    input3d_layer_type(input_shape=next%input_shape+&
                    [2*next%pad,0]))
            class default
               allocate(this%model(1)%layer, source=&
                    input3d_layer_type(input_shape=next%input_shape))
            end select
         case(4)
            select type(next)
            type is(conv3d_layer_type)
               allocate(this%model(1)%layer, source=&
                    input4d_layer_type(input_shape=next%input_shape+&
                    [2*next%pad,0]))
            class default
               allocate(this%model(1)%layer, source=&
                    input4d_layer_type(input_shape=next%input_shape))
            end select
         end select
       end associate
    end select


!!!-----------------------------------------------------------------------------
!!! ignore calcuation of input gradients for 1st non-input layer
!!!-----------------------------------------------------------------------------
    select type(second => this%model(2)%layer)
    type is(conv2d_layer_type)
       second%calc_input_gradients = .false.
    type is(conv3d_layer_type)
       second%calc_input_gradients = .false.
    end select


!!!-----------------------------------------------------------------------------
!!! initialise layers
!!!-----------------------------------------------------------------------------
    if(t_verb.gt.0)then
       write(*,*) "layer:",1, this%model(1)%name
       write(*,*) this%model(1)%layer%input_shape
       write(*,*) this%model(1)%layer%output_shape
    end if
    do i=2,size(this%model,dim=1)
       if(.not.allocated(this%model(i)%layer%input_shape)) &
            call this%model(i)%layer%init(this%model(i-1)%layer%output_shape)
       if(t_verb.gt.0)then
          write(*,*) "layer:",i, this%model(i)%name
          write(*,*) this%model(i)%layer%input_shape
          write(*,*) this%model(i)%layer%output_shape
       end if
    end do


!!!-----------------------------------------------------------------------------
!!! check for required reshape layers
!!!-----------------------------------------------------------------------------
    i = 1 !! starting for layer 2
    layer_loop: do
       if(i.ge.size(this%model,dim=1)) exit layer_loop
       i = i + 1
    
       flatten_layer_check: if(i.lt.size(this%model,dim=1))then
          if(allocated(this%model(i+1)%layer%input_shape).and.&
               allocated(this%model(i)%layer%output_shape))then
             if(size(this%model(i+1)%layer%input_shape).ne.&
                  size(this%model(i)%layer%output_shape))then

                select type(current => this%model(i)%layer)
                type is(flatten2d_layer_type)
                   cycle layer_loop
                type is(flatten3d_layer_type)
                   cycle layer_loop
                class default
                   this%model = [&
                        this%model(1:i),&
                        container_layer_type(name="flat"),&
                        this%model(i+1:size(this%model))&
                        ]
                   num_addit_inputs = 0
                   select type(next => this%model(i+1)%layer)
                   type is(full_layer_type)
                      num_addit_inputs = next%num_addit_inputs
                   end select
                   select case(size(this%model(i)%layer%output_shape))
                   case(3)
                      allocate(this%model(i+1)%layer, source=&
                           flatten2d_layer_type(input_shape=&
                           this%model(i)%layer%output_shape, &
                           num_addit_outputs = num_addit_inputs))
                   case(4)
                      allocate(this%model(i+1)%layer, source=&
                           flatten3d_layer_type(input_shape=&
                           this%model(i)%layer%output_shape, &
                           num_addit_outputs = num_addit_inputs))
                   end select
                   i = i + 1
                   cycle layer_loop
                end select
             end if
          else
             
          end if
       end if flatten_layer_check
    
    end do layer_loop
    
    !! update number of layers
    !!--------------------------------------------------------------------------
    this%num_layers = i


    !! set number of outputs
    !!--------------------------------------------------------------------------
    this%num_outputs = product(this%model(this%num_layers)%layer%output_shape)


  end subroutine compile
!!!#############################################################################


!!!#############################################################################
!!! return sample from any rank
!!!#############################################################################
  pure function get_sample(input, index) result(output)
    implicit none
    integer, intent(in) :: index
    real(real12), dimension(..), intent(in) :: input
    real(real12), allocatable, dimension(:) :: output

    select rank(input)
    rank(2)
       output = reshape(input(:,index), shape=[size(input(:,1))])
    rank(3)
       output = reshape(input(:,:,index), shape=[size(input(:,:,1))])
    rank(4)
       output = reshape(input(:,:,:,index), shape=[size(input(:,:,:,1))])
    rank(5)
       output = reshape(input(:,:,:,:,index), shape=[size(input(:,:,:,:,1))])
    end select

  end function get_sample
!!!#############################################################################


!!!#############################################################################
!!! forward pass
!!!#############################################################################
  pure subroutine forward_1d(this, input, addit_input, layer)
    implicit none
    class(network_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: input

    real(real12), dimension(:), optional, intent(in) :: addit_input
    integer, optional, intent(in) :: layer
    
    integer :: i


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(layer).and.present(addit_input))then
       select type(previous => this%model(layer-1)%layer)
       type is(flatten2d_layer_type)
          previous%output(size(previous%di)-size(addit_input)+1:) = addit_input
       type is(flatten3d_layer_type)
          previous%output(size(previous%di)-size(addit_input)+1:) = addit_input
       end select
    end if


    !! Forward pass (first layer)
    !!--------------------------------------------------------------------------
    select type(current => this%model(1)%layer)
    class is(input_layer_type)
       call current%set(input)
    end select

    !! Forward pass
    !!--------------------------------------------------------------------------
    do i=2,this%num_layers,1
       call this%model(i)%forward(this%model(i-1))
    end do

  end subroutine forward_1d
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
            this%get_loss_deriv(current%output,output))
    end select

    !! Backward pass
    !!-------------------------------------------------------------------
    do i=this%num_layers-1,2,-1
       select type(next => this%model(i+1)%layer)
       type is(conv2d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       type is(conv3d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
    
       type is(dropout_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       type is(dropblock2d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       type is(dropblock3d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
    
       type is(maxpool2d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       type is(maxpool3d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
    
       type is(flatten2d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       type is(flatten3d_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
    
       type is(full_layer_type)
          call this%model(i)%backward(this%model(i-1),next%di)
       end select
    end do

  end subroutine backward_1d
!!!#############################################################################


!!!#############################################################################
!!! update weights and biases
!!!#############################################################################
  subroutine update(this, batch_size)
    implicit none
    class(network_type), intent(inout) :: this
    integer, intent(in) :: batch_size

    integer :: i
    

    !!-------------------------------------------------------------------
    !! Update layers of learnable layer types
    !!-------------------------------------------------------------------
    do i=2, this%num_layers,1
       select type(current => this%model(i)%layer)
       class is(learnable_layer_type)
          call current%update(this%optimiser, batch_size)
       class is(drop_layer_type)
          call current%generate_mask()
       end select
    end do

    !! Increment optimiser iteration counter
    !!-------------------------------------------------------------------
    this%optimiser%iter = this%optimiser%iter + 1

  end subroutine update
!!!#############################################################################


!!!#############################################################################
!!! training loop
!!! ... loops over num_epoch number of epochs
!!! ... i.e. it trains on the same datapoints num_epoch times
!!!#############################################################################
  subroutine train(this, input, output, num_epochs, batch_size, &
       addit_input, addit_layer, &
       plateau_threshold, shuffle_batches, batch_print_step, verbose)
    use infile_tools, only: stop_check
    implicit none
    class(network_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    class(*), dimension(:,:), intent(in) :: output
    integer, intent(in) :: num_epochs, batch_size

    real(real12), dimension(:,:), optional, intent(in) :: addit_input
    integer, optional, intent(in) :: addit_layer

    real(real12), optional, intent(in) :: plateau_threshold
    logical, optional, intent(in) :: shuffle_batches
    integer, optional, intent(in) :: batch_print_step
    integer, optional, intent(in) :: verbose
    
    !! training and testing monitoring
    real(real12) :: batch_loss, batch_accuracy, avg_loss, avg_accuracy
    real(real12), allocatable, dimension(:,:) :: y_pred, y_true

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

#ifdef _OPENMP
    type(network_type) :: this_copy
    real(real12), allocatable, dimension(:,:) :: input_slice, addit_input_slice
#endif
    integer :: timer_start = 0, timer_stop = 0, timer_sum = 0, timer_tot = 0


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
    if(present(verbose))then
       t_verb = verbose
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
    allocate(y_pred(this%num_outputs,batch_size), source = 0._real12)
    allocate(y_true(this%num_outputs,batch_size), source = 0._real12)



!!!-----------------------------------------------------------------------------
!!! if parallel, initialise slices
!!!-----------------------------------------------------------------------------
  num_batches = size(output,dim=2) / batch_size
  allocate(batch_order(num_batches))
  do batch = 1, num_batches
     batch_order(batch) = batch
  end do


!!!-----------------------------------------------------------------------------
!!! set up parallel samples slices
!!!-----------------------------------------------------------------------------
#ifdef _OPENMP
  select rank(input)
  rank(2)
     allocate(input_slice(&
          size(input,1),&
          batch_size))
  rank(4)
     allocate(input_slice(&
          size(input,1)*size(input,2)*size(input,3),&
          batch_size))
  rank(5)
     allocate(input_slice(&
          size(input,1)*size(input,2)*size(input,3)*size(input,4),&
          batch_size))
  end select
  if(present(addit_input))then
     allocate(addit_input_slice(size(addit_input,1),batch_size))
  end if
  this_copy = this
#endif

    
!!!-----------------------------------------------------------------------------
!!! query system clock
!!!-----------------------------------------------------------------------------
    call system_clock(time, count_rate = clock_rate)


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
          y_pred = 0._real12
          select type(output)
          type is(integer)
             y_true(:,:) = real(output(:,start_index:end_index:1),real12)
          type is(real)
             y_true(:,:) = output(:,start_index:end_index:1)
          end select

#ifdef _OPENMP
          !! set up data slices for parallel
          !!--------------------------------------------------------------------
          do sample=start_index,end_index,1
             input_slice(:,sample - start_index + 1) = get_sample(input, sample)
          end do
          if(present(addit_input))then
             addit_input_slice(:,:) = addit_input(:,start_index:end_index)
          end if
          start_index = 1
          end_index = batch_size
#endif
          
          
          !$OMP PARALLEL DO & !! ORDERED
          !$OMP& DEFAULT(NONE) &
          !$OMP& SHARED(start_index, end_index) &
          !$OMP& SHARED(input_slice) &
          !$OMP& SHARED(addit_input_slice, addit_layer) &
          !$OMP& SHARED(y_pred) &
          !$OMP& SHARED(y_true) &
          !$OMP& SHARED(addit_input) &
          !$OMP& PRIVATE(sample) &
          !$OMP& REDUCTION(network_reduction:this_copy)
          !!--------------------------------------------------------------------
          !! sample loop
          !! ... test each sample and get gradients and losses from each
          !!--------------------------------------------------------------------
#ifdef _OPENMP
          train_loop: do sample=start_index,end_index,1
#else
          train_loop: do concurrent(sample=start_index:end_index:1)
#endif

             !! Forward pass
             !!-----------------------------------------------------------------
             if(present(addit_input).and.present(addit_layer))then
#ifdef _OPENMP
                call this_copy%forward(input_slice(:,sample),&
                     addit_input_slice(:,sample),addit_layer)
             else
                call this_copy%forward(input_slice(:,sample))
#else
                call this%forward(get_sample(input,sample),&
                     addit_input(:,sample),addit_layer)
             else
                call this%forward(get_sample(input,sample))
#endif
             end if

!!! SET UP LOSS TO APPLY A NORMALISER BY DEFAULT IF SOFTMAX NOT PREVIOUS
!!! (this is what keras does)
!!! ... USE current%transfer%name TO DETERMINE
!!! https://www.v7labs.com/blog/cross-entropy-loss-guide
!!! https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function
!!! https://math.stackexchange.com/questions/4367458/derivate-of-the-the-negative-log-likelihood-with-composition


             !! Backward pass and store predicted output
             !!-----------------------------------------------------------------
#ifdef _OPENMP
             call this_copy%backward(y_true(:,sample))
             select type(current => this_copy%model(this_copy%num_layers)%layer)
             type is(full_layer_type)
                y_pred(:,sample) = current%output
#else
             call this%backward(y_true(:,sample-start_index+1))
             select type(current => this%model(this%num_layers)%layer)
             type is(full_layer_type)
                y_pred(:,sample-start_index+1) = current%output
#endif
             end select

          end do train_loop
          !$OMP END PARALLEL DO


          !! compute loss and accuracy (for monitoring)
          !!-------------------------------------------------------------------
          batch_loss = 0._real12
          batch_accuracy = 0._real12
          do sample = 1, end_index-start_index+1, 1
             batch_loss = batch_loss + sum(&
#ifdef _OPENMP
                  this_copy%get_loss(&
#else
                  this%get_loss(&
#endif
                  y_pred(:,sample),y_true(:,sample)))
             select type(output)
             type is(integer)
                batch_accuracy = batch_accuracy + compute_accuracy(&
                     y_pred(:,sample),nint(y_true(:,sample)))
             type is(real)
                batch_accuracy = batch_accuracy + compute_accuracy(&
                     y_pred(:,sample),y_true(:,sample))
             end select

          end do


          !! Average metric over batch size and store
          !! Check metric convergence
          !!--------------------------------------------------------------------
          avg_loss = avg_loss + batch_loss
          avg_accuracy = avg_accuracy + batch_accuracy
#ifdef _OPENMP
          this_copy%metrics(1)%val = batch_loss / batch_size
          this_copy%metrics(2)%val = batch_accuracy / batch_size
          do i = 1, size(this_copy%metrics,dim=1)
             call this_copy%metrics(i)%check(t_plateau, converged)
#else
          this%metrics(1)%val = batch_loss / batch_size
          this%metrics(2)%val = batch_accuracy / batch_size
          do i = 1, size(this%metrics,dim=1)
             call this%metrics(i)%check(t_plateau, converged)
#endif
             if(converged.ne.0)then
                exit epoch_loop
             end if
          end do


          !! update weights and biases using optimization algorithm
          !! ... (gradient descent)
          !!--------------------------------------------------------------------
          !! STORE ADAM VALUES IN OPTIMISER
#ifdef _OPENMP
          call this_copy%update(batch_size)
#else
          call this%update(batch_size)
#endif


          !! print batch results
          !!--------------------------------------------------------------------
          if(abs(t_verb).gt.0.and.&
               (batch.eq.1.or.mod(batch,t_batch_print).eq.0.E0))then
             write(6,'("epoch=",I0,", batch=",I0,&
                  &", learning_rate=",F0.3,", loss=",F0.3,", accuracy=",F0.3)')&
                  epoch, batch, &
#ifdef _OPENMP
                  this_copy%optimiser%learning_rate, &
#else
                  this%optimiser%learning_rate, &
#endif
                  avg_loss/(batch*batch_size),  avg_accuracy/(batch*batch_size)
          end if
          
          
!!! TESTING
!#ifdef _OPENMP
!          call system_clock(timer_start)
!          call system_clock(timer_stop)
!          timer_sum = timer_sum + timer_stop - timer_start
!          timer_tot = timer_tot + timer_sum / omp_get_max_threads()
!#else
!          timer_tot = timer_tot + timer_sum
!#endif
          timer_sum = 0
           if(batch.gt.200)then
              time_old = time
              call system_clock(time)
              write(*,'("time check: ",F8.3," seconds")') real(time-time_old)/clock_rate
              !write(*,'("update timer: ",F8.3," seconds")') real(timer_tot)/clock_rate
              exit epoch_loop
              stop "THIS IS FOR TESTING PURPOSES"
           end if
!!!


          !! time check
          !!--------------------------------------------------------------------
          if(t_verb.eq.-2)then
             time_old = time
             call system_clock(time)
             write(*,'("time check: ",F5.3," seconds")') &
                  real(time-time_old)/clock_rate
             time_old = time
          end if


          !! check for user-name stop file
          !!--------------------------------------------------------------------
          if(stop_check())then
             write(0,*) "STOPCAR ENCOUNTERED"
             write(0,*) "Exiting training loop..."
             exit epoch_loop
          end if

       end do batch_loop


       !! print epoch summary results
       !!-----------------------------------------------------------------------
       if(t_verb.eq.0)then
          write(6,'("epoch=",I0,", batch=",I0,&
               &", learning_rate=",F0.3,", val_loss=",F0.3,&
               &", val_accuracy=",F0.3)') &
               epoch, batch, &
#ifdef _OPENMP
               this_copy%optimiser%learning_rate, &
               this_copy%metrics(1)%val, this_copy%metrics(2)%val
#else
               this%optimiser%learning_rate, &
               this%metrics(1)%val, this%metrics(2)%val
#endif
       end if


    end do epoch_loop

#ifdef _OPENMP
    !!--------------------------------------------------------------------------
    !! copy trained model back into original
    !!--------------------------------------------------------------------------
    this%optimiser = this_copy%optimiser
    this%model = this_copy%model
#endif

  end subroutine train
!!!#############################################################################


!!!#############################################################################
!!! testing loop
!!!#############################################################################
  subroutine test(this, input, output, &
       addit_input, addit_layer, &
       verbose)
    implicit none
    class(network_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    class(*), dimension(:,:), intent(in) :: output

    real(real12), dimension(:,:), optional, intent(in) :: addit_input
    integer, optional, intent(in) :: addit_layer

    integer, optional, intent(in) :: verbose

    integer :: sample, num_samples
    integer :: t_verb, unit
    real(real12) :: accuracy, loss
    real(real12), allocatable, dimension(:) :: accuracy_list
    real(real12), allocatable, dimension(:,:) :: predicted

#ifdef _OPENMP
    type(network_type) :: this_copy
#endif


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(verbose))then
       t_verb = verbose
    else
       t_verb = 0
    end if
    num_samples = size(output, dim=2)
    allocate(predicted(size(output,1), num_samples))

    this%metrics%val = 0._real12
    accuracy = 0._real12
    loss = 0._real12
    allocate(accuracy_list(num_samples))

#ifdef _OPENMP
    this_copy = this
#endif


!!!-----------------------------------------------------------------------------
!!! testing loop
!!!-----------------------------------------------------------------------------
    !$OMP PARALLEL DO & !! ORDERED
    !$OMP& DEFAULT(NONE) &
    !$OMP& SHARED(num_samples) &
    !$OMP& SHARED(unit, t_verb) &
    !$OMP& SHARED(input, output, predicted) &
    !$OMP& SHARED(addit_input, addit_layer) &
    !$OMP& SHARED(accuracy_list) &
    !$OMP& PRIVATE(sample) &
    !$OMP& PRIVATE(accuracy, loss) &
    !$OMP& REDUCTION(network_reduction:this_copy)
    test_loop: do sample = 1, num_samples

       !! Forward pass
       !!-----------------------------------------------------------------------
       if(present(addit_input).and.present(addit_layer))then
#ifdef _OPENMP
          call this_copy%forward(get_sample(input,sample),&
               addit_input(:,sample),addit_layer)
       else
          call this_copy%forward(get_sample(input,sample))
#else
          call this%forward(get_sample(input,sample),&
               addit_input(:,sample),addit_layer)
       else
          call this%forward(get_sample(input,sample))
#endif
       end if


       !! compute loss and accuracy (for monitoring)
       !!-----------------------------------------------------------------------
#ifdef _OPENMP
       select type(current => this_copy%model(this_copy%num_layers)%layer)
#else
       select type(current => this%model(this%num_layers)%layer)
#endif
       type is(full_layer_type)
          select type(output)
          type is(integer)
             accuracy = compute_accuracy(current%output, output(:,sample))
             loss = sum(&
#ifdef _OPENMP
                  this_copy%get_loss(&
#else
                  this%get_loss(&
#endif
                  predicted=current%output,expected=real(output(:,sample),real12)))
          type is(real)
             accuracy = compute_accuracy(current%output, output(:,sample))
             loss = sum(&
#ifdef _OPENMP
                  this_copy%get_loss(&
#else
                  this%get_loss(&
#endif
                  predicted=current%output,expected=output(:,sample)))
          end select
#ifdef _OPENMP
          this_copy%metrics(2)%val = this_copy%metrics(2)%val + accuracy
          this_copy%metrics(1)%val = this_copy%metrics(1)%val + loss
#else
          this%metrics(2)%val = this%metrics(2)%val + accuracy
          this%metrics(1)%val = this%metrics(1)%val + loss
#endif
          accuracy_list(sample) = accuracy
          predicted(:,sample) = current%output
       end select

    end do test_loop
    !$OMP END PARALLEL DO


#ifdef _OPENMP
    !! merge results back into original
    !!--------------------------------------------------------------------
    this%metrics  = this_copy%metrics
#endif
    
    
    !! print testing results
    !!--------------------------------------------------------------------
    if(abs(t_verb).gt.1)then
       open(file="test_output.out",newunit=unit)
       select type(final_layer => this%model(this%num_layers)%layer)
       type is(full_layer_type)
          test_loop: do concurrent(sample = 1:num_samples)
             select type(output)
             type is(integer)
                write(unit,'(I4," Expected=",I3,", Got=",I3,", Accuracy=",F0.3)') &
                     sample, &
                     maxloc(output(:,sample)), maxloc(predicted(:,sample),dim=1)-1, &
                     accuracy_list(sample)
             type is(real)
                write(unit,'(I4," Expected=",I3,", Got=",I3,", Accuracy=",F0.3)') &
                     sample, &
                     maxloc(output(:,sample)), maxloc(predicted(:,sample),dim=1)-1, &
                     accuracy_list(sample)
             end select
          end do test_loop
       end select
       close(unit)
    end if


    !! normalise metrics by number of samples
    !!--------------------------------------------------------------------
    this%accuracy = this%metrics(2)%val/real(num_samples)
    this%loss     = this%metrics(1)%val/real(num_samples)

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
    if (maxloc(expected,dim=1).eq.maxloc(output,dim=1)) then
       accuracy = 1._real12
    else
       accuracy = 0._real12
    end if

  end function compute_accuracy_int
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
!!! works for continuous datasets
  function compute_accuracy_real(output, expected) result(accuracy)
    implicit none
    real(real12), dimension(:), intent(in) :: output, expected
    real(real12) :: accuracy

    !! Compute the accuracy
    accuracy = sum(abs(expected - output)) !! should be for continuous data

  end function compute_accuracy_real
!!!#############################################################################


end module network
!!!#############################################################################
