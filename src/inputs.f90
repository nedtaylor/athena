!!!#############################################################################
!!! Module to define all global variables
!!! Code written by Ned Thaddeus Taylor and Francis Huw Davies
!!! Code part of the ARTEMIS group
!!!#############################################################################
module inputs
  use constants, only: real12, ierror
  use custom_types, only: clip_type, learning_parameters_type
  use misc, only: icount, flagmaker, file_check, to_lower
  implicit none
  integer :: verbosity    ! verbose printing
  integer :: seed         ! random seed
  integer :: num_threads  ! number of threads (FOR OPENMP PARALLEL ONLY!)
  real(real12) :: loss_threshold     ! threshold for loss convergence
  real(real12) :: plateau_threshold  ! threshold for plateau checking
  real(real12) :: learning_rate      ! rate of learning (larger = faster)
  real(real12) :: l1_lambda, l2_lambda  ! l1 and l2 regularisation parameters
  type(learning_parameters_type) :: learning_parameters
  logical :: batch_learning
  character(:), allocatable :: loss_method

  integer :: num_epochs  ! number of epochs
  integer :: batch_size  ! size of mini batches

  integer :: cv_num_filters                ! number of convolution filters
  character(:), allocatable :: convolution_method        ! type of convolution
  character(:), allocatable :: padding_method             ! type of convolution padding
  type(clip_type) :: cv_clip               ! convolution clipping thresholds
  integer, allocatable, dimension(:) :: cv_kernel_size  ! kernel size for each convolution layer (assume square)
  integer, allocatable, dimension(:) :: cv_stride       ! stride of kernels for convolution
  character(:), allocatable :: cv_dropout_method
  integer :: cv_block_size
  real(real12) :: cv_keep_prob
  character(:), allocatable :: cv_kernel_initialiser
  character(:), allocatable :: cv_bias_initialiser
  real(real12) :: cv_activation_scale
  character(:), allocatable :: cv_activation_function

  real(real12) :: bn_gamma, bn_beta  ! batch normalisation learning features

  integer :: pool_kernel_size    ! pooling size (assume square)
  integer :: pool_stride         ! pooling stride
  character(:), allocatable :: pool_normalisation  ! normalise output of pooling

  integer, allocatable, dimension(:) :: fc_num_hidden  ! number of fully connected hidden layers
  type(clip_type) :: fc_clip                           ! fully connected clipping thresholds
  character(len=10) :: fc_activation_function
  real(real12) :: fc_activation_scale
  character(:), allocatable :: fc_weight_initialiser


  real(real12) :: train_size  ! fraction of data to train on (NOT YET USED) 
  logical :: shuffle_dataset  ! shuffle train and test data
  


  private

  public :: seed, verbosity, num_threads
  public :: shuffle_dataset, train_size

  public :: batch_learning
  public :: loss_threshold
  public :: plateau_threshold
  public :: loss_method

  public :: learning_rate, l1_lambda, l2_lambda
  public :: num_epochs, batch_size
  public :: learning_parameters

  public :: cv_num_filters, cv_kernel_size, cv_stride
  public :: cv_clip
  public :: convolution_method, padding_method
  public :: cv_dropout_method
  public :: cv_block_size, cv_keep_prob
  public :: cv_activation_scale
  public :: cv_activation_function
  public :: cv_kernel_initialiser, cv_bias_initialiser

  public :: bn_gamma, bn_beta

  public :: pool_kernel_size, pool_stride
  public :: pool_normalisation

  public :: fc_num_hidden
  public :: fc_clip
  public :: fc_activation_scale
  public :: fc_activation_function
  public :: fc_weight_initialiser

  public :: set_global_vars


!!!updated  2023/06/28


contains
!!!#############################################################################
  subroutine set_global_vars()
    implicit none
    integer :: i,j
    integer :: num_hidden_layers
    character(1024) :: buffer,flag,input_file
    logical :: skip,empty


!!!-----------------------------------------------------------------------------
!!! initialises variables
!!!-----------------------------------------------------------------------------
    skip = .false.
    input_file = ""

    call system_clock(count=seed)
    verbosity = 1
    num_threads = 1

    loss_threshold = 1.E-1_real12
    plateau_threshold = 1.E-3_real12
    shuffle_dataset = .false.
    batch_learning = .true.

    learning_rate = 0.025_real12
    l1_lambda = 0._real12
    l2_lambda = 0._real12

    num_epochs = 20
    batch_size = 20
    cv_num_filters = 32
    cv_clip%l_min_max = .false.
    cv_clip%l_norm    = .false.
    cv_clip%min  = -huge(1._real12)
    cv_clip%max  =  huge(1._real12)
    cv_clip%norm =  huge(1._real12)
    cv_block_size = 5
    cv_keep_prob = 1._real12
    
    pool_kernel_size = 2
    pool_stride = 2
    pool_normalisation = "sum"

    fc_clip%l_min_max = .false.
    fc_clip%l_norm    = .false.
    fc_clip%min  = -huge(1._real12)
    fc_clip%max  =  huge(1._real12)
    fc_clip%norm =  huge(1._real12)
    fc_activation_function = "relu"
    fc_activation_scale = 1._real12
    !! gaussian, relu, piecewise, leaky_relu, sigmoid, tanh


!!!-----------------------------------------------------------------------------
!!! Reads flags and assigns to variables
!!!-----------------------------------------------------------------------------
    flagloop: do i=1,command_argument_count()
       empty=.false.
       if (skip) then
          skip=.false.
          cycle flagloop
       end if
       call get_command_argument(i,buffer)
!!!------------------------------------------------------------------------
!!! FILE AND DIRECTORY FLAGS
!!!------------------------------------------------------------------------
       if(index(buffer,'-f').eq.1)then
          flag="-f"
          call flagmaker(buffer,flag,i,skip,empty)
          if(.not.empty)then
             read(buffer,'(A)') input_file
          else
             write(6,'("ERROR: No input filename supplied, but the flag ''-f'' was used")')
             infilename_do: do j=1,3
                write(6,'("Please supply an input filename:")')
                read(5,'(A)') input_file
                if(trim(input_file).ne.'')then
                   write(6,'("Input filename supplied")')
                   exit infilename_do
                else
                   write(6,'(1X,"Not a valid filename")')
                end if
                if(j.eq.3)then
                   write(0,*) "ERROR: No valid input filename supplied"
                   stop "Exiting..."
                end if
             end do infilename_do
          end if
       !elseif(index(buffer,'-d').eq.1)then
       !   flag="-d"
       !   call flagmaker(buffer,flag,i,skip,empty)
       !   if(.not.empty) read(buffer,'(A)') dir
!!!------------------------------------------------------------------------
!!! NEURAL NETWORK FLAGS
!!!------------------------------------------------------------------------
       elseif(index(buffer,'-s').eq.1)then
          flag="-s"
          call flagmaker(buffer,flag,i,skip,empty)
          if(.not.empty) read(buffer,*) seed
       elseif(index(buffer,'-l').eq.1)then
          flag="-l"
          call flagmaker(buffer,flag,i,skip,empty)
          if(.not.empty)then
             num_hidden_layers = icount(buffer)
             allocate(fc_num_hidden(num_hidden_layers))
             read(buffer,*) (fc_num_hidden(j),j=1,num_hidden_layers)
          end if
!!!------------------------------------------------------------------------
!!! VERBOSE PRINTS
!!!------------------------------------------------------------------------
       elseif(index(buffer,'-v').eq.1)then
          flag="-v"
          call flagmaker(buffer,flag,i,skip,empty)
          if(.not.empty) read(buffer,*) ierror
       elseif(index(buffer,'-h').eq.1)then
          write(6,'("Flags:")')
          write(6,'(2X,"-h              : Prints the help for each flag.")')
          write(6,'(2X,"-v              : Verbose printing.")')
          write(6,'("-----------------FILE-NAME-FLAGS-----------------")')
          write(6,'(2X,"-f<STR>         : Input structure file name (Default = (empty)).")')
          !write(6,'(2X,"-d<STR>         : CHGCAR directory (Default = (empty)).")')
          write(6,'("------------------NETWORK-FLAGS------------------")')
          write(6,'(2X,"-s<INT>         : Random seed (Default = CLOCK).")')
          write(6,'(2X,"-l<INT,INT,...> : Hidden layer node size (Default = (empty) ).")')
          stop 0
       end if
    end do flagloop


!!!-----------------------------------------------------------------------------
!!! check if input file was specified and read if true
!!!-----------------------------------------------------------------------------
    if(trim(input_file).ne."")then
       call read_input_file(input_file)
    else
       stop "No input file given"
    end if


!!!-----------------------------------------------------------------------------
!!! read in dataset
!!!-----------------------------------------------------------------------------
    !! FOR LATER


    write(6,*) "======PARAMETERS======"
    write(6,*) "shuffle dataset:",shuffle_dataset
    write(6,*) "batch learning:",batch_learning
    write(6,*) "learning rate:",learning_rate
    write(6,*) "number of epochs:",num_epochs
    write(6,*) "number of filters:",cv_num_filters
    write(6,*) "hidden layers:",fc_num_hidden
    !write(6,*) "dataset size:",size(dataset,dim=1)
    !write(6,*) "training size:",nint(train_size*size(dataset,dim=1)),train_size
    write(6,*) "======================"


    return
  end subroutine set_global_vars
!!!#############################################################################



!!!#############################################################################
!!! read input file to get variables
!!!#############################################################################
  subroutine read_input_file(file_name)
    !use infile_tools, only: assign_val, assing_vec, rm_comments
    implicit none
    integer :: Reason,unit
    character(128) :: message
    
    integer :: num_filters
    
    integer :: block_size
    real(real12) ::  keep_prob
    real(real12) :: activation_scale

    real(real12) :: momentum, beta1, beta2, epsilon
    character(512) :: hidden_layers

    character(4)  :: loss
    character(9)  :: dropout
    character(6)  :: regularisation
    character(6)  :: normalisation
    character(20) :: adaptive_learning
    character(20) :: padding_type, convolution_type
    character(20) :: kernel_initialiser
    character(20) :: bias_initialiser
    character(20) :: weight_initialiser
    character(64) :: clip_min, clip_max, clip_norm
    character(512) :: kernel_size, stride
    character(20) :: activation_function

    character(*), intent(in) :: file_name


!!!-----------------------------------------------------------------------------
!!! set up namelists for input file
!!!-----------------------------------------------------------------------------
    namelist /setup/ seed, verbosity, num_threads !, dir
    namelist /training/ num_epochs, batch_size, &
         plateau_threshold, loss_threshold, &
         learning_rate, momentum, l1_lambda, l2_lambda, &
         shuffle_dataset, batch_learning, adaptive_learning, &
         beta1, beta2, epsilon, loss, &
         regularisation
    namelist /convolution/ num_filters, kernel_size, stride, &
         clip_min, clip_max, clip_norm, convolution_type, padding_type, &
         dropout, block_size, keep_prob, activation_function, &
         kernel_initialiser, bias_initialiser
    namelist /pooling/ kernel_size, stride, normalisation
    namelist /fully_connected/ hidden_layers, &
         clip_min, clip_max, clip_norm, &
         activation_function, activation_scale, &
         weight_initialiser


!!!-----------------------------------------------------------------------------
!!! check input file exists and open
!!!-----------------------------------------------------------------------------
    unit=20
    call file_check(unit,file_name)


!!!==========================================================================!!!
!!!                       read namelists from input file                     !!!
!!!==========================================================================!!!

!!!-----------------------------------------------------------------------------
!!! read setup namelist
!!!-----------------------------------------------------------------------------
    read(unit,NML=setup,iostat=Reason,iomsg=message)
    if(Reason.ne.0)then
       write(0,'("ERROR: Unexpected keyword found input file SETUP card")')
       stop trim(message)
    end if
    

!!!-----------------------------------------------------------------------------
!!! read training namelist
!!!-----------------------------------------------------------------------------
    loss="mse"
    dropout="none"
    block_size = 5
    keep_prob = 0.75_real12

    regularisation="" !! none, l1, l2, l1l2
    l1_lambda = 0._real12
    l2_lambda = 0._real12

    adaptive_learning = ""
    momentum = 0._real12
    beta1 = 0.9_real12
    beta2 = 0.999_real12
    epsilon = 1.E-8_real12
!!! ADD weight_decay (L2 penalty for AdamW)

    read(unit,NML=training,iostat=Reason,iomsg=message)
    if(.not.is_iostat_end(Reason).and.Reason.ne.0)then
       write(0,'("ERROR: Unexpected keyword found input file TRAINING card")')
       stop trim(message)
    end if
    if(batch_size.eq.1.and.batch_learning)then
       write(0,*) "WARNING: batch_learning=True whilst batch_size=1"
       write(0,*) " Changing to batch_learning=False"
       write(0,*) "(note: currently no input file way to specify alternative)"
    end if
    !! handle adaptive learning method
    !!---------------------------------------------------------------------------
    !! ce  = cross entropy (defaults to categorical)
    !! cce = categorical cross entropy
    !! mse = mean square error
    !! nll = negative log likelihood
    loss_method = to_lower(trim(loss))


!!!-----------------------------------------------------------------------------
!!! read convolution namelist
!!!-----------------------------------------------------------------------------
    num_filters = 32
    convolution_type    = "standard"
    padding_type        = "same"
    kernel_initialiser  = "he_uniform"
    bias_initialiser    = "zeros"
    activation_function = "none"
    activation_scale = 1._real12
    clip_min = ""; clip_max = ""; clip_norm = ""
    kernel_size = ""; stride = ""
    read(unit,NML=convolution,iostat=Reason,iomsg=message)
    if(.not.is_iostat_end(Reason).and.Reason.ne.0)then
       write(0,'("ERROR: Unexpected keyword found input file CONVOLUTION &
            &card")')
       stop trim(message)
    end if

    if(trim(kernel_size).ne."") call get_list(kernel_size, cv_kernel_size, cv_num_filters)
    if(trim(stride).ne."") call get_list(stride, cv_stride, cv_num_filters)
    call get_clip(clip_min, clip_max, clip_norm, cv_clip)
    cv_num_filters = num_filters
    cv_activation_scale = activation_scale
    cv_activation_function = to_lower(activation_function)
    cv_kernel_initialiser  = to_lower(kernel_initialiser)
    cv_bias_initialiser    = to_lower(bias_initialiser)
    !! handle convolution type
    !!---------------------------------------------------------------------------
    !! https://arxiv.org/pdf/1603.07285.pdf
    !! https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
    !! standard   = dot product operation between kernel and input data
    !! dilated    = spacing (dilation rate) between kernel values
    !! transposed = upsampling, sort of reverse of dilated
    !! depthwise  = separate filter to each input channel
    !! pointwise  = linear transform to adjust number of channels in feature map
    !!              ... 1x1 filter, not affecting spatial dimensions
    convolution_method = to_lower(trim(convolution_type))
!!! NOT SET UP YET
   
    !! handle padding type
    !!---------------------------------------------------------------------------
    !! none = alt. name for 'valid'
    !! zero = alt. name for 'same'
    !! half = alt. name for 'same'
    !! symmetric = alt.name for 'replication'
    !! valid = no padding
    !! same  = maintain spatial dimensions
    !!         ... (i.e. odd filter width, padding = (kernel_size - 1)/2)
    !!         ... (i.e. even filter width, padding = (kernel_size - 2)/2)
    !!         ... defaults to zeros in the padding
    !! full  = enough padding for filter to slide over every possible position
    !!         ... (i.e. padding = (kernel_size - 1)
    !! circular = maintain spatial dimensions
    !!            ... wraps data around for padding (periodic)
    !! reflection = maintains spatial dimensions
    !!              ... reflect data (about boundary index)
    !! replication = maintains spatial dimensions
    !!               ... reflect data (boundary included)
    padding_method = to_lower(trim(padding_type))


!!!-----------------------------------------------------------------------------
!!! read pooling namelist
!!!-----------------------------------------------------------------------------
    normalisation="none"
    kernel_size = ""; stride = ""
    read(unit,NML=pooling,iostat=Reason,iomsg=message)
    if(.not.is_iostat_end(Reason).and.Reason.ne.0)then
       write(0,'("ERROR: Unexpected keyword found input file POOLING card")')
       stop trim(message)
    end if

    if(trim(kernel_size).ne."") read(kernel_size,*) pool_kernel_size
    if(trim(stride).ne."") read(stride,*) pool_stride
    !! handle pooling normalisation
    !!--------------------------------------------------------------------------
    !! none
    !! linear
    !! sum
    !! norm
    pool_normalisation = to_lower(trim(normalisation))

    
!!!-----------------------------------------------------------------------------
!!! read fully connected namelist
!!!-----------------------------------------------------------------------------
    weight_initialiser = ""
    hidden_layers=""
    activation_scale = 1._real12
    activation_function = "relu"
    clip_min=""; clip_max=""; clip_norm=""
    read(unit,NML=fully_connected,iostat=Reason,iomsg=message)
    if(.not.is_iostat_end(Reason).and.Reason.ne.0)then
       write(0,'("ERROR: Unexpected keyword found input file FULLY_CONNECTED &
            &card")')
       stop trim(message)
    end if
    call get_clip(clip_min, clip_max, clip_norm, fc_clip)
    fc_activation_scale = activation_scale
    fc_activation_function = to_lower(activation_function)
    fc_weight_initialiser  = to_lower(weight_initialiser)
    !! convert hidden_layers string to dynamic array
    !!---------------------------------------------------------------------------
    call get_list(hidden_layers, fc_num_hidden)


!!!-----------------------------------------------------------------------------
!!! close input file
!!!-----------------------------------------------------------------------------
    close(unit)


!!!-----------------------------------------------------------------------------
!!! handle regularisation method
!!!-----------------------------------------------------------------------------
    !! none  = no regularisation
    !! l1    = l1 regularisation
    !! l2    = l2 regularisation
    !! l1l2  = l1 and l2 regularisation
    if(trim(regularisation).eq."")then
       if(l1_lambda.gt.0._real12.and.l2_lambda.gt.0._real12)then
          learning_parameters%regularisation = "l1l2"
          write(*,*) "l1_lambda and l2_lambda were set, but not regularisation"
          write(*,*) 'Setting regularisation = "l1l2"'
          write(*,*) 'If this is not desired, rerun with either:'
          write(*,*) '   regularisation = "none"'
          write(*,*) '   l1_lambda = 0.0, l2_lambda = 0.0'
       elseif(l1_lambda.gt.0._real12)then
          learning_parameters%regularisation = "l1"
          write(*,*) "l1_lambda was set, but not regularisation"
          write(*,*) 'Setting regularisation = "l1"'
          write(*,*) 'If this is not desired, rerun with either:'
          write(*,*) '   regularisation = "none"'
          write(*,*) '   l1_lambda = 0.0'
       elseif(l2_lambda.gt.0._real12)then
          learning_parameters%regularisation = "l2"
          write(*,*) "l2_lambda was set, but not regularisation"
          write(*,*) 'Setting regularisation = "l2"'
          write(*,*) 'If this is not desired, rerun with either:'
          write(*,*) '   regularisation = "none"'
          write(*,*) '   l2_lambda = 0.0'
       else
          learning_parameters%regularisation = "none"
       end if
    else
       learning_parameters%regularisation = to_lower(trim(regularisation))
    end if
    select case(learning_parameters%regularisation)
    case("none")
       write(*,*) "No regularisation set"
    case("l1l2")
       write(*,*) "L1L2 regularisation"
       if(abs(l1_lambda).le.1.E-8_real12.and.abs(l2_lambda).le.1.E-8_real12)then
          write(*,*) "ERROR: l1_lambda and l2_lambda set to = 0.0"
          write(*,*) "Please rerun with either a different regularisation or &
               &a larger values"
          stop "Exiting..."
       end if
       write(*,*) "l1_lambda =",l1_lambda
       write(*,*) "l2_lambda =",l2_lambda
       learning_parameters%l1 = l1_lambda
       learning_parameters%l2 = l2_lambda
    case("l1")
       write(*,*) "L1 regularisation"
       if(abs(l1_lambda).le.1.E-8_real12)then
          write(*,*) "ERROR: l1_lambda set to = 0.0"
          write(*,*) "Please rerun with either a different regularisation or &
               &a larger values"
          stop "Exiting..."
       end if
       write(*,*) "l1_lambda =",l1_lambda
       learning_parameters%l1 = l1_lambda
    case("l2")
       write(*,*) "L2 regularisation"
       if(abs(l2_lambda).le.1.E-8_real12)then
          write(*,*) "ERROR: l2_lambda set to = 0.0"
          write(*,*) "Please rerun with either a different regularisation or &
               &a larger values"
          stop "Exiting..."
       end if
       write(*,*) "l2_lambda =",l2_lambda
       learning_parameters%l2 = l2_lambda
    case default
       write(*,*) "ERROR: regularisation = "//learning_parameters%regularisation//" &
            &not known"
       stop "Exiting..."
    end select


!!!-----------------------------------------------------------------------------
!!! handle adaptive learning method
!!!-----------------------------------------------------------------------------
    !! none  = normal (stochastic) gradient descent
    !! adam  = adaptive moment estimation (adam) adaptive learning
    !! momentum   = momentum-based learning
    !! step_decay = step decay
    !! reduce_lr_on_plateau = reduce learning rate when output metric plateaus
    if(trim(adaptive_learning).eq."")then
       if(momentum.gt.0._real12)then
          learning_parameters%method = "momentum"
          write(*,*) "Momentum was set, but not adaptive_learning"
          write(*,*) 'Setting adaptive_learning = "momentum"'
          write(*,*) 'If this is not desired, rerun with either:'
          write(*,*) '   adaptive_learning = "none"'
          write(*,*) '   momentum = 0.0'
       else
          learning_parameters%method = "none"
       end if
    else
       learning_parameters%method = to_lower(trim(adaptive_learning))
    end if
    select case(learning_parameters%method)
    case("none")
       write(*,*) "No adaptive learning method"
    case("momentum")
       write(*,*) "Momentum-based adaptive learning method"
       if(abs(momentum).le.1.E-6_real12)then
          write(*,*) "ERROR: momentum adaptive learning set with momentum = 0"
          write(*,*) "Please rerun with either a different adaptive method or &
               &a larger momentum value"
          stop "Exiting..."
       end if
       write(*,*) "momentum =",momentum
    case("adam")
       write(*,*) "Adam-based adaptive learning method"
       write(*,*) "beta1 =", beta1
       write(*,*) "beta2 =", beta2
       write(*,*) "epsilon =", epsilon
       learning_parameters%beta1 = beta1
       learning_parameters%beta2 = beta2
       learning_parameters%epsilon = epsilon
    case("step_decay")
       !learning_parameters%decay_rate = decay_rate
       !learning_parameters%decay_steps = decay_steps
       stop "step_decay adaptive learning not yet set up"
    case("reduce_lr_on_plateau")
       stop "reduce_lr_on_plateau adaptive learning not yet set up"
    case default
       write(*,*) "ERROR: adaptive_learning = "//learning_parameters%method//" &
            &not known"
       stop "Exiting..."
    end select


!!!-----------------------------------------------------------------------------
!!! handle dropout method
!!!-----------------------------------------------------------------------------
    !! none
    !! dropblock
    if(trim(dropout).eq."")then
       cv_dropout_method = "none"
    else
       cv_dropout_method = to_lower(trim(dropout))
       if(cv_dropout_method.eq."dropblock")then
          cv_block_size = block_size
          cv_keep_prob = keep_prob
          write(*,*) "block_size =",cv_block_size
          write(*,*) "keep_prob =",cv_keep_prob
       end if
    end if
    write(*,*) "Dropout method: ",cv_dropout_method


    return
  end subroutine read_input_file
!!!#############################################################################


!!!#############################################################################
!!! get list from string
!!!#############################################################################
  subroutine get_list(string, output, length)
    implicit none
    character(512), intent(in) :: string
    integer, allocatable, dimension(:), intent(out) :: output
    integer, optional, intent(in) :: length

    character(1) :: fs
    
    if(scan(trim(string),",").ne.0)then
       fs = ","
    else
       fs = " "
    end if
    allocate(output(icount(trim(string),fs)))
    read(string,*) output

    if(present(length))then
       if(size(output,dim=1).ne.1.and.size(output,dim=1).ne.length)then
          stop "ERROR: LAYER PARAMETER DIMENSION SIZE MUST BE 1 OR &
               &NUMBER OF LAYERS" 
       end if
    end if
 
  end subroutine get_list
!!!#############################################################################


!!!#############################################################################
!!! get clipping information
!!!#############################################################################
  subroutine get_clip(min_str, max_str, norm_str, clip)
    implicit none
    character(*), intent(in) :: min_str, max_str, norm_str
    type(clip_type), intent(inout) :: clip

    if(trim(min_str).ne."") read(min_str,*) clip%min
    if(trim(max_str).ne."") read(max_str,*) clip%max

    if(trim(min_str).ne."".or.trim(max_str).ne."")then
       clip%l_min_max = .true.
    end if
    if(trim(norm_str).ne."")then
       read(norm_str,*) clip%norm
       clip%l_norm = .true.
    end if

  end subroutine get_clip
!!!#############################################################################

end module inputs
