module athena__batchnorm1d_layer
  !! Module containing implementation of 0D and 1D batch normalisation layers
  !!
  !! This module implements batch normalisation for 3D convolutional layers,
  !! normalizing activations across the batch dimension.
  !!
  !! Mathematical operation (training):
  !! \[ \mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m} x_i \]
  !! \[ \sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2 \]
  !! \[ \hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}} \]
  !! \[ y_i = \gamma \hat{x}_i + \beta \]
  !!
  !! where \(\gamma, \beta\) are learnable parameters, \(\epsilon\) is stability constant
  !!
  !! Inference: uses running statistics
  !! \(\mu_{\text{running}}, \sigma^2_{\text{running}}\) from training
  !!
  !! Benefits: Reduces internal covariate shift, enables higher learning rates,
  !! acts as regularisation, reduces dependence on initialisation
  !! Reference: Ioffe & Szegedy (2015), ICML
  use coreutils, only: real32, stop_program, print_warning
  use athena__base_layer, only: batch_layer_type, base_layer_type
  use athena__misc_types, only: base_init_type, &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use diffstruc, only: array_type
  use athena__diffstruc_extd, only: batchnorm_array_type, &
       batchnorm, batchnorm_inference
  implicit none


  private

  public :: batchnorm1d_layer_type
  public :: read_batchnorm1d_layer


  type, extends(batch_layer_type) :: batchnorm1d_layer_type
     !! Type for 0D or 1D batch normalisation layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_batchnorm1d
     !! Set hyperparameters for 1D batch normalisation layer
     procedure, pass(this) :: read => read_batchnorm1d
     !! Read 1D batch normalisation layer from file

     procedure, pass(this) :: forward => forward_batchnorm1d
     !! Forward propagation derived type handler

     final :: finalise_batchnorm1d
     !! Finalise 1D batch normalisation layer
  end type batchnorm1d_layer_type


  interface batchnorm1d_layer_type
     !! Interface for setting up the 1D batch normalisation layer
     module function layer_setup( &
          input_shape, &
          num_channels, num_inputs, &
          momentum, epsilon, &
          gamma_init_mean, gamma_init_std, &
          beta_init_mean, beta_init_std, &
          gamma_initialiser, beta_initialiser, &
          moving_mean_initialiser, moving_variance_initialiser, &
          verbose &
     ) result(layer)
       !! Set up the 1D batch normalisation layer
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: num_channels, num_inputs
       !! Number of channels and inputs
       real(real32), optional, intent(in) :: momentum, epsilon
       !! Momentum and epsilon
       real(real32), optional, intent(in) :: gamma_init_mean, gamma_init_std
       !! Gamma initialisation mean and standard deviation
       real(real32), optional, intent(in) :: beta_init_mean, beta_init_std
       !! Beta initialisation mean and standard deviation
       class(*), optional, intent(in) :: &
            gamma_initialiser, beta_initialiser, &
            moving_mean_initialiser, moving_variance_initialiser
       !! Initialisers
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(batchnorm1d_layer_type) :: layer
       !! Instance of the 1D batch normalisation layer
     end function layer_setup
  end interface batchnorm1d_layer_type



contains

!###############################################################################
  subroutine finalise_batchnorm1d(this)
    !! Finalise 1D batch normalisation layer
    implicit none

    ! Arguments
    type(batchnorm1d_layer_type), intent(inout) :: this
    !! Instance of the 1D batch normalisation layer

    if(allocated(this%mean)) deallocate(this%mean)
    if(allocated(this%variance)) deallocate(this%variance)
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)

  end subroutine finalise_batchnorm1d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       input_shape, &
       num_channels, num_inputs, &
       momentum, epsilon, &
       gamma_init_mean, gamma_init_std, &
       beta_init_mean, beta_init_std, &
       gamma_initialiser, beta_initialiser, &
       moving_mean_initialiser, moving_variance_initialiser, &
       verbose &
  ) result(layer)
    !! Set up the 1D batch normalisation layer
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: num_channels, num_inputs
    !! Number of channels and inputs
    real(real32), optional, intent(in) :: momentum, epsilon
    !! Momentum and epsilon
    real(real32), optional, intent(in) :: gamma_init_mean, gamma_init_std
    !! Gamma initialisation mean and standard deviation
    real(real32), optional, intent(in) :: beta_init_mean, beta_init_std
    !! Beta initialisation mean and standard deviation
    class(*), optional, intent(in) :: &
         gamma_initialiser, beta_initialiser, &
         moving_mean_initialiser, moving_variance_initialiser
    !! Initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(batchnorm1d_layer_type) :: layer
    !! Instance of the 1D batch normalisation layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    character(256) :: err_msg
    !! Error message
    class(base_init_type), allocatable :: &
         gamma_initialiser_, beta_initialiser_, &
         moving_mean_initialiser_, moving_variance_initialiser_
    !! Initialisers


    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Set up momentum and epsilon
    !---------------------------------------------------------------------------
    if(present(momentum))then
       layer%momentum = momentum
    else
       layer%momentum = 0._real32
    end if
    if(present(epsilon))then
       layer%epsilon = epsilon
    else
       layer%epsilon = 1.E-5_real32
    end if


    !---------------------------------------------------------------------------
    ! Set up initialiser mean and standard deviations
    !---------------------------------------------------------------------------
    if(present(gamma_init_mean)) layer%gamma_init_mean = gamma_init_mean
    if(present(gamma_init_std))  layer%gamma_init_std = gamma_init_std
    if(present(beta_init_mean))  layer%beta_init_mean = beta_init_mean
    if(present(beta_init_std))   layer%beta_init_std = beta_init_std


    !---------------------------------------------------------------------------
    ! Define gamma and beta initialisers
    !---------------------------------------------------------------------------
    if(present(gamma_initialiser))then
       gamma_initialiser_ = initialiser_setup(gamma_initialiser)
    end if
    if(present(beta_initialiser))then
       beta_initialiser_ = initialiser_setup(beta_initialiser)
    end if
    if(present(moving_mean_initialiser))then
       moving_mean_initialiser_ = initialiser_setup(moving_mean_initialiser)
    end if
    if(present(moving_variance_initialiser))then
       moving_variance_initialiser_ = initialiser_setup(moving_variance_initialiser)
    end if


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         momentum = layer%momentum, epsilon = layer%epsilon, &
         gamma_init_mean = layer%gamma_init_mean, &
         gamma_init_std = layer%gamma_init_std, &
         beta_init_mean = layer%beta_init_mean, &
         beta_init_std = layer%beta_init_std, &
         gamma_initialiser = gamma_initialiser_, &
         beta_initialiser = beta_initialiser_, &
         moving_mean_initialiser = moving_mean_initialiser_, &
         moving_variance_initialiser = moving_variance_initialiser_, &
         verbose = verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape))then
       if(present(num_channels).or.present(num_inputs))then
          write(err_msg,'(A)') &
               "both input_shape and num_channels/num_inputs present" // &
               achar(13) // achar(10) // &
               "These represent the same parameter, so are conflicting"
          call stop_program(err_msg)
          return
       end if
       if(size(input_shape).eq.1)then
          call layer%init(input_shape= [ 1, input_shape ] )
       else
          call layer%init(input_shape= input_shape)
       end if
    elseif(present(num_channels).and.present(num_inputs))then
       call layer%init(input_shape=[num_inputs, num_channels])
    elseif(present(num_channels))then
       call layer%init(input_shape=[1, num_channels])
    elseif(present(num_inputs))then
       call layer%init(input_shape=[num_inputs, 1])
    end if

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_batchnorm1d( &
       this, &
       momentum, epsilon, &
       gamma_init_mean, gamma_init_std, &
       beta_init_mean, beta_init_std, &
       gamma_initialiser, beta_initialiser, &
       moving_mean_initialiser, moving_variance_initialiser, &
       verbose )
    !! Set hyperparameters for 1D batch normalisation layer
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(inout) :: this
    !! Instance of the 1D batch normalisation layer
    real(real32), intent(in) :: momentum, epsilon
    !! Momentum and epsilon
    real(real32), intent(in) :: gamma_init_mean, gamma_init_std
    !! Gamma initialisation mean and standard deviation
    real(real32), intent(in) :: beta_init_mean, beta_init_std
    !! Beta initialisation mean and standard deviation
    class(base_init_type), allocatable, intent(in) :: &
         gamma_initialiser, beta_initialiser
    !! Gamma and beta initialisers
    class(base_init_type), allocatable, intent(in) :: &
         moving_mean_initialiser, moving_variance_initialiser
    !! Moving mean and variance initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level


    this%name = "batchnorm1d"
    this%type = "batc"
    this%input_rank = 2
    this%output_rank = 2
    this%use_bias = .true.
    this%momentum = momentum
    this%epsilon = epsilon
    if(allocated(this%kernel_init)) deallocate(this%kernel_init)
    if(.not.allocated(gamma_initialiser))then
       this%kernel_init = initialiser_setup('ones')
    else
       allocate(this%kernel_init, source=gamma_initialiser)
    end if
    if(allocated(this%bias_init)) deallocate(this%bias_init)
    if(.not.allocated(beta_initialiser))then
       this%bias_init = initialiser_setup('zeros')
    else
       allocate(this%bias_init, source=beta_initialiser)
    end if
    if(.not.allocated(moving_mean_initialiser))then
       this%moving_mean_init = initialiser_setup('zeros')
    else
       this%moving_mean_init = moving_mean_initialiser
    end if
    if(.not.allocated(moving_variance_initialiser))then
       this%moving_variance_init = initialiser_setup('ones')
    else
       this%moving_variance_init = moving_variance_initialiser
    end if
    this%gamma_init_mean = gamma_init_mean
    this%gamma_init_std  = gamma_init_std
    this%beta_init_mean  = beta_init_mean
    this%beta_init_std   = beta_init_std
    this%kernel_init%mean = this%gamma_init_mean
    this%kernel_init%std  = this%gamma_init_std
    this%bias_init%mean = this%beta_init_mean
    this%bias_init%std  = this%beta_init_std
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("BATCHNORM1D gamma (kernel) initialiser: ",A)') &
               trim(this%kernel_init%name)
          write(*,'("BATCHNORM1D beta (bias) initialiser: ",A)') &
               trim(this%bias_init%name)
          write(*,'("BATCHNORM1D moving mean initialiser: ",A)') &
               trim(this%moving_mean_init%name)
          write(*,'("BATCHNORM1D moving variance initialiser: ",A)') &
               trim(this%moving_variance_init%name)
       end if
    end if

  end subroutine set_hyperparams_batchnorm1d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine read_batchnorm1d(this, unit, verbose)
    !! Read 1D batch normalisation layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(inout) :: this
    !! Instance of the 1D batch normalisation layer
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat, verbose_ = 0
    !! File status and verbosity level
    integer :: i, j, k, c, itmp1, iline, final_line
    !! Temporary integers and loop indices
    integer :: num_channels
    !! Number of channels
    real(real32) :: momentum = 0._real32, epsilon = 1.E-5_real32
    !! Momentum and epsilon
    class(base_init_type), allocatable :: gamma_initialiser, beta_initialiser
    !! Initialisers
    class(base_init_type), allocatable :: &
         moving_mean_initialiser, moving_variance_initialiser
    !! Moving mean and variance initialisers
    character(14) :: gamma_initialiser_name='', beta_initialiser_name=''
    !! Initialisers
    character(14) :: &
         moving_mean_initialiser_name='', &
         moving_variance_initialiser_name=''
    !! Moving mean and variance initialisers
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    integer, dimension(2) :: input_shape
    !! Input shape
    real(real32), allocatable, dimension(:) :: data_list
    !! Data list
    integer, dimension(2) :: param_lines
    !! Lines where parameters are found


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
    iline = 0
    param_lines = 0
    final_line = 0
    tag_loop: do

       ! Check for end of file
       !------------------------------------------------------------------------
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg,'("file encountered error (EoF?) before END ",A)') &
               to_upper(this%name)
          call stop_program(err_msg)
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       ! Check for end of layer card
       !------------------------------------------------------------------------
       if(trim(adjustl(buffer)).eq."END "//to_upper(trim(this%name)))then
          final_line = iline
          backspace(unit)
          exit tag_loop
       end if
       iline = iline + 1

       tag = trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       ! Read parameters from save file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("INPUT_SHAPE")
          call assign_vec(buffer, input_shape, itmp1)
       case("MOMENTUM")
          call assign_val(buffer, momentum, itmp1)
       case("EPSILON")
          call assign_val(buffer, epsilon, itmp1)
       case("NUM_CHANNELS")
          call assign_val(buffer, num_channels, itmp1)
          write(0,*) "NUM_CHANNELS and INPUT_SHAPE are conflicting parameters"
          write(0,*) "NUM_CHANNELS will be ignored"
       case("GAMMA_INITIALISER", "KERNEL_INITIALISER")
          if(param_lines(1).ne.0)then
             write(err_msg,'("GAMMA and GAMMA_INITIALISER defined. Using GAMMA only.")')
             call print_warning(err_msg)
          end if
          call assign_val(buffer, gamma_initialiser_name, itmp1)
       case("BETA_INITIALISER", "BIAS_INITIALISER")
          if(param_lines(2).ne.0)then
             write(err_msg,'("BETA and BETA_INITIALISER defined. Using BETA only.")')
             call print_warning(err_msg)
          end if
          call assign_val(buffer, beta_initialiser_name, itmp1)
       case("MOVING_MEAN_INITIALISER")
          call assign_val(buffer, moving_mean_initialiser_name, itmp1)
       case("MOVING_VARIANCE_INITIALISER")
          call assign_val(buffer, moving_variance_initialiser_name, itmp1)
       case("GAMMA")
          gamma_initialiser_name = 'zeros'
          param_lines(1) = iline
       case("BETA")
          beta_initialiser_name   = 'zeros'
          param_lines(2) = iline
       case default
          ! Don't look for "e" due to scientific notation of numbers
          ! ... i.e. exponent (E+00)
          if(scan(to_lower(trim(adjustl(buffer))),&
               'abcdfghijklmnopqrstuvwxyz').eq.0)then
             cycle tag_loop
          elseif(tag(:3).eq.'END')then
             cycle tag_loop
          end if
          write(err_msg,'("Unrecognised line in input file: ",A)') &
               trim(adjustl(buffer))
          call stop_program(err_msg)
          return
       end select
    end do tag_loop
    gamma_initialiser = initialiser_setup(gamma_initialiser_name)
    beta_initialiser = initialiser_setup(beta_initialiser_name)
    moving_mean_initialiser = initialiser_setup(moving_mean_initialiser_name)
    moving_variance_initialiser = initialiser_setup(moving_variance_initialiser_name)


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    num_channels = input_shape(size(input_shape))
    call this%set_hyperparams( &
         momentum = momentum, &
         epsilon = epsilon, &
         gamma_init_mean = this%gamma_init_mean, &
         gamma_init_std = this%gamma_init_std, &
         beta_init_mean = this%beta_init_mean, &
         beta_init_std = this%beta_init_std, &
         gamma_initialiser = gamma_initialiser, &
         beta_initialiser = beta_initialiser, &
         moving_mean_initialiser = moving_mean_initialiser, &
         moving_variance_initialiser = moving_variance_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape = input_shape)


    ! Check if WEIGHTS card was found
    !---------------------------------------------------------------------------
    allocate(data_list(num_channels), source=0._real32)
    do i = 2, 1, -1
       if(param_lines(i).eq.0) cycle
       call move(unit, param_lines(i) - iline, iostat=stat)
       iline = param_lines(i) + 1
       c = 1
       k = 1
       data_list = 0._real32
       data_concat_loop: do while(c.le.num_channels)
          iline = iline + 1
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit data_concat_loop
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do data_concat_loop
       read(unit,'(A)',iostat=stat) buffer
       select case(i)
       case(1) ! gamma
          this%params(1)%val(1:this%num_channels,1) = data_list
          if(trim(adjustl(buffer)).ne."END GAMMA")then
             write(err_msg,'("END GAMMA not where expected: ",A)') &
                  trim(adjustl(buffer))
             call stop_program(err_msg)
             return
          end if
       case(2) ! beta
          this%params(1)%val(this%num_channels+1:this%num_channels*2,1) = &
               data_list
          if(trim(adjustl(buffer)).ne."END BETA")then
             write(err_msg,'("END BETA not where expected: ",A)') &
                  trim(adjustl(buffer))
             call stop_program(err_msg)
             return
          end if
       end select
    end do
    deallocate(data_list)


    ! Check for end of layer card
    !---------------------------------------------------------------------------
    call move(unit, final_line - iline, iostat=stat)
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_batchnorm1d
!###############################################################################


!###############################################################################
  function read_batchnorm1d_layer(unit, verbose) result(layer)
    !! Read 1D batch normalisation layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Allocatable instance of the base layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=batchnorm1d_layer_type())
    call layer%read(unit, verbose=verbose_)

  end function read_batchnorm1d_layer
!###############################################################################


!###############################################################################
  subroutine build_from_onnx_batchnorm1d( &
       this, node, initialisers, value_info, verbose &
  )
    !! Read ONNX attributes for 1D batch normalisation layer
    use athena__initialiser_data, only: data_init_type
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(inout) :: this
    !! Instance of the 1D batch normalisation layer
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info
    integer, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: i
    !! Loop index
    real(real32) :: epsilon, momentum
    !! Epsilon and momentum values
    character(256) :: val
    !! Attribute value
    class(base_init_type), allocatable :: gamma_initialiser, beta_initialiser
    class(base_init_type), allocatable :: &
         moving_mean_initialiser, moving_variance_initialiser

    ! Set default values
    epsilon = 1.E-5_real32
    momentum = 0.9_real32

    ! Parse ONNX attributes
    do i = 1, size(node%attributes)
       val = node%attributes(i)%val
       select case(trim(adjustl(node%attributes(i)%name)))
       case("epsilon")
          read(val,*) epsilon
       case("momentum")
          read(val,*) momentum
       case default
          ! Do nothing
          write(0,*) "WARNING: Unrecognised attribute in ONNX BATCHNORM1D &
               &layer: ", trim(adjustl(node%attributes(i)%name))
       end select
    end do

    ! Check for 4 initialisers: gamma, beta, mean, variance
    if(size(initialisers).ne.4)then
       call stop_program("ONNX BATCHNORM1D layer requires 4 initialisers &
            &(gamma, beta, mean, variance)")
       return
    end if

    ! ONNX BatchNormalization order: gamma, beta, mean, variance
    gamma_initialiser = data_init_type( data = initialisers(1)%data )
    beta_initialiser = data_init_type( data = initialisers(2)%data )
    moving_mean_initialiser = data_init_type( data = initialisers(3)%data )
    moving_variance_initialiser = data_init_type( data = initialisers(4)%data )

    call this%set_hyperparams( &
         momentum = momentum, &
         epsilon = epsilon, &
         gamma_init_mean = 1.0_real32, &
         gamma_init_std = 0.0_real32, &
         beta_init_mean = 0.0_real32, &
         beta_init_std = 0.0_real32, &
         gamma_initialiser = gamma_initialiser, &
         beta_initialiser = beta_initialiser, &
         moving_mean_initialiser = moving_mean_initialiser, &
         moving_variance_initialiser = moving_variance_initialiser, &
         verbose = verbose &
    )

  end subroutine build_from_onnx_batchnorm1d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_batchnorm1d(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(inout) :: this
    !! Instance of the 1D batch normalisation layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    class(batchnorm_array_type), pointer :: ptr
    ! Pointer array


    select case(this%inference)
    case(.true.)
       ! Do not perform the drop operation

       ptr => batchnorm_inference(input(1,1), this%params(1), &
            this%mean(:), this%variance(:), this%epsilon &
       )

    case default
       ! Perform the drop operation
       ptr => batchnorm( &
            input(1,1), this%params(1),&
            this%momentum, this%mean(:), this%variance(:), this%epsilon &
       )

    end select
    select type(output => this%output(1,1))
    type is(batchnorm_array_type)
       call output%assign_shallow(ptr)
       output%epsilon = ptr%epsilon
       output%mean = ptr%mean
       output%variance = ptr%variance
    end select
    deallocate(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_batchnorm1d
!###############################################################################

end module athena__batchnorm1d_layer
