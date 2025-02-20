module athena__batchnorm1d_layer
  !! Module containing implementation of 0D and 1D batch normalisation layers
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__base_layer, only: batch_layer_type, base_layer_type
  use athena__misc_types, only: initialiser_type, array3d_type
  implicit none
  

  private

  public :: batchnorm1d_layer_type
  public :: read_batchnorm1d_layer


  type, extends(batch_layer_type) :: batchnorm1d_layer_type
     !! Type for 0D or 1D batch normalisation layer with overloaded procedures
     integer :: num_inputs = 1
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_batchnorm1d
     !! Set hyperparameters for 1D batch normalisation layer
     procedure, pass(this) :: set_batch_size => set_batch_size_batchnorm1d
     !! Set batch size for 1D batch normalisation layer
     procedure, pass(this) :: print => print_batchnorm1d
     !! Print 1D batch normalisation layer to file
     procedure, pass(this) :: read => read_batchnorm1d
     !! Read 1D batch normalisation layer from file
     procedure, pass(this) :: forward  => forward_rank
     !! Forward propagation handler for 1D batch normalisation layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward propagation handler for 1D batch normalisation layer
     procedure, private, pass(this) :: forward_3d
     !! Forward propagation for 3D input
     procedure, private, pass(this) :: backward_3d
     !! Backward propagation for 3D input
     final :: finalise_batchnorm1d
     !! Finalise 1D batch normalisation layer
  end type batchnorm1d_layer_type


  interface batchnorm1d_layer_type
     !! Interface for setting up the 1D batch normalisation layer
     module function layer_setup( &
          input_shape, batch_size, &
          num_channels, num_inputs, &
          momentum, epsilon, &
          gamma_init_mean, gamma_init_std, &
          beta_init_mean, beta_init_std, &
          kernel_initialiser, bias_initialiser, &
          moving_mean_initialiser, moving_variance_initialiser, &
          verbose &
          ) result(layer)
       !! Set up the 1D batch normalisation layer
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: num_channels, num_inputs
       !! Number of channels and inputs
       real(real32), optional, intent(in) :: momentum, epsilon
       !! Momentum and epsilon
       real(real32), optional, intent(in) :: gamma_init_mean, gamma_init_std
       !! Gamma initialisation mean and standard deviation
       real(real32), optional, intent(in) :: beta_init_mean, beta_init_std
       !! Beta initialisation mean and standard deviation
       character(*), optional, intent(in) :: &
            kernel_initialiser, bias_initialiser, &
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

    if(associated(this%gamma)) nullify(this%gamma)
    if(associated(this%beta)) nullify(this%beta)
    if(allocated(this%mean)) deallocate(this%mean)
    if(allocated(this%variance)) deallocate(this%variance)
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(allocated(this%di)) deallocate(this%di)

  end subroutine finalise_batchnorm1d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine forward_rank(this, input)
    !! Forward propagation handler for 1D batch normalisation layer
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(inout) :: this
    !! Instance of the 1D batch normalisation layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values

    select rank(input)
    rank(2)
       call forward_3d(this, input)
    rank(3)
       call forward_3d(this, input)
    end select
  end subroutine forward_rank
!###############################################################################


!###############################################################################
  pure subroutine backward_rank(this, input, gradient)
    !! Backward propagation handler for 1D batch normalisation layer
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(inout) :: this
    !! Instance of the 1D batch normalisation layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient values

    select rank(input)
    rank(2)
       select rank(gradient)
       rank(2)
          call backward_3d(this, input, gradient)
       end select
    rank(3)
       select rank(gradient)
       rank(1)
          call backward_3d(this, input, gradient)
       rank(2)
          call backward_3d(this, input, gradient)
       rank(3)
          call backward_3d(this, input, gradient)
       end select
    end select
  end subroutine backward_rank
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       input_shape, batch_size, &
       num_channels, num_inputs, &
       momentum, epsilon, &
       gamma_init_mean, gamma_init_std, &
       beta_init_mean, beta_init_std, &
       kernel_initialiser, bias_initialiser, &
       moving_mean_initialiser, moving_variance_initialiser, &
       verbose &
       ) result(layer)
    !! Set up the 1D batch normalisation layer
    use athena__initialiser, only: get_default_initialiser
    implicit none

    ! Arguments
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: num_channels, num_inputs
    !! Number of channels and inputs
    real(real32), optional, intent(in) :: momentum, epsilon
    !! Momentum and epsilon
    real(real32), optional, intent(in) :: gamma_init_mean, gamma_init_std
    !! Gamma initialisation mean and standard deviation
    real(real32), optional, intent(in) :: beta_init_mean, beta_init_std
    !! Beta initialisation mean and standard deviation
    character(*), optional, intent(in) :: &
         kernel_initialiser, bias_initialiser, &
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
    if(present(kernel_initialiser)) &
         layer%kernel_initialiser = kernel_initialiser
    if(present(bias_initialiser)) layer%bias_initialiser = bias_initialiser

    if(present(moving_mean_initialiser)) &
         layer%moving_mean_initialiser = moving_mean_initialiser
    if(present(moving_variance_initialiser)) &
         layer%moving_variance_initialiser = moving_variance_initialiser


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         momentum = layer%momentum, epsilon = layer%epsilon, &
         gamma_init_mean = layer%gamma_init_mean, &
         gamma_init_std = layer%gamma_init_std, &
         beta_init_mean = layer%beta_init_mean, &
         beta_init_std = layer%beta_init_std, &
         kernel_initialiser = layer%kernel_initialiser, &
         bias_initialiser = layer%bias_initialiser, &
         moving_mean_initialiser = layer%moving_mean_initialiser, &
         moving_variance_initialiser = layer%moving_variance_initialiser, &
         verbose = verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise batch size
    !---------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


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
       kernel_initialiser, bias_initialiser, &
       moving_mean_initialiser, moving_variance_initialiser, &
       verbose )
    !! Set hyperparameters for 1D batch normalisation layer
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(inout) :: this
    !! Instance of the 1D batch normalisation layer
    real(real32), intent(in) :: momentum, epsilon
    !! Momentum and epsilon
    real(real32), intent(in) :: gamma_init_mean, gamma_init_std
    !! Gamma initialisation mean and standard deviation
    real(real32), intent(in) :: beta_init_mean, beta_init_std
    character(*), intent(in) :: kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    character(*), intent(in) :: &
         moving_mean_initialiser, moving_variance_initialiser
    !! Moving mean and variance initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level


    this%name = "batchnorm1d"
    this%type = "batc"
    this%input_rank = 2
    this%output = array3d_type()
    this%momentum = momentum
    this%epsilon = epsilon
    if(trim(this%kernel_initialiser).eq.'') &
         this%kernel_initialiser = 'ones'
         !get_default_initialiser("batch")
    if(trim(this%bias_initialiser).eq.'') &
         this%bias_initialiser = 'zeros'
         !get_default_initialiser("batch")

    if(trim(this%moving_mean_initialiser).eq.'') &
         this%moving_mean_initialiser = 'zeros'
         !get_default_initialiser("batch")
    if(trim(this%moving_variance_initialiser).eq.'') &
         this%moving_variance_initialiser = 'ones'
         !get_default_initialiser("batch")

    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("BATCHNORM1D kernel (gamma) initialiser: ",A)') &
               trim(this%kernel_initialiser)
          write(*,'("BATCHNORM1D bias (beta) initialiser: ",A)') &
               trim(this%bias_initialiser)
          write(*,'("BATCHNORM1D moving mean initialiser: ",A)') &
               trim(this%moving_mean_initialiser)
          write(*,'("BATCHNORM1D moving variance initialiser: ",A)') &
               trim(this%moving_variance_initialiser)
       end if
    end if

  end subroutine set_hyperparams_batchnorm1d
!###############################################################################


!###############################################################################
  subroutine set_batch_size_batchnorm1d(this, batch_size, verbose)
    !! Set batch size for 1D batch normalisation layer
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(inout), target :: this
    !! Instance of the 1D batch normalisation layer
    integer, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Set number of inputs
    !---------------------------------------------------------------------------
    if(size(this%input_shape).eq.2) this%num_inputs = this%input_shape(1)


    !---------------------------------------------------------------------------
    ! Initialise gamma and beta parameters
    !---------------------------------------------------------------------------
    this%gamma(1:this%num_channels) => this%params(1:this%num_channels)
    this%beta(1:this%num_channels) => &
         this%params(this%num_channels+1:this%num_channels*2)


    !---------------------------------------------------------------------------
    ! Set norm
    !---------------------------------------------------------------------------
    this%norm = real(this%batch_size, real32)


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(.not.allocated(this%output)) this%output = array3d_type()
       if(this%output%allocated) call this%output%deallocate(keep_shape=.true.)
       call this%output%allocate( array_shape = [ &
            this%num_inputs, &
            this%num_channels, &
            this%batch_size ], &
            source=0._real32 &
       )
       if(.not.allocated(this%di)) this%di = array3d_type()
       if(this%di%allocated) call this%di%deallocate()
       call this%di%allocate( source = this%output )
    end if

  end subroutine set_batch_size_batchnorm1d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_batchnorm1d(this, file)
    !! Print 1D batch normalisation layer to file
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(in) :: this
    !! Instance of the 1D batch normalisation layer
    character(*), intent(in) :: file
    !! File name

    ! Local variables
    integer :: unit
    !! File unit
    integer :: m
    !! Loop index


    ! Open file with new unit
    !---------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'("BATCHNORM1D")')
    write(unit,'(3X,"INPUT_SHAPE = ",2(1X,I0))') this%input_shape
    write(unit,'(3X,"MOMENTUM = ",F0.9)') this%momentum
    write(unit,'(3X,"EPSILON = ",F0.9)') this%epsilon
    write(unit,'(3X,"NUM_CHANNELS = ",I0)') this%num_channels
    write(unit,'("GAMMA")')
    do m=1,this%num_channels
       write(unit,'(5(E16.8E2))') this%gamma(m)
    end do
    write(unit,'("END GAMMA")')
    write(unit,'("BETA")')
    do m=1,this%num_channels
       write(unit,'(5(E16.8E2))') this%beta(m)
    end do
    write(unit,'("END BATCHNORM1D")')

    ! Close unit
    !---------------------------------------------------------------------------
    close(unit)

  end subroutine print_batchnorm1d
!###############################################################################


!###############################################################################
  subroutine read_batchnorm1d(this, unit, verbose)
    !! Read 1D batch normalisation layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
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
    integer :: itmp1, c, i, j, k
    !! Temporary integers and loop indices
    integer :: num_channels
    !! Number of channels
    real(real32) :: momentum = 0._real32, epsilon = 1.E-5_real32
    !! Momentum and epsilon
    logical :: found_gamma=.false., found_beta=.false.
    !! Flags for gamma and beta
    character(14) :: kernel_initialiser='', bias_initialiser=''
    !! Initialisers
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message

    integer, dimension(3) :: input_shape
    !! Input shape
    real(real32), allocatable, dimension(:) :: data_list
    !! Data list


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
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
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
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
       case("KERNEL_INITIALISER")
          call assign_val(buffer, kernel_initialiser, itmp1)
       case("BIAS_INITIALISER")
          call assign_val(buffer, bias_initialiser, itmp1)
       case("GAMMA")
          found_gamma = .true.
          kernel_initialiser = 'zeros'
          bias_initialiser   = 'zeros'
          exit tag_loop
       case("BETA")
          found_beta = .true.
          kernel_initialiser = 'zeros'
          bias_initialiser   = 'zeros'
          exit tag_loop
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
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser, &
         moving_mean_initialiser = this%moving_mean_initialiser, &
         moving_variance_initialiser = this%moving_variance_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape = input_shape)


    ! Check if WEIGHTS card was found
    !---------------------------------------------------------------------------
    allocate(data_list(num_channels), source=0._real32)
    do i=1,2
      if(found_gamma.or.found_beta)then
         c = 1
         k = 1
         data_list = 0._real32
         data_concat_loop: do while(c.le.num_channels)
            read(unit,'(A)',iostat=stat) buffer
            if(stat.ne.0) exit data_concat_loop
            k = icount(buffer)
            read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
            c = c + k
         end do data_concat_loop
         if(found_gamma)then
            this%gamma = data_list
            found_gamma = .false.
         elseif(found_beta)then
            this%beta = data_list
            found_beta = .false.
         end if
         read(unit,'(A)',iostat=stat) buffer
         if(index(trim(adjustl(buffer)),"GAMMA").eq.1) found_gamma = .true.
         if(index(trim(adjustl(buffer)),"BETA").eq.1) found_beta = .true.
      end if
    end do
    deallocate(data_list)


    ! Check for end of layer card
    !---------------------------------------------------------------------------
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


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine forward_3d(this, input)
    !! Forward propagation for 3D input
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(inout) :: this
    !! Instance of the 1D batch normalisation layer
    real(real32), dimension( &
         this%num_inputs, &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: input
    !! Input values

    ! Local variables
    integer :: m
    !! Loop index
    real(real32), dimension(this%num_channels) :: t_mean, t_variance
    !! Temporary mean and variance


    select type(output => this%output)
    type is (array3d_type)
       select case(this%inference)
       case(.true.)
         do concurrent(m=1:this%num_channels)
            ! Normalise each feature
            output%val_ptr(:,m,:) = &
                 this%gamma(m) * (input(:,m,:) - this%mean(m)) / &
                 sqrt( &
                 this%batch_size / (this%batch_size - 1) * this%variance(m) + &
                 this%epsilon) + &
                 this%beta(m)
         end do
       case default
         t_mean = 0._real32
         t_variance = 0._real32
         do concurrent(m=1:this%num_channels)
             ! Calculate current mean and variance
             t_mean(m) = sum(input(:,m,:)) / this%norm
             t_variance(m) = &
                  sum((input(:,m,:) - t_mean(m))**2._real32) / this%norm

             ! Convert to using inverse square root of variance (i.e. inv. std)
             ! Would also need to include epsilon in the sqrt denominator

             ! Update running averages
             if(this%momentum.gt.1.E-8_real32)then
                this%mean(m) = this%momentum * this%mean(m) + &
                      (1._real32 - this%momentum) * t_mean(m)
                this%variance(m) = this%momentum * this%variance(m) + &
                      (1._real32 - this%momentum) * t_variance(m)
             else
                this%mean(m) = t_mean(m)
                this%variance(m) = t_variance(m)
             end if

             ! Normalise each feature
             output%val_ptr(:,m,:) = &
                  this%gamma(m) * (input(:,m,:) - this%mean(m)) / &
                  sqrt(this%variance(m) + this%epsilon) + this%beta(m)

          end do
       end select
    end select

  end subroutine forward_3d
!###############################################################################


!###############################################################################
  pure subroutine backward_3d(this, input, gradient)
    !! Backward propagation for 3D input
    implicit none

    ! Arguments
    class(batchnorm1d_layer_type), intent(inout) :: this
    !! Instance of the 1D batch normalisation layer
    real(real32), dimension( &
         this%num_inputs, &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: input
    !! Input values
    real(real32), dimension( &
         this%num_inputs, &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: gradient
    !! Gradient values

    ! Local variables
    integer :: m
    !! Loop index
    real(real32), dimension( &
         this%num_inputs, &
         this%num_channels, &
         this%batch_size) :: x_hat, dx_hat
    !! Normalised input and gradient of normalised input


    select type(di => this%di)
    type is (array3d_type)
       do concurrent(m=1:this%num_channels)
          ! Recalculate x_hat (i.e. normalised input)
          x_hat(:,m,:) = (input(:,m,:) - this%mean(m)) / &
               sqrt(this%variance(m) + this%epsilon)

          ! Calculate gradient of normalised input
          dx_hat(:,m,:) = gradient(:,m,:) * this%gamma(m)

          ! Calculate gradient of inputs
          di%val_ptr(:,m,:) = &
               1._real32 / ( &
                    this%norm * sqrt(this%variance(m) + this%epsilon) &
               ) * &
               ( this%norm * dx_hat(:,m,:) - &
               sum(dx_hat(:,m,:)) - x_hat(:,m,:) * &
               sum(dx_hat(:,m,:) * x_hat(:,m,:)))

          ! Calculate gradient of gamma and beta
          this%dp(m,1) = sum(gradient(:,m,:) * x_hat(:,m,:))
          this%db(m,1) = sum(gradient(:,m,:))
       end do
    end select

  end subroutine backward_3d
!###############################################################################

end module athena__batchnorm1d_layer