module athena__batchnorm3d_layer
  !! Module containing implementation of 3D batch normalisation layers
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: batch_layer_type, base_layer_type
  use athena__misc_types, only: initialiser_type
  use diffstruc, only: array_type
  use athena__diffstruc_extd, only: batchnorm_array_type, &
       batchnorm, batchnorm_inference
  implicit none


  private

  public :: batchnorm3d_layer_type
  public :: read_batchnorm3d_layer


  type, extends(batch_layer_type) :: batchnorm3d_layer_type
     !! Type for 3D batch normalisation layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_batchnorm3d
     !! Set hyperparameters for 3D batch normalisation layer
     procedure, pass(this) :: set_batch_size => set_batch_size_batchnorm3d
     !! Set batch size for 3D batch normalisation layer
     procedure, pass(this) :: print_to_unit => print_to_unit_batchnorm3d
     !! Print 3D batch normalisation layer to unit
     procedure, pass(this) :: read => read_batchnorm3d
     !! Read 3D batch normalisation layer from file

     procedure, pass(this) :: forward => forward_batchnorm3d
     !! Forward propagation derived type handler

     final :: finalise_batchnorm3d
     !! Finalise 3D batch normalisation layer
  end type batchnorm3d_layer_type

  interface batchnorm3d_layer_type
     !! Interface for setting up the 3D batch normalisation layer
     module function layer_setup( &
          input_shape, batch_size, &
          momentum, epsilon, &
          gamma_init_mean, gamma_init_std, &
          beta_init_mean, beta_init_std, &
          kernel_initialiser, bias_initialiser, &
          moving_mean_initialiser, moving_variance_initialiser, &
          verbose &
     ) result(layer)
       !! Set up the 3D batch normalisation layer
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
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
       type(batchnorm3d_layer_type) :: layer
       !! Instance of the 3D batch normalisation layer
     end function layer_setup
  end interface batchnorm3d_layer_type



contains

!###############################################################################
  subroutine finalise_batchnorm3d(this)
    !! Finalise 3D batch normalisation layer
    implicit none

    ! Arguments
    type(batchnorm3d_layer_type), intent(inout) :: this
    !! Instance of the 3D batch normalisation layer

    if(allocated(this%mean)) deallocate(this%mean)
    if(allocated(this%variance)) deallocate(this%variance)
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)

  end subroutine finalise_batchnorm3d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       input_shape, batch_size, &
       momentum, epsilon, &
       gamma_init_mean, gamma_init_std, &
       beta_init_mean, beta_init_std, &
       kernel_initialiser, bias_initialiser, &
       moving_mean_initialiser, moving_variance_initialiser, &
       verbose &
  ) result(layer)
    !! Set up the 3D batch normalisation layer
    use athena__initialiser, only: get_default_initialiser
    implicit none

    ! Arguments
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
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

    type(batchnorm3d_layer_type) :: layer
    !! Instance of the 3D batch normalisation layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

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
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_batchnorm3d( &
       this, &
       momentum, epsilon, &
       gamma_init_mean, gamma_init_std, &
       beta_init_mean, beta_init_std, &
       kernel_initialiser, bias_initialiser, &
       moving_mean_initialiser, moving_variance_initialiser, &
       verbose )
    !! Set hyperparameters for 3D batch normalisation layer
    implicit none

    ! Arguments
    class(batchnorm3d_layer_type), intent(inout) :: this
    !! Instance of the 3D batch normalisation layer
    real(real32), intent(in) :: momentum, epsilon
    !! Momentum and epsilon
    real(real32), intent(in) :: gamma_init_mean, gamma_init_std
    !! Gamma initialisation mean and standard deviation
    real(real32), intent(in) :: beta_init_mean, beta_init_std
    !! Beta initialisation mean and standard deviation
    character(*), intent(in) :: kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    character(*), intent(in) :: &
         moving_mean_initialiser, moving_variance_initialiser
    !! Moving mean and variance initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "batchnorm3d"
    this%type = "batc"
    this%input_rank = 4
    this%output_rank = 4
    this%has_bias = .true.
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
          write(*,'("BATCHNORM3D kernel (gamma) initialiser: ",A)') &
               trim(this%kernel_initialiser)
          write(*,'("BATCHNORM3D bias (beta) initialiser: ",A)') &
               trim(this%bias_initialiser)
          write(*,'("BATCHNORM3D moving mean initialiser: ",A)') &
               trim(this%moving_mean_initialiser)
          write(*,'("BATCHNORM3D moving variance initialiser: ",A)') &
               trim(this%moving_variance_initialiser)
       end if
    end if

  end subroutine set_hyperparams_batchnorm3d
!###############################################################################


!###############################################################################
  subroutine set_batch_size_batchnorm3d(this, batch_size, verbose)
    !! Set batch size for 3D batch normalisation layer
    implicit none

    ! Arguments
    class(batchnorm3d_layer_type), intent(inout), target :: this
    !! Instance of the 3D batch normalisation layer
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
    ! Set the normalisation factor
    !---------------------------------------------------------------------------
    this%norm = real( &
         this%batch_size * &
         product(this%input_shape(1:this%input_rank-1) ),real32)

    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(this%use_graph_input)then
          call stop_program( &
               "Graph input not supported for 3D batch normalisation layer" &
          )
          return
       end if
       if(allocated(this%output)) deallocate(this%output)
       allocate( batchnorm_array_type :: this%output(1,1) )
       call this%output(1,1)%allocate( &
            array_shape = [ &
                 this%output_shape(1), &
                 this%output_shape(2), &
                 this%output_shape(3), this%num_channels, &
                 this%batch_size ], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_batchnorm3d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_batchnorm3d(this, unit)
    !! Print 3D batch normalisation layer to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(batchnorm3d_layer_type), intent(in) :: this
    !! Instance of the 3D batch normalisation layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    integer :: m
    !! Loop index


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"INPUT_SHAPE = ",4(1X,I0))') this%input_shape
    write(unit,'(3X,"MOMENTUM = ",F0.9)') this%momentum
    write(unit,'(3X,"EPSILON = ",F0.9)') this%epsilon
    write(unit,'(3X,"NUM_CHANNELS = ",I0)') this%num_channels
    write(unit,'("GAMMA")')
    do m=1,this%num_channels
       write(unit,'(5(E16.8E2))') this%params_array(1)%val(m,1)
    end do
    write(unit,'("END GAMMA")')
    write(unit,'("BETA")')
    do m=1,this%num_channels
       write(unit,'(5(E16.8E2))') this%params_array(1)%val(this%num_channels+m,1)
    end do
    write(unit,'("END BETA")')

  end subroutine print_to_unit_batchnorm3d
!###############################################################################


!###############################################################################
  subroutine read_batchnorm3d(this, unit, verbose)
    !! Read 3D batch normalisation layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(batchnorm3d_layer_type), intent(inout) :: this
    !! Instance of the 3D batch normalisation layer
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat, verbose_ = 0
    !! Status and verbosity level
    integer :: i, j, k, l, c, itmp1, iline, final_line
    !! Loop variables and temporary integer
    integer :: num_channels
    !! Number of channels
    real(real32) :: momentum = 0._real32, epsilon = 1.E-5_real32
    !! Momentum and epsilon
    character(14) :: kernel_initialiser='', bias_initialiser=''
    !! Kernel and bias initialisers
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    integer, dimension(4) :: input_shape
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
       case("NUM_CHANNELS")
          call assign_val(buffer, num_channels, itmp1)
          write(0,*) "NUM_CHANNELS and INPUT_SHAPE are conflicting parameters"
          write(0,*) "NUM_CHANNELS will be ignored"
       case("KERNEL_INITIALISER")
          call assign_val(buffer, kernel_initialiser, itmp1)
       case("BIAS_INITIALISER")
          call assign_val(buffer, bias_initialiser, itmp1)
       case("GAMMA")
          kernel_initialiser = 'zeros'
          param_lines(1) = iline
       case("BETA")
          bias_initialiser   = 'zeros'
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


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    num_channels = input_shape(size(input_shape,1))
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
          this%params_array(1)%val(1:this%num_channels,1) = data_list
          if(trim(adjustl(buffer)).ne."END GAMMA")then
             write(err_msg,'("END GAMMA not where expected: ",A)') &
                  trim(adjustl(buffer))
             call stop_program(err_msg)
             return
          end if
       case(2) ! beta
          this%params_array(1)%val(this%num_channels+1:this%num_channels*2,1) = &
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

  end subroutine read_batchnorm3d
!###############################################################################


!###############################################################################
  function read_batchnorm3d_layer(unit, verbose) result(layer)
    !! Read 3D batch normalisation layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    class(base_layer_type), allocatable :: layer
    !! Instance of the 3D batch normalisation layer
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=batchnorm3d_layer_type())
    call layer%read(unit, verbose=verbose_)

  end function read_batchnorm3d_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_batchnorm3d(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(batchnorm3d_layer_type), intent(inout) :: this
    !! Instance of the 3D batch normalisation layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    real(real32), dimension(this%num_channels) :: arr1
    class(batchnorm_array_type), pointer :: ptr


    select case(this%inference)
    case(.true.)
       ! Do not perform the drop operation

       ptr => batchnorm_inference(input(1,1), this%params_array(1), &
            this%norm, &
            this%mean(:), this%variance(:), this%epsilon &
       )

    case default
       ! Perform the drop operation
       ptr => batchnorm( &
            input(1,1), this%params_array(1),&
            this%norm, this%momentum, &
            this%mean(:), this%variance(:), this%epsilon &
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

  end subroutine forward_batchnorm3d
!###############################################################################

end module athena__batchnorm3d_layer
