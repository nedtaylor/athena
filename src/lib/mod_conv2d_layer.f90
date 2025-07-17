module athena__conv2d_layer
  !! Module containing implementation of a 2D convolutional layer
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__base_layer, only: conv_layer_type, base_layer_type
  use athena__pad2d_layer, only: pad2d_layer_type
  use athena__misc_types, only: initialiser_type, array4d_type
  implicit none


  private

  public :: conv2d_layer_type
  public :: read_conv2d_layer


  type, extends(conv_layer_type) :: conv2d_layer_type
     !! Type for 2D convolutional layer with overloaded procedures
     real(real32), pointer :: weight(:,:,:,:) => null()
     !! Weights of the convolutional layer
     real(real32), pointer :: dw(:,:,:,:,:) => null()
     !! Pointer to weight gradients
     real(real32), allocatable, dimension(:,:,:,:) :: z
     !! Activation values
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_conv2d
     !! Set hyperparameters for 2D convolutional layer
     procedure, pass(this), private :: &
          set_ptrs_hyperparams => set_ptrs_hyperparams_conv2d
     !! Set pointers to hyperparameters
     procedure, pass(this) :: set_batch_size => set_batch_size_conv2d
     !! Set batch size for 2D convolutional layer
     procedure, pass(this) :: print_to_unit => print_to_unit_conv2d
     !! Print 2D convolutional layer to unit
     procedure, pass(this) :: read => read_conv2d
     !! Read 2D convolutional layer from file
     procedure, pass(this) :: forward  => forward_rank
     !! Forward propagation handler for 2D convolutional layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward propagation handler for 2D convolutional layer
     procedure, private, pass(this) :: forward_4d
     !! Forward propagation for 4D input
     procedure, private, pass(this) :: backward_4d
     !! Backward propagation for 4D input
     final :: finalise_conv2d
     !! Finalise 2D convolutional layer
  end type conv2d_layer_type

  interface conv2d_layer_type
     !! Interface for setting up the 2D convolutional layer
     module function layer_setup( &
          input_shape, batch_size, &
          num_filters, kernel_size, stride, padding, &
          activation_function, activation_scale, &
          kernel_initialiser, bias_initialiser, &
          calc_input_gradients, &
          verbose ) result(layer)
       !! Set up the 2D convolutional layer
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: num_filters
       !! Number of filters
       integer, dimension(..), optional, intent(in) :: kernel_size
       !! Kernel size
       integer, dimension(..), optional, intent(in) :: stride
       !! Stride
       real(real32), optional, intent(in) :: activation_scale
       !! Activation scale
       character(*), optional, intent(in) :: activation_function, &
            kernel_initialiser, bias_initialiser, padding
       !! Activation function, kernel initialiser, bias initialiser, padding
       logical, optional, intent(in) :: calc_input_gradients
       !! Calculate input gradients
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(conv2d_layer_type) :: layer
       !! Instance of the 2D convolutional layer
     end function layer_setup
  end interface conv2d_layer_type



contains

!###############################################################################
!!! finalise layer
!###############################################################################
  subroutine finalise_conv2d(this)
    !! Finalise 2D convolutional layer
    implicit none

    ! Arguments
    type(conv2d_layer_type), intent(inout) :: this
    !! Instance of the 2D convolutional layer

    if(allocated(this%knl)) deallocate(this%knl)
    if(allocated(this%stp)) deallocate(this%stp)
    if(allocated(this%hlf)) deallocate(this%hlf)
    if(allocated(this%pad)) deallocate(this%pad)
    if(allocated(this%cen)) deallocate(this%cen)

    if(associated(this%bias)) nullify(this%bias)
    if(associated(this%weight)) nullify(this%weight)
    if(associated(this%dw)) nullify(this%dw)
    if(allocated(this%z)) deallocate(this%z)
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(allocated(this%di)) deallocate(this%di)
    if(allocated(this%di_padded)) deallocate(this%di_padded)

    if(allocated(this%pad_layer)) deallocate(this%pad_layer)

  end subroutine finalise_conv2d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
!!! forward propagation assumed rank handler
!###############################################################################
  subroutine forward_rank(this, input)
    !! Forward propagation handler for 2D convolutional layer
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout) :: this
    !! Instance of the 2D convolutional layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values

    select case(allocated(this%pad_layer))
    case(.true.)
       call this%pad_layer%forward(input)
       call forward_4d(this, this%pad_layer%output(1,1)%val)
    case default
       select rank(input)
       rank(2)
          call forward_4d(this, input)
       rank(4)
          call forward_4d(this, input)
       end select
    end select
  end subroutine forward_rank
!###############################################################################


!###############################################################################
!!! backward propagation assumed rank handler
!###############################################################################
  subroutine backward_rank(this, input, gradient)
    !! Backward propagation handler for 2D convolutional layer
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout) :: this
    !! Instance of the 2D convolutional layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient values

    select case(allocated(this%pad_layer))
    case(.true.)
       select rank(input)
       rank(2)
          select rank(gradient)
          rank(2)
             call backward_4d( &
                  this, this%pad_layer%output(1,1)%val, gradient, &
                  this%di_padded%val &
             )
          end select
          call this%pad_layer%backward(input, this%di_padded%val)
       rank(4)
          select rank(gradient)
          rank(1)
             call backward_4d( &
                  this, this%pad_layer%output(1,1)%val, gradient, &
                  this%di_padded%val &
             )
          rank(2)
             call backward_4d( &
                  this, this%pad_layer%output(1,1)%val, gradient, &
                  this%di_padded%val &
             )
          rank(4)
             call backward_4d( &
                  this, this%pad_layer%output(1,1)%val, gradient, &
                  this%di_padded%val &
             )
          end select
          call this%pad_layer%backward(input, this%di_padded%val)
       end select
       this%di(1,1)%val = this%di_padded%val
    case default
       select rank(input)
       rank(2)
          select rank(gradient)
          rank(2)
             call backward_4d(this, input, gradient, this%di(1,1)%val)
          end select
       rank(4)
          select rank(gradient)
          rank(1)
             call backward_4d(this, input, gradient, this%di(1,1)%val)
          rank(2)
             call backward_4d(this, input, gradient, this%di(1,1)%val)
          rank(4)
             call backward_4d(this, input, gradient, this%di(1,1)%val)
          end select
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
       num_filters, kernel_size, stride, padding, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser, &
       calc_input_gradients, &
       verbose ) result(layer)
    !! Set up the 2D convolutional layer
    implicit none

    ! Arguments
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: num_filters
    !! Number of filters
    integer, dimension(..), optional, intent(in) :: kernel_size
    !! Kernel size
    integer, dimension(..), optional, intent(in) :: stride
    !! Stride
    real(real32), optional, intent(in) :: activation_scale
    !! Activation scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser, bias_initialiser, padding
    !! Activation function, kernel initialiser, bias initialiser, padding
    logical, optional, intent(in) :: calc_input_gradients
    !! Calculate input gradients
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(conv2d_layer_type) :: layer
    !! Instance of the 2D convolutional layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    real(real32) :: scale
    !! Activation scale
    character(len=10) :: activation_function_
    !! Activation function
    character(len=20) :: padding_
    !! Padding
    integer, dimension(2) :: kernel_size_, stride_
    !! Kernel size and stride

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Determine whether to calculate input gradients
    !---------------------------------------------------------------------------
    if(present(calc_input_gradients))then
       layer%calc_input_gradients = calc_input_gradients
       write(*,*) "CONV2D input gradients turned off"
    else
       layer%calc_input_gradients = .true.
    end if


    !---------------------------------------------------------------------------
    ! Set up number of filters
    !---------------------------------------------------------------------------
    if(present(num_filters))then
       layer%num_filters = num_filters
    else
       layer%num_filters = 32
    end if


    !---------------------------------------------------------------------------
    ! Set up kernel size
    !---------------------------------------------------------------------------
    if(present(kernel_size))then
       select rank(kernel_size)
       rank(0)
          kernel_size_ = kernel_size
       rank(1)
          kernel_size_(1) = kernel_size(1)
          if(size(kernel_size,dim=1).eq.1)then
             kernel_size_(2) = kernel_size(1)
          elseif(size(kernel_size,dim=1).eq.2)then
             kernel_size_(2) = kernel_size(2)
          end if
       end select
    else
       kernel_size_ = 3
    end if


    !---------------------------------------------------------------------------
    ! Set up padding name
    !---------------------------------------------------------------------------
    if(present(padding))then
       padding_ = padding
    else
       padding_ = "valid"
    end if


    !---------------------------------------------------------------------------
    ! Set up stride
    !---------------------------------------------------------------------------
    if(present(stride))then
       select rank(stride)
       rank(0)
          stride_ = stride
       rank(1)
          stride_(1) = stride(1)
          if(size(stride,dim=1).eq.1)then
             stride_(2) = stride(1)
          elseif(size(stride,dim=1).eq.2)then
             stride_(2) = stride(2)
          end if
       end select
    else
       stride_ = 1
    end if


    !---------------------------------------------------------------------------
    ! Set activation and derivative functions based on input name
    !---------------------------------------------------------------------------
    if(present(activation_function))then
       activation_function_ = activation_function
    else
       activation_function_ = "none"
    end if
    if(present(activation_scale))then
       scale = activation_scale
    else
       scale = 1._real32
    end if


    !---------------------------------------------------------------------------
    ! Define weights (kernels) and biases initialisers
    !---------------------------------------------------------------------------
    if(present(kernel_initialiser)) layer%kernel_initialiser =kernel_initialiser
    if(present(bias_initialiser)) layer%bias_initialiser = bias_initialiser


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         num_filters = layer%num_filters, &
         kernel_size = kernel_size_, stride = stride_, &
         padding = padding_, &
         activation_function = activation_function_, &
         activation_scale = scale, &
         kernel_initialiser = layer%kernel_initialiser, &
         bias_initialiser = layer%bias_initialiser, &
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
  subroutine set_hyperparams_conv2d( &
       this, &
       num_filters, &
       kernel_size, stride, &
       padding, &
       activation_function, &
       activation_scale, &
       kernel_initialiser, &
       bias_initialiser, &
       verbose &
  )
    !! Set hyperparameters for 2D convolutional layer
    use athena__activation,  only: activation_setup
    use athena__initialiser, only: get_default_initialiser
    use athena__misc, only: to_lower
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout) :: this
    !! Instance of the 2D convolutional layer
    integer, intent(in) :: num_filters
    !! Number of filters
    integer, dimension(2), intent(in) :: kernel_size, stride
    !! Kernel size and stride
    character(*), intent(in) :: padding
    !! Padding
    character(*), intent(in) :: activation_function
    !! Activation function
    real(real32), intent(in) :: activation_scale
    !! Activation scale
    character(*), intent(in) :: kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    character(len=20) :: padding_

    this%name = "conv2d"
    this%type = "conv"
    this%input_rank = 3
    this%output_rank = 3
    this%has_bias = .true.
    allocate( &
         this%knl(this%input_rank-1), &
         this%stp(this%input_rank-1), &
         this%hlf(this%input_rank-1), &
         this%pad(this%input_rank-1), &
         this%cen(this%input_rank-1) &
    )
    this%knl = kernel_size
    this%stp = stride
    this%cen = 2 - mod(this%knl, 2)
    this%hlf   = (this%knl-1)/2
    padding_ = trim(adjustl(padding))
    select case(trim(adjustl(to_lower(padding_))))
    case("valid", "none", "")
       this%pad = 0
    case default
       this%pad_layer = pad2d_layer_type( &
            padding = [ this%hlf ], &
            method = padding_ &
       )
       this%pad = this%hlf
    end select
    allocate(this%transfer, &
         source=activation_setup(activation_function, activation_scale) &
    )

    if(trim(this%kernel_initialiser).eq.'') &
         this%kernel_initialiser=get_default_initialiser(activation_function)
    if(trim(this%bias_initialiser).eq.'') &
         this%bias_initialiser = get_default_initialiser(&
              activation_function, is_bias=.true.)

    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("CONV2D activation function: ",A)') &
               trim(activation_function)
          write(*,'("CONV2D kernel initialiser: ",A)') &
               trim(this%kernel_initialiser)
          write(*,'("CONV2D bias initialiser: ",A)') &
               trim(this%bias_initialiser)
       end if
    end if

  end subroutine set_hyperparams_conv2d
!###############################################################################


!###############################################################################
  subroutine set_ptrs_hyperparams_conv2d(this)
    !! Set pointers to hyperparameters for 2D convolutional layer
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout), target :: this
    !! Instance of the 2D convolutional layer

    if(allocated(this%params))then
       this%weight( &
            1:this%knl(1), &
            1:this%knl(2), &
            1:this%num_channels, &
            1:this%num_filters &
       ) => this%params(1:this%num_params-this%num_filters)
       this%bias(1:this%num_filters) => &
            this%params(this%num_params-this%num_filters+1:)
    end if
    if(allocated(this%dp))then
       this%dw( &
            1:this%knl(1), &
            1:this%knl(2), &
            1:this%num_channels, &
            1:this%num_filters, &
            1:this%batch_size &
       ) => this%dp(:,:)
    end if

  end subroutine set_ptrs_hyperparams_conv2d
!###############################################################################


!###############################################################################
  subroutine set_batch_size_conv2d(this, batch_size, verbose)
    !! Set batch size for 2D convolutional layer
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout), target :: this
    !! Instance of the 2D convolutional layer
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
    ! Set batch size of padding layer, if allocated
    !---------------------------------------------------------------------------
    if(allocated(this%pad_layer)) &
         call this%pad_layer%set_batch_size(this%batch_size, verbose=verbose_)


    !---------------------------------------------------------------------------
    ! Set weights and biases pointers to params array
    !---------------------------------------------------------------------------
    this%weight( &
         1:this%knl(1), &
         1:this%knl(2), &
         1:this%num_channels, &
         1:this%num_filters &
    ) => this%params(1:this%num_params-this%num_filters)
    this%bias(1:this%num_filters) => &
         this%params(this%num_params-this%num_filters+1:)


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(this%use_graph_input)then
          call stop_program( &
               "Graph input not supported for 2D convolutional layer" &
          )
          return
       end if
       if(allocated(this%output)) deallocate(this%output)
       allocate( this%output(1,1), source = array4d_type() )
       call this%output(1,1)%allocate( &
            array_shape = [ &
                 this%output_shape(1), &
                 this%output_shape(2), &
                 this%num_filters, &
                 this%batch_size ], &
            source=0._real32 &
       )
       if(allocated(this%z)) deallocate(this%z)
       select type(output => this%output(1,1))
       type is (array4d_type)
          allocate(this%z, source=output%val_ptr)
       end select
       if(allocated(this%di)) deallocate(this%di)
       allocate( this%di(1,1), source = array4d_type() )
       call this%di(1,1)%allocate( &
            array_shape = [ &
                 this%input_shape(1), &
                 this%input_shape(2), &
                 this%input_shape(3), &
                 this%batch_size ], &
            source=0._real32 &
       )

       if(allocated(this%pad_layer))then
          if(.not.allocated(this%di_padded)) this%di_padded = array4d_type()
          if(this%di_padded%allocated) call this%di_padded%deallocate()
          call this%di_padded%allocate( &
               array_shape = [ &
                    this%input_shape(1) + 2 * this%pad(1), &
                    this%input_shape(2) + 2 * this%pad(2), &
                    this%input_shape(3), &
                    this%batch_size ], &
               source=0._real32 &
          )
       end if

       if(allocated(this%dp)) deallocate(this%dp)
       allocate( &
            this%dp( this%num_params - this%num_filters, this%batch_size), &
            source=0._real32 &
       )
       this%dw( &
            1:this%knl(1), &
            1:this%knl(2), &
            1:this%num_channels, &
            1:this%num_filters, &
            1:this%batch_size &
       ) => this%dp(:,:)
       if(allocated(this%db)) deallocate(this%db)
       allocate(this%db(this%num_filters, this%batch_size), source=0._real32)
    end if

  end subroutine set_batch_size_conv2d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_conv2d(this, unit)
    !! Print 2D convolutional layer to unit
    use athena__misc, only: to_upper
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(in) :: this
    !! Instance of the 2D convolutional layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    integer :: l, i, itmp1, idx
    !! Loop indices
    character(:), allocatable :: padding_type
    !! Padding type


    ! Handle different width kernels for x, y, z
    !---------------------------------------------------------------------------
    itmp1 = -1
    do i=1,2
       if(this%pad(i).gt.itmp1)then
          itmp1 = this%pad(i)
          idx = i
       end if
    end do


    ! Determine padding method
    !---------------------------------------------------------------------------
    padding_type = ""
    if(this%pad(idx).eq.this%knl(idx)-1)then
       padding_type = "full"
    elseif(this%pad(idx).eq.0)then
       padding_type = "valid"
    else
       padding_type = "same"
    end if


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    write(unit,'(3X,"NUM_FILTERS = ",I0)') this%num_filters
    if(all(this%knl.eq.this%knl(1)))then
       write(unit,'(3X,"KERNEL_SIZE =",1X,I0)') this%knl(1)
    else
       write(unit,'(3X,"KERNEL_SIZE =",2(1X,I0))') this%knl
    end if
    if(all(this%stp.eq.this%stp(1)))then
       write(unit,'(3X,"STRIDE =",1X,I0)') this%stp(1)
    else
       write(unit,'(3X,"STRIDE =",2(1X,I0))') this%stp
    end if
    write(unit,'(3X,"PADDING = ",A)') padding_type

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') this%transfer%scale


    ! Write weights and biases
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    do l=1,this%num_filters
       write(unit,'(5(E16.8E2))', advance="no") this%weight(:,:,:,l)
       if(mod(size(this%weight(:,:,:,l)),5).eq.0) write(unit,*)
       write(unit,'(E16.8E2)') this%bias(l)
    end do

  end subroutine print_to_unit_conv2d
!###############################################################################


!###############################################################################
  subroutine read_conv2d(this, unit, verbose)
    !! Read 2D convolutional layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout) :: this
    !! Instance of the 2D convolutional layer
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat
    !! Status of read
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: j, k, l, c, itmp1
    !! Loop indices
    integer :: num_filters, num_inputs
    !! Number of filters and inputs
    real(real32) :: activation_scale
    !! Activation scale
    logical :: found_weights = .false.
    !! Boolean whether weights card was found
    character(14) :: kernel_initialiser='', bias_initialiser=''
    !! Kernel and bias initialisers
    character(20) :: padding, activation_function
    !! Padding and activation function
    character(256) :: buffer, tag, err_msg
    !! Buffer for reading lines
    integer, dimension(2) :: kernel_size, stride
    !! Kernel size and stride
    integer, dimension(3) :: input_shape
    !! Input shape
    real(real32), allocatable, dimension(:) :: data_list
    !! List of data values


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
       case("NUM_FILTERS")
          call assign_val(buffer, num_filters, itmp1)
       case("KERNEL_SIZE")
          call assign_vec(buffer, kernel_size, itmp1)
       case("STRIDE")
          call assign_vec(buffer, stride, itmp1)
       case("PADDING")
          call assign_val(buffer, padding, itmp1)
          padding = to_lower(padding)
       case("ACTIVATION")
          call assign_val(buffer, activation_function, itmp1)
       case("ACTIVATION_SCALE")
          call assign_val(buffer, activation_scale, itmp1)
       case("KERNEL_INITIALISER")
          call assign_val(buffer, kernel_initialiser, itmp1)
       case("BIAS_INITIALISER")
          call assign_val(buffer, bias_initialiser, itmp1)
       case("WEIGHTS")
          found_weights = .true.
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
    call this%set_hyperparams( &
         num_filters = num_filters, &
         kernel_size = kernel_size, stride = stride, &
         padding = padding, &
         activation_function = activation_function, &
         activation_scale = activation_scale, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape = input_shape)


    ! Check if WEIGHTS card was found
    !---------------------------------------------------------------------------
    if(.not.found_weights)then
       write(0,*) "WARNING: WEIGHTS card in "//to_upper(trim(this%name))//" not found"
    else
       do l=1,num_filters
          num_inputs = product(this%knl) + 1 !+1 for bias
          allocate(data_list(num_inputs), source=0._real32)
          c = 1
          k = 1
          data_concat_loop: do while(c.le.num_inputs)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop
          this%weight(:,:,:,l) = &
               reshape(&
                    data_list(1:num_inputs-1),&
                    shape(this%weight(:,:,:,l)))
          this%bias(l) = data_list(num_inputs)
          deallocate(data_list)
       end do

       ! Check for end of weights card
       !------------------------------------------------------------------------
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          write(0,*) trim(adjustl(buffer))
          call stop_program("END WEIGHTS not where expected")
          return
       end if
    end if


    ! Check for end of layer card
    !---------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_conv2d
!###############################################################################


!###############################################################################
  function read_conv2d_layer(unit, verbose) result(layer)
    !! Read 2D convolutional layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 2D convolutional layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=conv2d_layer_type())
    call layer%read(unit, verbose=verbose_)

  end function read_conv2d_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_4d(this, input)
    !! Forward propagation for 4D input
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout) :: this
    !! Instance of the 2D convolutional layer
    real(real32), &
         dimension( &
              1:this%input_shape(1) + 2 * this%pad(1), &
              1:this%input_shape(2) + 2 * this%pad(2), &
              this%num_channels,this%batch_size), &
         intent(in) :: input
    !! Input values

    ! Local variables
    integer :: i, j, l, s
    !! Loop indices
    integer, dimension(2) :: start_idx, end_idx
    !! Start and end indices for convolution


    ! Perform the convolution operation
    !---------------------------------------------------------------------------
    do concurrent( &
         i=1:this%output_shape(1):1, &
         j=1:this%output_shape(2):1)
#if defined(GFORTRAN)
       start_idx = ([i,j]-1)*this%stp + 1
#else
       start_idx(1) = (i-1)*this%stp(1) + 1
       start_idx(2) = (j-1)*this%stp(2) + 1
#endif
       end_idx   = start_idx + this%knl - 1

       do concurrent(s=1:this%batch_size)
          this%z(i,j,:,s) = this%bias(:)
       end do

       do concurrent(l=1:this%num_filters, s=1:this%batch_size)
          this%z(i,j,l,s) = this%z(i,j,l,s) + &
               sum( &
                    input( &
                         start_idx(1):end_idx(1),&
                         start_idx(2):end_idx(2),:,s &
                    ) * this%weight(:,:,:,l) &
               )
       end do
    end do


    ! Apply activation function to activation values (z)
    !---------------------------------------------------------------------------
    select type(output => this%output(1,1))
    type is (array4d_type)
       output%val_ptr = this%transfer%activate(this%z)
    end select

  end subroutine forward_4d
!###############################################################################


!###############################################################################
  subroutine backward_4d(this, input, gradient, di)
    !! Backward propagation for 4D input
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout) :: this
    !! Instance of the 2D convolutional layer
    real(real32), &
         dimension( &
              1:this%input_shape(1) + 2 * this%pad(1), &
              1:this%input_shape(2) + 2 * this%pad(2), &
              this%num_channels,this%batch_size), &
         intent(in) :: input
    !! Input values
    real(real32), &
         dimension( &
              this%output_shape(1), &
              this%output_shape(2), &
              this%num_filters,this%batch_size), &
         intent(in) :: gradient
    !! Gradient values
    real(real32), &
         dimension( &
              1:this%input_shape(1) + 2 * this%pad(1), &
              1:this%input_shape(2) + 2 * this%pad(2), &
              this%num_channels,this%batch_size), &
         intent(inout) :: di
    !! Input gradients

    ! Local variables
    integer :: l, m, i, j, x, y, s
    !! Loop indices
    integer, dimension(2) :: offset, n_stp
    !! Offset and number of steps
    integer, dimension(2,2) :: lim, lim_w, lim_g
    !! Limits for weights and gradients
    real(real32), &
         dimension( &
              this%output_shape(1),&
              this%output_shape(2),this%num_filters, &
              this%batch_size) :: grad_dz
    !! Gradient multiplied by differential of Z (aka delta values)

    ! Local variables
    real(real32), dimension(1) :: bias_diff
    !! Differential of bias


    bias_diff = this%transfer%differentiate([1._real32])


    ! Get gradient multiplied by differential of Z
    !---------------------------------------------------------------------------
    grad_dz = gradient * &
         this%transfer%differentiate(this%z)
    do concurrent( &
         l=1:this%num_filters, s=1:this%batch_size)
       this%db(l,s) = this%db(l,s) + sum(grad_dz(:,:,l,s)) * bias_diff(1)
    end do


    ! Apply convolution to compute weight gradients
    ! Offset applied as centre of kernel is 0 ...
    ! ... whilst the starting index for input is 1
    !---------------------------------------------------------------------------
    do concurrent( &
         s = 1 : this%batch_size, &
         l = 1 : this%num_filters, &
         m = 1 : this%num_channels &
    )
       do y = 1, this%knl(2), 1
          do j = 1, this%output_shape(2)
             do x = 1, this%knl(1), 1
                do i = 1, this%output_shape(1)
                   this%dw(x,y,m,l,s) = this%dw(x,y,m,l,s) + &
                        grad_dz(i,j,l,s) * &
                        input( &
                             x + ( i - 1 ) * this%stp(1), &
                             y + ( j - 1 ) * this%stp(2), &
                             m, s &
                        )
                end do
             end do
          end do
       end do
    end do


    ! Apply strided convolution to obtain input gradients
    !---------------------------------------------------------------------------
    if(this%calc_input_gradients)then
       offset  = 1 + this%hlf + (this%cen - 1)
       lim(1,:) = this%knl + this%hlf
       lim(2,:) = (this%output_shape(:2) - 1) * this%stp + 1 + this%knl
       n_stp = this%output_shape(:2) * this%stp
       di = 0._real32
       ! All elements of the output are separated by stride_x, stride_y
       do concurrent( &
            s = 1 : this%batch_size, &
            l = 1 : this%num_filters, &
            m = 1 : this%num_channels, &
            i = 1 : size(di,dim=1) : 1, &
            j = 1 : size(di,dim=2) : 1 &
       )

          ! Set weight bounds (o/p = output)
          ! max( ...
          ! ... 1. offset of 1st o/p idx from centre of knl     (lim)
          ! ... 2. lwst o/p idx overlap with <<- knl idx (rpt. pattern)
          ! ...)
          lim_w(2,:) = max( &
               lim(1,:)-[i,j], &
               1 + mod(n_stp+this%knl-[i,j],this%stp) &
          )
          ! min( ...
          ! ... 1. offset of last o/p idx from centre of knl    (lim)
          ! ... 2. hghst o/p idx overlap with ->> knl idx (rpt. pattern)
          ! ...)
          lim_w(1,:) = min( &
               lim(2,:)-[i,j], &
               this%knl - mod(n_stp-1+[i,j],this%stp) &
          )
          if(any(lim_w(2,:).gt.lim_w(1,:))) cycle

          ! Set gradient bounds
          lim_g(1,:) = max(1, [i,j] - offset)
          lim_g(2,:) = min( &
               this%output_shape(:2), &
               [i,j] - offset + this%knl - 1 &
          )

          ! Apply full convolution to compute input gradients
          di(i,j,m,s) = di(i,j,m,s) + &
               sum( &
                    grad_dz( &
                         lim_g(1,1):lim_g(2,1), &
                         lim_g(1,2):lim_g(2,2), &
                         l, s &
                    ) * this%weight( &
                         lim_w(1,1):lim_w(2,1):-this%stp(1), &
                         lim_w(1,2):lim_w(2,2):-this%stp(2), &
                         m, l &
                    ) &
               )
       end do
    end if

  end subroutine backward_4d
!###############################################################################

end module athena__conv2d_layer
