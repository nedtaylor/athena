module athena__conv3d_layer
  !! Module containing implementation of a 3D convolutional layer
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: conv_layer_type, base_layer_type
  use athena__pad3d_layer, only: pad3d_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type
  use diffstruc, only: array_type
  use athena__diffstruc_extd, only: conv3d, add_bias
  implicit none


  private

  public :: conv3d_layer_type
  public :: read_conv3d_layer


  type, extends(conv_layer_type) :: conv3d_layer_type
     !! Type for 3D convolutional layer with overloaded procedures
     type(array_type), dimension(2) :: z
     !! Temporary arrays for forward propagation
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_conv3d
     !! Set hyperparameters for 3D convolutional layer
     procedure, pass(this) :: set_batch_size => set_batch_size_conv3d
     !! Set batch size for 3D convolutional layer
     procedure, pass(this) :: print_to_unit => print_to_unit_conv3d
     !! Print 3D convolutional layer to unit
     procedure, pass(this) :: read => read_conv3d
     !! Read 3D convolutional layer from file

     procedure, pass(this) :: forward => forward_conv3d
     !! Forward propagation derived type handler

     final :: finalise_conv3d
     !! Finalise 3D convolutional layer
  end type conv3d_layer_type

  interface conv3d_layer_type
     !! Interface for setting up the 3D convolutional layer
     module function layer_setup( &
          input_shape, batch_size, &
          num_filters, kernel_size, stride, dilation, padding, &
          use_bias, &
          activation, &
          kernel_initialiser, bias_initialiser, &
          verbose ) result(layer)
       !! Set up the 3D convolutional layer
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
       integer, dimension(..), optional, intent(in) :: dilation
       !! Dilation
       logical, optional, intent(in) :: use_bias
       !! Use bias
       class(*), optional, intent(in) :: activation, &
            kernel_initialiser, bias_initialiser
       !! Activation function, kernel initialiser, bias initialiser
       character(*), optional, intent(in) :: padding
       !! Padding method
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(conv3d_layer_type) :: layer
       !! Instance of the 3D convolutional layer
     end function layer_setup
  end interface conv3d_layer_type



contains

!###############################################################################
  subroutine finalise_conv3d(this)
    !! Finalise 3D convolutional layer
    implicit none

    ! Arguments
    type(conv3d_layer_type), intent(inout) :: this
    !! Instance of the 3D convolutional layer

    if(allocated(this%dil)) deallocate(this%dil)
    if(allocated(this%knl)) deallocate(this%knl)
    if(allocated(this%stp)) deallocate(this%stp)
    if(allocated(this%hlf)) deallocate(this%hlf)
    if(allocated(this%pad)) deallocate(this%pad)
    if(allocated(this%cen)) deallocate(this%cen)

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(allocated(this%pad_layer)) deallocate(this%pad_layer)

  end subroutine finalise_conv3d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       input_shape, batch_size, &
       num_filters, kernel_size, stride, dilation, padding, &
       use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, &
       verbose ) result(layer)
    !! Set up the 3D convolutional layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
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
    integer, dimension(..), optional, intent(in) :: dilation
    !! Dilation
    logical, optional, intent(in) :: use_bias
    !! Use bias
    class(*), optional, intent(in) :: activation
    !! Activation function
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    !! Activation function, kernel initialiser, and bias initialiser
    character(*), optional, intent(in) :: padding
    !! Padding method
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(conv3d_layer_type) :: layer
    !! Instance of the 3D convolutional layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: num_filters_
    !! Number of filters
    logical :: use_bias_ = .true.
    !! Use bias
    character(len=20) :: padding_
    !! Padding
    integer, dimension(3) :: kernel_size_, stride_, dilation_
    !! Kernel size and stride
    class(base_actv_type), allocatable :: activation_
    !! Activation function
    class(base_init_type), allocatable :: kernel_initialiser_, bias_initialiser_
    !! Kernel and bias initialisers

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set use_bias
    !---------------------------------------------------------------------------
    if(present(use_bias)) use_bias_ = use_bias


    !---------------------------------------------------------------------------
    ! Set activation and derivative functions based on input name
    !---------------------------------------------------------------------------
    if(present(activation))then
       activation_ = activation_setup(activation)
    else
       activation_ = activation_setup("none")
    end if


    !---------------------------------------------------------------------------
    ! Define weights (kernels) and biases initialisers
    !---------------------------------------------------------------------------
    if(present(kernel_initialiser))then
       kernel_initialiser_ = initialiser_setup(kernel_initialiser)
    end if
    if(present(bias_initialiser))then
       bias_initialiser_ = initialiser_setup(bias_initialiser)
    end if


    !---------------------------------------------------------------------------
    ! Set up number of filters
    !---------------------------------------------------------------------------
    if(present(num_filters))then
       num_filters_ = num_filters
    else
       num_filters_ = 32
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
             kernel_size_(2:) = kernel_size(1)
          elseif(size(kernel_size,dim=1).eq.3)then
             kernel_size_(2:) = kernel_size(2:)
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
             stride_(2:) = stride(1)
          elseif(size(stride,dim=1).eq.3)then
             stride_(2:) = stride(2:)
          end if
       end select
    else
       stride_ = 1
    end if


    !---------------------------------------------------------------------------
    ! Set up dilation
    !---------------------------------------------------------------------------
    if(present(dilation))then
       select rank(dilation)
       rank(0)
          dilation_ = dilation
       rank(1)
          dilation_(1) = dilation(1)
          if(size(dilation,dim=1).eq.1)then
             dilation_(2:) = dilation(1)
          elseif(size(dilation,dim=1).eq.3)then
             dilation_(2:) = dilation(2:)
          end if
       end select
    else
       dilation_ = 1
    end if


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         num_filters = num_filters_, &
         kernel_size = kernel_size_, stride = stride_, dilation = dilation_, &
         padding = padding_, &
         use_bias = use_bias_, &
         activation = activation_, &
         kernel_initialiser = kernel_initialiser_, &
         bias_initialiser = bias_initialiser_, &
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
  subroutine set_hyperparams_conv3d( &
       this, &
       num_filters, &
       kernel_size, stride, dilation, &
       padding, &
       use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    !! Set hyperparameters for 3D convolutional layer
    use athena__activation,  only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    use coreutils, only: to_lower
    implicit none

    ! Arguments
    class(conv3d_layer_type), intent(inout) :: this
    !! Instance of the 3D convolutional layer
    integer, intent(in) :: num_filters
    !! Number of filters
    integer, dimension(3), intent(in) :: kernel_size, stride, dilation
    !! Kernel size, stride, dilation
    character(*), intent(in) :: padding
    !! Padding
    logical, intent(in) :: use_bias
    !! Use bias
    class(base_actv_type), allocatable, intent(in) :: activation
    !! Activation function
    class(base_init_type), allocatable, intent(in) :: &
         kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    character(len=20) :: padding_
    character(len=256) :: buffer

    this%name = "conv3d"
    this%type = "conv"
    this%input_rank = 4
    this%output_rank = 4
    this%use_bias = use_bias
    if(allocated(this%dil)) deallocate(this%dil)
    if(allocated(this%knl)) deallocate(this%knl)
    if(allocated(this%stp)) deallocate(this%stp)
    if(allocated(this%hlf)) deallocate(this%hlf)
    if(allocated(this%pad)) deallocate(this%pad)
    if(allocated(this%cen)) deallocate(this%cen)
    allocate( &
         this%dil(this%input_rank-1), &
         this%knl(this%input_rank-1), &
         this%stp(this%input_rank-1), &
         this%hlf(this%input_rank-1), &
         this%pad(this%input_rank-1), &
         this%cen(this%input_rank-1) &
    )
    this%dil = dilation
    this%knl = kernel_size
    this%stp = stride
    this%cen = 2 - mod(this%knl, 2)
    this%hlf   = (this%knl-1)/2
    this%num_filters = num_filters
    padding_ = trim(adjustl(padding))

    select case(trim(adjustl(to_lower(padding_))))
    case("valid", "none", "")
       this%pad = 0
    case default
       this%pad_layer = pad3d_layer_type( &
            padding = [ this%hlf ], &
            method = padding_ &
       )
       this%pad = this%hlf
    end select
    if(.not.allocated(activation))then
       this%activation = activation_setup("none")
    else
       this%activation = activation
    end if
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser(this%activation%name)
       this%kernel_init = initialiser_setup(buffer)
    else
       this%kernel_init = kernel_initialiser
    end if
    if(.not.allocated(bias_initialiser))then
       buffer = get_default_initialiser( &
            this%activation%name, &
            is_bias=.true. &
       )
       this%bias_init = initialiser_setup(buffer)
    else
       this%bias_init = bias_initialiser
    end if
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("CONV3D activation function: ",A)') &
               trim(this%activation%name)
          write(*,'("CONV3D kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
          write(*,'("CONV3D bias initialiser: ",A)') &
               trim(this%bias_init%name)
       end if
    end if

  end subroutine set_hyperparams_conv3d
!###############################################################################


!###############################################################################
  subroutine set_batch_size_conv3d(this, batch_size, verbose)
    !! Set batch size for 3D convolutional layer
    implicit none

    ! Arguments
    class(conv3d_layer_type), intent(inout), target :: this
    !! Instance of the 3D convolutional layer
    integer, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    integer :: verbose_ = 0
    !! Verbosity level
    integer :: i
    !! Loop index

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
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(this%use_graph_input)then
          call stop_program( &
               "Graph input not supported for 3D convolutional layer" &
          )
          return
       end if
       if(allocated(this%output)) deallocate(this%output)
       allocate( this%output(1,1) )
       call this%output(1,1)%allocate( &
            array_shape = [ &
                 this%output_shape(1), &
                 this%output_shape(2), &
                 this%output_shape(3), &
                 this%num_filters, &
                 this%batch_size ], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_conv3d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_conv3d(this, unit)
    !! Print 3D convolutional layer to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(conv3d_layer_type), intent(in) :: this
    !! Instance of the 3D convolutional layer
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
    do i=1,3
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
    write(unit,'(3X,"INPUT_SHAPE = ",4(1X,I0))') this%input_shape
    write(unit,'(3X,"NUM_FILTERS = ",I0)') this%num_filters
    if(all(this%knl.eq.this%knl(1)))then
       write(unit,'(3X,"KERNEL_SIZE =",1X,I0)') this%knl(1)
    else
       write(unit,'(3X,"KERNEL_SIZE =",3(1X,I0))') this%knl
    end if
    if(all(this%stp.eq.this%stp(1)))then
       write(unit,'(3X,"STRIDE =",1X,I0)') this%stp(1)
    else
       write(unit,'(3X,"STRIDE =",3(1X,I0))') this%stp
    end if
    if(all(this%dil.eq.this%dil(1)))then
       write(unit,'(3X,"DILATION =",1X,I0)') this%dil(1)
    else
       write(unit,'(3X,"DILATION =",3(1X,I0))') this%dil
    end if
    write(unit,'(3X,"PADDING = ",A)') padding_type

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%activation%name)


    ! Write weights and biases
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params_array(1)%val(:,1)
    write(unit,'(5(E16.8E2))') this%params_array(2)%val(:,1)
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_conv3d
!###############################################################################


!###############################################################################
  subroutine read_conv3d(this, unit, verbose)
    !! Read 3D convolutional layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(conv3d_layer_type), intent(inout) :: this
    !! Instance of the 3D convolutional layer
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat
    !! Status of read
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: j, k, l, c, itmp1, iline, num_params
    !! Loop variables and temporary integer
    integer :: num_filters
    !! Number of filters
    logical :: use_bias = .true.
    !! Whether to use bias
    character(14) :: kernel_initialiser_name='', bias_initialiser_name=''
    !! Kernel and bias initialisers
    character(20) :: padding, activation_name=''
    !! Padding and activation function
    class(base_actv_type), allocatable :: activation
    !! Activation function
    class(base_init_type), allocatable :: kernel_initialiser, bias_initialiser
    !! Initialisers
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    integer, dimension(3) :: kernel_size, stride, dilation
    !! Kernel size and stride
    integer, dimension(4) :: input_shape
    !! Input shape
    real(real32), allocatable, dimension(:) :: data_list
    !! Data list
    integer :: param_line, final_line
    !! Parameter line number


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
    iline = 0
    param_line = 0
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
       case("DILATION")
          call assign_vec(buffer, dilation, itmp1)
       case("USE_BIAS")
          call assign_val(buffer, use_bias, itmp1)
       case("PADDING")
          call assign_val(buffer, padding, itmp1)
          padding = to_lower(padding)
       case("ACTIVATION")
          call assign_val(buffer, activation_name, itmp1)
       case("KERNEL_INITIALISER", "KERNEL_INIT", "KERNEL_INITIALIZER")
          call assign_val(buffer, kernel_initialiser_name, itmp1)
       case("BIAS_INITIALISER", "BIAS_INIT", "BIAS_INITIALIZER")
          call assign_val(buffer, bias_initialiser_name, itmp1)
       case("WEIGHTS")
          kernel_initialiser_name = 'zeros'
          bias_initialiser_name   = 'zeros'
          param_line = iline
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
    activation = activation_setup(activation_name)
    kernel_initialiser = initialiser_setup(kernel_initialiser_name)
    bias_initialiser = initialiser_setup(bias_initialiser_name)


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    call this%set_hyperparams( &
         num_filters = num_filters, &
         kernel_size = kernel_size, stride = stride, dilation = dilation, &
         padding = padding, &
         use_bias = use_bias, &
         activation = activation, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape = input_shape)


    ! Check if WEIGHTS card was found
    !---------------------------------------------------------------------------
    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in "//to_upper(trim(this%name))//" not found"
    else
       call move(unit, param_line - iline, iostat=stat)
       num_params = product(this%knl) * input_shape(4) * num_filters
       allocate(data_list(num_params), source=0._real32)
       c = 1
       k = 1
       data_concat_loop1: do while(c.le.num_params)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit data_concat_loop1
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do data_concat_loop1
       this%params_array(1)%val(:,1) = data_list
       deallocate(data_list)
       allocate(data_list(num_filters), source=0._real32)
       c = 1
       k = 1
       data_concat_loop2: do while(c.le.num_filters)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit data_concat_loop2
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do data_concat_loop2
       this%params_array(2)%val(:,1) = data_list
       deallocate(data_list)

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
    call move(unit, final_line - iline, iostat=stat)
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_conv3d
!###############################################################################


!###############################################################################
  function read_conv3d_layer(unit, verbose) result(layer)
    !! Read 3D convolutional layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the base layer

    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=conv3d_layer_type())
    call layer%read(unit, verbose=verbose_)

  end function read_conv3d_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_conv3d(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(conv3d_layer_type), intent(inout) :: this
    !! Instance of the 3D convolutional layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    type(array_type), pointer :: ptr
    !! Pointer array


    ! Generate outputs from weights, biases, and inputs
    !---------------------------------------------------------------------------
    select case(allocated(this%pad_layer))
    case(.true.)
       call this%pad_layer%forward(input)
       ptr => conv3d(this%pad_layer%output(1,1), this%params_array(1), &
            this%stp, this%dil &
       )
    case default
       ptr => conv3d(input(1,1), this%params_array(1), this%stp, this%dil)
    end select
    ptr => add_bias(ptr, this%params_array(2), dim=4, dim_act_on_shape=.true.)

    ! Apply activation function to activation
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    if(trim(this%activation%name) .eq. "none") then
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    else
       ptr => this%activation%apply(ptr)
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    end if
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_conv3d
!###############################################################################

end module athena__conv3d_layer
