module athena__conv2d_layer
  !! Module containing implementation of a 2D convolutional layer
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: conv_layer_type, base_layer_type
  use athena__pad2d_layer, only: pad2d_layer_type
  use athena__misc_types, only: initialiser_type, array4d_type, &
       onnx_node_type, onnx_initialiser_type
  use athena__misc_types, only: activation_type, initialiser_type
  use diffstruc, only: array_type
  use athena__diffstruc_extd, only: conv2d, add_bias
  implicit none


  private

  public :: conv2d_layer_type
  public :: read_conv2d_layer, create_from_onnx_conv2d_layer


  type, extends(conv_layer_type) :: conv2d_layer_type
     !! Type for 2D convolutional layer with overloaded procedures
     type(array_type), dimension(2) :: z
     !! Temporary arrays for forward propagation
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_conv2d
     !! Set hyperparameters for 2D convolutional layer
     procedure, pass(this) :: set_batch_size => set_batch_size_conv2d
     !! Set batch size for 2D convolutional layer
     procedure, pass(this) :: print_to_unit => print_to_unit_conv2d
     !! Print 2D convolutional layer to unit
     procedure, pass(this) :: read => read_conv2d
     !! Read 2D convolutional layer from file
     procedure, pass(this) :: build_from_onnx => build_from_onnx_conv2d
     !! Build 2D convolutional layer from ONNX node and initialiser

     procedure, pass(this) :: forward => forward_conv2d
     !! Forward propagation derived type handler

     final :: finalise_conv2d
     !! Finalise 2D convolutional layer
  end type conv2d_layer_type

  interface conv2d_layer_type
     !! Interface for setting up the 2D convolutional layer
     module function layer_setup( &
          input_shape, batch_size, &
          num_filters, kernel_size, stride, dilation, padding, &
          activation_function, activation_scale, &
          kernel_initialiser, bias_initialiser, &
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
       integer, dimension(..), optional, intent(in) :: dilation
       !! Dilation
       real(real32), optional, intent(in) :: activation_scale
       !! Activation scale
       character(*), optional, intent(in) :: activation_function, &
            kernel_initialiser, bias_initialiser, padding
       !! Activation function, kernel initialiser, bias initialiser, padding
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

    if(allocated(this%dil)) deallocate(this%dil)
    if(allocated(this%knl)) deallocate(this%knl)
    if(allocated(this%stp)) deallocate(this%stp)
    if(allocated(this%hlf)) deallocate(this%hlf)
    if(allocated(this%pad)) deallocate(this%pad)
    if(allocated(this%cen)) deallocate(this%cen)

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(allocated(this%pad_layer)) deallocate(this%pad_layer)

  end subroutine finalise_conv2d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       input_shape, batch_size, &
       num_filters, kernel_size, stride, dilation, padding, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser, &
       verbose ) result(layer)
    !! Set up the 2D convolutional layer
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
    real(real32), optional, intent(in) :: activation_scale
    !! Activation scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser, bias_initialiser, padding
    !! Activation function, kernel initialiser, bias initialiser, padding
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(conv2d_layer_type) :: layer
    !! Instance of the 2D convolutional layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: num_filters_
    !! Number of filters
    real(real32) :: scale = 1._real32
    !! Activation scale
    character(len=10) :: activation_function_ = "none"
    !! Activation function
    character(len=20) :: padding_
    !! Padding
    integer, dimension(2) :: kernel_size_, stride_, dilation_
    !! Kernel size and stride
    class(initialiser_type), allocatable :: kernel_initialiser_, bias_initialiser_
    !! Kernel and bias initialisers

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set activation and derivative functions based on input name
    !---------------------------------------------------------------------------
    if(present(activation_function)) activation_function_ = activation_function
    if(present(activation_scale)) scale = activation_scale


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
    ! Set up dilation
    !---------------------------------------------------------------------------
    if(present(dilation))then
       select rank(dilation)
       rank(0)
          dilation_ = dilation
       rank(1)
          dilation_(1) = dilation(1)
          if(size(dilation,dim=1).eq.1)then
             dilation_(2) = dilation(1)
          elseif(size(dilation,dim=1).eq.2)then
             dilation_(2) = dilation(2)
          end if
       end select
    else
       dilation_ = 1
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
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         num_filters = num_filters_, &
         kernel_size = kernel_size_, stride = stride_, dilation = dilation_, &
         padding = padding_, &
         activation_function = activation_function_, &
         activation_scale = scale, &
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
  subroutine set_hyperparams_conv2d( &
       this, &
       num_filters, &
       kernel_size, stride, dilation, &
       padding, &
       activation_function, &
       activation_scale, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    !! Set hyperparameters for 2D convolutional layer
    use athena__activation,  only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    use coreutils, only: to_lower
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout) :: this
    !! Instance of the 2D convolutional layer
    integer, intent(in) :: num_filters
    !! Number of filters
    integer, dimension(2), intent(in) :: kernel_size, stride, dilation
    !! Kernel size and stride
    character(*), intent(in) :: padding
    !! Padding
    character(*), intent(in) :: activation_function
    !! Activation function
    real(real32), intent(in) :: activation_scale
    !! Activation scale
    class(initialiser_type), allocatable, intent(in) :: &
         kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    character(len=20) :: padding_
    character(len=256) :: buffer

    this%name = "conv2d"
    this%type = "conv"
    this%input_rank = 3
    this%output_rank = 3
    this%has_bias = .true.
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
       this%pad_layer = pad2d_layer_type( &
            padding = [ this%hlf ], &
            method = padding_ &
       )
       this%pad = this%hlf
    end select
    if(allocated(this%transfer)) deallocate(this%transfer)
    allocate(this%transfer, &
         source=activation_setup(activation_function, activation_scale) &
    )
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser(activation_function)
       this%kernel_init = initialiser_setup(buffer)
    else
       this%kernel_init = kernel_initialiser
    end if
    if(.not.allocated(bias_initialiser))then
       buffer = get_default_initialiser( &
            activation_function, &
            is_bias=.true. &
       )
       this%bias_init = initialiser_setup(buffer)
    else
       this%bias_init = bias_initialiser
    end if
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("CONV2D activation function: ",A)') &
               trim(activation_function)
          write(*,'("CONV2D kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
          write(*,'("CONV2D bias initialiser: ",A)') &
               trim(this%bias_init%name)
       end if
    end if

  end subroutine set_hyperparams_conv2d
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
               "Graph input not supported for 2D convolutional layer" &
          )
          return
       end if
       if(allocated(this%output)) deallocate(this%output)
       allocate( this%output(1,1) )
       call this%output(1,1)%allocate( &
            array_shape = [ &
                 this%output_shape(1), &
                 this%output_shape(2), &
                 this%num_filters, &
                 this%batch_size ], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_conv2d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_conv2d(this, unit)
    !! Print 2D convolutional layer to unit
    use coreutils, only: to_upper
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
    if(all(this%dil.eq.this%dil(1)))then
       write(unit,'(3X,"DILATION =",1X,I0)') this%dil(1)
    else
       write(unit,'(3X,"DILATION =",2(1X,I0))') this%dil
    end if
    write(unit,'(3X,"PADDING = ",A)') padding_type

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') this%transfer%scale


    ! Write weights and biases
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params_array(1)%val(:,1)
    write(unit,'(5(E16.8E2))') this%params_array(2)%val(:,1)
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_conv2d
!###############################################################################


!###############################################################################
  subroutine read_conv2d(this, unit, verbose)
    !! Read 2D convolutional layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__initialiser, only: initialiser_setup
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
    integer :: j, k, l, c, itmp1, iline, num_params
    !! Loop variables and temporary integer
    integer :: num_filters
    !! Number of filters
    real(real32) :: activation_scale
    !! Activation scale
    character(14) :: kernel_initialiser_name='', bias_initialiser_name=''
    !! Kernel and bias initialisers
    character(20) :: padding, activation_function
    !! Padding and activation function
    class(initialiser_type), allocatable :: kernel_initialiser, bias_initialiser
    !! Initialisers
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    integer, dimension(2) :: kernel_size, stride, dilation
    !! Kernel size and stride
    integer, dimension(3) :: input_shape
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
       case("PADDING")
          call assign_val(buffer, padding, itmp1)
          padding = to_lower(padding)
       case("ACTIVATION")
          call assign_val(buffer, activation_function, itmp1)
       case("ACTIVATION_SCALE")
          call assign_val(buffer, activation_scale, itmp1)
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
    kernel_initialiser = initialiser_setup(kernel_initialiser_name)
    bias_initialiser = initialiser_setup(bias_initialiser_name)


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    call this%set_hyperparams( &
         num_filters = num_filters, &
         kernel_size = kernel_size, stride = stride, dilation = dilation, &
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
    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in "//to_upper(trim(this%name))//" not found"
    else
       call move(unit, param_line - iline, iostat=stat)
       num_params = product(this%knl) * input_shape(3) * num_filters
       allocate(data_list(num_params), source=0._real32)
       c = 1
       k = 1
       data_concat_loop: do while(c.le.num_params)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit data_concat_loop
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do data_concat_loop
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


!###############################################################################
  subroutine build_from_onnx_conv2d(this, node, initialisers, verbose )
    !! Read ONNX attributes for 2D convolutional layer
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout) :: this
    !! Instance of the 2D convolutional layer
    type(onnx_node_type), intent(in) :: node
    !! Instance of ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! Instance of ONNX initialiser information
    integer, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: i
    !! Loop index and temporary integer
    integer :: num_filters
    !! Number of filters
    integer, dimension(2) :: padding, stride, kernel_size
    !! Padding, stride, and kernel size
    character(256) :: val
    !! Attribute value

    do i = 1, size(node%attributes)
       val = node%attributes(i)%value
       select case(trim(adjustl(node%attributes(i)%name)))
       case("pads")
          read(val,*) padding
       case("strides")
          read(val,*) stride
       case("kernel_shape")
          read(val,*) kernel_size
       case("dilations")
          write(0,*) "WARNING: dilations not yet implemented for conv2d layer"
       case default
          ! Do nothing
          write(0,*) "WARNING: Unrecognised attribute in ONNX CONV2D layer: ", &
               trim(adjustl(node%attributes(i)%name))
       end select
    end do


    ! Initialise parameters from initialisers
    write(0,*) "WARNING: Weights initialisation from ONNX not yet implemented &
         &for conv2d layer"

    call this%set_hyperparams( &
         num_filters = num_filters, &
         kernel_size = kernel_size, stride = stride, &
         padding = "valid", &
         activation_function = "none", &
         activation_scale = 1._real32, &
         verbose = verbose_, &
         kernel_initialiser = "zeros", &
         bias_initialiser = "zeros" &
    )

  end subroutine build_from_onnx_conv2d
!###############################################################################


!###############################################################################
  function create_from_onnx_conv2d_layer(node, initialisers, verbose) result(layer)
    !! Build 2D convolutional layer from attributes and return layer
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! Instance of ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! Instance of ONNX initialiser information
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 2D convolutional layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=conv2d_layer_type())
    call layer%build_from_onnx(node, initialisers, verbose=verbose_)

  end function create_from_onnx_conv2d_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_conv2d(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(conv2d_layer_type), intent(inout) :: this
    !! Instance of the 2D convolutional layer
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
       ptr => conv2d(this%pad_layer%output(1,1), this%params_array(1), &
            this%stp, this%dil &
       )
    case default
       ptr => conv2d(input(1,1), this%params_array(1), this%stp, this%dil)
    end select
    ptr => add_bias(ptr, this%params_array(2), dim=3, dim_act_on_shape=.true.)

    ! Apply activation function to activation
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    if(trim(this%transfer%name) .eq. "none") then
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    else
       ptr => this%transfer%activate(ptr)
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    end if
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_conv2d
!###############################################################################

end module athena__conv2d_layer
