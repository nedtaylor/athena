module athena__conv3d_layer
  !! Module containing implementation of a 3D convolutional layer
  !!
  !! This module implements 3D convolution for processing volumetric data
  !! such as video, medical imaging, or 3D point clouds.
  !!
  !! Mathematical operation:
  !! \[
  !!   y_{i,j,k,\,q}
  !!   =
  !!   \sigma\left(
  !!     \sum_{c=1}^{C_{in}}
  !!     \sum_{a=0}^{K_d-1}
  !!     \sum_{b=0}^{K_h-1}
  !!     \sum_{d=0}^{K_w-1}
  !!       x_{i+a,\,j+b,\,k+d,\,c}\;
  !!       w_{a,\,b,\,d,\,c,\,q}
  !!     + b_q
  !!   \right)
  !! \]
  !!
  !! where:
  !!   - \((i,j,k)\) are spatial coordinates in the output
  !!   - \(q\) is the output channel (filter) index
  !!   - \((a,b,d)\) are kernel offsets: depth, height, width
  !!   - \(c\) is the input channel index
  !!   - \(K_d, K_h, K_w\) are kernel dimensions
  !!   - \(\sigma\) is the activation function
  !!
  !! Shape: \((D, H, W, C_{in}) \rightarrow (D', H', W', C_{out})\)
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: conv_layer_type, base_layer_type, &
       learnable_layer_type
  use athena__pad3d_layer, only: pad3d_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type, &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use diffstruc, only: array_type
  use athena__diffstruc_extd, only: conv3d, add_bias
  implicit none


  private

  public :: conv3d_layer_type
  public :: read_conv3d_layer


  type, extends(conv_layer_type) :: conv3d_layer_type
     !! Type for 3D convolutional layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_conv3d
     !! Set hyperparameters for 3D convolutional layer
     procedure, pass(this) :: read => read_conv3d
     !! Read 3D convolutional layer from file

#ifdef __INTEL_COMPILER
     procedure, pass(this) :: reduce => reduce_conv3d
     !! Merge another 3D convolutional layer into this one
     procedure :: add_t_t => add_conv3d
     !! Add two 3D convolutional layers
#endif

     procedure, pass(this) :: forward => forward_conv3d
     !! Forward propagation derived type handler

     final :: finalise_conv3d
     !! Finalise 3D convolutional layer
  end type conv3d_layer_type

  interface conv3d_layer_type
     !! Interface for setting up the 3D convolutional layer
     module function layer_setup( &
          input_shape, &
          num_filters, kernel_size, stride, dilation, padding, &
          use_bias, &
          activation, &
          kernel_initialiser, bias_initialiser, &
          verbose ) result(layer)
       !! Set up the 3D convolutional layer
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
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

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(allocated(this%pad_layer)) deallocate(this%pad_layer)
    if(this%z(1)%allocated) call this%z(1)%deallocate()
    if(this%z(2)%allocated) call this%z(2)%deallocate()

  end subroutine finalise_conv3d
!###############################################################################

#ifdef __INTEL_COMPILER
!###############################################################################
  subroutine reduce_conv3d(this, input)
    !! Merge two 3D convolutional layers via parameter summation
    implicit none

    class(conv3d_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: input

    real(real32), allocatable :: params(:), gradients(:)

    select type(input)
    type is (conv3d_layer_type)
       params = this%get_params() + input%get_params()
       call this%set_params(params)

       gradients = this%get_gradients() + input%get_gradients()
       call this%set_gradients(gradients)
    class default
       call stop_program("reduce_conv3d: incompatible layer type")
    end select

  end subroutine reduce_conv3d
!###############################################################################


!###############################################################################
  function add_conv3d(a, b) result(output)
    !! Add two 3D convolutional layers without whole-object allocatable copy
    implicit none

    class(conv3d_layer_type), intent(in) :: a
    class(learnable_layer_type), intent(in) :: b
    class(learnable_layer_type), allocatable :: output

    type(conv3d_layer_type) :: layer
    character(len=20) :: padding
    real(real32), allocatable :: params(:), gradients(:)

    select type(b)
    type is (conv3d_layer_type)
       padding = 'valid'
       if(allocated(a%pad_layer)) padding = a%pad_layer%method

       call layer%set_hyperparams( &
            num_filters = a%num_filters, &
            kernel_size = a%knl, stride = a%stp, dilation = a%dil, &
            padding = padding, &
            use_bias = a%use_bias, &
            activation = a%activation, &
            kernel_initialiser = a%kernel_init, &
            bias_initialiser = a%bias_init &
       )

       if(allocated(a%input_shape)) call layer%init(a%input_shape)

       params = a%get_params() + b%get_params()
       call layer%set_params(params)

       gradients = a%get_gradients() + b%get_gradients()
       call layer%set_gradients(gradients)

       layer%id = a%id
       layer%inference = a%inference
       layer%use_graph_input = a%use_graph_input
       layer%use_graph_output = a%use_graph_output
       layer%subtype = a%subtype

       allocate(output, source=layer)
    class default
       call stop_program("add_conv3d: incompatible layer type")
    end select

  end function add_conv3d
!###############################################################################
#endif

!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       input_shape, &
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
    ! Set activation functions based on input name
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
    allocate( &
         this%dil(this%input_rank-1), &
         this%knl(this%input_rank-1), &
         this%stp(this%input_rank-1) &
    )
    this%dil = dilation
    this%knl = kernel_size
    this%stp = stride
    this%num_filters = num_filters
    padding_ = trim(adjustl(padding))

    select case(trim(adjustl(to_lower(padding_))))
    case("valid", "none", "")
    case default
       this%pad_layer = pad3d_layer_type( &
            padding = [ (this%knl-1)/2 ], &
            method = padding_ &
       )
    end select
    if(allocated(this%activation)) deallocate(this%activation)
    if(.not.allocated(activation))then
       this%activation = activation_setup("none")
    else
       allocate(this%activation, source=activation)
    end if
    if(allocated(this%kernel_init)) deallocate(this%kernel_init)
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser(this%activation%name)
       this%kernel_init = initialiser_setup(buffer)
    else
       allocate(this%kernel_init, source=kernel_initialiser)
    end if
    if(allocated(this%bias_init)) deallocate(this%bias_init)
    if(.not.allocated(bias_initialiser))then
       buffer = get_default_initialiser( &
            this%activation%name, &
            is_bias=.true. &
       )
       this%bias_init = initialiser_setup(buffer)
    else
       allocate(this%bias_init, source=bias_initialiser)
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


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine read_conv3d(this, unit, verbose)
    !! Read 3D convolutional layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
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
    character(20) :: padding='', activation_name=''
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
          iline = iline - 1
          backspace(unit)
          activation = read_activation(unit, iline)
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
       this%params(1)%val(:,1) = data_list
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
       this%params(2)%val(:,1) = data_list
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
  subroutine build_from_onnx_conv3d( &
       this, node, initialisers, value_info, verbose &
  )
    !! Read ONNX attributes for 3D convolutional layer
    use athena__activation, only: activation_setup
    use athena__initialiser_data, only: data_init_type
    implicit none

    ! Arguments
    class(conv3d_layer_type), intent(inout) :: this
    !! Instance of the 3D convolutional layer
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info information
    integer, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: i, weight_idx, bias_idx
    !! Loop index and temporary integer
    integer :: num_filters
    !! Number of filters
    logical :: use_bias = .true.
    !! Whether to use bias
    integer, dimension(3) :: padding, stride, kernel_size, dilation
    !! Padding, stride, kernel size, and dilation
    integer, dimension(:), allocatable :: dims
    !! Dimensions
    character(256) :: val
    !! Attribute value
    class(base_actv_type), allocatable :: activation
    !! Activation function
    class(base_init_type), allocatable :: kernel_initialiser, bias_initialiser

    ! Set default values
    padding = 0
    stride = 1
    kernel_size = 3
    dilation = 1

    do i = 1, size(node%attributes)
       val = node%attributes(i)%val
       select case(trim(adjustl(node%attributes(i)%name)))
       case("pads")
          read(val,*) padding
       case("strides")
          read(val,*) stride
       case("kernel_shape")
          read(val,*) kernel_size
       case("dilations")
          read(val,*) dilation
       case default
          ! Do nothing
          write(0,*) "WARNING: Unrecognised attribute in ONNX CONV3D layer: ", &
               trim(adjustl(node%attributes(i)%name))
       end select
    end do

    weight_idx = -1
    bias_idx = -1
    allocate(dims(0))
    if(size(initialisers).lt.1)then
       call stop_program("ONNX CONV3D layer requires at least 1 initialiser")
       return
    else
       ! check which initialiser has weights and which has biases
       do i = 1, size(initialisers)
          if(allocated(initialisers(i)%dims))then
             dims = [ dims, product(initialisers(i)%dims) ]
          end if
       end do
    end if

    select case(size(dims))
    case(1)
       if(mod(dims(1), product(kernel_size)).eq.0)then
          weight_idx = 1
       else
          call stop_program("ONNX CONV3D layer initialiser dimensions do not &
               &match kernel size")
          return
       end if
       use_bias = .false.
    case(2)
       ! check which is weight and which is bias
       if(mod(dims(1), product(kernel_size)).eq.0 .and. &
            dims(1)/product(kernel_size).eq.dims(2))then
          weight_idx = 1
          bias_idx = 2
       elseif(mod(dims(2), product(kernel_size)).eq.0 .and. &
            dims(2)/product(kernel_size).eq.dims(1))then
          weight_idx = 2
          bias_idx = 1
       else
          call stop_program("ONNX CONV3D layer initialiser dimensions do not &
               &match kernel size")
          return
       end if
    case default
       call stop_program("ONNX CONV3D layer number of initialisers not &
            &supported")
       return
    end select

    num_filters = dims(weight_idx) / product(kernel_size)
    if(num_filters .ne. value_info(1)%dims(2))then
       call stop_program("ONNX CONV3D layer number of filters does not match &
            &value info")
       return
    end if

    kernel_initialiser = data_init_type( data = initialisers(weight_idx)%data )
    if(use_bias)then
       bias_initialiser = data_init_type( data = initialisers(bias_idx)%data )
    end if

    activation = activation_setup("none")
    call this%set_hyperparams( &
         num_filters = num_filters, &
         kernel_size = kernel_size, stride = stride, &
         dilation = dilation, &
         padding = "valid", &
         use_bias = use_bias, &
         activation = activation, &
         verbose = verbose, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser &
    )

  end subroutine build_from_onnx_conv3d
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
       ptr => conv3d(this%pad_layer%output(1,1), this%params(1), &
            this%stp, this%dil &
       )
    case default
       ptr => conv3d(input(1,1), this%params(1), this%stp, this%dil)
    end select
    call this%z(1)%zero_grad()
    call this%z(1)%assign_and_deallocate_source(ptr)
    this%z(1)%is_temporary = .false.
    ptr => add_bias(this%z(1), this%params(2), dim=4, dim_act_on_shape=.true.)

    ! Apply activation function to activation
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    if(trim(this%activation%name) .eq. "none") then
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    else
       call this%z(2)%zero_grad()
       call this%z(2)%assign_and_deallocate_source(ptr)
       this%z(2)%is_temporary = .false.
       ptr => this%activation%apply(this%z(2))
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    end if
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_conv3d
!###############################################################################

end module athena__conv3d_layer
