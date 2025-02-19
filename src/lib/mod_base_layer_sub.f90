submodule(athena__base_layer) athena__base_layer_submodule
  !! Submodule containing the implementation of the base layer types
  !!
  !! This submodule contains the implementation of the base layer types
  !! used in the ATHENA library. The base layer types are the abstract
  !! types from which all other layer types are derived. The submodule
  !! contains the implementation of the procedures that are common to
  !! all layer types, such as setting the input shape, getting the
  !! number of parameters, and printing the layer to a file.
  !!
  !! The following procedures are based on code from the neural-fortran library
  !! https://github.com/modern-fortran/neural-fortran/blob/main/src/nf/nf_layer.f90
  !! procedures:
  !! - get_num_params*
  !! - get_params*
  !! - set_params*
  !! - get_gradients*
  !! - set_gradients*
  use athena__io_utils, only: stop_program
  use athena__misc, only: to_lower, to_upper, icount
  use athena__tools_infile, only: assign_val, assign_vec
  implicit none

contains

!###############################################################################
  module subroutine print_base(this, file)
    !! Print the layer to a file
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    character(*), intent(in) :: file
    !! File name

    ! No need to write anything for the default layer
    return
  end subroutine print_base
!-------------------------------------------------------------------------------
  module subroutine print_pool(this, file)
    !! Print pooling layer to a file
    implicit none

    ! Arguments
    class(pool_layer_type), intent(in) :: this
    !! Instance of the layer
    character(*), intent(in) :: file
    !! File name

    ! Local variables
    integer :: unit
    !! Unit number
    character(100) :: fmt
    !! Format string

    ! open file with new unit
    !---------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    ! write convolution initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(A)') to_upper(trim(this%name))
    write(fmt,'("(3X,""INPUT_SHAPE ="",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    if(all(this%pool.eq.this%pool(1)))then
       write(unit,'(3X,"POOL_SIZE =",1X,I0)') this%pool(1)
    else
       write(fmt,'("(3X,""STRIDE ="",",I0,"(1X,I0))")') size(this%pool)
       write(unit,fmt) this%pool
    end if
    if(all(this%strd.eq.this%strd(1)))then
       write(unit,'(3X,"STRIDE =",1X,I0)') this%strd(1)
    else
       write(fmt,'("(3X,""STRIDE ="",",I0,"(1X,I0))")') size(this%strd)
       write(unit,fmt) this%strd
    end if
    write(unit,'("END ",A)') to_upper(trim(this%name))

    ! close unit
    !---------------------------------------------------------------------------
    close(unit)

  end subroutine print_pool
!###############################################################################


!###############################################################################
  module subroutine set_shape_base(this, input_shape)
    !! Set the input shape of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    character(len=100) :: err_msg
    !! Error message

    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(size(input_shape,dim=1).eq.this%input_rank)then
       this%input_shape = input_shape
    else
       write(err_msg,'("ERROR: invalid size of input_shape in ",A,&
            &" expected (",I0,"), got (",I0")")')  &
            trim(this%name), this%input_rank, size(input_shape,dim=1)
       call stop_program(err_msg)
       return
    end if
 
  end subroutine set_shape_base
!###############################################################################


!###############################################################################
  pure subroutine get_output_base(this, output)
    !! Get the output of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    real(real32), allocatable, dimension(..), intent(out) :: output
    !! Output of the layer
  
    call this%output%get(output)
  end subroutine get_output_base
!###############################################################################


!###############################################################################
  module subroutine set_ptrs(this)
    !! Set the pointers of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout), target :: this
    !! Instance of the layer

    ! Local variables
    character(256) :: err_msg
    !! Error message

    if(allocated(this%output))then
       if(.not.this%output%allocated)then
          write(err_msg,'("output not allocated for layer ",A," ",I0)') &
               trim(this%name), this%id
          call stop_program(err_msg)
          return
       end if
       call this%output%set_ptr()
    end if
    if(allocated(this%di))then
       if(.not.this%di%allocated)then
          write(err_msg,'("di not allocated for layer ",A," ",I0)') &
               trim(this%name), this%id
          call stop_program(err_msg)
          return
       end if
       call this%di%set_ptr()
    end if

    call this%set_ptrs_hyperparams()

  end subroutine set_ptrs
!-------------------------------------------------------------------------------
  module subroutine set_ptrs_hyperparams(this)
    !! Set the hyperparameter pointers of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout), target :: this
    !! Instance of the layer

    ! No hyperparameters to set for the base layer
    return
  end subroutine set_ptrs_hyperparams
!###############################################################################


!###############################################################################
  pure module function get_num_params_base(this) result(num_params)
    !! Get the number of parameters in the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    integer :: num_params
    !! Number of parameters
    
    ! No parameters in the base layer
    num_params = 0

  end function get_num_params_base
!-------------------------------------------------------------------------------
  pure module function get_num_params_conv(this) result(num_params)
    !! Get the number of parameters in convolutional layer
    implicit none

    ! Arguments
    class(conv_layer_type), intent(in) :: this
    !! Instance of the layer
    integer :: num_params
    !! Number of parameters
    
    ! num_filters x num_channels x kernel_size + num_biases
    ! num_biases = num_filters
    num_params = this%num_filters * this%num_channels * product(this%knl) + &
         this%num_filters

  end function get_num_params_conv
!-------------------------------------------------------------------------------
  pure module function get_num_params_batch(this) result(num_params)
    !! Get the number of parameters in batch normalisation layer
    implicit none
    
    ! Arguments
    class(batch_layer_type), intent(in) :: this
    !! Instance of the layer
    integer :: num_params
    !! Number of parameters
    
    ! num_filters x num_channels x kernel_size + num_biases
    ! num_biases = num_filters
    num_params = 2 * this%num_channels

  end function get_num_params_batch
!###############################################################################


!###############################################################################
  module subroutine reduce_learnable(this, rhs)
    !! Reduce two learnable layers to a single one via summation
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(inout) :: this
    !! Instance of the layer
    class(learnable_layer_type), intent(in) :: rhs
    !! Instance of a layer

    this%dp = this%dp + rhs%dp
    this%db = this%db + rhs%db

  end subroutine  reduce_learnable
!###############################################################################


!###############################################################################
  module function add_learnable(a, b) result(output)
    !! Add two learnable layers together
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: a, b
    !! Instances of layers
    class(learnable_layer_type), allocatable :: output
    !! Output layer

    output = a
    output%dp = output%dp + b%dp
    output%db = output%db + b%db

  end function add_learnable
!###############################################################################


!###############################################################################
  module subroutine merge_learnable(this, input)
    !! Merge two learnable layers via summation
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(inout) :: this
    !! Instance of the layer
    class(learnable_layer_type), intent(in) :: input
    !! Instance of a layer

    this%dp = this%dp + input%dp
    this%db = this%db + input%db

  end subroutine merge_learnable
!###############################################################################


!###############################################################################
  pure module function get_params(this) result(params)
    !! Get the learnable parameters of the layer
    !!
    !! This function returns the learnable parameters of the layer
    !! as a single array.
    !! This has been modified from the neural-fortran library
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: this
    !! Instance of the layer
    real(real32), dimension(this%num_params) :: params
    !! Learnable parameters

    params = this%params

  end function get_params
!###############################################################################


!###############################################################################
  module subroutine set_params(this, params)
    !! Set the learnable parameters of the layer
    !!
    !! This function sets the learnable parameters of the layer
    !! from a single array.
    !! This has been modified from the neural-fortran library
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(inout) :: this
    !! Instance of the layer
    real(real32), dimension(this%num_params), intent(in) :: params
    !! Learnable parameters

    this%params = params

  end subroutine set_params
!###############################################################################


!###############################################################################
  pure module function get_gradients(this, clip_method) result(gradients)
    !! Get the gradients of the layer
    !!
    !! This function returns the gradients of the layer as a single array.
    !! This has been modified from the neural-fortran library
    use athena__clipper, only: clip_type
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: this
    !! Instance of the layer
    type(clip_type), optional, intent(in) :: clip_method
    !! Method to clip the gradients
    real(real32), dimension(this%num_params) :: gradients
    !! Gradients of the layer
  
    gradients = [ sum(this%dp, dim=2) / this%batch_size, &
         sum(this%db, dim=2) / this%batch_size ]
  
    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)

  end function get_gradients
!###############################################################################


!###############################################################################
  module subroutine set_gradients(this, gradients)
    !! Set the gradients of the layer
    !!
    !! This function sets the gradients of the layer from a single array.
    !! This has been modified from the neural-fortran library
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(inout) :: this
    !! Instance of the layer
    real(real32), dimension(..), intent(in) :: gradients
    !! Gradients of the layer
  
    select rank(gradients)
    rank(0)
       this%dp = gradients
       this%db = gradients
    rank(1)
       this%dp = spread( &
            gradients(1:this%num_params - size(this%db,1)), &
            2, &
            this%batch_size &
       )
       this%db = spread( &
            gradients(this%num_params - size(this%db,1) + 1:), &
            2, &
            this%batch_size &
       )
    end select
  
  end subroutine set_gradients
!###############################################################################


!###############################################################################
  module subroutine set_gradients_batch(this, gradients)
    !! Set the gradients of a batch normalisation layer
    !!
    !! This function sets the gradients of a batch normalisation layer
    !! from a single array.
    !! This has been modified from the neural-fortran library
    implicit none

    ! Arguments
    class(batch_layer_type), intent(inout) :: this
    !! Instance of the layer
    real(real32), dimension(..), intent(in) :: gradients
    !! Gradients of the layer

    select rank(gradients)
    rank(0)
       this%dp = gradients * this%batch_size
       this%db = gradients * this%batch_size
    rank(1)
        this%dp(:,1) = gradients(:this%num_channels) * this%batch_size
        this%db(:,1) = gradients(this%num_channels+1:) * this%batch_size
    end select
  
  end subroutine set_gradients_batch
!###############################################################################


!###############################################################################
  module subroutine init_conv(this, input_shape, batch_size, verbose)
    !! Initialise convolutional layer
    use athena__initialiser, only: initialiser_setup
    use athena__misc_types, only: initialiser_type
    implicit none

    ! Arguments
    class(conv_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    integer :: verbose_ = 0
    class(initialiser_type), allocatable :: initialiser_


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! initialise padding layer, if allocated
    !---------------------------------------------------------------------------
    if(allocated(this%pad_layer)) &
         call this%pad_layer%init(this%input_shape, this%batch_size, verbose_)


    !---------------------------------------------------------------------------
    ! allocate output, activation, bias, and weight shapes
    !---------------------------------------------------------------------------
    ! NOTE: INPUT SHAPE DOES NOT INCLUDE PADDING WIDTH
    ! THIS IS HANDLED AUTOMATICALLY BY THE CODE
    ! ... provide the initial input data shape and let us deal with the padding
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output))then
       if(this%output%allocated) call this%output%deallocate()
    end if
    this%output%shape(this%input_rank) = this%num_filters
    this%output%shape(:this%input_rank-1) = floor( &
         ( &
              this%input_shape(:this%input_rank-1) + 2 * this%pad - this%knl &
         ) / real(this%stp) &
    ) + 1
    this%num_params = this%get_num_params()
    if(allocated(this%params)) deallocate(this%params)
    allocate(this%params(this%num_params), source=0._real32)


    !---------------------------------------------------------------------------
    ! initialise weights (kernels)
    !---------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%kernel_initialiser))
    call initialiser_%initialise( &
         this%params(:this%num_params-this%num_filters), &
         fan_in=product(this%knl)+1, fan_out=1, &
         spacing = [ this%knl, this%num_channels, this%num_filters ] &
    )
    deallocate(initialiser_)

    ! initialise biases
    !---------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%bias_initialiser))
    call initialiser_%initialise( &
         this%params(this%num_params-this%num_filters+1:), &
         fan_in=product(this%knl)+1, fan_out=1 &
    )
    deallocate(initialiser_)


    !---------------------------------------------------------------------------
    ! initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_conv
!###############################################################################


!###############################################################################
  module subroutine init_batch(this, input_shape, batch_size, verbose)
    !! Initialise batch normalisation layer
    use athena__initialiser, only: initialiser_setup
    use athena__misc_types, only: initialiser_type
    implicit none

    ! Arguments
    class(batch_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    integer :: verbose_ = 0
    class(initialiser_type), allocatable :: t_initialiser


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! set up number of channels, width, height
    !---------------------------------------------------------------------------
    if(allocated(this%output))then
       if(this%output%allocated) call this%output%deallocate()
    end if
    if(size(this%input_shape).eq.1)then
       this%output%shape(1) = this%input_shape(1)
       this%output%shape(2) = 1
    else
       this%output%shape = this%input_shape
    end if
    this%num_channels = this%input_shape(this%input_rank)
    this%num_params = this%get_num_params()
    allocate(this%params(2 * this%num_channels), source=0._real32)
    allocate(this%dp(this%num_channels,1), source=0._real32)
    allocate(this%db(this%num_channels,1), source=0._real32)


    !---------------------------------------------------------------------------
    ! allocate mean and variance
    !---------------------------------------------------------------------------
    allocate(this%mean(this%num_channels), source=0._real32)
    allocate(this%variance, source=this%mean)


    !---------------------------------------------------------------------------
    ! initialise gamma
    !---------------------------------------------------------------------------
    allocate(t_initialiser, source=initialiser_setup(this%kernel_initialiser))
    t_initialiser%mean = this%gamma_init_mean
    t_initialiser%std  = this%gamma_init_std
    call t_initialiser%initialise(this%params(1:this%num_channels), &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)

    ! initialise beta
    !---------------------------------------------------------------------------
    allocate(t_initialiser, source=initialiser_setup(this%bias_initialiser))
    t_initialiser%mean = this%beta_init_mean
    t_initialiser%std  = this%beta_init_std
    call t_initialiser%initialise(this%params(this%num_channels+1:), &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)


    !---------------------------------------------------------------------------
    ! initialise moving mean
    !---------------------------------------------------------------------------
    allocate(t_initialiser, &
         source=initialiser_setup(this%moving_mean_initialiser))
    call t_initialiser%initialise(this%mean, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)

    ! initialise moving variance
    !---------------------------------------------------------------------------
    allocate(t_initialiser, &
         source=initialiser_setup(this%moving_variance_initialiser))
    call t_initialiser%initialise(this%variance, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)


    !---------------------------------------------------------------------------
    ! initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_batch
!###############################################################################


!###############################################################################
  module subroutine set_ptrs_hyperparams_batch(this)
    !! Set the hyperparameter pointers of a batch normalisation layer
    implicit none

    ! Arguments
    class(batch_layer_type), intent(inout), target :: this
    !! Instance of the layer

    if(allocated(this%params))then
       this%gamma(1:this%num_channels) => this%params(1:this%num_channels)
       this%beta(1:this%num_channels) => &
            this%params(this%num_channels+1:this%num_channels*2)
    end if

  end subroutine set_ptrs_hyperparams_batch
!###############################################################################

end submodule athena__base_layer_submodule