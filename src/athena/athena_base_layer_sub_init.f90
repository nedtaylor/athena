submodule(athena__base_layer) athena__base_layer_submodule_init
  !! Submodule containing the implementation of the base layer types
  !!
  !! This submodule contains the implementation of the base layer types
  !! used in the ATHENA library. The base layer types are the abstract
  !! types from which all other layer types are derived. The submodule
  !! contains the implementation of the initialisation procedures
  use coreutils, only: stop_program
  use athena__diffstruc_extd, only: batchnorm_array_type

contains

!###############################################################################
  module subroutine init_pad(this, input_shape, verbose)
    !! Initialise padding layer
    implicit none

    ! Arguments
    class(pad_layer_type), intent(inout) :: this
    !! Instance of the padding layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: i
    !! Loop index
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    if(.not.allocated(this%orig_bound)) then
       allocate(this%orig_bound(2,this%input_rank-1))
       allocate(this%dest_bound(2,this%input_rank-1))
    end if
    do i = 1, this%input_rank - 1
       this%orig_bound(:,i) = [ 1, this%input_shape(i) ]
       this%dest_bound(:,i) = [ 1, this%input_shape(i) + this%pad(i) * 2 ]
       call this%facets(i)%setup_bounds( &
            length = this%input_shape(:this%input_rank-1), &
            pad = this%pad, &
            imethod = this%imethod &
       )
    end do


    !---------------------------------------------------------------------------
    ! Set up number of channels, width, height
    !---------------------------------------------------------------------------
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    allocate( this%output_shape(this%input_rank) )
    this%output_shape(this%input_rank) = this%input_shape(this%input_rank)
    this%output_shape(:this%input_rank-1) = &
         this%input_shape(:this%input_rank-1) + this%pad(:) * 2


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program("Graph input not supported for padding layer")
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate( this%output(1,1) )

  end subroutine init_pad
!###############################################################################


!###############################################################################
  module subroutine init_pool(this, input_shape, verbose)
    !! Initialise pooling layer
    implicit none

    ! Arguments
    class(pool_layer_type), intent(inout) :: this
    !! Instance of the pooling layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! Set up number of channels, width, height
    !---------------------------------------------------------------------------
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    allocate( this%output_shape(this%input_rank) )
    this%output_shape(this%input_rank) = this%input_shape(this%input_rank)
    this%output_shape(:this%input_rank-1) = &
         floor( &
              ( &
                   this%input_shape(:this%input_rank-1) - this%pool &
              ) / real(this%strd) &
         ) + 1


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program( &
            "Graph input not supported for pooling layer" &
       )
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate( this%output(1,1) )

  end subroutine init_pool
!###############################################################################


!###############################################################################
  module subroutine init_conv(this, input_shape, verbose)
    !! Initialise convolutional layer
    use athena__initialiser, only: initialiser_setup
    use athena__misc_types, only: base_init_type
    implicit none

    ! Arguments
    class(conv_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, dimension(this%input_rank-1) :: pad_shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! initialise padding layer, if allocated
    !---------------------------------------------------------------------------
    if(allocated(this%pad_layer))then
       call this%pad_layer%init(this%input_shape, verbose_)
       pad_shape = pad_shape + 2 * this%pad_layer%pad
    else
       pad_shape = 0
    end if


    !---------------------------------------------------------------------------
    ! allocate output, activation, bias, and weight shapes
    !---------------------------------------------------------------------------
    ! NOTE: INPUT SHAPE DOES NOT INCLUDE PADDING WIDTH
    ! THIS IS HANDLED AUTOMATICALLY BY THE CODE
    ! ... provide the initial input data shape and let us deal with the padding
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    allocate( this%output_shape(this%input_rank) )
    this%output_shape(this%input_rank) = this%num_filters
    this%output_shape(:this%input_rank-1) = floor( &
         ( &
              this%input_shape(:this%input_rank-1) + 2 * pad_shape - this%knl &
         ) / real(this%stp) &
    ) + 1
    this%num_params = this%get_num_params()
    allocate(this%weight_shape(this%input_rank + 1,1))
    this%weight_shape(:,1) = [ this%knl, this%num_channels, this%num_filters ]
    this%bias_shape = [this%num_filters]

    if(allocated(this%params)) deallocate(this%params)
    allocate(this%params(2))
    call this%params(1)%allocate([this%weight_shape(:,1), 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%fix_pointer = .true.
    this%params(1)%is_sample_dependent = .false.
    this%params(1)%is_temporary = .false.
    call this%params(2)%allocate([this%bias_shape, 1])
    call this%params(2)%set_requires_grad(.true.)
    this%params(2)%fix_pointer = .true.
    this%params(2)%is_sample_dependent = .false.
    this%params(2)%is_temporary = .false.


    !---------------------------------------------------------------------------
    ! initialise weights (kernels)
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params(1)%val(:,1), &
         fan_in = product(this%knl)+1, fan_out = 1, &
         spacing = [ this%knl, this%num_channels, this%num_filters ] &
    )

    ! initialise biases
    !---------------------------------------------------------------------------
    call this%bias_init%initialise( &
         this%params(2)%val(:,1), &
         fan_in = product(this%knl)+1, fan_out = 1 &
    )


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program( &
            "Graph input not supported for convolutional layer" &
       )
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate( this%output(1,1) )
    if(this%z(1)%allocated) call this%z(1)%deallocate()
    if(this%z(2)%allocated) call this%z(2)%deallocate()

  end subroutine init_conv
!###############################################################################


!###############################################################################
  module subroutine init_batch(this, input_shape, verbose)
    !! Initialise batch normalisation layer
    use athena__initialiser, only: initialiser_setup
    use athena__misc_types, only: base_init_type
    implicit none

    ! Arguments
    class(batch_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    integer :: verbose_ = 0


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! set up number of channels, width, height
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output_shape(this%input_rank))
    if(size(this%input_shape).eq.1)then
       this%output_shape(1) = this%input_shape(1)
       this%output_shape(2) = 1
    else
       this%output_shape = this%input_shape
    end if
    this%num_channels = this%input_shape(this%input_rank)
    this%num_params = this%get_num_params()
    allocate(this%params(1))
    call this%params(1)%allocate([2 * this%num_channels, 1])
    call this%params(1)%set_requires_grad(.true.)
    allocate(this%weight_shape(1,1))
    this%weight_shape(:,1) = [ this%num_channels ]
    this%bias_shape = [this%num_channels]


    !---------------------------------------------------------------------------
    ! allocate mean and variance
    !---------------------------------------------------------------------------
    allocate(this%mean(this%num_channels), source=0._real32)
    allocate(this%variance, source=this%mean)


    !---------------------------------------------------------------------------
    ! initialise gamma
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise(this%params(1)%val(1:this%num_channels,1), &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)

    ! initialise beta
    !---------------------------------------------------------------------------
    call this%bias_init%initialise(this%params(1)%val(this%num_channels+1:,1), &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)


    !---------------------------------------------------------------------------
    ! initialise moving mean
    !---------------------------------------------------------------------------
    call this%moving_mean_init%initialise(this%mean, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)

    ! initialise moving variance
    !---------------------------------------------------------------------------
    call this%moving_variance_init%initialise(this%variance, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program( &
            "Graph input not supported for batch normalisation layer" &
       )
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate( batchnorm_array_type :: this%output(1,1) )

  end subroutine init_batch
!###############################################################################

end submodule athena__base_layer_submodule_init
