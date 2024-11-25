!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! definition of the abstract base layer type, from which all other layers ...
!!! ... are derived
!!! module includes the following public abstract types:
!!! base_layer_type      - abstract type for all layers
!!! pool_layer_type      - abstract type for spatial pooling layers
!!! drop_layer_type      - abstract type for dropout layers
!!! learnable_layer_type - abstract type for layers with learnable parameters
!!! conv_layer_type      - abstract type for spatial convolutional layers
!!! batch_layer_type     - abstract type for batch normalisation layers
!!!##################
!!! base_layer_type includes the following procedures:
!!! set_shape            - set the input shape of the layer
!!! get_num_params       - get the number of parameters in the layer
!!! print                - print the layer to a file
!!! get_output           - get the output of the layer
!!! init                 - initialise the layer
!!! set_batch_size       - set the batch size of the layer
!!! forward              - forward pass of layer
!!! backward             - backward pass of layer
!!!##################
!!! learnable_layer_type includes the following unique procedures:
!!! layer_reduction      - reduce the layer to a single value
!!! layer_merge          - merge the layer with another layer
!!! get_params           - get the learnable parameters of the layer
!!! set_params           - set the learnable parameters of the layer
!!! get_gradients        - get the gradients of the layer
!!! set_gradients        - set the gradients of the layer
!!!#############################################################################
submodule(base_layer) base_layer_submodule
  implicit none

contains

!!!#############################################################################
!!! print layer to file (do nothing for a base layer)
!!!#############################################################################
!!! this = (T, in) base_layer_type
!!! file = (I, in) file name
  module subroutine print_base(this, file)
    implicit none
    class(base_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    !! NO NEED TO WRITE ANYTHING FOR A DEFAULT LAYER
    return
  end subroutine print_base
!!!#############################################################################


!!!#############################################################################
!!! setup input layer shape
!!!#############################################################################
!!! this        = (T, inout) base_layer_type
!!! input_shape = (I, in) input shape
  module subroutine set_shape_base(this, input_shape)
    implicit none
    class(base_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    character(len=100) :: err_msg

    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(size(input_shape,dim=1).eq.this%input_rank)then
       this%input_shape = input_shape
    else
       write(err_msg,'("ERROR: invalid size of input_shape in ",A,&
            &" expected (",I0,"), got (",I0")")')  &
            trim(this%name), this%input_rank, size(input_shape,dim=1)
       stop trim(err_msg)
    end if
 
  end subroutine set_shape_base
!!!#############################################################################


!!!#############################################################################
!!! get layer outputs
!!!#############################################################################
  pure subroutine get_output_base(this, output)
    implicit none
    class(base_layer_type), intent(in) :: this
    real(real32), allocatable, dimension(..), intent(out) :: output
  
    call this%output%get(output)
  end subroutine get_output_base
!!!#############################################################################


!!!#############################################################################
!!! set the pointers of the layer
!!!#############################################################################
  module subroutine set_ptrs(this)
    implicit none
    class(base_layer_type), intent(inout), target :: this

    if(allocated(this%output)) call this%output%set_ptr()
    if(allocated(this%di)) call this%di%set_ptr()

    call this%set_ptrs_hyperparams()

  end subroutine set_ptrs
!!!-----------------------------------------------------------------------------
  module subroutine set_ptrs_hyperparams(this)
    implicit none
    class(base_layer_type), intent(inout), target :: this
  end subroutine set_ptrs_hyperparams
!!!#############################################################################


!!!#############################################################################
!!! get number of parameters in layer
!!!#############################################################################
!!! this       = (T, in) layer_type
!!! num_params = (I, out) number of parameters
  pure module function get_num_params_base(this) result(num_params)
    implicit none
    class(base_layer_type), intent(in) :: this
    integer :: num_params
    
    !! NO PARAMETERS IN A BASE LAYER
    num_params = 0

  end function get_num_params_base
!!!-----------------------------------------------------------------------------
  pure module function get_num_params_conv(this) result(num_params)
    implicit none
    class(conv_layer_type), intent(in) :: this
    integer :: num_params
    
    !! num_filters x num_channels x kernel_size + num_biases
    !! num_biases = num_filters
    num_params = this%num_filters * this%num_channels * product(this%knl) + &
         this%num_filters

  end function get_num_params_conv
!!!-----------------------------------------------------------------------------
  pure module function get_num_params_batch(this) result(num_params)
    implicit none
    class(batch_layer_type), intent(in) :: this
    integer :: num_params
    
    !! num_filters x num_channels x kernel_size + num_biases
    !! num_biases = num_filters
    num_params = 2 * this%num_channels

  end function get_num_params_batch
!!!#############################################################################


!!!#############################################################################
!!! get learnable parameters of layer
!!!#############################################################################
  pure module function get_params(this) result(params)
  implicit none
  class(learnable_layer_type), intent(in) :: this
  real(real32), dimension(this%num_params) :: params

  params = this%params

end function get_params
!!!#############################################################################


!!!#############################################################################
!!! set learnable parameters of layer
!!!#############################################################################
module subroutine set_params(this, params)
  implicit none
  class(learnable_layer_type), intent(inout) :: this
  real(real32), dimension(this%num_params), intent(in) :: params

  this%params = params

end subroutine set_params
!!!#############################################################################


!!!#############################################################################
!!! get gradients of layer
!!!#############################################################################
  pure module function get_gradients(this, clip_method) result(gradients)
    use clipper, only: clip_type
    implicit none
    class(learnable_layer_type), intent(in) :: this
    type(clip_type), optional, intent(in) :: clip_method
    real(real32), dimension(this%num_params) :: gradients
  
    gradients = [ sum(this%dp, dim=2) / this%batch_size, &
         sum(this%db, dim=2) / this%batch_size ]
  
    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)

  end function get_gradients
!!!#############################################################################


!!!#############################################################################
!!! set gradients of layer
!!!#############################################################################
  module subroutine set_gradients(this, gradients)
    implicit none
    class(learnable_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: gradients
  
    select rank(gradients)
    rank(0)
       this%dp = gradients
       this%db = gradients
    rank(1)
       this%dp = spread(gradients(1:this%num_params - size(this%db,1)), 2, this%batch_size)
       this%db = spread(gradients(this%num_params - size(this%db,1) + 1:), 2, this%batch_size)
    end select
  
  end subroutine set_gradients
!!!#############################################################################


!!!#############################################################################
!!! set gradients of layer
!!!#############################################################################
  module subroutine set_gradients_batch(this, gradients)
    implicit none
    class(batch_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: gradients
  
    select rank(gradients)
    rank(0)
       this%dp = gradients * this%batch_size
       this%db = gradients * this%batch_size
    rank(1)
        this%dp(:,1) = gradients(:this%num_channels) * this%batch_size
        this%db(:,1) = gradients(this%num_channels+1:) * this%batch_size
    end select
  
  end subroutine set_gradients_batch
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  module subroutine init_conv(this, input_shape, batch_size, verbose)
    use initialiser, only: initialiser_setup
    use custom_types, only: initialiser_type
    implicit none
    class(conv_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0
    class(initialiser_type), allocatable :: initialiser_


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !!--------------------------------------------------------------------------
    !! allocate output, activation, bias, and weight shapes
    !!--------------------------------------------------------------------------
    !! NOTE: INPUT SHAPE DOES NOT INCLUDE PADDING WIDTH
    !! THIS IS HANDLED AUTOMATICALLY BY THE CODE
    !! ... provide the initial input data shape and let us deal with the padding
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output))then
       if(this%output%allocated) call this%output%deallocate()
    end if
    this%output%shape(this%input_rank) = this%num_filters
    this%output%shape(:this%input_rank-1) = floor( &
         ( &
              this%input_shape(:this%input_rank-1) + &
              2.0 * this%pad - &
              this%knl &
         ) / real(this%stp) &
    ) + 1
    this%num_params = this%get_num_params()
    allocate(this%params(this%num_params), source=0._real32)


    !!--------------------------------------------------------------------------
    !! initialise weights (kernels)
    !!--------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%kernel_initialiser))
    call initialiser_%initialise( &
         this%params(:this%num_params-this%num_filters), &
         fan_in=product(this%knl)+1, fan_out=1 &
    )
    deallocate(initialiser_)

    !! initialise biases
    !!--------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%bias_initialiser))
    call initialiser_%initialise( &
         this%params(this%num_params-this%num_filters+1:), &
         fan_in=product(this%knl)+1, fan_out=1 &
    )
    deallocate(initialiser_)


    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_conv
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  module subroutine init_batch(this, input_shape, batch_size, verbose)
    use initialiser, only: initialiser_setup
    use custom_types, only: initialiser_type
    implicit none
    class(batch_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0
    class(initialiser_type), allocatable :: t_initialiser


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !!--------------------------------------------------------------------------
    !! set up number of channels, width, height
    !!--------------------------------------------------------------------------
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


    !!--------------------------------------------------------------------------
    !! allocate mean, variance, gamma, beta, dg, db
    !!--------------------------------------------------------------------------
    allocate(this%mean(this%num_channels), source=0._real32)
    allocate(this%variance, source=this%mean)
    ! allocate(this%gamma, source=this%mean)
    ! allocate(this%beta, source=this%mean)


    !!--------------------------------------------------------------------------
    !! initialise gamma
    !!--------------------------------------------------------------------------
    allocate(t_initialiser, source=initialiser_setup(this%kernel_initialiser))
    t_initialiser%mean = this%gamma_init_mean
    t_initialiser%std  = this%gamma_init_std
    call t_initialiser%initialise(this%params(1:this%num_channels), &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)

    !! initialise beta
    !!--------------------------------------------------------------------------
    allocate(t_initialiser, source=initialiser_setup(this%bias_initialiser))
    t_initialiser%mean = this%beta_init_mean
    t_initialiser%std  = this%beta_init_std
    call t_initialiser%initialise(this%params(this%num_channels+1:), &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)


    !!--------------------------------------------------------------------------
    !! initialise moving mean
    !!--------------------------------------------------------------------------
    allocate(t_initialiser, &
         source=initialiser_setup(this%moving_mean_initialiser))
    call t_initialiser%initialise(this%mean, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)

    !! initialise moving variance
    !!--------------------------------------------------------------------------
    allocate(t_initialiser, &
         source=initialiser_setup(this%moving_variance_initialiser))
    call t_initialiser%initialise(this%variance, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)


    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_batch
!!!#############################################################################


!!!#############################################################################
!!! set the pointers of the layer
!!!#############################################################################
  module subroutine set_ptrs_hyperparams_batch(this)
    implicit none
    class(batch_layer_type), intent(inout), target :: this

    if(allocated(this%params))then
       this%gamma(1:this%num_channels) => this%params(1:this%num_channels)
       this%beta(1:this%num_channels) => &
            this%params(this%num_channels+1:this%num_channels*2)
    end if

  end subroutine set_ptrs_hyperparams_batch
!!!#############################################################################

end submodule base_layer_submodule
!!!#############################################################################
