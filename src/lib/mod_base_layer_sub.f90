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
  pure module function get_params_batch(this) result(params)
    implicit none
    class(batch_layer_type), intent(in) :: this
    real(real32), dimension(this%num_params) :: params
  
    params = [this%gamma, this%beta]
  
  end function get_params_batch
!!!#############################################################################


!!!#############################################################################
!!! set learnable parameters of layer
!!!#############################################################################
  module subroutine set_params_batch(this, params)
    implicit none
    class(batch_layer_type), intent(inout) :: this
    real(real32), dimension(this%num_params), intent(in) :: params
  
    this%gamma = params(1:this%num_channels)
    this%beta  = params(this%num_channels+1:2*this%num_channels)
  
  end subroutine set_params_batch
!!!#############################################################################


!!!#############################################################################
!!! get gradients of layer
!!!#############################################################################
  pure module function get_gradients_batch(this, clip_method) result(gradients)
    use clipper, only: clip_type
    implicit none
    class(batch_layer_type), intent(in) :: this
    type(clip_type), optional, intent(in) :: clip_method
    real(real32), dimension(this%num_params) :: gradients
  
    gradients = [this%dg/this%batch_size, this%db/this%batch_size]
  
    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)

  end function get_gradients_batch
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
       this%dg = gradients * this%batch_size
       this%db = gradients * this%batch_size
    rank(1)
        this%dg = gradients(:this%batch_size) * this%batch_size
        this%db = gradients(this%batch_size+1:) * this%batch_size
    end select
  
  end subroutine set_gradients_batch
!!!#############################################################################

end submodule base_layer_submodule
!!!#############################################################################
