!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module base_layer
  use constants, only: real12
  use optimiser, only: optimiser_type
  use custom_types, only: activation_type
  implicit none

!!!------------------------------------------------------------------------
!!! layer abstract type
!!!------------------------------------------------------------------------
  type, abstract :: base_layer_type !! give it parameterised values?
     integer :: batch_size = 0
     integer :: input_rank = 0
     character(:), allocatable :: name
     integer, allocatable, dimension(:) :: input_shape, output_shape
  contains
     procedure, pass(this) :: set_shape => set_shape_base
     procedure, pass(this) :: get_num_params => get_num_params_base
     procedure, pass(this) :: print => print_base
     procedure(initialise), deferred, pass(this) :: init
     procedure(set_batch_size), deferred, pass(this) :: set_batch_size
     procedure(forward), deferred, pass(this) :: forward
     procedure(backward), deferred, pass(this) :: backward
     !! NO NEED FOR DEFERRED PRODECURES
     !! instead, make this a generic type that just has a set of interfaces for (module) procedures that call 1D, 3D, and 4D forms
     !! Use subroutines because output data is trickier for function tricker to handle
     !! Use a general train subroutine that is called by the main model, which internally goes through forward and backward passes
     !! Input sizes have to be 1D, 3D, or 4D (any 2D data is simply 3D with num_channels=1)
     !! Output sizes defined by user
     !! For every forward, just pass in the whole previous layer container
     !! ... reverse for backward
     !! In each layer container, you know what size you are expecting for the input, so just take that based on a select type (or of a previous?)
  end type base_layer_type

  abstract interface
     subroutine initialise(this, input_shape, batch_size, verbose)
       import :: base_layer_type
       class(base_layer_type), intent(inout) :: this
       integer, dimension(:), intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
     end subroutine initialise

    subroutine set_batch_size(this, batch_size, verbose)
      import :: base_layer_type
      class(base_layer_type), intent(inout) :: this
      integer, intent(in) :: batch_size
      integer, optional, intent(in) :: verbose
    end subroutine set_batch_size
  end interface

  abstract interface
    pure function get_num_params(this) result(num_params)
      import :: base_layer_type
      class(base_layer_type), intent(in) :: this
      integer :: num_params
    end function get_num_params
  end interface

  abstract interface
     pure subroutine forward(this, input)
       import :: base_layer_type, real12
       class(base_layer_type), intent(inout) :: this
       real(real12), dimension(..), intent(in) :: input
     end subroutine forward

     pure subroutine backward(this, input, gradient)
       import :: base_layer_type, real12
       class(base_layer_type), intent(inout) :: this
       real(real12), dimension(..), intent(in) :: input
       real(real12), dimension(..), intent(in) :: gradient
     end subroutine backward
  end interface


!!!-----------------------------------------------------------------------------
!!! input derived extended type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(base_layer_type) :: input_layer_type
     integer :: num_outputs
   contains
     procedure(set), deferred, pass(this) :: set
  end type input_layer_type

  abstract interface
     pure subroutine set(this, input)
       import :: input_layer_type, real12
       class(input_layer_type), intent(inout) :: this
       real(real12), dimension(this%num_outputs), intent(in) :: input
     end subroutine set
  end interface


!!!-----------------------------------------------------------------------------
!!! flatten derived extended type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(base_layer_type) :: flatten_layer_type
     integer :: num_outputs, num_addit_outputs = 0
     real(real12), allocatable, dimension(:,:) :: output
  end type flatten_layer_type


!!!-----------------------------------------------------------------------------
!!! dropout derived extended type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(base_layer_type) :: drop_layer_type
   contains
     procedure(generate_mask), deferred, pass(this) :: generate_mask
  end type drop_layer_type

  abstract interface
     subroutine generate_mask(this)
       import :: drop_layer_type
       class(drop_layer_type), intent(inout) :: this
     end subroutine generate_mask
  end interface


!!!-----------------------------------------------------------------------------
!!! learnable derived extended type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(base_layer_type) :: learnable_layer_type
     character(len=14) :: kernel_initialiser='', bias_initialiser=''
     class(activation_type), allocatable :: transfer
   contains
     procedure(update), deferred, pass(this) :: update
     procedure(layer_reduction), deferred, pass(this) :: reduce
     procedure(layer_merge), deferred, pass(this) :: merge
     procedure(get_params), deferred, pass(this) :: get_params
     procedure(set_params), deferred, pass(this) :: set_params
  end type learnable_layer_type

  abstract interface
     pure subroutine update(this, method)
       import :: learnable_layer_type, optimiser_type
       class(learnable_layer_type), intent(inout) :: this
       type(optimiser_type), intent(in) :: method
     end subroutine update
  end interface

  abstract interface
     subroutine layer_reduction(this, rhs)
       import :: learnable_layer_type
       class(learnable_layer_type), intent(inout) :: this
       class(learnable_layer_type), intent(in) :: rhs
     end subroutine layer_reduction

     subroutine layer_merge(this, input)
       import :: learnable_layer_type
       class(learnable_layer_type), intent(inout) :: this
       class(learnable_layer_type), intent(in) :: input
     end subroutine layer_merge
  end interface

  abstract interface
     pure function get_params(this) result(params)
       import :: learnable_layer_type, real12
       class(learnable_layer_type), intent(in) :: this
       real(real12), allocatable, dimension(:) :: params
     end function get_params

     subroutine set_params(this, params)
       import :: learnable_layer_type, real12
       class(learnable_layer_type), intent(inout) :: this
       real(real12), dimension(:), intent(in) :: params
    end subroutine set_params
  end interface

  !!!-----------------------------------------------------------------------------
  !!! convolution extended derived type
  !!!-----------------------------------------------------------------------------
    type, abstract, extends(learnable_layer_type) :: conv_layer_type
       !! knl = kernel
       !! stp = stride (step)
       !! hlf = half
       !! pad = pad
       !! cen = centre
       !! output_shape = dimension (height, width, depth)
       logical :: calc_input_gradients = .true.
       integer :: num_channels
       integer :: num_filters
       integer, allocatable, dimension(:) :: knl, stp, hlf, pad, cen
       real(real12), allocatable, dimension(:) :: bias
       real(real12), allocatable, dimension(:,:) :: db  ! bias gradient
     contains
       procedure, pass(this) :: get_num_params => get_num_params_conv
    end type conv_layer_type


!!!-----------------------------------------------------------------------------
!!! batch extended derived type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(learnable_layer_type) :: batch_layer_type
     !! gamma = scale factor (learnable)
     !! beta = shift factor (learnable)
     !! dg = gradient of gamma
     !! db = gradient of beta
     !! mean = mean of each feature (not learnable)
     !! variance = variance of each feature (not learnable)
     !! NOTE: if momentum = 0, mean and variance batch-dependent values
     !! NOTE: if momentum > 0, mean and variance are running averages
     !! NED: NEED TO KEEP TRACK OF EXPONENTIAL MOVING AVERAGE (EMA)
     !!   ... FOR INFERENCE
     logical :: inference = .false.
     integer :: num_channels
     real(real12) :: norm, momentum = 0._real12
     real(real12) :: epsilon = 1.E-5_real12
     real(real12) :: gamma_init_mean = 1._real12, gamma_init_std = 0.01_real12
     real(real12) :: beta_init_mean  = 0._real12, beta_init_std  = 0.01_real12
     character(len=14) :: moving_mean_initialiser='', &
          moving_variance_initialiser=''
     real(real12), allocatable, dimension(:) :: mean, variance !! not learnable
     real(real12), allocatable, dimension(:) :: gamma_incr, beta_incr !! not learnable
     real(real12), allocatable, dimension(:) :: gamma, beta !! learnable
     real(real12), allocatable, dimension(:) :: dg, db !! learnable
   contains
     procedure, pass(this) :: get_num_params => get_num_params_batch
     procedure, pass(this) :: get_params => get_params_batch
     procedure, pass(this) :: set_params => set_params_batch
  end type batch_layer_type
  

  private

  public :: base_layer_type
  public :: input_layer_type
  public :: flatten_layer_type
  public :: drop_layer_type
  public :: learnable_layer_type
  public :: conv_layer_type
  public :: batch_layer_type


contains

!!!#############################################################################
!!! print layer to file (do nothing for a base layer)
!!!#############################################################################
  subroutine print_base(this, file)
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
  subroutine set_shape_base(this, input_shape)
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
!!! get number of parameters in layer
!!!#############################################################################
  pure function get_num_params_base(this) result(num_params)
    implicit none
    class(base_layer_type), intent(in) :: this
    integer :: num_params
    
    !! NO PARAMETERS IN A BASE LAYER
    num_params = 0

  end function get_num_params_base
!!!-----------------------------------------------------------------------------
  pure function get_num_params_conv(this) result(num_params)
    implicit none
    class(conv_layer_type), intent(in) :: this
    integer :: num_params
    
    !! num_filters x num_channels x kernel_size + num_biases
    !! num_biases = num_filters
    num_params = this%num_filters * this%num_channels * product(this%knl) + &
         this%num_filters

  end function get_num_params_conv
!!!-----------------------------------------------------------------------------
  pure function get_num_params_batch(this) result(num_params)
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
pure function get_params_batch(this) result(params)
  implicit none
  class(batch_layer_type), intent(in) :: this
  real(real12), allocatable, dimension(:) :: params

  params = [this%gamma, this%beta]

end function get_params_batch
!!!#############################################################################


!!!#############################################################################
!!! set learnable parameters of layer
!!!#############################################################################
subroutine set_params_batch(this, params)
  implicit none
  class(batch_layer_type), intent(inout) :: this
  real(real12), dimension(:), intent(in) :: params

  this%gamma = params(1:this%num_channels)
  this%beta  = params(this%num_channels+1:2*this%num_channels)

end subroutine set_params_batch
!!!#############################################################################

end module base_layer
!!!#############################################################################
