!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module flatten3d_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  implicit none
  
  
  type, extends(base_layer_type) :: flatten3d_layer_type
     integer :: num_outputs, num_addit_outputs = 0
     real(real12), allocatable, dimension(:) :: output
     real(real12), allocatable, dimension(:,:,:,:) :: di
   contains
     procedure, pass(this) :: init => init_flatten3d
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
  end type flatten3d_layer_type

  interface flatten3d_layer_type
     module function layer_setup(input_shape) result(layer)
       integer, dimension(:), optional, intent(in) :: input_shape
       type(flatten3d_layer_type) :: layer
     end function layer_setup
  end interface flatten3d_layer_type

  
  private
  public :: flatten3d_layer_type


contains

!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(flatten3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select rank(input); rank(4)
       this%output(:this%num_outputs) = reshape(input, [this%num_outputs])
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(flatten3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(gradient); rank(1)
       this%di = reshape(gradient(:this%num_outputs), shape(this%di))
    end select
  end subroutine backward_rank
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up layer
!!!#############################################################################
  module function layer_setup(input_shape, num_addit_outputs) result(layer)
    implicit none
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: num_addit_outputs

    type(flatten3d_layer_type) :: layer


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(num_addit_outputs)) layer%num_addit_outputs = num_addit_outputs
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_flatten3d(this, input_shape, verbose)
    implicit none
    class(flatten3d_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: verbose

    integer :: t_verb


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose))then
       t_verb = verbose
    else
       t_verb = 0
    end if


    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(size(input_shape,dim=1).eq.4)then
       this%input_shape = input_shape
    else
       stop "ERROR: invalid size of input_shape in flatten3d, expected (4)"
    end if
    
    this%num_outputs = product(this%input_shape)

    allocate(this%output(this%num_outputs + this%num_addit_outputs), &
         source=0._real12)
    allocate(this%di(&
         input_shape(1), input_shape(2), &
         input_shape(3), input_shape(4)), &
         source=0._real12)

  end subroutine init_flatten3d
!!!#############################################################################

end module flatten3d_layer
!!!#############################################################################
