!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module input1d_layer
  use constants, only: real12
  use base_layer, only: input_layer_type
  implicit none
  
  
  type, extends(input_layer_type) :: input1d_layer_type
     real(real12), allocatable, dimension(:) :: output

   contains
     procedure, pass(this) :: init => init_input1d
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, pass(this) :: set => set_input1d
  end type input1d_layer_type

  interface input1d_layer_type
     module function layer_setup(input_shape) result(layer)
       integer, dimension(:), optional, intent(in) :: input_shape
       type(input1d_layer_type) :: layer
     end function layer_setup
  end interface input1d_layer_type

  
  private
  public :: input1d_layer_type


contains

!!!#############################################################################
!!! forward propagation assumed rank handler
!!! placeholder to satisfy deferred
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(input1d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    return
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!! placeholder to satisfy deferred
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(input1d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient
    return
  end subroutine backward_rank
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up layer
!!!#############################################################################
  module function layer_setup(input_shape) result(layer)
    implicit none
    integer, dimension(:), optional, intent(in) :: input_shape

    type(input1d_layer_type) :: layer


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_input1d(this, input_shape)
    implicit none
    class(input1d_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape


    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(size(input_shape,dim=1).eq.3)then
       this%input_shape = input_shape
       this%output_shape = input_shape
    else
       stop "ERROR: invalid size of input_shape in input3d, expected (3)"
    end if
    
    this%num_outputs = product(input_shape)

    allocate(this%output(input_shape(1)))

  end subroutine init_input1d
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set input layer values
!!!#############################################################################
  pure subroutine set_input1d(this, input)
    implicit none
    class(input1d_layer_type), intent(inout) :: this
    real(real12), dimension(this%num_outputs), intent(in) :: input

    this%output = reshape(input, shape=shape(this%output))
  end subroutine set_input1d
!!!#############################################################################


end module input1d_layer
!!!#############################################################################
