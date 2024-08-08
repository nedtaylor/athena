!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module to setup the activation function
!!! module includes the following procedures:
!!! - activation_setup - set up the activation function
!!!#############################################################################
module activation
  use constants, only: real32
  use misc, only: to_lower
  use custom_types, only: activation_type
  use activation_gaussian, only: gaussian_setup
  use activation_linear, only: linear_setup
  use activation_piecewise, only: piecewise_setup
  use activation_relu, only: relu_setup
  use activation_leaky_relu, only: leaky_relu_setup
  use activation_sigmoid, only: sigmoid_setup
  use activation_softmax, only: softmax_setup
  use activation_tanh, only: tanh_setup
  use activation_none, only: none_setup
  implicit none


  private

  public :: activation_setup


contains

!!!#############################################################################
!!! function to setup the activation function
!!!#############################################################################
  pure function activation_setup(name, scale) result(transfer)
    implicit none
    real(real32), optional, intent(in) :: scale
    class(activation_type), allocatable :: transfer
    character(*), intent(in) :: name

    real(real32) :: scale_


    !!--------------------------------------------------------------------------
    !! set defaults if not present
    !!--------------------------------------------------------------------------
    if(present(scale))then
       scale_ = scale
    else
       scale_ = 1._real32
    end if


    !!--------------------------------------------------------------------------
    !! select desired activation function
    !!--------------------------------------------------------------------------
    select case(trim(to_lower(name)))
    case("gaussian")
       transfer = gaussian_setup(scale = scale_)
    case ("linear")
       transfer = linear_setup(scale = scale_)
    case ("piecewise")
       transfer = piecewise_setup(scale = scale_)
    case ("relu")
       transfer = relu_setup(scale = scale_)
    case ("leaky_relu")
       transfer = leaky_relu_setup(scale = scale_)
    case ("sigmoid")
       transfer = sigmoid_setup(scale = scale_)
    case ("softmax")
       transfer = softmax_setup(scale = scale_)
    case ("tanh")
       transfer = tanh_setup(scale = scale_)
    case ("none")
       transfer = none_setup(scale = scale_)
    case default
       transfer = none_setup(scale = scale_)
    end select

  end function activation_setup
!!!#############################################################################

end module activation
!!!#############################################################################
