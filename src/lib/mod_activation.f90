!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation
  use constants, only: real12
  use misc, only: to_lower
  use custom_types, only: activation_type
  use activation_gaussian, only: gaussian_setup
  use activation_linear, only: linear_setup
  use activation_piecewise, only: piecewise_setup
  use activation_relu, only: relu_setup
  use activation_leaky_relu, only: leaky_relu_setup
  use activation_sigmoid, only: sigmoid_setup
  use activation_tanh, only: tanh_setup
  use activation_none, only: none_setup
  implicit none


  private

  public :: activation_setup

!!!!! ALSO, HAVE THE CV, PL, FC, etc, LAYERS AS CLASSES
!!!!! ... they may be able to be appended on to each other

  
  !! make an initialiser that takes in an assumed rank
  !! it then does product(shape(weight)) OR size(weight)
  !! could always use select rank(x) statement if needed
  !! https://keras.io/api/layers/initializers/
  
  

contains

!!!#############################################################################
!!! 
!!!#############################################################################
  function activation_setup(name, scale) result(transfer)
    implicit none
    real(real12), optional, intent(in) :: scale
    class(activation_type), allocatable :: transfer
    character(*), intent(in) :: name

    real(real12) :: t_scale

    if(present(scale))then
       t_scale = scale
    else
       t_scale = 1._real12
    end if

    
    select case(trim(to_lower(name)))
    case("gaussian")
       transfer = gaussian_setup(scale = t_scale)
    case ("linear")
       transfer = linear_setup(scale = t_scale)
    case ("piecewise")
       transfer = piecewise_setup(scale = t_scale)
    case ("relu")
       transfer = relu_setup(scale = t_scale)
    case ("leaky_relu")
       transfer = leaky_relu_setup(scale = t_scale)
    case ("sigmoid")
       transfer = sigmoid_setup(scale = t_scale)
    case ("tanh")
       transfer = tanh_setup(scale = t_scale)
    case ("none")
       transfer = none_setup(scale = t_scale)
    case default
       transfer = none_setup(scale = t_scale)
    end select

  end function activation_setup
!!!#############################################################################

end module activation
!!!#############################################################################
