module athena__activation
  !! Module containing the activation function setup
  use coreutils, only: stop_program, to_lower
  use athena__misc_types, only: activation_type
  use athena__activation_gaussian, only: gaussian_actv_type
  use athena__activation_linear, only: linear_actv_type
  use athena__activation_piecewise, only: piecewise_actv_type
  use athena__activation_relu, only: relu_actv_type
  use athena__activation_leaky_relu, only: leaky_relu_actv_type
  use athena__activation_sigmoid, only: sigmoid_actv_type
  use athena__activation_softmax, only: softmax_actv_type
  use athena__activation_swish, only: swish_actv_type
  use athena__activation_tanh, only: tanh_actv_type
  use athena__activation_none, only: none_actv_type
  implicit none


  private

  public :: activation_setup



contains

!###############################################################################
  function activation_setup(input, error) result(activation)
    !! Setup the desired activation function
    implicit none

    ! Arguments
    class(*), intent(in) :: input
    !! Name of the activation function or activation object
    class(activation_type), allocatable :: activation
    !! Activation function object
    integer, optional, intent(out) :: error
    !! Error code

    ! Local variables
    character(256) :: err_msg
    !! Error message


    !---------------------------------------------------------------------------
    ! select desired activation function
    !---------------------------------------------------------------------------
    select type(input)
    class is(activation_type)
       activation = input
    type is(character(*))
       select case(trim(to_lower(input)))
       case("gaussian")
          activation = gaussian_actv_type()
       case ("linear")
          activation = linear_actv_type()
       case ("piecewise")
          activation = piecewise_actv_type()
       case ("relu")
          activation = relu_actv_type()
       case ("leaky_relu")
          activation = leaky_relu_actv_type()
       case ("sigmoid")
          activation = sigmoid_actv_type()
       case ("softmax")
          activation = softmax_actv_type()
       case("swish")
          activation = swish_actv_type()
       case ("tanh")
          activation = tanh_actv_type()
       case ("none")
          activation = none_actv_type()
       case default
          if(present(error))then
             error = -1
             return
          else
             write(err_msg,'("Incorrect activation name given ''",A,"''")') &
                  trim(to_lower(input))
             call stop_program(trim(err_msg))
             return
          end if
       end select
    class default
       if(present(error))then
          error = -1
          return
       else
          write(err_msg,'("Unknown input type given for activation setup")')
          call stop_program(trim(err_msg))
          return
       end if
    end select

  end function activation_setup
!###############################################################################

end module athena__activation
