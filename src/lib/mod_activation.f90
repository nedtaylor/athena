module athena__activation
  !! Module containing the activation function setup
  use coreutils, only: stop_program, to_lower
  use athena__misc_types, only: activation_type
  use athena__activation_gaussian, only: gaussian_setup
  use athena__activation_linear, only: linear_setup
  use athena__activation_piecewise, only: piecewise_setup
  use athena__activation_relu, only: relu_setup
  use athena__activation_leaky_relu, only: leaky_relu_setup
  use athena__activation_sigmoid, only: sigmoid_setup
  use athena__activation_softmax, only: softmax_setup
  use athena__activation_swish, only: swish_setup
  use athena__activation_tanh, only: tanh_setup
  use athena__activation_none, only: none_setup
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
          activation = gaussian_setup()
       case ("linear")
          activation = linear_setup()
       case ("piecewise")
          activation = piecewise_setup()
       case ("relu")
          activation = relu_setup()
       case ("leaky_relu")
          activation = leaky_relu_setup()
       case ("sigmoid")
          activation = sigmoid_setup()
       case ("softmax")
          activation = softmax_setup()
       case("swish")
          activation = swish_setup()
       case ("tanh")
          activation = tanh_setup()
       case ("none")
          activation = none_setup()
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
