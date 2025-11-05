module athena__initialiser_zeros
  !! Module containing the implementation of the Zeros initialiser
  !!
  !! This module contains the implementation of the Zeros initialiser
  !! for the weights and biases of a layer
  use athena__constants, only: real32
  use athena__misc_types, only: initialiser_type
  implicit none


  private

  public :: zeros_type


  type, extends(initialiser_type) :: zeros_type
     !! Type for the Zeros initialiser
   contains
     procedure, pass(this) :: initialise => zeros_initialise
     !! Initialise the weights and biases using the Zeros distribution
  end type zeros_type


  interface zeros_type
     module function initialiser_zeros_setup() result(initialiser)
       !! Interface for the Zeros initialiser
       type(zeros_type) :: initialiser
       !! Zeros initialiser object
     end function initialiser_zeros_setup
  end interface zeros_type



contains

!###############################################################################
  module function initialiser_zeros_setup() result(initialiser)
    !! Interface for the Zeros initialiser
    implicit none

    type(zeros_type) :: initialiser
    !! Zeros initialiser object

    initialiser%name = "zeros"

  end function initialiser_zeros_setup
!###############################################################################


!###############################################################################
  pure subroutine zeros_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the Zeros distribution
    implicit none

    ! Arguments
    class(zeros_type), intent(inout) :: this
    !! Instance of the Zeros initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    select rank(input)
    rank(0)
       input = 0._real32
    rank(1)
       input = 0._real32
    rank(2)
       input = 0._real32
    rank(3)
       input = 0._real32
    rank(4)
       input = 0._real32
    rank(5)
       input = 0._real32
    rank(6)
       input = 0._real32
    end select

  end subroutine zeros_initialise
!###############################################################################

end module athena__initialiser_zeros
