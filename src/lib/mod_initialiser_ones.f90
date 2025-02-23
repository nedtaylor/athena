module athena__initialiser_ones
  !! Module containing the implementation of the Ones initialiser
  !!
  !! This module contains the implementation of the Ones initialiser
  !! for the weights and biases of a layer
  use athena__constants, only: real32
  use athena__misc_types, only: initialiser_type
  implicit none


  private

  public :: ones


  type, extends(initialiser_type) :: ones_type
     !! Type for the Ones initialiser
   contains
     procedure, pass(this) :: initialise => ones_initialise
     !! Initialise the weights and biases using the Ones distribution
  end type ones_type

  type(ones_type) :: ones
  !! Ones initialiser object



contains

!###############################################################################
  pure subroutine ones_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the Ones distribution
    implicit none

    ! Arguments
    class(ones_type), intent(inout) :: this
    !! Instance of the Ones initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    select rank(input)
    rank(0)
       input = 1._real32
    rank(1)
       input = 1._real32
    rank(2)
       input = 1._real32
    rank(3)
       input = 1._real32
    rank(4)
       input = 1._real32
    rank(5)
       input = 1._real32
    rank(6)
       input = 1._real32
    end select

  end subroutine ones_initialise
!###############################################################################

end module athena__initialiser_ones
