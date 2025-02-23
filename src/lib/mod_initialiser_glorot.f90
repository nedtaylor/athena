module athena__initialiser_glorot
  !! Module containing the implementation of the Glorot initialiser
  !!
  !! This module contains the implementation of the Glorot initialiser
  !! for the weights and biases of a layer
  !! The Glorot initialiser is also known as the Xavier initialiser
  !! Reference: https://proceedings.mlr.press/v9/glorot10a.html
  use athena__constants, only: real32, pi
  use athena__misc_types, only: initialiser_type
  implicit none


  private

  public :: glorot_uniform
  public :: glorot_normal


  type, extends(initialiser_type) :: glorot_uniform_type
     !! Type for the Glorot initialiser (uniform)
   contains
     procedure, pass(this) :: initialise => glorot_uniform_initialise
     !! Initialise the weights and biases using the Glorot uniform distribution
  end type glorot_uniform_type

  type, extends(initialiser_type) :: glorot_normal_type
     !! Type for the Glorot initialiser (normal)
   contains
     procedure, pass(this) :: initialise => glorot_normal_initialise
     !! Initialise the weights and biases using the Glorot normal distribution
  end type glorot_normal_type

  type(glorot_uniform_type) :: glorot_uniform
  !! Glorot initialiser object
  type(glorot_normal_type) :: glorot_normal
  !! Glorot initialiser object


contains

!###############################################################################
  subroutine glorot_uniform_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the Glorot uniform distribution
    implicit none

    ! Arguments
    class(glorot_uniform_type), intent(inout) :: this
    !! Instance of the Glorot initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) ::  fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    ! Local variables
    real(real32) :: scale
    !! Scaling factor

    scale = sqrt(6._real32/real(fan_in+fan_out,real32))
    select rank(input)
    rank(0)
       call random_number(input)
       input = (input * 2._real32 - 1._real32) * scale
    rank(1)
       call random_number(input)
       input = (input * 2._real32 - 1._real32) * scale
    rank(2)
       call random_number(input)
       input = (input * 2._real32 - 1._real32) * scale
    rank(3)
       call random_number(input)
       input = (input * 2._real32 - 1._real32) * scale
    rank(4)
       call random_number(input)
       input = (input * 2._real32 - 1._real32) * scale
    rank(5)
       call random_number(input)
       input = (input * 2._real32 - 1._real32) * scale
    rank(6)
       call random_number(input)
       input = (input * 2._real32 - 1._real32) * scale
    end select

  end subroutine glorot_uniform_initialise
!###############################################################################


!###############################################################################
  subroutine glorot_normal_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the Glorot normal distribution
    implicit none

    ! Arguments
    class(glorot_normal_type), intent(inout) :: this
    !! Instance of the Glorot initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) ::  fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    ! Local variables
    real(real32) :: scale, norm
    !! Scaling factor, normalisation factor

    scale = sqrt(2._real32/real(fan_in+fan_out,real32)) ! standard deviation
    scale = 2._real32 * scale**2._real32                ! 2*variance
    norm  = 1._real32 / (sqrt(pi*scale))                ! normalisation
    select rank(input)
    rank(0)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real32 - 1._real32)**2._real32) / scale )
    rank(1)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real32 - 1._real32)**2._real32) / scale )
    rank(2)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real32 - 1._real32)**2._real32) / scale )
    rank(3)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real32 - 1._real32)**2._real32) / scale )
    rank(4)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real32 - 1._real32)**2._real32) / scale )
    rank(5)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real32 - 1._real32)**2._real32) / scale )
    rank(6)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real32 - 1._real32)**2._real32) / scale )
    end select

  end subroutine glorot_normal_initialise
!###############################################################################

end module athena__initialiser_glorot
