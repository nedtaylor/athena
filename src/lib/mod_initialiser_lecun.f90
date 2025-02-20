module athena__initialiser_lecun
  !! Module containing the implementation of the LeCun initialiser
   !!
   !! This module contains the implementation of the LeCun initialiser
   !! for the weights and biases of a layer
   !! Reference: https://dl.acm.org/doi/10.5555/645754.668382
  use athena__constants, only: real32, pi
  use athena__misc_types, only: initialiser_type
  implicit none


  private

  public :: lecun_uniform
  public :: lecun_normal


  type, extends(initialiser_type) :: lecun_uniform_type
     !! Type for the LeCun initialiser (uniform)
   contains
     procedure, pass(this) :: initialise => lecun_uniform_initialise
     !! Initialise the weights and biases using the LeCun uniform distribution
  end type lecun_uniform_type
  type, extends(initialiser_type) :: lecun_normal_type
     !! Type for the LeCun initialiser (normal)
   contains
     procedure, pass(this) :: initialise => lecun_normal_initialise
     !! Initialise the weights and biases using the LeCun normal distribution
  end type lecun_normal_type

  type(lecun_uniform_type) :: lecun_uniform
  !! LeCun initialiser object
  type(lecun_normal_type) :: lecun_normal
  !! LeCun initialiser object



contains

!###############################################################################
  subroutine lecun_uniform_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the LeCun uniform distribution
    implicit none

    ! Arguments
    class(lecun_uniform_type), intent(inout) :: this
    !! Instance of the LeCun initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    ! Local variables
    real(real32) :: scale
    !! Scaling factor

    scale = sqrt(3._real32/real(fan_in,real32))
    select rank(input)
    rank(0)
       call random_number(input)
       input = (input *2._real32 - 1._real32) * scale
    rank(1)
       call random_number(input)
       input = (input *2._real32 - 1._real32) * scale
    rank(2)
       call random_number(input)
       input = (input *2._real32 - 1._real32) * scale
    rank(3)
       call random_number(input)
       input = (input *2._real32 - 1._real32) * scale
    rank(4)
       call random_number(input)
       input = (input *2._real32 - 1._real32) * scale
    rank(5)
       call random_number(input)
       input = (input *2._real32 - 1._real32) * scale
    rank(6)
       call random_number(input)
       input = (input *2._real32 - 1._real32) * scale
    end select

  end subroutine lecun_uniform_initialise
!###############################################################################


!###############################################################################
  subroutine lecun_normal_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the LeCun normal distribution
    implicit none

    ! Arguments
    class(lecun_normal_type), intent(inout) :: this
    !! Instance of the LeCun initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    ! Local variables
    real(real32) :: scale, norm
    !! Scaling factor, normalisation factor

    scale = sqrt(1._real32/real(fan_in,real32))  ! standard deviation
    scale = 2._real32 * scale**2._real32         ! 2*variance
    norm  = 1._real32 / (sqrt(pi*scale))         ! normalisation
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

  end subroutine lecun_normal_initialise
!###############################################################################

end module athena__initialiser_lecun
