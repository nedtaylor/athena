module athena__initialiser_he
  !! Module containing the implementation of the He initialiser
  !!
  !! This module contains the implementation of the He initialiser
  !! for the weights and biases of a layer
  !! The He initialiser is also known as the Kaiming initialiser
  !! The He initialiser is also known as the MSRA initialiser
  !! Reference: https://doi.org/10.48550/arXiv.1502.01852
  use athena__constants, only: real32, pi
  use athena__misc_types, only: initialiser_type
  implicit none


  private

  public :: he_uniform
  public :: he_normal


  type, extends(initialiser_type) :: he_uniform_type
     !! Type for the He initialiser (uniform)
   contains
     procedure, pass(this) :: initialise => he_uniform_initialise
     !! Initialise the weights and biases using the He uniform distribution
  end type he_uniform_type

  type, extends(initialiser_type) :: he_normal_type
     !! Type for the He initialiser (normal)
   contains
     procedure, pass(this) :: initialise => he_normal_initialise
     !! Initialise the weights and biases using the He normal distribution
  end type he_normal_type

  type(he_uniform_type) :: he_uniform
  !! He initialiser object
  type(he_normal_type) :: he_normal
  !! He initialiser object



contains

!###############################################################################
  subroutine he_uniform_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the He uniform distribution
    implicit none

    ! Arguments
    class(he_uniform_type), intent(inout) :: this
    !! Instance of the He initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    ! Local variables
    real(real32) :: scale
    !! Scaling factor

    scale = sqrt(6._real32/real(fan_in,real32))
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

  end subroutine he_uniform_initialise
!###############################################################################


!###############################################################################
  subroutine he_normal_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the He normal distribution
    implicit none

    ! Arguments
    class(he_normal_type), intent(inout) :: this
    !! Instance of the He initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    ! Local variables
    real(real32) :: scale, norm
    !! Scaling factor

    scale = sqrt(2._real32/real(fan_in,real32))  ! standard deviation
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

  end subroutine he_normal_initialise
!###############################################################################

end module athena__initialiser_he
!###############################################################################
