!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the He initialiser
!!!#############################################################################
!!! He initialiser reference: https://doi.org/10.48550/arXiv.1502.01852
!!!#############################################################################
module initialiser_he
  use constants, only: real32, pi
  use custom_types, only: initialiser_type
  implicit none


  type, extends(initialiser_type) :: he_uniform_type
   contains
     procedure, pass(this) :: initialise => he_uniform_initialise
  end type he_uniform_type
  type(he_uniform_type) :: he_uniform
  type, extends(initialiser_type) :: he_normal_type
   contains
     procedure, pass(this) :: initialise => he_normal_initialise
  end type he_normal_type
  type(he_normal_type) :: he_normal

  
  private

  public :: he_uniform
  public :: he_normal


contains

!!!#############################################################################
!!! He initialisation (uniform)
!!! Kaiming initialisation
!!! MSRA initialisation
!!!#############################################################################
  subroutine he_uniform_initialise(this, input, fan_in, fan_out)
    implicit none
    class(he_uniform_type), intent(inout) :: this
    real(real32), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    real(real32) :: scale

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
!!!#############################################################################


!!!#############################################################################
!!! He initialisation (normal)
!!! Kaiming initialisation
!!! MSRA initialisation
!!!#############################################################################
  subroutine he_normal_initialise(this, input, fan_in, fan_out)
    implicit none
    class(he_normal_type), intent(inout) :: this
    real(real32), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    real(real32) :: scale, norm

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
!!!#############################################################################

end module initialiser_he
!!!#############################################################################
