!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the LeCun initialiser
!!!#############################################################################
!!! LeCun initialiser reference: https://dl.acm.org/doi/10.5555/645754.668382
!!!#############################################################################
module initialiser_lecun
  use constants, only: real32, pi
  use custom_types, only: initialiser_type
  implicit none


  type, extends(initialiser_type) :: lecun_uniform_type
   contains
     procedure, pass(this) :: initialise => lecun_uniform_initialise
  end type lecun_uniform_type
  type(lecun_uniform_type) :: lecun_uniform
  type, extends(initialiser_type) :: lecun_normal_type
   contains
     procedure, pass(this) :: initialise => lecun_normal_initialise
  end type lecun_normal_type
  type(lecun_normal_type) :: lecun_normal

  
  private

  public :: lecun_uniform
  public :: lecun_normal


contains

!!!#############################################################################
!!! LeCun initialisation (uniform)
!!!#############################################################################
  subroutine lecun_uniform_initialise(this, input, fan_in, fan_out)
    implicit none
    class(lecun_uniform_type), intent(inout) :: this
    real(real32), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    real(real32) :: scale

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
!!!#############################################################################


!!!#############################################################################
!!! LeCun initialisation (normal)
!!!#############################################################################
  subroutine lecun_normal_initialise(this, input, fan_in, fan_out)
    implicit none
    class(lecun_normal_type), intent(inout) :: this
    real(real32), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    real(real32) :: scale, norm

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
!!!#############################################################################

end module initialiser_lecun
!!!#############################################################################
