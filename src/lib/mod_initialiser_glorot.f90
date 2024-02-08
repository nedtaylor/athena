!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
!!! Glorot initialiser reference: https://proceedings.mlr.press/v9/glorot10a.html
module initialiser_glorot
  use constants, only: real12, pi
  use custom_types, only: initialiser_type
  implicit none


  type, extends(initialiser_type) :: glorot_uniform_type
   contains
     procedure, pass(this) :: initialise => glorot_uniform_initialise
  end type glorot_uniform_type
  type(glorot_uniform_type) :: glorot_uniform
  type, extends(initialiser_type) :: glorot_normal_type
   contains
     procedure, pass(this) :: initialise => glorot_normal_initialise
  end type glorot_normal_type
  type(glorot_normal_type) :: glorot_normal

  
  private

  public :: glorot_uniform
  public :: glorot_normal


contains

!!!#############################################################################
!!! Xavier Glorot initialisation (uniform)
!!!#############################################################################
  subroutine glorot_uniform_initialise(this, input, fan_in, fan_out)
    implicit none
    class(glorot_uniform_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) ::  fan_in, fan_out  !no. in and out params

    real(real12) :: scale

    scale = sqrt(6._real12/real(fan_in+fan_out,real12))
    select rank(input)
    rank(0)
       call random_number(input)
       input = (input * 2._real12 - 1._real12) * scale
    rank(1)
       call random_number(input)
       input = (input * 2._real12 - 1._real12) * scale
    rank(2)
       call random_number(input)
       input = (input * 2._real12 - 1._real12) * scale
    rank(3)
       call random_number(input)
       input = (input * 2._real12 - 1._real12) * scale
    rank(4)
       call random_number(input)
       input = (input * 2._real12 - 1._real12) * scale
    rank(5)
       call random_number(input)
       input = (input * 2._real12 - 1._real12) * scale
    rank(6)
       call random_number(input)
       input = (input * 2._real12 - 1._real12) * scale
    end select
    
  end subroutine glorot_uniform_initialise
!!!#############################################################################


!!!#############################################################################
!!! Xavier Glorot initialisation (normal)
!!!#############################################################################
  subroutine glorot_normal_initialise(this, input, fan_in, fan_out)
    implicit none
    class(glorot_normal_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) ::  fan_in, fan_out  !no. in and out params

    real(real12) :: scale, norm

    scale = sqrt(2._real12/real(fan_in+fan_out,real12)) ! standard deviation
    scale = 2._real12 * scale**2._real12                ! 2*variance
    norm  = 1._real12 / (sqrt(pi*scale))                ! normalisation
    select rank(input)
    rank(0)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real12 - 1._real12)**2._real12) / scale )
    rank(1)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real12 - 1._real12)**2._real12) / scale )
    rank(2)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real12 - 1._real12)**2._real12) / scale )
    rank(3)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real12 - 1._real12)**2._real12) / scale )
    rank(4)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real12 - 1._real12)**2._real12) / scale )
    rank(5)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real12 - 1._real12)**2._real12) / scale )
    rank(6)
       call random_number(input)
       input = norm * &
            exp( (-(input * 2._real12 - 1._real12)**2._real12) / scale )
    end select
    
  end subroutine glorot_normal_initialise
!!!#############################################################################

end module initialiser_glorot
!!!#############################################################################
