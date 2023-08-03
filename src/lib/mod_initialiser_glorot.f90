!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module initialiser_glorot
  use constants, only: real12
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
!!! https://proceedings.mlr.press/v9/glorot10a.html
!!!#############################################################################
  subroutine glorot_uniform_initialise(this, input, fan_in, fan_out)
    implicit none
    class(glorot_uniform_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) ::  fan_in, fan_out  !no. in and out params

    real(real12) :: scale
!!! HAVE ASSUMED RANK

    scale = sqrt(6._real12/real(fan_in+fan_out,real12))
    select rank(input)
    rank(0)
       call random_number(input)
       input = input * scale
    rank(1)
       call random_number(input)
       input = input * scale
    rank(2)
       call random_number(input)
       input = input * scale
    rank(3)
       call random_number(input)
       input = input * scale
    end select
    
  end subroutine glorot_uniform_initialise
!!!#############################################################################


!!!#############################################################################
!!! Xavier Glorot initialisation (normal)
!!! https://proceedings.mlr.press/v9/glorot10a.html
!!!#############################################################################
  subroutine glorot_normal_initialise(this, input, fan_in, fan_out)
    implicit none
    class(glorot_normal_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) ::  fan_in, fan_out  !no. in and out params

    real(real12) :: scale
!!! HAVE ASSUMED RANK

    scale = sqrt(2._real12/real(fan_in+fan_out,real12))
    select rank(input)
    rank(0)
       call random_number(input)
       input = input * scale
    rank(1)
       call random_number(input)
       input = input * scale
    rank(2)
       call random_number(input)
       input = input* scale
    rank(3)
       call random_number(input)
       input = input * scale
    end select
    
  end subroutine glorot_normal_initialise
!!!#############################################################################

end module initialiser_glorot
!!!#############################################################################
