!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module initialiser_ones
  use constants, only: real12
  use custom_types, only: initialiser_type
  implicit none


  type, extends(initialiser_type) :: ones_type
   contains
     procedure, pass(this) :: initialise => ones_initialise
  end type ones_type
  type(ones_type) :: ones

  
  private

  public :: ones


contains

!!!#############################################################################
!!! Ones initialisation
!!!#############################################################################
  pure subroutine ones_initialise(this, input, fan_in, fan_out)
    implicit none
    class(ones_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    select rank(input)
    rank(0)
       input = 1._real12
    rank(1)
       input = 1._real12
    rank(2)
       input = 1._real12
    rank(3)
       input = 1._real12
    rank(4)
       input = 1._real12
    rank(5)
       input = 1._real12
    end select
    
  end subroutine ones_initialise
!!!#############################################################################

end module initialiser_ones
!!!#############################################################################
