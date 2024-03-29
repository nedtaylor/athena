!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the zeros initialiser
!!!#############################################################################
module initialiser_zeros
  use constants, only: real12
  use custom_types, only: initialiser_type
  implicit none


  type, extends(initialiser_type) :: zeros_type
   contains
     procedure, pass(this) :: initialise => zeros_initialise
  end type zeros_type
  type(zeros_type) :: zeros

  
  private

  public :: zeros


contains

!!!#############################################################################
!!! Zeros initialisation
!!!#############################################################################
  pure subroutine zeros_initialise(this, input, fan_in, fan_out)
    implicit none
    class(zeros_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    select rank(input)
    rank(0)
       input = 0._real12
    rank(1)
       input = 0._real12
    rank(2)
       input = 0._real12
    rank(3)
       input = 0._real12
    rank(4)
       input = 0._real12
    rank(5)
       input = 0._real12
    rank(6)
       input = 0._real12
    end select
    
  end subroutine zeros_initialise
!!!#############################################################################

end module initialiser_zeros
!!!#############################################################################
