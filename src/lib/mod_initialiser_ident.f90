!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the identity initialiser
!!!#############################################################################
module initialiser_ident
  use constants, only: real32
  use custom_types, only: initialiser_type
  implicit none


  type, extends(initialiser_type) :: ident_type
   contains
     procedure, pass(this) :: initialise => ident_initialise
  end type ident_type
  type(ident_type) :: ident

  
  private

  public :: ident


contains

!!!#############################################################################
!!! Ident initialisation
!!!#############################################################################
  subroutine ident_initialise(this, input, fan_in, fan_out)
    implicit none
    class(ident_type), intent(inout) :: this
    real(real32), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    integer :: i

    if(all(shape(input).ne.size(input,1))) stop &
         "ERROR: A non-square tensor cannot be initialised as an &
         &identity matrix"

    select rank(input)
    rank(0)
       input = 1._real32
    rank(1)
       if(size(input).ne.1) stop &
            "ERROR: A vector of length greater than 1 cannot be &
            &initialised as an identity matrix"
       input = 1._real32
    rank(2)
       input = 0._real32
       do i=1,size(input,1)
          input(i,i) = 1._real32
       end do
    rank(3)
       input = 0._real32
       do i=1,size(input,1)
          input(i,i,i) = 1._real32
       end do
    rank(4)
       input = 0._real32
       do i=1,size(input,1)
          input(i,i,i,i) = 1._real32
       end do
    rank(5)
       input = 0._real32
       do i=1,size(input,1)
          input(i,i,i,i,i) = 1._real32
       end do
    rank(6)
       input = 0._real32
       do i=1,size(input,1)
          input(i,i,i,i,i,i) = 1._real32
       end do
    end select
    
  end subroutine ident_initialise
!!!#############################################################################

end module initialiser_ident
!!!#############################################################################
