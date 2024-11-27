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
  subroutine ident_initialise(this, input, fan_in, fan_out, spacing)
    implicit none
    class(ident_type), intent(inout) :: this
    real(real32), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params
    integer, dimension(:), optional, intent(in) :: spacing

    integer :: i, j
    integer :: ndim
    integer, dimension(:), allocatable :: iprime, iprime2

    if(all(shape(input).ne.size(input,1)))then
       write(0,*) &
            "ERROR: A non-square tensor cannot be initialised as an &
            &identity matrix"
       stop 1
    end if

    select rank(input)
    rank(0)
       input = 1._real32
    rank(1)
       if(size(input).ne.1)then
          if(.not.present(spacing))then
             write(0,*) &
                  "ERROR: A vector of length greater than 1 cannot be &
                  &initialised as an identity matrix"
             stop 1
          else
             ndim = size(spacing)
             if(ndim.eq.1)then
                do i = 1, size(input)/spacing(1)
                   write(*,*) i, 1 + ( i - 1 ) * ( spacing(1) + 1)
                   input(1 + ( i - 1 ) * ( spacing(1) + 1) ) = 1._real32
                end do
             elseif(ndim.gt.1)then
                allocate(iprime(ndim))
                allocate(iprime2(ndim))
                iprime2 = 0
                iprime2(1) = 1
                do i = 1, size(input)/spacing(1)
                   iprime(ndim) = mod((i - 1)/product(spacing(:ndim-1)),product(spacing(:ndim)))
                   iprime(ndim) = iprime(ndim) * product(spacing(:ndim-1))
                   do j = ndim - 1, 1, -1
                     iprime(j) = mod((i - 1),sum(iprime(j+1:))) / product(spacing(:j-1))
                     iprime(j) = iprime(j) * product(spacing(:j-1))
                  end do
                  input(1 + sum(iprime * ( spacing(1) + iprime2 ))) = 1._real32
                end do
             end if
          end if
       else
          input = 1._real32
       end if
    rank(2)
       input = 0._real32
       do i = 1, size(input,1)
          input(i,i) = 1._real32
       end do
    rank(3)
       input = 0._real32
       do i = 1, size(input,1)
          input(i,i,i) = 1._real32
       end do
    rank(4)
       input = 0._real32
       do i = 1, size(input,1)
          input(i,i,i,i) = 1._real32
       end do
    rank(5)
       input = 0._real32
       do i = 1, size(input,1)
          input(i,i,i,i,i) = 1._real32
       end do
    rank(6)
       input = 0._real32
       do i = 1, size(input,1)
          input(i,i,i,i,i,i) = 1._real32
       end do
    end select
    
  end subroutine ident_initialise
!!!#############################################################################

end module initialiser_ident
!!!#############################################################################
