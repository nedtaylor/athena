!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the Gaussian initialisation
!!!#############################################################################
module initialiser_gaussian
  use constants, only: real12, pi
  use custom_types, only: initialiser_type
  implicit none


  type, extends(initialiser_type) :: gaussian_type
   contains
     procedure, pass(this) :: initialise => gaussian_initialise
  end type gaussian_type
  type(gaussian_type) :: gaussian

  
  private

  public :: gaussian


contains

!!!#############################################################################
!!! Gaussian initialisation
!!!#############################################################################
  subroutine gaussian_initialise(this, input, fan_in, fan_out)
    implicit none
    class(gaussian_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) ::  fan_in, fan_out

    integer :: i, j, k, l, m, o


    select rank(input)
    rank(0)
       call box_muller(input, this%mean, this%std)
    rank(1)
       do i=1, size(input)
          call box_muller(input(i), this%mean, this%std)
       end do
    rank(2)
       do i=1, size(input,1)
           do j=1, size(input,2)
             call box_muller(input(i,j), this%mean, this%std)
           end do
       end do
    rank(3)
       do i=1, size(input,1)
           do j=1, size(input,2)
             do k=1, size(input,3)
               call box_muller(input(i,j,k), this%mean, this%std)
             end do
           end do
       end do
    rank(4)
       do i=1, size(input,1)
           do j=1, size(input,2)
             do k=1, size(input,3)
               do l=1, size(input,4)
                 call box_muller(input(i,j,k,l), this%mean, this%std)
               end do
             end do
           end do
       end do
    rank(5)
       do i=1, size(input,1)
           do j=1, size(input,2)
             do k=1, size(input,3)
               do l=1, size(input,4)
                 do m=1, size(input,5)
                   call box_muller(input(i,j,k,l,m), &
                        this%mean, this%std)
                 end do
               end do
             end do
           end do
       end do
    rank(6)
       do i=1, size(input,1)
           do j=1, size(input,2)
             do k=1, size(input,3)
               do l=1, size(input,4)
                 do m=1, size(input,5)
                   do o=1, size(input,6)
                     call box_muller(input(i,j,k,l,m,o), &
                          this%mean, this%std)
                    end do
                 end do
               end do
             end do
           end do
       end do
    end select
    
  end subroutine gaussian_initialise
!!!#############################################################################


!!!#############################################################################
!!! element-wise Gaussian initialisation
!!!#############################################################################
  subroutine box_muller(input, mean, std)
    implicit none
    real(real12), intent(out) :: input
    real(real12), intent(in) :: mean, std

    real(real12) :: r1, r2

    call random_number(r1)
    call random_number(r2)
    input = sqrt(-2._real12 * log(r1))
    input = input * cos(2._real12 * pi * r2)
    input = mean + std * input
    
  end subroutine box_muller
!!!#############################################################################

end module initialiser_gaussian
!!!#############################################################################
