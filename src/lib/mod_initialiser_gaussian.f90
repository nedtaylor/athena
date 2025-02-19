module athena__initialiser_gaussian
  !! Module containing the Gaussian initialisation
  !!
  !! This module contains the implementation of the Gaussian initialisation
  !! for the weights and biases of a layer
  use athena__constants, only: real32, pi
  use athena__misc_types, only: initialiser_type
  implicit none


  private

  public :: gaussian


  type, extends(initialiser_type) :: gaussian_type
     !! Type for the Gaussian initialiser
   contains
     procedure, pass(this) :: initialise => gaussian_initialise
     !! Initialise the weights and biases using the Gaussian distribution
  end type gaussian_type
  type(gaussian_type) :: gaussian
  !! Gaussian initialiser object



contains

!###############################################################################
  subroutine gaussian_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the Gaussian distribution
    implicit none

    ! Arguments
    class(gaussian_type), intent(inout) :: this
    !! Instance of the Gaussian initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) ::  fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

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
!###############################################################################


!###############################################################################
  subroutine box_muller(input, mean, std)
    !! Generate a random number using the Box-Muller transform
    !!
    !! This subroutine generates a random number using the Box-Muller
    !! transform. The random number is generated using two random numbers
    !! generated using the random_number subroutine
    !! This is used for element-wise Gaussian initialisation
    implicit none

    ! Arguments
    real(real32), intent(out) :: input
    !! Random number
    real(real32), intent(in) :: mean, std
    !! Mean and standard deviation

    real(real32) :: r1, r2

    call random_number(r1)
    call random_number(r2)
    input = sqrt(-2._real32 * log(r1))
    input = input * cos(2._real32 * pi * r2)
    input = mean + std * input
    
  end subroutine box_muller
!###############################################################################

end module athena__initialiser_gaussian