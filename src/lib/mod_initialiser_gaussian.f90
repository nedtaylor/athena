module athena__initialiser_gaussian
  !! Module containing the Gaussian initialisation
  !!
  !! This module contains the implementation of the Gaussian initialisation
  !! for the weights and biases of a layer
  use coreutils, only: real32, pi
  use athena__misc_types, only: base_init_type
  implicit none


  private

  public :: gaussian_init_type


  type, extends(base_init_type) :: gaussian_init_type
     !! Type for the Gaussian initialiser
   contains
     procedure, pass(this) :: initialise => gaussian_initialise
     !! Initialise the weights and biases using the Gaussian distribution
  end type gaussian_init_type


  interface gaussian_init_type
     module function initialiser_gaussian_type(name) result(initialiser)
       !! Interface for the Gaussian initialiser
       type(gaussian_init_type) :: initialiser
       !! Gaussian initialiser object
       character(*), optional, intent(in) :: name
       !! Name of the initialiser
     end function initialiser_gaussian_type
  end interface gaussian_init_type



contains

!###############################################################################
  module function initialiser_gaussian_type(name) result(initialiser)
    !! Interface for the Gaussian initialiser
    implicit none
    ! Arguments
    character(*), optional, intent(in) :: name
    !! Name of the initialiser


    type(gaussian_init_type) :: initialiser
    !! Gaussian initialiser object

    if(present(name)) then
       initialiser%name = trim(adjustl(name))
    else
       initialiser%name = "gaussian"
    end if

  end function initialiser_gaussian_type
!###############################################################################


!###############################################################################
  subroutine gaussian_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the Gaussian distribution
    implicit none

    ! Arguments
    class(gaussian_init_type), intent(inout) :: this
    !! Instance of the Gaussian initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) ::  fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    ! Local variables
    integer :: n
    !! Number of elements in the input array
    real(real32), dimension(:), allocatable :: u1, u2, z
    !! Temporary arrays for the random numbers

    n = size(input)
    allocate(u1(n), u2(n), z(n))

    call random_number(u1)
    call random_number(u2)
    where (u1 .lt. 1.E-7_real32)
       u1 = 1.E-7_real32
    end where

    ! Box-Muller transform for normal distribution
    z = sqrt(-2._real32 * log(u1)) * cos(2._real32 * pi * u2)
    z = this%mean + this%std * z

    ! Assign according to rank
    select rank(input)
    rank(0)
       input = z(1)
    rank(1)
       input = z
    rank(2)
       input = reshape(z, shape(input))
    rank(3)
       input = reshape(z, shape(input))
    rank(4)
       input = reshape(z, shape(input))
    rank(5)
       input = reshape(z, shape(input))
    rank(6)
       input = reshape(z, shape(input))
    end select

    deallocate(u1, u2, z)

  end subroutine gaussian_initialise
!###############################################################################

end module athena__initialiser_gaussian
