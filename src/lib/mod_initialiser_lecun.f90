module athena__initialiser_lecun
  !! Module containing the implementation of the LeCun initialiser
  !!
  !! This module contains the implementation of the LeCun initialiser
  !! for the weights and biases of a layer
  !! Reference: https://dl.acm.org/doi/10.5555/645754.668382
  use athena__constants, only: real32, pi
  use athena__io_utils, only: stop_program
  use athena__misc_types, only: initialiser_type
  implicit none


  private

  public :: lecun_uniform_type
  public :: lecun_normal_type


  type, extends(initialiser_type) :: lecun_uniform_type
     !! Type for the LeCun initialiser (uniform)
   contains
     procedure, pass(this) :: initialise => lecun_uniform_initialise
     !! Initialise the weights and biases using the LeCun uniform distribution
  end type lecun_uniform_type
  type, extends(initialiser_type) :: lecun_normal_type
     !! Type for the LeCun initialiser (normal)
   contains
     procedure, pass(this) :: initialise => lecun_normal_initialise
     !! Initialise the weights and biases using the LeCun normal distribution
  end type lecun_normal_type


  interface lecun_uniform_type
     module function initialiser_lecun_uniform_setup() result(initialiser)
       !! Interface for the LeCun uniform initialiser
       type(lecun_uniform_type) :: initialiser
       !! LeCun uniform initialiser object
     end function initialiser_lecun_uniform_setup
  end interface lecun_uniform_type

  interface lecun_normal_type
     module function initialiser_lecun_normal_setup() result(initialiser)
       !! Interface for the LeCun normal initialiser
       type(lecun_normal_type) :: initialiser
       !! LeCun normal initialiser object
     end function initialiser_lecun_normal_setup
  end interface lecun_normal_type



contains

!###############################################################################
  module function initialiser_lecun_uniform_setup() result(initialiser)
    !! Interface for the LeCun uniform initialiser
    implicit none

    type(lecun_uniform_type) :: initialiser
    !! LeCun uniform initialiser object

    initialiser%name = "lecun_uniform"

  end function initialiser_lecun_uniform_setup
!-------------------------------------------------------------------------------
  module function initialiser_lecun_normal_setup() result(initialiser)
    !! Interface for the LeCun normal initialiser
    implicit none

    type(lecun_normal_type) :: initialiser
    !! LeCun normal initialiser object

    initialiser%name = "lecun_normal"

  end function initialiser_lecun_normal_setup
!###############################################################################


!###############################################################################
  subroutine lecun_uniform_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the LeCun uniform distribution
    implicit none

    ! Arguments
    class(lecun_uniform_type), intent(inout) :: this
    !! Instance of the Glorot initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output units
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units (not used)

    ! Local variables
    integer :: n
    !! Number of elements in the input array
    real(real32) :: limit
    !! Scaling factor
    real(real32), dimension(:), allocatable :: r
    !! Temporary uniform random numbers

    if(.not.present(fan_in)) &
         call stop_program("lecun_uniform_initialise: fan_in not present")

    limit = sqrt(3._real32 / real(fan_in, real32))
    n = size(input)
    allocate(r(n))
    call random_number(r)
    r = (2._real32 * r - 1._real32) * limit

    ! Assign according to rank
    select rank(input)
    rank(0)
       input = r(1)
    rank(1)
       input = r
    rank(2)
       input = reshape(r, shape(input))
    rank(3)
       input = reshape(r, shape(input))
    rank(4)
       input = reshape(r, shape(input))
    rank(5)
       input = reshape(r, shape(input))
    rank(6)
       input = reshape(r, shape(input))
    end select

    deallocate(r)
  end subroutine lecun_uniform_initialise
!###############################################################################


!###############################################################################
  subroutine lecun_normal_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the LeCun normal distribution
    implicit none

    ! Arguments
    class(lecun_normal_type), intent(inout) :: this
    !! Instance of the LeCun initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units (not used)

    ! Local variables
    integer :: n
    !! Number of elements in the input array
    real(real32) :: sigma
    !! Scaling factor
    real(real32), dimension(:), allocatable :: u1, u2, z
    !! Temporary arrays for the random numbers

    if(.not.present(fan_in)) &
         call stop_program("lecun_normal_initialise: fan_in not present")

    sigma = sqrt(1._real32/real(fan_in,real32))
    n = size(input)
    allocate(u1(n), u2(n), z(n))

    call random_number(u1)
    call random_number(u2)
    where (u1 .lt. 1.E-7_real32)
       u1 = 1.E-7_real32
    end where

    ! Box-Muller transform
    z = sqrt(-2._real32 * log(u1)) * cos(2._real32 * pi * u2)
    z = sigma * z

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

  end subroutine lecun_normal_initialise
!###############################################################################

end module athena__initialiser_lecun
