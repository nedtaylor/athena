module athena__initialiser_glorot
  !! Module containing the implementation of the Glorot initialiser
  !!
  !! This module contains the implementation of the Glorot initialiser
  !! for the weights and biases of a layer
  !! The Glorot initialiser is also known as the Xavier initialiser
  !! Reference: https://proceedings.mlr.press/v9/glorot10a.html
  use coreutils, only: real32, pi, stop_program
  use athena__misc_types, only: initialiser_type
  implicit none


  private

  public :: glorot_uniform_init_type
  public :: glorot_normal_init_type


  type, extends(initialiser_type) :: glorot_uniform_init_type
     !! Type for the Glorot initialiser (uniform)
   contains
     procedure, pass(this) :: initialise => glorot_uniform_initialise
     !! Initialise the weights and biases using the Glorot uniform distribution
  end type glorot_uniform_init_type

  type, extends(initialiser_type) :: glorot_normal_init_type
     !! Type for the Glorot initialiser (normal)
   contains
     procedure, pass(this) :: initialise => glorot_normal_initialise
     !! Initialise the weights and biases using the Glorot normal distribution
  end type glorot_normal_init_type


  interface glorot_uniform_init_type
     module function initialiser_uniform_setup() result(initialiser)
       !! Interface for the Glorot uniform initialiser
       type(glorot_uniform_init_type) :: initialiser
       !! Glorot uniform initialiser object
     end function initialiser_uniform_setup
  end interface glorot_uniform_init_type

  interface glorot_normal_init_type
     module function initialiser_normal_setup() result(initialiser)
       !! Interface for the Glorot normal initialiser
       type(glorot_normal_init_type) :: initialiser
       !! Glorot normal initialiser object
     end function initialiser_normal_setup
  end interface glorot_normal_init_type



contains

!###############################################################################
  module function initialiser_uniform_setup() result(initialiser)
    implicit none
    ! Arguments
    type(glorot_uniform_init_type) :: initialiser
    !! Glorot uniform initialiser object

    initialiser%name = "glorot_uniform"

  end function initialiser_uniform_setup
!-------------------------------------------------------------------------------
  module function initialiser_normal_setup() result(initialiser)
    implicit none
    ! Arguments
    type(glorot_normal_init_type) :: initialiser
    !! Glorot normal initialiser object

    initialiser%name = "glorot_normal"

  end function initialiser_normal_setup
!###############################################################################


!###############################################################################
  subroutine glorot_uniform_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the Glorot uniform distribution
    implicit none

    ! Arguments
    class(glorot_uniform_init_type), intent(inout) :: this
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

    ! Validate inputs
    if(.not.present(fan_in)) &
         call stop_program("glorot_uniform_initialise: fan_in not present")
    if(.not.present(fan_out)) &
         call stop_program("glorot_uniform_initialise: fan_out not present")

    limit = sqrt(6._real32 / real(fan_in + fan_out, real32))
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
  end subroutine glorot_uniform_initialise
!###############################################################################


!###############################################################################
  subroutine glorot_normal_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the Glorot normal distribution
    implicit none

    ! Arguments
    class(glorot_normal_init_type), intent(inout) :: this
    !! Instance of the Glorot initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output units
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units (not used here, included for compatibility)

    ! Local variables
    integer :: n
    !! Number of elements in the input array
    real(real32) :: sigma
    !! Scaling factor
    real(real32), dimension(:), allocatable :: u1, u2, z
    !! Temporary arrays for the random numbers

    ! Default fallback values (to avoid division by zero)
    if(.not.present(fan_in)) &
         call stop_program("glorot_normal_initialise: fan_in not present")
    if(.not.present(fan_out)) &
         call stop_program("glorot_normal_initialise: fan_out not present")

    sigma = sqrt(2._real32 / real(fan_in + fan_out, real32))
    n = size(input)
    allocate(u1(n), u2(n), z(n))

    call random_number(u1)
    call random_number(u2)
    where (u1 .lt. 1.E-7_real32)
       u1 = 1.E-7_real32
    end where

    ! Box-Muller transform for normal distribution
    z = sqrt(-2._real32 * log(u1)) * cos(2._real32 * pi * u2)
    z = sigma * z

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

  end subroutine glorot_normal_initialise
!###############################################################################

end module athena__initialiser_glorot
