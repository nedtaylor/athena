module athena__initialiser_he
  !! Module containing the implementation of the He initialiser
  !!
  !! This module contains the implementation of the He initialiser
  !! for the weights and biases of a layer
  !! The He initialiser is also known as the Kaiming initialiser
  !! The He initialiser is also known as the MSRA initialiser
  !! Reference: https://doi.org/10.48550/arXiv.1502.01852
  use athena__constants, only: real32, pi
  use athena__io_utils, only: stop_program
  use athena__misc_types, only: initialiser_type
  use athena__misc, only: to_lower
  implicit none


  private

  public :: he_uniform, he_normal
  public :: he_uniform_type, he_normal_type


  type, extends(initialiser_type) :: he_uniform_type
     !! Type for the He initialiser (uniform)
     integer, private :: mode = 1
   contains
     procedure, pass(this) :: initialise => he_uniform_initialise
     !! Initialise the weights and biases using the He uniform distribution
  end type he_uniform_type

  type, extends(initialiser_type) :: he_normal_type
     !! Type for the He initialiser (normal)
     integer, private :: mode = 1
   contains
     procedure, pass(this) :: initialise => he_normal_initialise
     !! Initialise the weights and biases using the He normal distribution
  end type he_normal_type

  type(he_uniform_type) :: he_uniform
  !! He initialiser object
  type(he_normal_type) :: he_normal
  !! He initialiser object


  interface he_uniform_type
     module function initialiser_uniform_setup(scale, mode) result(initialiser)
       !! Interface for the He uniform initialiser
       real(real32), intent(in), optional :: scale
       !! Scaling factor (default: 1.0)
       character(len=*), intent(in), optional :: mode
       !! Mode for calculating the scaling factor (default: "fan_in")
       type(he_uniform_type) :: initialiser
       !! He uniform initialiser object
     end function initialiser_uniform_setup
  end interface he_uniform_type

  interface he_normal_type
     module function initialiser_normal_setup(scale) result(initialiser)
       !! Interface for the He normal initialiser
       real(real32), intent(in), optional :: scale
       !! Scaling factor (default: 1.0)
       type(he_normal_type) :: initialiser
       !! He normal initialiser object
     end function initialiser_normal_setup
  end interface he_normal_type

contains

!###############################################################################
  module function initialiser_uniform_setup(scale, mode) result(initialiser)
    implicit none
    ! Arguments
    real(real32), intent(in), optional :: scale
    !! Scaling factor (default: 1.0)
    character(len=*), intent(in), optional :: mode
    !! Mode for calculating the scaling factor (default: "fan_in")
    type(he_uniform_type) :: initialiser
    !! He uniform initialiser object

    ! Local variables
    character(len=20) :: mode_
    !! Mode for calculating the scaling factor

    initialiser%name = "he_uniform"
    if(present(scale)) initialiser%scale = scale
    if(present(mode))then
       mode_ = to_lower(trim(mode))
       select case(mode_)
       case("fan_in")
          initialiser%mode = 1
       case("fan_out")
          initialiser%mode = 2
       case default
          call stop_program("initialiser_setup: invalid mode")
       end select
    end if

  end function initialiser_uniform_setup
!###############################################################################


!###############################################################################
  module function initialiser_normal_setup(scale, mode) result(initialiser)
    implicit none
    ! Arguments
    real(real32), intent(in), optional :: scale
    !! Scaling factor (default: 1.0)
    character(len=*), intent(in), optional :: mode
    !! Mode for calculating the scaling factor (default: "fan_in")
    type(he_normal_type) :: initialiser
    !! He normal initialiser object

    ! Local variables
    character(len=20) :: mode_
    !! Mode for calculating the scaling factor

    initialiser%name = "he_normal"
    if(present(scale)) initialiser%scale = scale
    if(present(mode))then
       mode_ = to_lower(trim(mode))
       select case(mode_)
       case("fan_in")
          initialiser%mode = 1
       case("fan_out")
          initialiser%mode = 2
       case default
          call stop_program("initialiser_setup: invalid mode")
       end select
    end if
  end function initialiser_normal_setup
!###############################################################################


!###############################################################################
  subroutine he_uniform_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the He uniform distribution
    implicit none

    ! Arguments
    class(he_uniform_type), intent(inout) :: this
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
         call stop_program("he_uniform_initialise: fan_in not present")

    select case(this%mode)
    case(1)
       limit = this%scale * sqrt(6._real32 / real(fan_in, real32))
    case(2)
       limit = this%scale * sqrt(6._real32 / real(fan_out, real32))
    case default
       call stop_program("he_uniform_initialise: invalid mode")
    end select
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
  end subroutine he_uniform_initialise
!###############################################################################


!###############################################################################
  subroutine he_normal_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the He normal distribution
    implicit none

    ! Arguments
    class(he_normal_type), intent(inout) :: this
    !! Instance of the He initialiser
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
         call stop_program("he_normal_initialise: fan_in not present")

    select case(this%mode)
    case(1)
       sigma = this%scale * sqrt(2._real32/real(fan_in,real32))
    case(2)
       sigma = this%scale * sqrt(2._real32/real(fan_out,real32))
    case default
       call stop_program("he_uniform_initialise: invalid mode")
    end select
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

  end subroutine he_normal_initialise
!###############################################################################

end module athena__initialiser_he
