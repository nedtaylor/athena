!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module initialiser
  use constants, only: real12
  use misc, only: to_lower
  use custom_types, only: initialiser_type
  use initialiser_glorot, only: glorot_uniform, glorot_normal
  use initialiser_he, only: he_uniform, he_normal
  use initialiser_lecun, only: lecun_uniform, lecun_normal
  use initialiser_zeros, only: zeros
  implicit none


  private

  public :: initialiser_setup

!!!!! ALSO, HAVE THE CV, PL, FC, etc, LAYERS AS CLASSES
!!!!! ... they may be able to be appended on to each other

  
  !! make an initialiser that takes in an assumed rank
  !! it then does product(shape(weight)) OR size(weight)
  !! could always use select rank(x) statement if needed
  !! https://keras.io/api/layers/initializers/
  
  

contains

!!!#############################################################################
!!! 
!!!#############################################################################
  function initialiser_setup(name) result(initialiser)
    implicit none
    class(initialiser_type), allocatable :: initialiser
    character(*), intent(in) :: name
    real(real12), dimension(2) :: rnum


    !!--------------------------------------------------------------------------
    !! set initialiser function
    !!--------------------------------------------------------------------------
    select case(trim(to_lower(name)))
    case("glorot_uniform")
       initialiser = glorot_uniform
    case("glorot_normal")
       initialiser = glorot_normal
    case("he_uniform")
       initialiser = he_uniform
    case("he_normal")
       initialiser = he_normal
    case("lecun_uniform")
       initialiser = lecun_uniform
       !initialiser%initialise() => lecun_uniform()
    case("lecun_normal")
       initialiser = lecun_normal
    case("zeros")
       initialiser = zeros
    case default
       stop "Incorrect initialiser name given '"//trim(to_lower(name))//"'"
    end select

  end function initialiser_setup
!!!#############################################################################

end module initialiser
!!!#############################################################################
