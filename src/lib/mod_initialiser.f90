!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
!!! module contains initialiser functions
!!! module includes the following procedures:
!!! initialiser_setup - set up initialiser
!!! get_default_initialiser - get default initialiser based on activation ...
!!!                           ... function
!!!#############################################################################
!! Examples of initialsers in keras: https://keras.io/api/layers/initializers/
!!!#############################################################################
module initialiser
  use misc, only: to_lower
  use custom_types, only: initialiser_type
  use initialiser_glorot, only: glorot_uniform, glorot_normal
  use initialiser_he, only: he_uniform, he_normal
  use initialiser_lecun, only: lecun_uniform, lecun_normal
  use initialiser_ones, only: ones
  use initialiser_zeros, only: zeros
  use initialiser_ident, only: ident
  use initialiser_gaussian, only: gaussian
  implicit none


  private

  public :: initialiser_setup, get_default_initialiser


contains

!!!#############################################################################
!!! get default initialiser based on activation function (and if a bias)
!!!#############################################################################
!!! activation = (S, in) activation function name
!!! is_bias    = (B, in) if true, then initialiser is for bias
!!! name       = (S, out) name of default initialiser
  function get_default_initialiser(activation, is_bias) result(name)
    implicit none
    character(*), intent(in) :: activation
    logical, optional, intent(in) :: is_bias

    character(:), allocatable :: name


    !!--------------------------------------------------------------------------
    !! if bias, use default initialiser of zero
    !!--------------------------------------------------------------------------
    if(present(is_bias))then
       if(is_bias) name = "zeros"
       return
    end if


    !!--------------------------------------------------------------------------
    !! set default initialiser based on activation
    !!--------------------------------------------------------------------------
    if(trim(activation).eq."selu")then
       name = "lecun_normal"
    elseif(index(activation,"elu").ne.0)then
       name = "he_uniform"
    elseif(trim(activation).eq."batch")then
       name = "gaussian"
    else
       name = "glorot_uniform"
    end if

  end function get_default_initialiser
!!!#############################################################################


!!!#############################################################################
!!! set up initialiser
!!!#############################################################################
!!! name        = (S, in) name of initialiser
!!! error       = (I, out) error code
!!! initialiser = (O, out) initialiser function
  function initialiser_setup(name, error) result(initialiser)
    implicit none
    class(initialiser_type), allocatable :: initialiser
    character(*), intent(in) :: name
    integer, optional, intent(out) :: error


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
    case("lecun_normal")
       initialiser = lecun_normal
    case("ones")
       initialiser = ones
    case("zeros")
       initialiser = zeros
    case("ident")
       initialiser = ident
    case("gaussian")
       initialiser = gaussian
    case("normal")
       initialiser = gaussian
    case default
       if(present(error))then
          error = -1
          return
       else
          stop "Incorrect initialiser name given '"//trim(to_lower(name))//"'"
       end if
    end select

  end function initialiser_setup
!!!#############################################################################

end module initialiser
!!!#############################################################################
