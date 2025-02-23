module athena__initialiser
  !! Module containing functions to set up initialisers
  !!
  !! This module contains functions to set up initialisers for the weights and
  !! biases of a neural network model
  !! Examples of initialsers in keras: https://keras.io/api/layers/initializers/
  use athena__io_utils, only: stop_program
  use athena__misc, only: to_lower
  use athena__misc_types, only: initialiser_type
  use athena__initialiser_glorot, only: glorot_uniform, glorot_normal
  use athena__initialiser_he, only: he_uniform, he_normal
  use athena__initialiser_lecun, only: lecun_uniform, lecun_normal
  use athena__initialiser_ones, only: ones
  use athena__initialiser_zeros, only: zeros
  use athena__initialiser_ident, only: ident
  use athena__initialiser_gaussian, only: gaussian
  implicit none


  private

  public :: initialiser_setup, get_default_initialiser


contains

!###############################################################################
  function get_default_initialiser(activation, is_bias) result(name)
    !! Get the default initialiser based on the activation function
    implicit none

    ! Arguments
    character(*), intent(in) :: activation
    !! Activation function
    logical, optional, intent(in) :: is_bias
    !! Boolean whether initialiser is for bias

    character(:), allocatable :: name


    !---------------------------------------------------------------------------
    ! If bias, use default initialiser of zero
    !---------------------------------------------------------------------------
    if(present(is_bias))then
       if(is_bias) name = "zeros"
       return
    end if


    !---------------------------------------------------------------------------
    ! Set default initialiser based on activation
    !---------------------------------------------------------------------------
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
!###############################################################################


!###############################################################################
  function initialiser_setup(name, error) result(initialiser)
    !! Set up the initialiser function
    implicit none

    ! Arguments
    class(initialiser_type), allocatable :: initialiser
    !! Initialiser function
    character(*), intent(in) :: name
    !! Name of initialiser
    integer, optional, intent(out) :: error
    !! Error code

    ! Local variables
    character(256) :: err_msg
    !! Error message


    !---------------------------------------------------------------------------
    ! Set initialiser function
    !---------------------------------------------------------------------------
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
          write(err_msg,'("Incorrect initialiser name given ''",A,"''")') &
               trim(to_lower(name))
          call stop_program(trim(err_msg))
          return
       end if
    end select

  end function initialiser_setup
!###############################################################################

end module athena__initialiser
