module athena__initialiser
  !! Module containing functions to set up initialisers
  !!
  !! This module contains functions to set up initialisers for the weights and
  !! biases of a neural network model
  !! Examples of initialsers in keras: https://keras.io/api/layers/initializers/
  use coreutils, only: stop_program, to_lower
  use athena__misc_types, only: initialiser_type
  use athena__initialiser_glorot, only: &
       glorot_uniform_init_type, glorot_normal_init_type
  use athena__initialiser_he, only: he_uniform_init_type, he_normal_init_type
  use athena__initialiser_lecun, only: &
       lecun_uniform_init_type, lecun_normal_init_type
  use athena__initialiser_ones, only: ones_init_type
  use athena__initialiser_zeros, only: zeros_init_type
  use athena__initialiser_ident, only: ident_init_type
  use athena__initialiser_gaussian, only: gaussian_init_type
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
  function initialiser_setup(input, error) result(initialiser)
    !! Set up the initialiser function
    implicit none

    ! Arguments
    class(initialiser_type), allocatable :: initialiser
    !! Initialiser function
    class(*) :: input
    !! Name of initialiser or initialiser object
    integer, optional, intent(out) :: error
    !! Error code

    ! Local variables
    character(256) :: err_msg
    !! Error message


    !---------------------------------------------------------------------------
    ! Set initialiser function
    !---------------------------------------------------------------------------
    select type(input)
    class is(initialiser_type)
       initialiser = input
    type is(character(*))
       select case(trim(to_lower(input)))
       case("glorot_uniform")
          initialiser = glorot_uniform_init_type()
       case("glorot_normal")
          initialiser = glorot_normal_init_type()
       case("he_uniform")
          initialiser = he_uniform_init_type()
       case("he_normal")
          initialiser = he_normal_init_type()
       case("lecun_uniform")
          initialiser = lecun_uniform_init_type()
       case("lecun_normal")
          initialiser = lecun_normal_init_type()
       case("ones")
          initialiser = ones_init_type()
       case("zeros")
          initialiser = zeros_init_type()
       case("ident")
          initialiser = ident_init_type()
       case("gaussian")
          initialiser = gaussian_init_type()
       case("normal")
          initialiser = gaussian_init_type(name="normal")
       case default
          if(present(error))then
             error = -1
             return
          else
             write(err_msg,'("Incorrect initialiser name given ''",A,"''")') &
                  trim(to_lower(input))
             call stop_program(trim(err_msg))
             return
          end if
       end select
    class default
       if(present(error))then
          error = -1
          return
       else
          write(err_msg,'("Unknown input type given for initialiser setup")')
          call stop_program(trim(err_msg))
          return
       end if
    end select

  end function initialiser_setup
!###############################################################################

end module athena__initialiser
