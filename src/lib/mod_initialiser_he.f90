!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module initialiser_he
  use constants, only: real12
  use custom_types, only: initialiser_type
  implicit none


  type, extends(initialiser_type) :: he_uniform_type
   contains
     procedure, pass(this) :: initialise => he_uniform_initialise
  end type he_uniform_type
  type(he_uniform_type) :: he_uniform
  type, extends(initialiser_type) :: he_normal_type
   contains
     procedure, pass(this) :: initialise => he_normal_initialise
  end type he_normal_type
  type(he_normal_type) :: he_normal

  
  private

  public :: he_uniform
  public :: he_normal


contains

!!!#############################################################################
!!! He initialisation (uniform)
!!! Kaiming initialisation
!!! MSRA initialisation
!!! https://doi.org/10.48550/arXiv.1502.01852
!!!#############################################################################
  subroutine he_uniform_initialise(this, input, fan_in, fan_out)
    implicit none
    class(he_uniform_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    real(real12) :: scale
!!! HAVE ASSUMED RANK

    scale = sqrt(6._real12/real(fan_in,real12))
    select rank(input)
    rank(0)
       call random_number(input)
       input = (input *2._real12 - 1._real12) * scale
    rank(1)
       call random_number(input)
       input = (input *2._real12 - 1._real12) * scale
    rank(2)
       call random_number(input)
       input = input * scale
    rank(3)
       call random_number(input)
       input = input * scale
    end select
    
  end subroutine he_uniform_initialise
!!!#############################################################################


!!!#############################################################################
!!! He initialisation (normal)
!!! Kaiming initialisation
!!! MSRA initialisation
!!!#############################################################################
  subroutine he_normal_initialise(this, input, fan_in, fan_out)
    implicit none
    class(he_normal_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    real(real12) :: scale
!!! HAVE ASSUMED RANK

    scale = sqrt(2._real12/real(fan_in,real12))
    select rank(input)
    rank(0)
       call random_number(input)
       input = input * scale
    rank(1)
       call random_number(input)
       input = input * scale
    rank(2)
       call random_number(input)
       input = input * scale
    rank(3)
       call random_number(input)
       input = input * scale
    end select
    
  end subroutine he_normal_initialise
!!!#############################################################################

end module initialiser_he
!!!#############################################################################
