!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module initialiser_lecun
  use constants, only: real12
  use custom_types, only: initialiser_type
  implicit none


  type, extends(initialiser_type) :: lecun_uniform_type
   contains
     procedure, pass(this) :: initialise => lecun_uniform_initialise
  end type lecun_uniform_type
  type(lecun_uniform_type) :: lecun_uniform
  type, extends(initialiser_type) :: lecun_normal_type
   contains
     procedure, pass(this) :: initialise => lecun_normal_initialise
  end type lecun_normal_type
  type(lecun_normal_type) :: lecun_normal

  
  private

  public :: lecun_uniform
  public :: lecun_normal


contains

!!!#############################################################################
!!! LeCun initialisation (uniform)
!!! https://dl.acm.org/doi/10.5555/645754.668382
!!!#############################################################################
  subroutine lecun_uniform_initialise(this, input, fan_in, fan_out)
    implicit none
    class(lecun_uniform_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    real(real12) :: scale

!!! HAVE ASSUMED RANK

    scale = sqrt(3._real12/real(fan_in,real12))
    select rank(input)
    rank(0)
       call random_number(input)
       input = (input *2._real12 - 1._real12) * scale
    rank(1)
       call random_number(input)
       input = (input *2._real12 - 1._real12) * scale
    rank(2)
       call random_number(input)
       input = (input *2._real12 - 1._real12) * scale
    rank(3)
       call random_number(input)
       input = (input *2._real12 - 1._real12) * scale
    rank(4)
       call random_number(input)
       input = (input *2._real12 - 1._real12) * scale
    rank(5)
       call random_number(input)
       input = (input *2._real12 - 1._real12) * scale
    end select
    
  end subroutine lecun_uniform_initialise
!!!#############################################################################


!!!#############################################################################
!!! LeCun initialisation (normal)
!!! https://dl.acm.org/doi/10.5555/645754.668382
!!!#############################################################################
  subroutine lecun_normal_initialise(this, input, fan_in, fan_out)
    implicit none
    class(lecun_normal_type), intent(inout) :: this
    real(real12), dimension(..), intent(out) :: input
    integer, optional, intent(in) :: fan_in, fan_out ! no. in and out params

    real(real12) :: scale

!!! HAVE ASSUMED RANK

    scale = sqrt(1._real12/real(fan_in,real12))
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
    rank(4)
       call random_number(input)
       input = input * scale
    rank(5)
       call random_number(input)
       input = input * scale
    end select
    
  end subroutine lecun_normal_initialise
!!!#############################################################################

end module initialiser_lecun
!!!#############################################################################
