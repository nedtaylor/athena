!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module weight_initialiser
  use constants, only: real12
  implicit none


  private

  public :: he_uniform
  public :: zeros

!!!!! HAVE A FUNCTION THAT SETS IT UP BASED ON THE NAME PROVIDED
!!!!! DO THE SAME FOR ACTIVATION (much neater than one per code)
!!!!! ALSO, HAVE THE CV, PL, FC, etc, LAYERS AS CLASSES
!!!!! ... they may be able to be appended on to each other

  
  !! normalise (kernel_initialise?) to number of input units
  !! He uniform initialiser
  !! make an initialiser that takes in an assumed rank
  !! it then does product(shape(weight)) OR size(weight)
  !! could always use select rank(x) statement if needed
  !! https://keras.io/api/layers/initializers/
  
  

contains

!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine he_uniform(input, fan_in)
    implicit none
    integer, intent(in) :: fan_in  !number of input parameters
    real(real12), dimension(..), intent(out) :: input

    real(real12) :: scale
!!! HAVE ASSUMED RANK

!!! DO TESTS TO CONFIRM THAT RANDOM NUMBER IS CARRIED THROUGH
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
       input = (input *2._real12 - 1._real12) * scale
    end select
    
  end subroutine he_uniform  
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine zeros(input)
    implicit none
    real(real12), intent(out) :: input

    input = 0._real12

  end subroutine zeros
!!!#############################################################################




end module weight_initialiser
!!!#############################################################################
