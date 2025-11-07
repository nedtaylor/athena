module athena__normalisation
  !! Module containing procedures for normalising input and output data
  use coreutils, only: real32
  implicit none


  private

  public :: linear_renormalise
  public :: renormalise_norm
  public :: renormalise_sum



contains

!###############################################################################
  subroutine linear_renormalise(input, min, max)
    !! Renormalise input data to a specified range
    implicit none

    ! Arguments
    real(real32), dimension(:), intent(inout) :: input
    !! Input data to be renormalised
    real(real32), optional, intent(in) :: min, max
    !! Minimum and maximum values for renormalisation

    ! Local variables
    real(real32) :: lower, width
    !! Lower bound and width of the range
    real(real32) :: min_val, max_val
    !! Minimum and maximum values of the input data

    min_val = minval(input)
    max_val = maxval(input)

    if(present(min))then
       lower = min
    else
       lower = -1._real32
    end if
    if(present(max))then
       width = max - min
    else
       width = 2._real32
    end if

    input = lower + width * (input - min_val)/(max_val - min_val)
  end subroutine linear_renormalise
!###############################################################################


!###############################################################################
  subroutine renormalise_norm(input, norm, mirror)
    !! Renormalise input data to a unit norm
    implicit none

    ! Arguments
    real(real32), dimension(:), intent(inout) :: input
    !! Input data to be renormalised
    real(real32), optional, intent(in) :: norm
    !! Desired norm value
    logical, optional, intent(in) :: mirror
    !! Boolean whether the data should be mirrored

    ! Local variables
    real(real32) :: scale
    !! Scaling factor

    if(present(norm))then
       scale = norm
    else
       scale = 1._real32
    end if

    if(present(mirror))then
       if(mirror) call linear_renormalise(input)
    end if
    input = input * scale/sqrt(dot_product(input,input))
  end subroutine renormalise_norm
!###############################################################################


!###############################################################################
  subroutine renormalise_sum(input, norm, mirror, magnitude)
    !! Renormalise input data to a unit sum
    implicit none

    ! Arguments
    real(real32), dimension(:), intent(inout) :: input
    !! Input data to be renormalised
    real(real32), optional, intent(in) :: norm
    !! Desired sum value
    logical, optional, intent(in) :: mirror, magnitude
    !! Booleans whether the data should be mirrored or use magnitude

    ! Local variables
    logical :: magnitude_
    !! Flag to indicate if magnitude should be used
    real(real32) :: scale
    !! Scaling factor

    if(present(norm))then
       scale = norm
    else
       scale = 1._real32
    end if

    if(present(mirror))then
       if(mirror) call linear_renormalise(input)
    end if

    if(present(magnitude)) magnitude_ = magnitude
    if(present(magnitude))then
       scale = scale/sum(abs(input))
    else
       scale = scale/sum(input)
    end if
    input = input * scale
  end subroutine renormalise_sum
!###############################################################################

end module athena__normalisation
