!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains procedures for normalising input and output data
!!! module contains the following procedures:
!!! - linear_renormalise - renormalises input data to a range
!!! - renormalise_norm   - renormalises input data to a unit norm
!!! - renormalise_sum    - renormalises input data to a unit sum
!!!#############################################################################
module normalisation
  use constants, only: real32
  implicit none


  private

  public :: linear_renormalise
  public :: renormalise_norm
  public :: renormalise_sum
  
contains

!!!#############################################################################
!!! 
!!!#############################################################################
subroutine linear_renormalise(input, min, max)
  implicit none
  real(real32), dimension(:), intent(inout) :: input
  real(real32), optional, intent(in) :: min, max

  real(real32) :: lower, width
  real(real32) :: min_val, max_val

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
!!!#############################################################################


!!!#############################################################################
!!!
!!!#############################################################################
subroutine renormalise_norm(input, norm, mirror)
  implicit none
  real(real32), dimension(:), intent(inout) :: input
  real(real32), optional, intent(in) :: norm
  logical, optional, intent(in) :: mirror
 
  real(real32) :: scale

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
!!!#############################################################################


!!!#############################################################################
!!!
!!!#############################################################################
subroutine renormalise_sum(input, norm, mirror, magnitude)
  implicit none
  real(real32), dimension(:), intent(inout) :: input
  real(real32), optional, intent(in) :: norm
  logical, optional, intent(in) :: mirror, magnitude

  logical :: magnitude_
  
  real(real32) :: scale

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
!!!#############################################################################


end module normalisation
!!!#############################################################################
