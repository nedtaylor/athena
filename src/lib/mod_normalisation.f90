!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module normalisation
  use constants, only: real12
  implicit none


  private

  public :: linear_renormalise
  public :: renormalise_norm
  public :: renormalise_sum


contains

!!!########################################################################
!!! 
!!!########################################################################
subroutine linear_renormalise(input, min, max)
  implicit none
  real(real12), dimension(:), intent(inout) :: input
  real(real12), optional, intent(in) :: min, max

  real(real12) :: lower, width
  real(real12) :: min_val, max_val

  min_val = minval(input)
  max_val = maxval(input)

  if(present(min))then
     lower = min
  else
     lower = -1._real12
  end if
  if(present(max))then
     width = max - min
  else
     width = 2._real12
  end if

  input = lower + width * (input - min_val)/(max_val - min_val)
 
end subroutine linear_renormalise
!!!########################################################################


!!!########################################################################
!!!
!!!########################################################################
subroutine renormalise_norm(input, norm, mirror)
  implicit none
  real(real12), dimension(:), intent(inout) :: input
  real(real12), optional, intent(in) :: norm
  logical, optional, intent(in) :: mirror
 
  real(real12) :: scale

  if(present(norm))then
     scale = norm
  else
     scale = 1._real12
  end if
  
  if(mirror)then
     call linear_renormalise(input)
  end if
  input = input * scale/sqrt(dot_product(input,input))

end subroutine renormalise_norm
!!!########################################################################


!!!########################################################################
!!!
!!!########################################################################
subroutine renormalise_sum(input, norm, mirror, magnitude)
  implicit none
  real(real12), dimension(:), intent(inout) :: input
  real(real12), optional, intent(in) :: norm
  logical, optional, intent(in) :: mirror, magnitude
  
  real(real12) :: scale

  if(present(norm))then
     scale = norm
  else
     scale = 1._real12
  end if

  if(present(mirror))then
     if(mirror) &
          call linear_renormalise(input)
  end if
  
  if(present(magnitude))then
     if(magnitude)then
        scale = scale/sum(abs(input))
     else
        scale = scale/sum(input)
     end if
  else
     scale = scale/sum(input)
  end if
  input = input * scale

end subroutine renormalise_sum
!!!########################################################################


end module normalisation
!!!#############################################################################
