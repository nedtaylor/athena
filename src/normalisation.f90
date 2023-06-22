module normalisation
  use constants, only: real12
  implicit none


contains

subroutine linear_renormalise(input)
  implicit none
  real(real12), dimension(:), intent(inout) :: input

  real(real12) :: min_val, max_val

  min_val = minval(input)
  max_val = maxval(input)

  input = -1._real12 + 2._real12 * (input - min_val)/(max_val - min_val)
  

end subroutine linear_renormalise

end module normalisation
