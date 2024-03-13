!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor and Francis Huw Davies
!!!#############################################################################
!!! module contains various constants used in the code
!!!#############################################################################
module constants
  implicit none
  integer, parameter, public :: real12 = Selected_real_kind(6,37)!(15,307)
  real(real12), parameter, public :: pi = 4.e0_real12*atan(1._real12)
  real(real12), parameter, public :: INF = huge(0._real12)
  complex(real12), parameter, public :: imag=(0._real12, 1._real12)
  integer, public :: ierror = -1
end module constants
