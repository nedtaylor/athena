module athena__constants
  !! Module with global constants
  !!
  !! This module contains global constants that may be used throughout the
  !! library.
  !! Code copied from ARTEMIS with permission of the authors
  !! Ned Thaddeus Taylor and Francis Huw Davies
  !! https://github.com/ExeQuantCode/ARTEMIS
  use coreutils, only: real32
  implicit none
  real(real32), parameter, public :: pi = 4.e0_real32*atan(1._real32)
  real(real32), parameter, public :: INF = huge(0._real32)
  complex(real32), parameter, public :: imag=(0._real32, 1._real32)
  integer, public :: ierror = -1
end module athena__constants
