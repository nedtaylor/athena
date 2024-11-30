module constants_mnist
  implicit none
  integer, parameter, public :: real32 = Selected_real_kind(6,37)!(15,307)
  real(real32), parameter, public :: k_b = 1.3806503e-23_real32
  real(real32), parameter, public :: k_b_ev = 8.61733326e-5_real32
  real(real32), parameter, public :: hbar = 1.05457148e-34_real32
  real(real32), parameter, public :: hbar_ev = 6.58211957e-16_real32
  real(real32), parameter, public :: h = 6.626068e-34_real32
  real(real32), parameter, public :: atomic_mass=1.66053907e-27_real32
  real(real32), parameter, public :: neutron_mass=1.67262158e-27_real32
  real(real32), parameter, public :: electron_mass=9.109383562e-31_real32
  real(real32), parameter, public :: elem_charge=1.60217662e-19_real32
  real(real32), parameter, public :: avogadros=6.022e23_real32
  real(real32), parameter, public :: bohrtoang=0.529177249_real32
  real(real32), parameter, public :: pi = 4.e0_real32*atan(1._real32)
  real(real32), parameter, public :: c = 0.26246582250210965422e0_real32
  real(real32), parameter, public :: c_vasp = 0.262465831e0_real32
  real(real32), parameter, public :: INF = huge(0._real32)
  complex(real32), parameter, public :: imag=(0._real32, 1._real32)
  integer, public :: ierror = -1
end module constants_mnist
