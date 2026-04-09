module athena__io_utils
  !! Module for handling errors and io calls in the program.
  !!
  !! This module provides the expected procedure for stopping a program.
  implicit none

  character(len=*), parameter :: athena__version__ = "2.1.0"

  private

  public :: athena__version__
  public :: print_version, print_build_info


contains

!###############################################################################
  subroutine print_version()
    !! Print the version number of the program.
    implicit none

    write(*,'("version: ",A)') athena__version__
  end subroutine print_version
!###############################################################################


!###############################################################################
  subroutine print_build_info()
    !! Print the build information of the program.
    implicit none

    write(*,'("ATHENA: &
         &Adaptive Training for High Efficiency Neural network Applications")')
    write(*,'(" version: ",A)') athena__version__
    write(*,'(" (build ",A,1X,A,")")') __DATE__, __TIME__

  end subroutine print_build_info
!###############################################################################

end module athena__io_utils
