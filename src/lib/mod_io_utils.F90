module athena__io_utils
  !! Module for handling errors and io calls in the program.
  !!
  !! This module provides the expected procedure for stopping a program.
  !! If in testing mode, the stop can be suppressed.
  implicit none
  logical :: test_error_handling = .false.

  character(len=*), parameter :: athena__version__ = "2.0-alpha"

  private

  public :: athena__version__
  public :: test_error_handling
  public :: stop_program, print_warning
  public :: print_version, print_build_info


contains

!###############################################################################
  subroutine stop_program(message, exit_code)
    !! Stop the program and print an error message.
    implicit none
    character(len=*), intent(in) :: message
    integer, intent(in), optional :: exit_code

    integer :: exit_code_

    if(present(exit_code)) then
       exit_code_ = exit_code
    else
       exit_code_ = 1
    end if

    write(0,*) 'ERROR: ', trim(message)
    if(.not.test_error_handling)then
       stop exit_code_
    end if
  end subroutine stop_program
!###############################################################################


!###############################################################################
  subroutine print_warning(message)
    !! Print a warning message
    implicit none
    character(len=*), intent(in) :: message

    write(0,*) 'WARNING: ', trim(message)
  end subroutine print_warning
!###############################################################################


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
