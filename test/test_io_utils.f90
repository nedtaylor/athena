program test_io_utils
  use athena__io_utils
  implicit none

  ! Test variables
  logical :: success = .true.
  character(100) :: message

  ! Test stop_program subroutine
  test_error_handling = .true.
  message = "Test error message"
  call stop_program(message)

  ! Test print_warning subroutine
  call print_warning("This is a test warning message")
  
  ! Test print_version subroutine
  call print_version()

  ! Test print_build_info subroutine
  call print_build_info()

  !-----------------------------------------------------------------------------
  ! check for any failed tests
  !-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_misc_linalg passed all tests'
  else
     write(0,*) 'test_misc_linalg failed one or more tests'
     stop 1
  end if

end program test_io_utils