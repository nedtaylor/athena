program test_io_utils
  use athena__io_utils
  implicit none

  ! Test variables
  logical :: success = .true.

  ! Test print_version subroutine
  call print_version()

  ! Test print_build_info subroutine
  call print_build_info()

  !-----------------------------------------------------------------------------
  ! check for any failed tests
  !-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_io_utils passed all tests'
  else
     write(0,*) 'test_io_utils failed one or more tests'
     stop 1
  end if

end program test_io_utils
