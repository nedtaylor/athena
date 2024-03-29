program test_misc
  use misc
  implicit none

  logical :: success = .true.


!!!-----------------------------------------------------------------------------
!!! test to_upper and to_lower
!!!-----------------------------------------------------------------------------
  call check_strings(to_upper('abc'), 'ABC')
  call check_strings(to_upper('AbC'), 'ABC')
  call check_strings(to_upper('123'), '123')

  call check_strings(to_lower('ABC'), 'abc')
  call check_strings(to_lower('AbC'), 'abc')
  call check_strings(to_lower('123'), '123')

  call check_icount('abc def ghi', ' ', 3)
  call check_icount('abc   def   ghi', ' ', 3)
  call check_icount('abc,def,ghi', ',', 3)
  call check_icount('abc,,,def,,,ghi', ',', 3)
  call check_icount('', ' ', 0)


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_shuffle passed all tests'
  else
     write(0,*) 'test_shuffle failed one or more tests'
     stop 1
  end if

contains

!!!-----------------------------------------------------------------------------
!!! compare two strings
!!!-----------------------------------------------------------------------------
  subroutine check_strings(actual, expected)
    character(*), intent(in) :: actual
    character(*), intent(in) :: expected
  
    if (actual .ne. expected) then
      write(0,*) "Error: Strings are not equal."
      write(0,*) "Actual: ", actual
      write(0,*) "Expected: ", expected
      success = .false.
    end if
  end subroutine check_strings

!!!-----------------------------------------------------------------------------
!!! check counter returns correct number of words in string
!!!-----------------------------------------------------------------------------
  subroutine check_icount(full_line, tmpchar, expected)
    character(*), intent(in) :: full_line
    character(*), intent(in) :: tmpchar
    integer, intent(in) :: expected
    integer :: actual
  
    actual = Icount(full_line, tmpchar)
  
    if (actual .ne. expected) then
      write(0,*) "Error: Word counts are not equal."
      write(0,*) "Line: ", full_line
      write(0,*) "Char: ", tmpchar
      write(0,*) "Actual: ", actual
      write(0,*) "Expected: ", expected
      success = .false.
    end if
  end subroutine check_icount

end program test_misc
