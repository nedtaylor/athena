program test_misc
  use athena__misc
  implicit none

  logical :: success = .true.


!-------------------------------------------------------------------------------
! test to_upper and to_lower
!-------------------------------------------------------------------------------
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

  call check_camel_case('abc def ghi', 'AbcDefGhi', .true.)
  call check_camel_case('abc   def   ghi', 'AbcDefGhi', .true.)
  call check_camel_case('abc_def_ghi', 'AbcDefGhi', .true.)
  call check_camel_case('abc__def__ghi', 'abcDefGhi', .false.)


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_shuffle passed all tests'
  else
     write(0,*) 'test_shuffle failed one or more tests'
     stop 1
  end if

contains

!-------------------------------------------------------------------------------
! compare two strings
!-------------------------------------------------------------------------------
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

!-------------------------------------------------------------------------------
! check counter returns correct number of words in string
!-------------------------------------------------------------------------------
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

!-------------------------------------------------------------------------------
! check camel case conversion
!-------------------------------------------------------------------------------
  subroutine check_camel_case(input_string, expected, first_letter_capitalised)
    character(*), intent(in) :: input_string
    character(*), intent(in) :: expected
    character(len=:), allocatable :: actual
    logical :: first_letter_capitalised

    actual = to_camel_case(input_string, first_letter_capitalised)

    if (actual .ne. expected) then
       write(0,*) "Error: Camel case conversion is not equal."
       write(0,*) "Input: ", input_string
       write(0,*) "Actual: ", actual
       write(0,*) "Expected: ", expected
       success = .false.
    end if
  end subroutine check_camel_case

end program test_misc
