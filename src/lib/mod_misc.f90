module athena__misc
  !! Module contains various miscellaneous functions and subroutines.
  !!
  !! This module contains various miscellaneous functions and subroutines that
  !! are used throughout the library. These include mathematical functions,
  !! string manipulation functions, and file I/O functions.
  !! Code copied from ARTEMIS with permission of the authors
  !! Ned Thaddeus Taylor and Francis Huw Davies
  !! https://github.com/ExeQuantCode/ARTEMIS
  use athena__constants, only: real32
  implicit none


  private

  public :: outer_product
  public :: Icount, grep, to_upper, to_lower, to_camel_case


contains

!###############################################################################
  pure function outer_product(a,b) result(c)
    !! Compute the outer product of two vectors
    implicit none

    ! Arguments
    real(real32), dimension(:), intent(in) :: a,b
    !! Input vectors
    real(real32), dimension(size(a),size(b)) :: c
    !! Outer product of the two vectors

    ! Local variables
    integer :: i,j
    !! Loop indices

    do i = 1, size(a)
       do j = 1, size(b)
          c(i,j) = a(i) * b(j)
       end do
    end do

    return
  end function outer_product
!###############################################################################


!###############################################################################
  integer function Icount(full_line,tmpchar)
    !! Count the number of words in a line
    implicit none

    ! Arguments
    character(*), intent(in) :: full_line
    !! Line to count the words of
    character(*), optional, intent(in) :: tmpchar
    !! Optional field separator

    ! Local variables
    character(len=:), allocatable :: fs
    !! Field separator
    integer ::items,pos,k,length
    !! Number of items, position, loop index, length of field separator


    items = 0
    pos = 1

    length=1
    if(present(tmpchar)) length=len(trim(tmpchar))
    allocate(character(len=length) :: fs)
    if(present(tmpchar)) then
       if(trim(tmpchar).ne." ")then
          fs=trim(tmpchar)
       else
          fs = tmpchar
       end if
    else
       fs = " "
    end if

    loop: do
       k = verify( full_line(pos:), fs )
       if (k.eq.0) exit loop
       items = items + 1
       pos = k + pos - 1
       k = scan( full_line(pos:), fs )
       if (k.eq.0) exit loop
       pos = k + pos - 1
    end do loop
    Icount = items
  end function Icount
!###############################################################################


!###############################################################################
  subroutine grep(unit,input,lstart)
    !! Search a file for a matching pattern
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number of the file to search
    character(*), intent(in) :: input
    !! Pattern to search for
    logical, optional, intent(in) :: lstart
    !! Optional boolean whether to rewind the file

    ! Local variables
    integer :: iostat
    !! I/O status
    character(1024) :: buffer
    !! Buffer to read lines into


    if(present(lstart))then
       if(lstart) rewind(unit)
    else
       rewind(unit)
    end if

    greploop: do
       read(unit,'(A100)',iostat=iostat) buffer
       if(iostat.lt.0) return
       if(index(trim(buffer),trim(input)).ne.0) exit greploop
    end do greploop
  end subroutine grep
!###############################################################################


!###############################################################################
  pure function to_upper(buffer) result(upper)
    !! Convert all characters in a string to upper case
    implicit none

    ! Arguments
    character(*), intent(in) :: buffer
    !! Input string
    character(len=:),allocatable :: upper
    !! Output string

    ! Local variables
    integer :: i
    !! Loop index
    integer :: j
    !! ASCII value of character


    allocate(character(len=len(buffer)) :: upper)
    do i = 1, len(buffer)
       j = iachar(buffer(i:i))
       if( j .ge. iachar("a") .and. j .le. iachar("z") )then
          upper(i:i) = achar( j - 32 )
       else
          upper(i:i) = buffer(i:i)
       end if
    end do

    return
  end function to_upper
!###############################################################################


!###############################################################################
  pure function to_lower(buffer) result(lower)
    !! Convert all characters in a string to lower case
    implicit none

    ! Arguments
    character(*), intent(in) :: buffer
    !! Input string
    character(len=:),allocatable :: lower
    !! Output string

    ! Local variables
    integer :: i
    !! Loop index
    integer :: j
    !! ASCII value of character


    allocate(character(len=len(buffer)) :: lower)
    do i = 1, len(buffer)
       j = iachar(buffer(i:i))
       if( j .ge. iachar("A") .and. j .le. iachar("Z") )then
          lower(i:i) = achar( j + 32 )
       else
          lower(i:i)=buffer(i:i)
       end if
    end do

    return
  end function to_lower
!###############################################################################


!###############################################################################
  pure function to_camel_case(input, capitalise_first_letter) result(output)
    !! Convert a string to camel case
    implicit none

    ! Arguments
    character(*), intent(in) :: input
    !! Input string
    logical, intent(in), optional :: capitalise_first_letter
    !! Boolean to capitalise the first letter
    character(len=:), allocatable :: output
    !! Output string

    ! Local variables
    integer :: i, j, len_input, idx
    !! Loop indices and length of input string
    character(:), allocatable :: input_lower
    !! Lowercase version of input string
    logical :: capitalise_first_letter_
    !! Local copy of capitalise_first_letter


    ! Default value for capitalise_first_letter
    capitalise_first_letter_ = .true.
    if(present(capitalise_first_letter)) &
         capitalise_first_letter_ = capitalise_first_letter

    ! Convert input to lowercase and allocate output
    input_lower = to_lower(trim(adjustl(input)))
    len_input = len_trim(input_lower)
    allocate(character(len=len_input) :: output)
    output(:) = ' '  ! Initialise output with spaces

    ! Convert to camel case
    i = 1
    j = 1
    do while ( i .lt. len_input )
       ! find the next word after the separator
       idx = verify(input_lower(i:), '_, ')
       if (idx .eq. 0) exit
       i = i + idx - 1

       ! Capitalise the first letter of the word
       if (i .le. len_input) then
          if (iachar(input_lower(i:i)) .ge. iachar('a') .and. &
               iachar(input_lower(i:i)) .le. iachar('z')) then
             output(j:j) = achar(iachar(input_lower(i:i)) - 32)
          else
             output(j:j) = input_lower(i:i)
          end if
          j = j + 1
          i = i + 1
       end if

       ! find the next word separator (underscore or space)
       idx = scan(input_lower(i:), '_, ')
       ! get the smallest of the two indices that is not zero
       if(idx .eq. 0) then
          output(j:len_input-i+j) = input_lower(i:len_input)
          exit
       else
          output(j:j + idx - 1) = input_lower(i:i + idx - 1)
          j = j + idx - 1
          i = i + idx - 1
       end if
    end do
    output = trim(adjustl(output))

    ! Capitalise the first letter if required
    if (capitalise_first_letter_.and. len(output) .gt. 0 .and. &
         iachar(output(1:1)) .ge. iachar("a") .and. &
         iachar(output(1:1)) .le. iachar("z") &
    ) then
       ! Capitalise the first letter if required
       output(1:1) = achar(iachar(output(1:1)) - 32)
    elseif(.not. capitalise_first_letter_ .and. &
         len(output) .gt. 0 .and. &
         iachar(output(1:1)) .ge. iachar("A") .and. &
         iachar(output(1:1)) .le. iachar("Z") &
    ) then
       ! Convert the first letter to lowercase if not capitalising
       output(1:1) = achar(iachar(output(1:1)) + 32)
    end if

    return
  end function to_camel_case
!###############################################################################

end module athena__misc
