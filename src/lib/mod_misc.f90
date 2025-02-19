module athena__misc
  !! Module contains various miscellaneous functions and subroutines.
  !!
  !! Procedures in this module have been written by Ned Thaddeus Taylor and
  !! Francis Huw Davies
  use athena__constants, only: real32
  implicit none


  private

  public :: outer_product
  public :: Icount, grep, to_upper, to_lower


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

end module athena__misc