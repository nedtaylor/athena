!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor and Francis Huw Davies
!!!#############################################################################
!!! module contains various miscellaneous functions and subroutines.
!!! module includes the following functions and subroutines:
!!! - outer_product - calculates the outer product of two vectors
!!! - Icount        - counts words on line
!!! - grep          - finds 1st line containing the pattern
!!! - to_upper      - converts all characters in string to upper case
!!! - to_lower      - converts all characters in string to lower case
!!!#############################################################################
module misc
  use constants, only: real12
  implicit none

  private
  public :: outer_product
  public :: Icount, grep, to_upper, to_lower


contains

!!!#####################################################
!!! outer product
!!!#####################################################
  pure function outer_product(a,b) result(c)
    implicit none
    real(real12), dimension(:), intent(in) :: a,b
    real(real12), dimension(size(a),size(b)) :: c
    integer :: i,j

    do i=1,size(a)
       do j=1,size(b)
          c(i,j)=a(i)*b(j)
       end do
    end do

    return
  end function outer_product 
!!!#####################################################


!!!#####################################################
!!! counts the number of words on a line
!!!#####################################################
  integer function Icount(full_line,tmpchar)
    implicit none
    character(*), intent(in) :: full_line
    character(*), optional, intent(in) :: tmpchar

    character(len=:), allocatable :: fs
    integer ::items,pos,k,length
    items=0
    pos=1

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
       fs=" "
    end if

    loop: do
       k=verify(full_line(pos:),fs)
       if (k.eq.0) exit loop
       items=items+1
       pos=k+pos-1
       k=scan(full_line(pos:),fs)
       if (k.eq.0) exit loop
       pos=k+pos-1
    end do loop
    Icount=items
  end function Icount
!!!#####################################################


!!!#####################################################
!!! grep 
!!!#####################################################
!!! searches a file untill it finds the mattching patern
  subroutine grep(unit,input,lstart)
    implicit none
    integer, intent(in) :: unit
    character(*), intent(in) :: input
    logical, optional, intent(in) :: lstart

    integer :: Reason
    character(1024) :: buffer
    !  character(1024), intent(out), optional :: linechar
    if(present(lstart))then
       if(lstart) rewind(unit)
    else
       rewind(unit)
    end if

    greploop: do
       read(unit,'(A100)',iostat=Reason) buffer
       if(Reason.lt.0) return
       if(index(trim(buffer),trim(input)).ne.0) exit greploop
    end do greploop
  end subroutine grep
!!!#####################################################


!!!#####################################################
!!! converts all characters in string to upper case
!!!#####################################################
  pure function to_upper(buffer) result(upper)
    implicit none
    character(*), intent(in) :: buffer
    character(len=:),allocatable :: upper

    integer :: i,j


    allocate(character(len=len(buffer)) :: upper)
    do i=1,len(buffer)
       j=iachar(buffer(i:i))
       if(j.ge.iachar("a").and.j.le.iachar("z"))then
          upper(i:i)=achar(j-32)
       else
          upper(i:i)=buffer(i:i)
       end if
    end do

    return
  end function to_upper
!!!#####################################################


!!!#####################################################
!!! converts all characters in string to lower case
!!!#####################################################
  pure function to_lower(buffer) result(lower)
    implicit none
    character(*), intent(in) :: buffer
    character(len=:),allocatable :: lower

    integer :: i,j


    allocate(character(len=len(buffer)) :: lower)
    do i=1,len(buffer)
       j=iachar(buffer(i:i))
       if(j.ge.iachar("A").and.j.le.iachar("Z"))then
          lower(i:i)=achar(j+32)
       else
          lower(i:i)=buffer(i:i)
       end if
    end do

    return
  end function to_lower
!!!#####################################################

end module misc
