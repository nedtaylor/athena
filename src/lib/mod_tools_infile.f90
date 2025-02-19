!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor and Francis Huw Davies
!!! Code part of the ARTEMIS group (Hepplestone research group).
!!! Think Hepplestone, think HRG.
!!!#############################################################################
!!! module contains customn input file reading functions and subroutines.
!!! module includes the following functionsand subroutines:
!!! - assign_val  - assign a value to a variable
!!! - assign_vec  - assign a vector to a variable
!!! - getline     - return line using grep and goes back to start of line
!!! - rm_comments - remove comments from a string (anything after ! or #)
!!! - cat         - cat lines until user-defined end string is encountered
!!! - stop_check  - check for <STOP> file and LSTOP or LABORT tags inside
!!!#############################################################################
module athena__tools_infile
  !! Module containing custom input file reading functions and subroutines
  use athena__misc, only: grep, icount
  implicit none


  private

  public :: assign_val, assign_vec
  public :: getline, rm_comments
  public :: stop_check


  interface assign_val
     !! Interface for assigning a value to a variable
     procedure assignI, assignR, assignD, assignS, assignL
  end interface assign_val

  interface assign_vec
     !! Interface for assigning a vector to a variable
     procedure assignIvec, assignRvec, assignDvec
  end interface assign_vec

contains

!###############################################################################
  function val(buffer) result(output)
    !! Extract the section of buffer that occurs after an "="
    implicit none

    ! Arguments
    character(*), intent(in) :: buffer
    !! Input buffer

    ! Local variables
    character(100) :: output
    !! Extracted value

    output = trim(adjustl(buffer((scan(buffer, "=") + 1):)))
  end function val
!###############################################################################


!###############################################################################
  subroutine getline(unit, pattern, buffer)
    !! Get the line from a grep and assign it to buffer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit to read from
    character(*), intent(in) :: pattern
    !! Pattern to grep for
    character(*), intent(out) :: buffer
    !! Buffer to assign line to

    ! Local variables
    integer :: iostat
    !! I/O status

    call grep(unit, pattern)
    backspace(unit)
    read(unit, '(A)', iostat=iostat) buffer
  end subroutine getline
!###############################################################################


!###############################################################################
  subroutine assignI(buffer, variable, found, keyword)
    !! Assign an integer to variable
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    integer, intent(out) :: variable
    !! Variable to assign data to
    integer, intent(inout) :: found
    !! Count for finding variable
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from

    ! Local variables
    character(1024) :: buffer2
    !! Temporary buffer

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, "=") /= 0) buffer2 = val(buffer)
    if(trim(adjustl(buffer2)) /= '') then
       found = found + 1
       read(buffer2, *) variable
    end if
  end subroutine assignI
!###############################################################################


!###############################################################################
  subroutine assignIvec(buffer, variable, found, keyword)
    !! Assign an arbitrary length vector of integers to variable
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    integer, dimension(:), intent(out) :: variable
    !! Variable to assign data to
    integer, intent(inout) :: found
    !! Count for finding variable
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from

    ! Local variables
    integer :: i
    !! Loop index
    character(1024) :: buffer2
    !! Temporary buffer

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, "=") /= 0) buffer2 = val(buffer)
    if(trim(adjustl(buffer2)) /= '') then
       found = found + 1
       if(icount(buffer2) == 1 .and. icount(buffer2) /= size(variable)) then
          read(buffer2, *) variable(1)
          variable = variable(1)
       else
          read(buffer2, *) (variable(i), i = 1, size(variable))
       end if
    end if
  end subroutine assignIvec
!###############################################################################


!###############################################################################
  subroutine assignR(buffer, variable, found, keyword)
    !! Assign a real to variable
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    real, intent(out) :: variable
    !! Variable to assign data to
    integer, intent(inout) :: found
    !! Count for finding variable
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from

    ! Local variables
    character(1024) :: buffer2
    !! Temporary buffer

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, "=") /= 0) buffer2 = val(buffer)
    if(trim(adjustl(buffer2)) /= '') then
       found = found + 1
       read(buffer2, *) variable
    end if
  end subroutine assignR
!###############################################################################


!###############################################################################
  subroutine assignRvec(buffer, variable, found, keyword)
    !! Assign an arbitrary length vector of reals to variable
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    real, dimension(:), intent(out) :: variable
    !! Variable to assign data to
    integer, intent(inout) :: found
    !! Count for finding variable
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from

    ! Local variables
    integer :: i
    !! Loop index
    character(1024) :: buffer2
    !! Temporary buffer

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, "=") /= 0) buffer2 = val(buffer)
    if(trim(adjustl(buffer2)) /= '') then
       found = found + 1
       if(icount(buffer2) == 1 .and. icount(buffer2) /= size(variable)) then
          read(buffer2, *) variable(1)
          variable = variable(1)
       else
          read(buffer2, *) (variable(i), i = 1, size(variable))
       end if
    end if
  end subroutine assignRvec
!###############################################################################


!###############################################################################
  subroutine assignD(buffer, variable, found, keyword)
    !! Assign a double precision to variable
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    double precision, intent(out) :: variable
    !! Variable to assign data to
    integer, intent(inout) :: found
    !! Count for finding variable
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from

    ! Local variables
    character(1024) :: buffer2
    !! Temporary buffer

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, "=") /= 0) buffer2 = val(buffer)
    if(trim(adjustl(buffer2)) /= '') then
       found = found + 1
       read(buffer2, *) variable
    end if
  end subroutine assignD
!###############################################################################


!###############################################################################
  subroutine assignDvec(buffer, variable, found, keyword)
    !! Assign an arbitrary length vector of double precision to variable
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    double precision, dimension(:), intent(out) :: variable
    !! Variable to assign data to
    integer, intent(inout) :: found
    !! Count for finding variable
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from

    ! Local variables
    integer :: i
    !! Loop index
    character(1024) :: buffer2
    !! Temporary buffer

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, "=") /= 0) buffer2 = val(buffer)
    if(trim(adjustl(buffer2)) /= '') then
       found = found + 1
       if(icount(buffer2) == 1 .and. icount(buffer2) /= size(variable)) then
          read(buffer2, *) variable(1)
          variable = variable(1)
       else
          read(buffer2, *) (variable(i), i = 1, size(variable))
       end if
    end if
  end subroutine assignDvec
!###############################################################################


!###############################################################################
  subroutine assignS(buffer, variable, found, keyword)
    !! Assign a string to variable
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    character(*), intent(out) :: variable
    !! Variable to assign data to
    integer, intent(inout) :: found
    !! Count for finding variable
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from

    ! Local variables
    character(1024) :: buffer2
    !! Temporary buffer

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, "=") /= 0) buffer2 = val(buffer)
    if(trim(adjustl(buffer2)) /= '') then
       found = found + 1
       read(buffer2, '(A)') variable
    end if
  end subroutine assignS
!###############################################################################


!###############################################################################
  subroutine assignL(buffer, variable, found, keyword)
    !! Assign a logical to variable (T/t/1 and F/f/0 accepted)
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    logical, intent(out) :: variable
    !! Variable to assign data to
    integer, intent(inout) :: found
    !! Count for finding variable
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from

    ! Local variables
    character(1024) :: buffer2
    !! Temporary buffer

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, "=") /= 0) buffer2 = val(buffer)
    if(trim(adjustl(buffer2)) /= '') then
       found = found + 1
       if(index(buffer2, "T") /= 0 .or. index(buffer2, "t") /= 0 .or. index(buffer2, "1") /= 0) then
          variable = .TRUE.
       end if
       if(index(buffer2, "F") /= 0 .or. index(buffer2, "f") /= 0 .or. index(buffer2, "0") /= 0) then
          variable = .FALSE.
       end if
    end if
  end subroutine assignL
!###############################################################################


!###############################################################################
  subroutine rm_comments(buffer, iline)
    !! Remove comment from a string (anything after ! or #)
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    integer, optional, intent(in) :: iline
    !! Line number

    ! Local variables
    integer :: lbracket, rbracket, iline_
    !! Bracket positions and line number

    iline_ = 0
    if(present(iline)) iline_ = iline

    if(scan(buffer, '!') /= 0) buffer = buffer(:(scan(buffer, '!') - 1))
    if(scan(buffer, '#') /= 0) buffer = buffer(:(scan(buffer, '#') - 1))
    do while(scan(buffer, '(') /= 0 .or. scan(buffer, ')') /= 0)
       lbracket = scan(buffer, '(', back = .true.)
       rbracket = scan(buffer(lbracket:), ')')
       if(lbracket == 0 .or. rbracket == 0) then
          write(6, '(A,I0)') ' NOTE: a bracketing error was encountered on line ', iline_
          buffer = ""
          return
       end if
       rbracket = rbracket + lbracket - 1
       buffer = buffer(:(lbracket - 1)) // buffer((rbracket + 1):)
    end do
  end subroutine rm_comments
!###############################################################################


!###############################################################################
  function stop_check(file) result(output)
    !! Logical check for stop file
    implicit none

    ! Arguments
    character(*), optional, intent(in) :: file
    !! File to check for

    ! Local variables
    integer :: Reason, itmp1, unit
    !! I/O status, temporary integer, and unit
    logical :: lfound, output
    !! File found flag and output
    character(248) :: file_
    !! File name
    character(128) :: buffer, tagname
    !! Buffer and tag name

    unit = 999
    file_ = "STOPCAR"
    if(present(file)) file_ = file

    output = .false.
    !! Check if file exists
    inquire(file = trim(file_), exist = lfound)
    if(lfound) then
       itmp1 = 0
       open(unit = unit, file = trim(file_))
       !! Read line-by-line
       do
          read(unit, '(A)', iostat = Reason) buffer
          if(Reason /= 0) exit
          call rm_comments(buffer)
          if(trim(buffer) == "") cycle
          tagname = trim(adjustl(buffer))
          if(scan(buffer, "=") /= 0) tagname = trim(tagname(:scan(tagname, "=") - 1))
          select case(trim(tagname))
          case("LSTOP")
             call assignL(buffer, output, itmp1)
             exit
          case("LABORT")
             call assignL(buffer, output, itmp1)
             if(output) then
                close(unit, status = 'delete')
                stop "LABORT ENCOUNTERED IN STOP FILE (" // trim(file_) // ")"
             end if
          end select
       end do
       close(unit, status = 'delete')
    end if
  end function stop_check
!###############################################################################

end module athena__tools_infile
