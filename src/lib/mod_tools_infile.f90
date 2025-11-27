module athena__tools_infile
  !! Module containing custom input file reading functions and subroutines
  !!
  !! This module contains custom input file reading functions and subroutines
  !! for reading and assigning values from a file.
  !! Code copied from ARTEMIS with permission of the authors
  !! Ned Thaddeus Taylor and Francis Huw Davies
  !! https://github.com/ExeQuantCode/ARTEMIS
  use athena__constants, only: real32
  use athena__misc, only: grep, icount
  use athena__io_utils, only: stop_program
  implicit none


  private

  public :: get_val
  public :: assign_val, assign_vec, allocate_and_assign_vec
  public :: getline, rm_comments
  public :: stop_check
  public :: move


  interface assign_val
     !! Interface for assigning a value to a variable
     procedure assignI, assignR, assignS, assignL
  end interface assign_val

  interface assign_vec
     !! Interface for assigning a vector to a variable
     procedure assignIvec, assignRvec
  end interface assign_vec

  interface allocate_and_assign_vec
     !! Interface for allocating and assigning a vector to a variable
     procedure allocate_and_assignRvec
  end interface allocate_and_assign_vec


contains

!###############################################################################
  function get_val(buffer, fs) result(output)
    !! Extract the section of buffer that occurs after the field separator fs
    implicit none

    ! Arguments
    character(*), intent(in) :: buffer
    !! Input buffer
    character(1), intent(in), optional :: fs
    !! Field separator

    ! Local variables
    character(:), allocatable :: output
    !! Extracted value
    character(1) :: fs_
    !! Field separator

    fs_ = '='
    if(present(fs)) fs_ = fs

    output = trim(adjustl(buffer((scan(buffer, fs_) + 1):)))
  end function get_val
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
  subroutine assignI(buffer, variable, found, keyword, fs)
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
    character(1), optional, intent(in) :: fs
    !! Field separator

    ! Local variables
    character(1024) :: buffer2
    !! Temporary buffer
    character(1) :: fs_
    !! Field separator

    fs_ = '='
    if(present(fs)) fs_ = fs

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, fs_) .ne. 0) buffer2 = get_val(buffer, fs_)
    if(trim(adjustl(buffer2)) .ne. '') then
       found = found + 1
       read(buffer2, *) variable
    end if
  end subroutine assignI
!###############################################################################


!###############################################################################
  subroutine assignIvec(buffer, variable, found, keyword, fs)
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
    character(1), optional, intent(in) :: fs
    !! Field separator

    ! Local variables
    integer :: i
    !! Loop index
    character(1024) :: buffer2
    !! Temporary buffer
    character(1) :: fs_
    !! Field separator

    fs_ = '='
    if(present(fs)) fs_ = fs

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, fs_) .ne. 0) buffer2 = get_val(buffer, fs_)
    if(trim(adjustl(buffer2)) .ne. '') then
       found = found + 1
       if(icount(buffer2) == 1 .and. icount(buffer2) .ne. size(variable)) then
          read(buffer2, *) variable(1)
          variable = variable(1)
       else
          read(buffer2, *) (variable(i), i = 1, size(variable))
       end if
    end if
  end subroutine assignIvec
!###############################################################################


!###############################################################################
  subroutine assignR(buffer, variable, found, keyword, fs)
    !! Assign a real to variable
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    real(real32), intent(out) :: variable
    !! Variable to assign data to
    integer, intent(inout) :: found
    !! Count for finding variable
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from
    character(1), optional, intent(in) :: fs
    !! Field separator

    ! Local variables
    character(1024) :: buffer2
    !! Temporary buffer
    character(1) :: fs_
    !! Field separator

    fs_ = '='
    if(present(fs)) fs_ = fs

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, fs_) .ne. 0) buffer2 = get_val(buffer, fs_)
    if(trim(adjustl(buffer2)) .ne. '') then
       found = found + 1
       read(buffer2, *) variable
    end if
  end subroutine assignR
!###############################################################################


!###############################################################################
  subroutine assignRvec(buffer, variable, found, keyword, fs)
    !! Assign an arbitrary length vector of reals to variable
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    real(real32), dimension(:), intent(out) :: variable
    !! Variable to assign data to
    integer, intent(inout) :: found
    !! Count for finding variable
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from
    character(1), optional, intent(in) :: fs
    !! Field separator

    ! Local variables
    integer :: i
    !! Loop index
    character(1024) :: buffer2
    !! Temporary buffer
    character(1) :: fs_
    !! Field separator

    fs_ = '='
    if(present(fs)) fs_ = fs

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, fs_) .ne. 0) buffer2 = get_val(buffer, fs_)
    if(trim(adjustl(buffer2)) .ne. '') then
       found = found + 1
       if(icount(buffer2) == 1 .and. icount(buffer2) .ne. size(variable)) then
          read(buffer2, *) variable(1)
          variable = variable(1)
       else
          read(buffer2, *) (variable(i), i = 1, size(variable))
       end if
    end if
  end subroutine assignRvec
!###############################################################################


!###############################################################################
  subroutine allocate_and_assignRvec(buffer, variable, keyword, fs)
    !! Allocate and assign an arbitrary length vector of reals to variable
    implicit none

    ! Arguments
    character(*), intent(inout) :: buffer
    !! Input buffer
    real(real32), dimension(:), allocatable, intent(out) :: variable
    !! Variable to assign data to
    character(*), optional, intent(in) :: keyword
    !! Keyword to start from
    character(1), optional, intent(in) :: fs
    !! Field separator

    ! Local variables
    integer :: i
    !! Number of values and loop index
    character(1024) :: buffer2
    !! Temporary buffer
    character(1) :: fs_
    !! Field separator
    character(1), parameter :: open_brackets(3) = ['[', '(', '{']
    character(1), parameter :: close_brackets(3) = [']', ')', '}']

    fs_ = '='
    if(present(fs)) fs_ = fs

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, fs_) .ne. 0)then
       buffer2 = get_val(buffer, fs_)
    else
       buffer2 = buffer
    end if
    buffer2 = adjustl(buffer2)
    if(any(index(buffer2,open_brackets).eq.1)) then
       do i = 1, size(open_brackets)
          if(index(buffer2, open_brackets(i)) .eq. 1) then
             buffer2 = buffer2(2:)
          end if
       end do
    end if
    if(any(index(trim(buffer2),close_brackets).eq.len(trim(buffer2)))) then
       do i = 1, size(close_brackets)
          if(index(trim(buffer2), close_brackets(i)) .eq. len(trim(buffer2))) then
             buffer2 = buffer2(:len(trim(buffer2))-1)
          end if
       end do
    end if
    ! count number of values
    i = icount(buffer2)
    allocate(variable(i))
    read(buffer2, *) (variable(i), i = 1, size(variable))
  end subroutine allocate_and_assignRvec
!###############################################################################


!###############################################################################
  subroutine assignS(buffer, variable, found, keyword, fs)
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
    character(1), optional, intent(in) :: fs
    !! Field separator

    ! Local variables
    character(1024) :: buffer2
    !! Temporary buffer
    character(1) :: fs_
    !! Field separator

    fs_ = '='
    if(present(fs)) fs_ = fs

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, fs_) .ne. 0) buffer2 = get_val(buffer, fs_)
    if(trim(adjustl(buffer2)) .ne. '')then
       found = found + 1
       if( &
            ( &
                 buffer2(1:1) .eq. '"' .and. &
                 buffer2(len(trim(buffer2)):len(trim(buffer2))) .eq. '"' &
            ) .or. ( &
                 buffer2(1:1) .eq. '''' .and. &
                 buffer2(len(trim(buffer2)):len(trim(buffer2))) .eq. '''' &
            ) &
       )then
          buffer2 = buffer2(2:len(trim(buffer2))-1)
       end if
       read(buffer2, '(A)') variable
    end if
  end subroutine assignS
!###############################################################################


!###############################################################################
  subroutine assignL(buffer, variable, found, keyword, fs)
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
    character(1), optional, intent(in) :: fs
    !! Field separator

    ! Local variables
    character(1024) :: buffer2
    !! Temporary buffer
    character(1) :: fs_
    !! Field separator

    fs_ = '='
    if(present(fs)) fs_ = fs

    if(present(keyword)) buffer = buffer(index(buffer, keyword):)
    if(scan(buffer, fs_) .ne. 0) buffer2 = get_val(buffer, fs_)
    if(trim(adjustl(buffer2)) .ne. '') then
       found = found + 1
       if( &
            index(buffer2, "T") .ne. 0 .or. &
            index(buffer2, "t") .ne. 0 .or. &
            index(buffer2, "1") .ne. 0 &
       ) then
          variable = .TRUE.
       end if
       if( &
            index(buffer2, "F") .ne. 0 .or. &
            index(buffer2, "f") .ne. 0 .or. &
            index(buffer2, "0") .ne. 0 &
       ) then
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

    if(scan(buffer, '!') .ne. 0) buffer = buffer(:(scan(buffer, '!') - 1))
    if(scan(buffer, '#') .ne. 0) buffer = buffer(:(scan(buffer, '#') - 1))
    do while(scan(buffer, '(') .ne. 0 .or. scan(buffer, ')') .ne. 0)
       lbracket = scan(buffer, '(', back = .true.)
       rbracket = scan(buffer(lbracket:), ')')
       if(lbracket == 0 .or. rbracket == 0) then
          write(6, '(A,I0)') &
               ' NOTE: a bracketing error was encountered on line ', iline_
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
          if(Reason .ne. 0) exit
          call rm_comments(buffer)
          if(trim(buffer) == "") cycle
          tagname = trim(adjustl(buffer))
          if(scan(buffer, "=") .ne. 0) &
               tagname = trim(tagname(:scan(tagname, "=") - 1))
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


!###############################################################################
  subroutine move(unit, change, iostat, err_msg)
    !! Move current position in file based on relative change
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit to read from
    integer, intent(in) :: change
    !! Relative change in position
    integer, intent(out), optional :: iostat
    !! I/O status
    character(*), intent(out), optional :: err_msg
    !! Error message

    ! Local variables
    integer :: iostat_
    !! I/O status
    integer :: i
    !! Loop index
    character(256) :: err_msg_
    !! Error message

    if(present(iostat)) iostat = 0
    if(present(err_msg)) err_msg = ""
    if(change.eq.0) return
    inquire(unit = unit, iostat = iostat_)
    if(iostat_ .ne. 0) then
       write(err_msg_, '(A,I0)') &
            'ERROR: cannot move in file, unit ', unit
       if(present(iostat)) iostat = iostat_
       if(present(err_msg))then
          err_msg = err_msg_
       else
          call stop_program(err_msg_)
       end if
       return
    end if
    if(change.gt.0)then
       do i = 1, change
          read(unit, '(A)', iostat = iostat_)
          if(iostat_ .ne. 0) then
             write(err_msg_, '(A,I0)') &
                  'ERROR: cannot move forward in file, unit ', unit
             if(present(iostat)) iostat = iostat_
             if(present(err_msg))then
                err_msg = err_msg_
             else
                call stop_program(err_msg_)
             end if
             return
          end if
       end do
    else
       do i = 1, abs(change)
          backspace(unit)
          if(iostat .ne. 0) then
             write(err_msg_, '(A,I0)') &
                  'ERROR: cannot move backward in file, unit ', unit
             if(present(iostat)) iostat = iostat_
             if(present(err_msg))then
                err_msg = err_msg_
             else
                call stop_program(err_msg_)
             end if
             return
          end if
       end do
    end if

  end subroutine move
!###############################################################################

end module athena__tools_infile
