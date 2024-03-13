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
module infile_tools
  use misc, only: grep,icount
  implicit none

  private
  
!!!-----------------------------------------------------
!!! assign a value to variable
!!!-----------------------------------------------------
!!! buffer   = (S, io) sacrifical input character string
!!! variable = (*, in) variable to assign data to
!!! found    = (I, io) count for finding variable
!!! keyword  = (S, in, opt) keyword to start from
!!! num      = (I, in, opt) number of tags in tag_list
!!! tag_list = (S, in, opt) list of tags to search for
!!!-----------------------------------------------------
  interface assign_val
     procedure assignI,assignR,assignD,assignS,assignL
  end interface assign_val
  interface assign_vec
     procedure assignIvec,assignRvec,assignDvec
  end interface assign_vec


  public :: getline, rm_comments
  public :: assign_val, assign_vec
  public :: stop_check

  
!!!updated 2024/03/04


contains
!!!#############################################################################
!!! val outputs the section of buffer that occurs after an "="
!!!#############################################################################
  function val(buffer)
    character(*), intent(in) :: buffer
    character(100) :: val

    val=trim(adjustl(buffer((scan(buffer,"=",back=.false.)+1):)))
    return
  end function val
!!!#############################################################################


!!!#############################################################################
!!! gets the line from a grep and assigns it to buffer
!!!#############################################################################
!!! unit    = (I, in) unit to read from
!!! pattern = (S, in) pattern to grep for
!!! buffer  = (S, io) buffer to assign line to
  subroutine getline(unit,pattern,buffer)
    integer, intent(in) :: unit
    character(*), intent(in) :: pattern
    character(*), intent(out) :: buffer
    
    integer :: Reason

    call grep(unit,pattern)
    backspace(unit);read(unit,'(A)',iostat=Reason) buffer
    
  end subroutine getline
!!!#############################################################################


!!!#############################################################################
!!! assign an integer to variable
!!!#############################################################################
  subroutine assignI(buffer,variable,found,keyword)
    integer, intent(inout) :: found
    character(*), intent(inout) :: buffer
    integer, intent(out) :: variable
    character(*), optional, intent(in) :: keyword

    character(1024) :: buffer2

    if(present(keyword))then
       buffer=buffer(index(buffer,keyword):)
    end if
    if(scan("=",buffer).ne.0) buffer2=val(buffer)
    if(trim(adjustl(buffer2)).ne.'') then
       found=found+1
       read(buffer2,*) variable
    end if
  end subroutine assignI
!!!#############################################################################


!!!#############################################################################
!!! assign an arbitrary length vector of integers to variable
!!!#############################################################################
  subroutine assignIvec(buffer,variable,found,keyword)
    integer, intent(inout) :: found
    character(*), intent(inout) :: buffer
    integer, dimension(:) :: variable
    character(*), optional, intent(in) :: keyword

    integer :: i
    character(1024) :: buffer2

    if(present(keyword))then
       buffer=buffer(index(buffer,keyword):)
    end if
    if(scan("=",buffer).ne.0) buffer2=val(buffer)
    if(trim(adjustl(buffer2)).ne.'') then
       found=found+1
       if(icount(buffer2).eq.1.and.&
            icount(buffer2).ne.size(variable))then
          read(buffer2,*) variable(1)
          variable = variable(1)
       else
          read(buffer2,*) (variable(i),i=1,size(variable))
       end if
    end if
  end subroutine assignIvec
!!!#############################################################################


!!!#############################################################################
!!! assign a real to variable
!!!#############################################################################
  subroutine assignR(buffer,variable,found,keyword)
    integer, intent(inout) :: found
    character(*), intent(inout) :: buffer
    real, intent(out) :: variable
    character(*), optional, intent(in) :: keyword

    character(1024) :: buffer2

    if(present(keyword))then
       buffer=buffer(index(buffer,keyword):)
    end if
    if(scan("=",buffer).ne.0) buffer2=val(buffer)
    if(trim(adjustl(buffer2)).ne.'') then
       found=found+1
       read(buffer2,*) variable
    end if
  end subroutine assignR
!!!#############################################################################


!!!#############################################################################
!!! assign a DP value to variable
!!!#############################################################################
  subroutine assignRvec(buffer,variable,found,keyword)
    integer, intent(inout) :: found
    character(*), intent(inout) :: buffer
    real, dimension(:), intent(out) :: variable
    character(*), optional, intent(in) :: keyword

    integer :: i
    character(1024) :: buffer2

    if(present(keyword))then
       buffer=buffer(index(buffer,keyword):)
    end if
    if(scan("=",buffer).ne.0) buffer2=val(buffer)
    if(trim(adjustl(buffer2)).ne.'') then
       found=found+1
       if(icount(buffer2).eq.1.and.&
            icount(buffer2).ne.size(variable))then
          read(buffer2,*) variable(1)
          variable = variable(1)
       else
          read(buffer2,*) (variable(i),i=1,size(variable))
       end if
    end if
  end subroutine assignRvec
!!!#############################################################################


!!!#############################################################################
!!! assign a double precision to variable
!!!#############################################################################
  subroutine assignD(buffer,variable,found,keyword)
    integer, intent(inout) :: found
    character(*), intent(inout) :: buffer
    double precision, intent(out) :: variable
    character(*), optional, intent(in) :: keyword

    character(1024) :: buffer2

    if(present(keyword))then
       buffer=buffer(index(buffer,keyword):)
    end if
    if(scan("=",buffer).ne.0) buffer2=val(buffer)
    if(trim(adjustl(buffer2)).ne.'') then
       found=found+1
       read(buffer2,*) variable
    end if
  end subroutine assignD
!!!#############################################################################


!!!#############################################################################
!!! assign an arbitrary length vector of DP to variable
!!!#############################################################################
  subroutine assignDvec(buffer,variable,found,keyword)
    integer, intent(inout) :: found
    character(*), intent(inout) :: buffer
    double precision, dimension(:), intent(out) :: variable
    character(*), optional, intent(in) :: keyword

    integer :: i
    character(1024) :: buffer2

    if(present(keyword))then
       buffer=buffer(index(buffer,keyword):)
    end if
    if(scan("=",buffer).ne.0) buffer2=val(buffer)
    if(trim(adjustl(buffer2)).ne.'') then
       found=found+1
       if(icount(buffer2).eq.1.and.&
            icount(buffer2).ne.size(variable))then
          read(buffer2,*) variable(1)
          variable = variable(1)
       else
          read(buffer2,*) (variable(i),i=1,size(variable))
       end if
    end if
  end subroutine assignDvec
!!!#############################################################################


!!!#############################################################################
!!! assign a string to variable
!!!#############################################################################
  subroutine assignS(buffer,variable,found,keyword)
    integer, intent(inout) :: found
    character(*), intent(inout) :: buffer
    character(*), intent(out) :: variable
    character(*), optional, intent(in) :: keyword

    character(1024)::buffer2

    if(present(keyword))then
       buffer=buffer(index(buffer,keyword):)
    end if
    if(scan("=",buffer).ne.0) buffer2=val(buffer)
    if(trim(adjustl(buffer2)).ne.'') then
       found=found+1
       read(buffer2,'(A)') variable
    end if
  end subroutine assignS
!!!#############################################################################


!!!#############################################################################
!!! assign a logical to variable (T/t/1 and F/f/0 accepted
!!!#############################################################################
  subroutine assignL(buffer,variable,found,keyword)
    integer, intent(inout) :: found
    character(*), intent(inout) :: buffer
    logical, intent(out) :: variable
    character(*), optional, intent(in) :: keyword

    character(1024)::buffer2

    if(present(keyword))then
       buffer=buffer(index(buffer,keyword):)
    end if
    if(scan("=",buffer).ne.0) buffer2=val(buffer)
    if(trim(adjustl(buffer2)).ne.'') then
       found=found+1
       if(index(buffer2,"T").ne.0.or.&
            index(buffer2,"t").ne.0.or.&
            index(buffer2,"1").ne.0) then
          variable=.TRUE.
       end if
       if(index(buffer2,"F").ne.0.or.&
            index(buffer2,"f").ne.0.or.&
            index(buffer2,"0").ne.0) then
          variable=.FALSE.
       end if
    end if
  end subroutine assignL
!!!#############################################################################


!!!#############################################################################
!!! remove comment from a string (anything after ! or #) 
!!!#############################################################################
!!! buffer = (S, io) sacrifical input character string
!!! iline  = (I, in, opt) line number
  subroutine rm_comments(buffer,iline)
    implicit none
    character(*), intent(inout) :: buffer
    integer, optional, intent(in) :: iline

    integer :: lbracket,rbracket,iline_

    iline_=0
    if(present(iline)) iline_=iline

    if(scan(buffer,'!').ne.0) buffer=buffer(:(scan(buffer,'!')-1))
    if(scan(buffer,'#').ne.0) buffer=buffer(:(scan(buffer,'#')-1))
    do while(scan(buffer,'(').ne.0.or.scan(buffer,')').ne.0)
       lbracket=scan(buffer,'(',back=.true.)
       rbracket=scan(buffer(lbracket:),')')
       if(lbracket.eq.0.or.rbracket.eq.0)then
          write(6,'(A,I0)') &
               ' NOTE: a bracketing error was encountered on line ',iline_
          buffer=""
          return
       end if
       rbracket=rbracket+lbracket-1
       buffer=buffer(:(lbracket-1))//buffer((rbracket+1):)
    end do

    return
  end subroutine rm_comments
!!!#############################################################################


!!!#############################################################################
!!! logical check for stop file
!!!#############################################################################
!!! file   = (S, in, opt) file to check for
  function stop_check(file) result(output)
    implicit none
    integer :: Reason,itmp1
    integer :: unit
    logical :: lfound
    logical :: output
    character(*), optional, intent(in) :: file
    character(248) :: file_
    character(128) :: buffer, tagname

    unit = 999
    file_ = "STOPCAR"
    if(present(file)) file_ = file

    output = .false.
    !! check if file exists
    inquire(file=trim(file_),exist=lfound)
    file_if: if(lfound)then
       itmp1 = 0
       open(unit=unit, file=trim(file_))
       !! read line-by-line
       file_loop: do
          read(unit,'(A)',iostat=Reason) buffer
          if(Reason.ne.0) exit file_loop
          call rm_comments(buffer)
          if(trim(buffer).eq.'') cycle file_loop
          tagname=trim(adjustl(buffer))
          if(scan(buffer,"=").ne.0) tagname=trim(tagname(:scan(tagname,"=")-1))
          select case(trim(tagname))
          case("LSTOP")
             call assignL(buffer,output,itmp1)
             exit file_loop
          case("LABORT")
             call assignL(buffer,output,itmp1)
             if(output)then
                close(unit,status='delete')
                stop "LABORT ENCOUNTERED IN STOP FILE ("//trim(file_)//")"
             end if
          end select
       end do file_loop
       close(unit,status='delete')
    end if file_if

  end function stop_check
!!!#############################################################################

end module infile_tools
