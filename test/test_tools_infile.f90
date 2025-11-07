program test_infile_tools
  use coreutils, only: real32
  use athena__tools_infile
  implicit none

  integer :: found
  integer :: ivar, ivar_vec(3)
  real(real32) :: rvar, rvar_vec(3)
  logical :: lvar
  character(len=1024) :: svar
  character(len=1024) :: buffer
  character(len=1024) :: keyword

  logical :: success = .true.


!-------------------------------------------------------------------------------
! set up test line
!-------------------------------------------------------------------------------
  found = 0
  buffer = "test=5"
  keyword = "test"


!-------------------------------------------------------------------------------
! test scalar assign procedures
!-------------------------------------------------------------------------------
  !! test integer
  call assign_val(buffer, ivar, found, trim(keyword))
  if(ivar.ne.5)then
     write(0,*) "integer variable not set correctly"
     success = .false.
  end if

  !! test real
  call assign_val(buffer, rvar, found, trim(keyword))
  if(abs(rvar-5.E0).gt.1.E-6)then
     write(0,*) "real variable not set correctly"
     success = .false.
  end if

  !! test string
  call assign_val(buffer, svar, found, trim(keyword))
  if(trim(svar).ne."5")then
     write(0,*) "string variable not set correctly"
     success = .false.
  end if

  buffer = "test=T"
  !! test logical
  call assign_val(buffer, lvar, found, trim(keyword))
  if(.not.lvar)then
     write(0,*) "logical variable not set correctly"
     success = .false.
  end if



!-------------------------------------------------------------------------------
! test vector assign procedures
!-------------------------------------------------------------------------------
  buffer = "test= 1 2 3"

  !! test integer vector
  call assign_vec(buffer, ivar_vec, found, trim(keyword))
  if(any(ivar_vec.ne.[1,2,3]))then
     write(0,*) "integer vector not set correctly"
     success = .false.
  end if

  !! test real vector
  call assign_vec(buffer, rvar_vec, found, trim(keyword))
  if(any(abs(rvar_vec-[1.E0,2.E0,3.E0]).gt.1.E-6))then
     write(0,*) "real vector not set correctly"
     success = .false.
  end if

  buffer = "test=1"

  !! test integer vector
  call assign_vec(buffer, ivar_vec, found, trim(keyword))
  if(any(ivar_vec.ne.1))then
     write(0,*) "integer vector not set correctly"
     success = .false.
  end if

  !! test real vector single value
  call assign_vec(buffer, rvar_vec, found, trim(keyword))
  if(any(abs(rvar_vec-1.E0).gt.1.E-6))then
     write(0,*) "real vector not set correctly"
     success = .false.
  end if

  !! test remove comments
  buffer = "keep this!comment"
  call rm_comments(buffer)
  if(trim(buffer).ne."keep this")then
     write(0,*) "remove_comment failed"
     success = .false.
  end if


!-------------------------------------------------------------------------------
! test move subroutine
!-------------------------------------------------------------------------------
  block
    integer :: unit, iostat_val
    character(256) :: err_msg_val
    character(100) :: line

    ! Create a temporary test file with multiple lines
    open(newunit=unit, file='test_move.tmp', &
         status='replace', action='write')
    write(unit, '(A)') 'line 1'
    write(unit, '(A)') 'line 2'
    write(unit, '(A)') 'line 3'
    write(unit, '(A)') 'line 4'
    write(unit, '(A)') 'line 5'
    close(unit)

    ! Test forward movement
    open(newunit=unit, file='test_move.tmp', &
         status='old', action='read')

    ! Read first line
    read(unit, '(A)') line
    if(trim(line) .ne. 'line 1')then
       write(0,*) "move test setup failed - wrong first line"
       success = .false.
    end if

    ! Move forward 2 lines
    call move(unit, 2, iostat=iostat_val, err_msg=err_msg_val)
    if(iostat_val .ne. 0)then
       write(0,*) "move forward failed: ", trim(err_msg_val)
       success = .false.
    end if

    ! Should now be at line 4
    read(unit, '(A)') line
    if(trim(line) .ne. 'line 4')then
       write(0,*) "move forward test failed - expected 'line 4', got: ", &
                  trim(line)
       success = .false.
    end if

    ! Move backward 1 line
    call move(unit, -1, iostat=iostat_val, err_msg=err_msg_val)
    if(iostat_val .ne. 0)then
       write(0,*) "move backward failed: ", trim(err_msg_val)
       success = .false.
    end if

    ! After backspace -1, we should be positioned to re-read line 4
    read(unit, '(A)') line
    if(trim(line) .ne. 'line 4')then
       write(0,*) "move backward test failed - expected 'line 4', got: ", &
                  trim(line)
       success = .false.
    end if

    ! Test zero movement (should be no-op) - we're now at line 5
    call move(unit, 0, iostat=iostat_val, err_msg=err_msg_val)
    if(iostat_val .ne. 0)then
       write(0,*) "move zero failed: ", trim(err_msg_val)
       success = .false.
    end if

    ! Position should be unchanged - next line should be line 5
    read(unit, '(A)') line
    if(trim(line) .ne. 'line 5')then
       write(0,*) "move zero test failed - expected 'line 5', got: ", &
                  trim(line)
       success = .false.
    end if

    close(unit)

    ! Test error conditions - try to move with invalid unit
    call move(999, 1, iostat=iostat_val, err_msg=err_msg_val)
    if(iostat_val .eq. 0)then
       write(0,*) "move should fail with invalid unit"
       success = .false.
    end if

    ! Clean up temporary file
    open(newunit=unit, file='test_move.tmp', status='old')
    close(unit, status='delete')
  end block


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_infile_tools passed all tests'
  else
     write(0,*) 'test_infile_tools failed one or more tests'
     stop 1
  end if

end program test_infile_tools
