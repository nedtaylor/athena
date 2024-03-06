program test_infile_tools
  use infile_tools
  implicit none

  integer :: found
  integer :: ivar, ivar_vec(3)
  real :: rvar, rvar_vec(3)
  double precision :: dvar, dvar_vec(3)
  logical :: lvar
  character(len=1024) :: svar
  character(len=1024) :: buffer
  character(len=1024) :: keyword

  logical :: success = .true.

  !! set up test
  found = 0
  buffer = "test=5"
  keyword = "test"

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

  !! test double precision
  call assign_val(buffer, dvar, found, trim(keyword))
  if(abs(dvar-5.E0).gt.1.E-6)then
     write(0,*) "double precision variable not set correctly"
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

  !! test double precision vector
  call assign_vec(buffer, dvar_vec, found, trim(keyword))
  if(any(abs(ivar_vec-[1.E0,2.E0,3.E0]).gt.1.E-6))then
     write(0,*) "double precision vector not set correctly"
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

  !! test double precision vector single value
  call assign_vec(buffer, dvar_vec, found, trim(keyword))
  if(any(abs(dvar_vec-1.E0).gt.1.E-6))then
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

!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_infile_tools passed all tests'
  else
     write(0,*) 'test_infile_tools failed one or more tests'
     stop 1
  end if

end program test_infile_tools