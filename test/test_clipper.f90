program test_clipper
  use clipper
  implicit none

  type(clip_type) :: clip1, clip2
  type(clip_type), allocatable :: clip3

  integer, parameter :: length = 3
  real, dimension(length) :: gradient = [1.0, 2.0, 3.0]
  real, dimension(1) :: bias = [1.0]

  logical :: success = .true.


!!!-----------------------------------------------------------------------------
!!! check clip manual setup
!!!-----------------------------------------------------------------------------
  allocate(clip3, source=clip_type( &
       clip_min = 0.1, &
       clip_max = 0.2, &
       clip_norm = 0.3 &
       ))
  if(abs(clip3%min - 0.1) .gt. 1.E-6)then
     write(0,*) "min setup failed", clip3%min
     success = .false.
  end if
  if(abs(clip3%max - 0.2) .gt. 1.E-6)then
     write(0,*) "max setup failed", clip3%max
     success = .false.
  end if
  if(abs(clip3%norm - 0.3) .gt. 1.E-6)then
     write(0,*) "norm setup failed", clip3%norm
     success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! test clip read
!!!-----------------------------------------------------------------------------
  call clip2%read("0.2", "0.4", "0.6")
  if(abs(clip2%min - 0.2) .gt. 1.E-6)then
     write(0,*) "read min failed"
     success = .false.
  end if
  if(abs(clip2%max - 0.4) .gt. 1.E-6)then
     write(0,*) "read max failed"
     success = .false.
  end if
  if(abs(clip2%norm - 0.6) .gt. 1.E-6)then
     write(0,*) "read norm failed"
     success = .false.
  end if
  

!!!-----------------------------------------------------------------------------
!!! test clip set
!!!-----------------------------------------------------------------------------
  call clip1%set(clip_min=0.0, clip_max=1.0, clip_norm=1.0)
  if(abs(clip1%min) .gt. 1.E-6)then
     write(0,*) "set min failed"
     success = .false.
  end if
  if(abs(clip1%max - 1.0) .gt. 1.E-6)then
     write(0,*) "set max failed"
     success = .false.
  end if
  if(abs(clip1%norm - 1.0) .gt. 1.E-6)then
     write(0,*) "set norm failed"
     success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! test copying clip
!!!-----------------------------------------------------------------------------
  call clip2%set(clip_dict=clip1)
  if(abs(clip2%min - clip1%min) .gt. 1.E-6)then
     write(0,*) "copy min failed"
     success = .false.
  end if
  if(abs(clip2%max - clip1%max) .gt. 1.E-6)then
     write(0,*) "copy max failed"
     success = .false.
  end if
  if(abs(clip2%norm - clip1%norm) .gt. 1.E-6)then
     write(0,*) "copy norm failed"
     success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! test clip apply
!!!-----------------------------------------------------------------------------
  call clip1%apply(length, gradient, bias)
  if(any(abs(gradient - 0.5E0) .gt. 1.E-6))then
     write(0,*) "gradient apply failed"
     success = .false.
  end if
  if(any(abs(bias - 0.5E0) .gt. 1.E-6))then
     write(0,*) "bias apply failed"
     success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_clipper passed all tests'
  else
     write(0,*) 'test_clipper failed one or more tests'
     stop 1
  end if

end program test_clipper