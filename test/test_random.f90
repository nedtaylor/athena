program test_random
  use random, only: random_setup
  implicit none

  integer :: i
  integer, allocatable, dimension(:) :: seed, seed_check
  integer :: num_seed
  logical :: already_initialised
  logical :: success = .true.

  call random_seed(size=num_seed)
  allocate(seed(num_seed))
  seed = (/ (i, i=1, num_seed) /)

  !! test with seed and num_seed
  call random_setup(seed, restart=.false., &
       already_initialised=already_initialised)
  if(already_initialised) then
     write(*,*) "Random_setup did not set already_initialised to .false."
     success = .false.
  end if

  !! check seed properly set
  allocate(seed_check(num_seed))
  call random_seed(get=seed_check)
  if (any(seed .ne. seed_check)) then
     write(*,*) "Error: seed is not as expected."
     write(*,*) "Actual: ", seed_check
     write(*,*) "Expected: ", seed
     success = .false.
  end if

  !! test without restart
  call random_setup(seed, restart=.false., &
       already_initialised=already_initialised)
  if(.not.already_initialised) then
     write(*,*) "Random_setup did not set already_initialised to .true."
     success = .false.
  end if

  !! test with restart
  call random_setup(restart = .true., already_initialised = already_initialised)

!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_random passed all tests'
  else
     write(0,*) 'test_random failed one or more tests'
     stop 1
  end if

end program test_random
