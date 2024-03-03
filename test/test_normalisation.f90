program test_normalisation
  use normalisation
  implicit none

  logical :: success = .true.

  real, allocatable, dimension(:) :: input
  real, allocatable, dimension(:) :: expected_output

  !! test linear renormalisation
  input = [1.0E0, 2.0E0, 3.0E0]
  expected_output = [-1.0E0, 0.0E0, 1.0E0]
  call linear_renormalise(input)
  if (any(abs(input - expected_output) .gt. 1.E-6)) then
     write(0,*) "Linear renormalisation failed"
     success = .false.
  end if

  !! test linear renormalisation
  input = [0.0E0, 5.0E0, 10.0E0, 15.0E0]
  expected_output = [-1.0E0, -1.E0/3.E0, 1.0E0/3.E0, 1.E0]
  call linear_renormalise(input)
  if (any(abs(input - expected_output) .gt. 1.E-6)) then
     write(0,*) "Linear renormalisation failed"
     success = .false.
  end if

  !! test norm renormalisation
  input = [-3.0E0, 0.0E0, 4.0E0, 12.0E0, 15.0E0]
  expected_output = input * 1.E0/sqrt(dot_product(input,input))
  call renormalise_norm(input)
  if (any(abs(input - expected_output) .gt. 1.E-6)) then
     write(0,*) "Norm renormalisation failed"
     success = .false.
  end if
  
  !! test sum renormalisation
  input = [1.0E0, 2.0E0, 3.0E0]
  expected_output = input / sum(input)
  call renormalise_sum(input)
  if (any(abs(input - expected_output) .gt. 1.E-6)) then
     write(0,*) "Sum renormalisation failed"
     success = .false.
  end if
  
!!!-----------------------------------------------------------------------------
!!! Final printing array shuffle tests
!!!-----------------------------------------------------------------------------
  
  !! check for any fails
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_shuffle passed all tests'
  else
     write(0,*) 'test_shuffle failed one or more tests'
     stop 1
  end if

end program test_normalisation

