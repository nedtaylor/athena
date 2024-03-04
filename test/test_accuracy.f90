program test_accuracy
  use accuracy
  implicit none

  real, dimension(3,3) :: output, expected
  real, dimension(3) :: result
  integer, dimension(3,3) :: expected_categorical

  logical :: success = .true.


  !! initialize output test data
  output = &
       reshape([1.E0, 2.E0, 3.E0, 4.E0, 5.E0, 6.E0, 7.E0, 8.E0, 9.E0], [3,3])

  !! test categorical_score
  expected_categorical = reshape([1, 2, 3, 1, 2, 3, 1, 2, 3], [3,3])
  result = categorical_score(output, expected_categorical)
  if (any(abs(result - 1.E0) .gt. 1.E-6)) then
     write(0,*) 'categorical_score failed'
     success = .false.
  end if
  expected_categorical = reshape([1, 2, 3, 1, 1, 1, 1, 1, 1], [3,3])
  result = categorical_score(output, expected_categorical)
  if (any(abs(result - [1.E0, 0.E0, 0.E0]) .gt. 1.E-6)) then
     write(0,*) 'categorical_score failed'
     success = .false.
  end if

  !! initialize output test data
  output = reshape([0.1E0, 0.2E0, 0.3E0, &
                    0.4E0, 0.5E0, 0.6E0, &
                    0.7E0, 0.8E0, 0.9E0], [3,3])

  !! test mae_score
  expected = reshape([0.1E0, 0.2E0, 0.3E0, &
                      0.4E0, 0.5E0, 0.6E0, &
                      0.7E0, 0.8E0, 0.9E0], [3,3])
  result = mae_score(output, expected)
  if (any(abs(result - 1.E0) .gt. 1.E-6)) then
     write(0,*) 'mae_score failed'
     success = .false.
  end if
  expected = reshape([0.1E0, 0.2E0, 0.1E0, &
                      0.4E0, 0.5E0, 0.6E0, &
                      0.0E0, 0.0E0, 0.0E0], [3,3])
  result = mae_score(output, expected)
  if (any(abs(result - [14.E0/15.E0 ,1.E0, 0.2E0] ) .gt. 1.E-6)) then
     write(0,*) 'mae_score failed'
     write(*,*) result
     success = .false.
  end if

  !! test mse_score
  expected = reshape([0.1E0, 0.2E0, 0.3E0, &
                      0.4E0, 0.5E0, 0.6E0, &
                      0.7E0, 0.8E0, 0.9E0], [3,3])
  result = mse_score(output, expected)
  if (any(abs(result - 1.E0) .gt. 1.E-6)) then
     write(0,*) 'mse_score failed'
     success = .false.
  end if
  expected = reshape([0.0E0, 0.1E0, 0.2E0, &
                      0.4E0, 0.5E0, 0.6E0, &
                      0.7E0, 0.8E0, 0.9E0], [3,3])
  result = mse_score(output, expected)
  if (any(abs(result - [0.99E0, 1.E0, 1.E0]) .gt. 1.E-6)) then
     write(0,*) 'mse_score failed'
     success = .false.
     write(*,*) result
  end if

  !! test r2_score
  expected = reshape([0.1E0, 0.2E0, 0.3E0, &
                      0.4E0, 0.5E0, 0.6E0, &
                      0.7E0, 0.8E0, 0.9E0], [3,3])
  result = r2_score(output, expected)
  if (any(abs(result - 1.E0) .gt. 1.E-6)) then
     write(0,*) 'r2_score failed'
     success = .false.
  end if
  expected = reshape([0.1E0, 0.15E0, 0.3E0, &
                      0.4E0, 0.5E0, 0.6E0, &
                      0.7E0, 0.8E0, 0.9E0], [3,3])
  result = r2_score(output, expected)
  if (any(abs(result - [0.884615421E0, 1.E0, 1.E0]) .gt. 1.E-6)) then
     write(0,*) 'r2_score failed'
     success = .false.
  end if

!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_accuracy passed all tests'
  else
     write(0,*) 'test_accuracy failed one or more tests'
     stop 1
  end if


end program test_accuracy