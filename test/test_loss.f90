program test_loss
  use loss
  implicit none

  real, dimension(3, 3) :: predicted = &
       reshape([0.1, 0.2, 0.7, 0.3, 0.4, 0.3, 0.6, 0.2, 0.2], [3, 3])
  real, dimension(3, 3) :: expected = &
       reshape([0.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0, 0.E0], [3, 3])
  real, dimension(3, 3) :: expected_loss, actual_loss
  real, dimension(3) :: expected_total_loss, actual_total_loss

  logical :: success = .true.


!!!-----------------------------------------------------------------------------
!!! test BCE loss
!!!-----------------------------------------------------------------------------
  expected_loss = -expected * log(predicted + 1.E-7)
  actual_loss = compute_loss_bce(predicted, expected)
  if (any(abs(actual_loss - expected_loss) .gt. 1.E-6)) then
    write(0,*) "Error: compute_loss_bce did not return the expected result."
    write(0,*) "actual_loss: ", actual_loss
    write(0,*) "expected_loss: ", expected_loss
    success = .false.
  end if
  expected_total_loss = sum(expected_loss, dim=1)
  actual_total_loss = total_loss_bce(predicted, expected)
  if (any(abs(actual_total_loss - expected_total_loss) .gt. 1.E-6)) then
    write(0,*) "Error: total_loss_bce did not return the expected result."
    write(0,*) "actual_total_loss: ", actual_total_loss
    success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! test CCE loss
!!!-----------------------------------------------------------------------------
  expected_loss = -expected * log(predicted + 1.E-7)
  actual_loss = compute_loss_cce(predicted, expected)
  if (any(abs(actual_loss - expected_loss) .gt. 1.E-6)) then
    write(0,*) "Error: compute_loss_cce did not return the expected result."
    write(0,*) "actual_loss: ", actual_loss
    write(0,*) "expected_loss: ", expected_loss
    success = .false.
  end if
  expected_total_loss = sum(expected_loss, dim=1)
  actual_total_loss = total_loss_cce(predicted, expected)
  if (any(abs(actual_total_loss - expected_total_loss) .gt. 1.E-6)) then
    write(0,*) "Error: total_loss_cce did not return the expected result."
    write(0,*) "actual_total_loss: ", actual_total_loss
    success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! test MAE loss
!!!-----------------------------------------------------------------------------
  expected_loss = abs(predicted - expected)
  actual_loss = compute_loss_mae(predicted, expected)
  if (any(abs(actual_loss - expected_loss) .gt. 1.E-6)) then
    write(0,*) "Error: compute_loss_mae did not return the expected result."
    write(0,*) "actual_loss: ", actual_loss
    write(0,*) "expected_loss: ", expected_loss
    success = .false.
  end if
  expected_total_loss = sum(expected_loss, dim=1) / size(predicted, 1)
  actual_total_loss = total_loss_mae(predicted, expected)
  if (any(abs(actual_total_loss - expected_total_loss) .gt. 1.E-6)) then
    write(0,*) "Error: total_loss_mae did not return the expected result."
    write(0,*) "actual_total_loss: ", actual_total_loss
    success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! test MSE loss
!!!-----------------------------------------------------------------------------
  expected_loss = ((predicted - expected)**2.E0) /(2.E0)
  actual_loss = compute_loss_mse(predicted, expected)
  if (any(abs(actual_loss - expected_loss) .gt. 1.E-6)) then
    write(0,*) "Error: compute_loss_mse did not return the expected result."
    write(0,*) "actual_loss: ", actual_loss
    write(0,*) "expected_loss: ", expected_loss
    success = .false.
  end if
  expected_total_loss = sum(expected_loss, dim=1) * 2.E0 / size(predicted, 1)
  actual_total_loss = total_loss_mse(predicted, expected)
  if (any(abs(actual_total_loss - expected_total_loss) .gt. 1.E-6)) then
    write(0,*) "Error: total_loss_mse did not return the expected result."
    write(0,*) "actual_total_loss: ", actual_total_loss
    success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! test NLL loss
!!!-----------------------------------------------------------------------------
  expected_loss = - log(expected - predicted + 1.E-7)
  actual_loss = compute_loss_nll(predicted, expected)
  if (any(abs(actual_loss - expected_loss) .gt. 1.E-6)) then
    write(0,*) "Error: compute_loss_nll did not return the expected result."
    write(0,*) "actual_loss: ", actual_loss
    write(0,*) "expected_loss: ", expected_loss
    success = .false.
  end if
  expected_total_loss = sum(expected_loss, dim=1)
  actual_total_loss = total_loss_nll(predicted, expected)
  if (any(abs(actual_total_loss - expected_total_loss) .gt. 1.E-6)) then
    write(0,*) "Error: total_loss_nll did not return the expected result."
    write(0,*) "actual_total_loss: ", actual_total_loss
    success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_loss passed all tests'
  else
     write(0,*) 'test_loss failed one or more tests'
     stop 1
  end if

end program test_loss