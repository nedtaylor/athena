program test_loss
  use athena__loss
  use diffstruc, only: array_type
  implicit none

  real, dimension(3, 3) :: predicted = &
       reshape([0.1, 0.2, 0.7, 0.3, 0.4, 0.3, 0.6, 0.2, 0.2], [3, 3])
  real, dimension(3, 3) :: expected = &
       reshape([0.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0, 0.E0], [3, 3])
  real, dimension(3, 3) :: expected_loss

  logical :: success = .true.
  class(base_loss_type), allocatable :: loss
  type(array_type), dimension(1,1) :: predicted_array, expected_array
  type(array_type), pointer :: actual_loss


  call predicted_array(1,1)%allocate(array_shape=[3,3])
  call predicted_array(1,1)%set(predicted)
  call expected_array(1,1)%allocate(array_shape=[3,3])
  call expected_array(1,1)%set(expected)


!-------------------------------------------------------------------------------
! test BCE loss
!-------------------------------------------------------------------------------
  loss = bce_loss_type()
  expected_loss = -expected * log(predicted + 1.E-7)
  actual_loss => loss%compute(predicted_array, expected_array)
  if (any(abs(actual_loss%val(:,1) - sum(expected_loss,2) / 3.0) .gt. 1.E-6)) then
     write(0,*) "Error: compute_loss_bce did not return the expected result."
     write(0,*) "actual_loss: ", actual_loss%val(:,1)
     write(0,*) "expected_loss: ", sum(expected_loss,2) / 3.0
     success = .false.
  end if


!-------------------------------------------------------------------------------
! test CCE loss
!-------------------------------------------------------------------------------
  loss = cce_loss_type()
  expected_loss = -expected * log(predicted + 1.E-7)
  actual_loss => loss%compute(predicted_array, expected_array)
  if (any(abs(actual_loss%val(:,1) - sum(expected_loss,2) / 3.0) .gt. 1.E-6)) then
     write(0,*) "Error: compute_loss_cce did not return the expected result."
     write(0,*) "actual_loss: ", actual_loss%val(:,1)
     write(0,*) "expected_loss: ", sum(expected_loss,2) / 3.0
     success = .false.
  end if


!-------------------------------------------------------------------------------
! test MAE loss
!-------------------------------------------------------------------------------
  loss = mae_loss_type()
  expected_loss = abs(predicted - expected)
  actual_loss => loss%compute(predicted_array, expected_array)
  if (any(abs(actual_loss%val(:,1) - sum(expected_loss,2) / 6.0) .gt. 1.E-6)) then
     write(0,*) "Error: compute_loss_mae did not return the expected result."
     write(0,*) "actual_loss: ", actual_loss%val(:,1)
     write(0,*) "expected_loss: ", sum(expected_loss,2) / 6.0
     success = .false.
  end if


!-------------------------------------------------------------------------------
! test MSE loss
!-------------------------------------------------------------------------------
  loss = mse_loss_type()
  expected_loss = ((predicted - expected)**2.E0) /(2.E0)
  actual_loss => loss%compute(predicted_array, expected_array)
  if (any(abs(actual_loss%val(:,1) - sum(expected_loss,2) / 3.0) .gt. 1.E-6)) then
     write(0,*) "Error: compute_loss_mse did not return the expected result."
     write(0,*) "actual_loss: ", actual_loss%val(:,1)
     write(0,*) "expected_loss: ", sum(expected_loss,2) / 3.0
     success = .false.
  end if


!-------------------------------------------------------------------------------
! test NLL loss
!-------------------------------------------------------------------------------
  loss = nll_loss_type()
  expected_loss = - log(expected - predicted + 1.E-7)
  actual_loss => loss%compute(predicted_array, expected_array)
  if (any(abs(actual_loss%val(:,1) - sum(expected_loss,2) / 3.0) .gt. 1.E-6)) then
     write(0,*) "Error: compute_loss_nll did not return the expected result."
     write(0,*) "actual_loss: ", actual_loss%val(:,1)
     write(0,*) "expected_loss: ", sum(expected_loss,2) / 3.0
     success = .false.
  end if


!-------------------------------------------------------------------------------
! test HUB loss
!-------------------------------------------------------------------------------
  loss = huber_loss_type()
  where (abs(predicted - expected) .le. 1.0)
     expected_loss = 0.5 * (predicted - expected)**2.0
  elsewhere
     expected_loss = 1.0 * (abs(predicted - expected) - 0.5 * 1.0)
  end where
  actual_loss => loss%compute(predicted_array, expected_array)
  if (any(abs(actual_loss%val(:,1) - sum(expected_loss,2) / 3.0) .gt. 1.E-6)) then
     write(0,*) "Error: compute_loss_huber did not return the expected result."
     write(0,*) "actual_loss: ", actual_loss%val(:,1)
     write(0,*) "expected_loss: ", sum(expected_loss,2) / 3.0
     success = .false.
  end if


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_loss passed all tests'
  else
     write(0,*) 'test_loss failed one or more tests'
     stop 1
  end if

end program test_loss
