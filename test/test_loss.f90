program test_loss
  use athena__loss
  use diffstruc, only: array_type
  implicit none

  real, dimension(3, 3) :: predicted = &
       reshape([0.1, 0.2, 0.7, 0.3, 0.4, 0.3, 0.6, 0.2, 0.2], [3, 3])
  real, dimension(3, 3) :: expected = &
       reshape([0.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0, 0.E0], [3, 3])
  real, dimension(3, 3) :: expected_loss
  logical :: success
  class(base_loss_type), allocatable :: loss
  type(array_type), dimension(1,1) :: predicted_array, expected_array
  type(array_type), pointer :: actual_loss
   type(array_type), allocatable :: predicted_grid(:,:), expected_grid(:,:)
   type(array_type), allocatable :: huber_predicted(:,:), huber_expected(:,:)
   real :: predicted_values(2,2), expected_values(2,2)
   real :: huber_predicted_values(2,2), huber_expected_values(2,2)
   real :: expected_scalar
   integer :: i, j

  success = .true.

  call predicted_array(1,1)%allocate(array_shape=[3,3])
  call predicted_array(1,1)%set(predicted)
  call expected_array(1,1)%allocate(array_shape=[3,3])
  call expected_array(1,1)%set(expected)

  loss = bce_loss_type()
  expected_loss = -expected * log(predicted + 1.E-7)
  actual_loss => loss%compute(predicted_array, expected_array)
  if(any(abs(actual_loss%val(:,1) - sum(expected_loss,2) / 3.0) .gt. 1.E-6))then
     write(0,*) 'Error: compute_loss_bce did not return the expected result.'
     write(0,*) 'actual_loss: ', actual_loss%val(:,1)
     write(0,*) 'expected_loss: ', sum(expected_loss,2) / 3.0
     success = .false.
  end if

  loss = cce_loss_type()
  expected_loss = -expected * log(predicted + 1.E-7)
  actual_loss => loss%compute(predicted_array, expected_array)
  if(any(abs(actual_loss%val(:,1) - sum(expected_loss) / 3.0) .gt. 1.E-6))then
     write(0,*) 'Error: compute_loss_cce did not return the expected result.'
     write(0,*) 'actual_loss: ', actual_loss%val(:,1)
     write(0,*) 'expected_loss: ', sum(expected_loss,2) / 3.0
     success = .false.
  end if

  loss = mae_loss_type()
  expected_loss = abs(predicted - expected)
  actual_loss => loss%compute(predicted_array, expected_array)
  if(any(abs(actual_loss%val(:,1) - sum(expected_loss) / 18.0) .gt. 1.E-6))then
     write(0,*) 'Error: compute_loss_mae did not return the expected result.'
     write(0,*) 'actual_loss: ', actual_loss%val(:,1)
     write(0,*) 'expected_loss: ', sum(expected_loss) / 18.0
     success = .false.
  end if

  loss = mse_loss_type()
  expected_loss = ((predicted - expected)**2.E0) /(2.E0)
  actual_loss => loss%compute(predicted_array, expected_array)
  if(any(abs(actual_loss%val(:,1) - sum(expected_loss) / 9.0) .gt. 1.E-6))then
     write(0,*) 'Error: compute_loss_mse did not return the expected result.'
     write(0,*) 'actual_loss: ', actual_loss%val(:,1)
     write(0,*) 'expected_loss: ', sum(expected_loss) / 9.0
     success = .false.
  end if

  loss = nll_loss_type()
  expected_loss = - log(expected - predicted + 1.E-7)
  actual_loss => loss%compute(predicted_array, expected_array)
  if(any(abs(actual_loss%val(:,1) - sum(expected_loss) / 9.0) .gt. 1.E-6))then
     write(0,*) 'Error: compute_loss_nll did not return the expected result.'
     write(0,*) 'actual_loss: ', actual_loss%val(:,1)
     write(0,*) 'expected_loss: ', sum(expected_loss) / 9.0
     success = .false.
  end if

  loss = huber_loss_type()
  where (abs(predicted - expected) .le. 1.0)
     expected_loss = 0.5 * (predicted - expected)**2.0
  elsewhere
     expected_loss = 1.0 * (abs(predicted - expected) - 0.5 * 1.0)
  end where
  actual_loss => loss%compute(predicted_array, expected_array)
  if(any(abs(actual_loss%val(:,1) - sum(expected_loss) / 9.0) .gt. 1.E-6))then
     write(0,*) 'Error: compute_loss_huber did not return the expected result.'
     write(0,*) 'actual_loss: ', actual_loss%val(:,1)
     write(0,*) 'expected_loss: ', sum(expected_loss) / 9.0
     success = .false.
  end if

  predicted_values = reshape([0.2, 0.4, 0.6, 0.8], [2, 2])
  expected_values = 1.0
  allocate(predicted_grid(2, 2), expected_grid(2, 2))
  do j = 1, 2
     do i = 1, 2
        call predicted_grid(i,j)%allocate(array_shape=[1, 1])
        predicted_grid(i,j)%val(1,1) = predicted_values(i,j)
        call expected_grid(i,j)%allocate(array_shape=[1, 1])
        expected_grid(i,j)%val(1,1) = expected_values(i,j)
     end do
  end do

  loss = bce_loss_type()
  actual_loss => loss%compute(predicted_grid, expected_grid)
  expected_scalar = sum(-expected_values * log(predicted_values + 1.E-7))
  if(abs(actual_loss%val(1,1) - expected_scalar) .gt. 5.E-6)then
     write(0,*) 'Error: multi-cell BCE accumulation failed.'
     write(0,*) 'actual_loss: ', actual_loss%val(1,1)
     write(0,*) 'expected_loss: ', expected_scalar
     success = .false.
  end if

  loss = cce_loss_type()
  actual_loss => loss%compute(predicted_grid, expected_grid)
  expected_scalar = sum(-expected_values * log(predicted_values + 1.E-7))
  if(abs(actual_loss%val(1,1) - expected_scalar) .gt. 5.E-6)then
     write(0,*) 'Error: multi-cell CCE accumulation failed.'
     write(0,*) 'actual_loss: ', actual_loss%val(1,1)
     write(0,*) 'expected_loss: ', expected_scalar
     success = .false.
  end if

  loss = mae_loss_type()
  actual_loss => loss%compute(predicted_grid, expected_grid)
  expected_scalar = sum(abs(predicted_values - expected_values) / 2.0)
  if(abs(actual_loss%val(1,1) - expected_scalar) .gt. 5.E-6)then
     write(0,*) 'Error: multi-cell MAE accumulation failed.'
     write(0,*) 'actual_loss: ', actual_loss%val(1,1)
     write(0,*) 'expected_loss: ', expected_scalar
     success = .false.
  end if

  loss = mse_loss_type()
  actual_loss => loss%compute(predicted_grid, expected_grid)
  expected_scalar = sum(((predicted_values - expected_values)**2.0) / 2.0)
  if(abs(actual_loss%val(1,1) - expected_scalar) .gt. 5.E-6)then
     write(0,*) 'Error: multi-cell MSE accumulation failed.'
     write(0,*) 'actual_loss: ', actual_loss%val(1,1)
     write(0,*) 'expected_loss: ', expected_scalar
     success = .false.
  end if

  loss = nll_loss_type()
  actual_loss => loss%compute(predicted_grid, expected_grid)
  expected_scalar = sum(-log(expected_values - predicted_values + 1.E-7))
  if(abs(actual_loss%val(1,1) - expected_scalar) .gt. 5.E-6)then
     write(0,*) 'Error: multi-cell NLL accumulation failed.'
     write(0,*) 'actual_loss: ', actual_loss%val(1,1)
     write(0,*) 'expected_loss: ', expected_scalar
     success = .false.
  end if

  huber_predicted_values = reshape([0.2, 0.6, 1.5, 2.0], [2, 2])
  huber_expected_values = 0.0
  allocate(huber_predicted(2, 2), huber_expected(2, 2))
  do j = 1, 2
     do i = 1, 2
        call huber_predicted(i,j)%allocate(array_shape=[1, 1])
        huber_predicted(i,j)%val(1,1) = huber_predicted_values(i,j)
        call huber_expected(i,j)%allocate(array_shape=[1, 1])
        huber_expected(i,j)%val(1,1) = huber_expected_values(i,j)
     end do
  end do

  loss = huber_loss_type()
  actual_loss => loss%compute(huber_predicted, huber_expected)
  expected_scalar = 0.0
  do j = 1, 2
     do i = 1, 2
        if(abs(huber_predicted_values(i,j) - huber_expected_values(i,j)) &
             .le. 1.0)then
           expected_scalar = expected_scalar + 0.5 * &
                (huber_predicted_values(i,j) - huber_expected_values(i,j))**2
        else
           expected_scalar = expected_scalar + abs( &
                huber_predicted_values(i,j) - huber_expected_values(i,j)) &
                - 0.5
        end if
     end do
  end do
  if(abs(actual_loss%val(1,1) - expected_scalar) .gt. 5.E-6)then
     write(0,*) 'Error: multi-cell Huber accumulation failed.'
     write(0,*) 'actual_loss: ', actual_loss%val(1,1)
     write(0,*) 'expected_loss: ', expected_scalar
     success = .false.
  end if

  write(*,*) '----------------------------------------'
  if(success)then
     write(*,*) 'test_loss passed all tests'
  else
     write(0,*) 'test_loss failed one or more tests'
     stop 1
  end if

end program test_loss
