program test_onnx_pytorch_nop_import
  !! Regression test for importing PyTorch-style NOP/LNO ONNX graphs.
  use athena
  use coreutils, only: real32, pi
  implicit none

  type(network_type) :: python_network
  real(real32), parameter :: expected_output(48) = [ &
       -0.0251456797_real32, 0.0474197380_real32, 0.0004574030_real32, &
       -0.0493903905_real32, -0.0105340499_real32, -0.0110056456_real32, &
       -0.0125177167_real32, -0.0097822305_real32, -0.0371379144_real32, &
       0.0008492495_real32, 0.0408220515_real32, 0.0328212492_real32, &
       0.0256083440_real32, 0.0447822064_real32, -0.0200615879_real32, &
       0.0358736217_real32, -0.0024657752_real32, 0.0041111750_real32, &
       -0.0288147666_real32, -0.0122372089_real32, 0.0319091938_real32, &
       -0.0097788516_real32, -0.0434338450_real32, 0.0155056855_real32, &
       0.0345000625_real32, -0.0333463550_real32, -0.0352671184_real32, &
       -0.0199292917_real32, -0.0400573909_real32, -0.0229577031_real32, &
       -0.0381513983_real32, -0.0154445041_real32, 0.0130559439_real32, &
       -0.0334145315_real32, 0.0306734405_real32, -0.0022374154_real32, &
       -0.0088451607_real32, 0.0259310342_real32, -0.0421876572_real32, &
       -0.0040429756_real32, -0.0209704451_real32, -0.0190024618_real32, &
       0.0499222241_real32, -0.0148923658_real32, 0.0497453213_real32, &
       0.0272012949_real32, -0.0002294979_real32, 0.0041748742_real32 ]
  real(real32) :: x(48, 1), python_output(48, 1)
  real(real32) :: max_abs_diff, tol, x_value
  integer :: i

  tol = 1.0e-6_real32

  python_network = read_onnx('example/lno_rollout/shared/model.json')

  call python_network%compile( &
       optimiser=base_optimiser_type(learning_rate=0.001_real32), &
       loss_method='mse', batch_size=1)

  if(python_network%num_layers .ne. 3)then
     write(*,*) 'FAIL: imported Python model layer count is unexpected.'
     write(*,*) 'Imported layers:', python_network%num_layers
     stop 1
  end if

  do i = 1, 48
     x_value = real(i - 1, real32) / 47.0_real32
     x(i,1) = sin(2.0_real32 * pi * x_value) + 0.25_real32 * &
          cos(3.0_real32 * pi * x_value)
  end do
  x(1,1) = -1.0_real32
  x(48,1) = 1.0_real32

  python_output = python_network%predict(x)

  max_abs_diff = maxval(abs(python_output(:,1) - expected_output))
  write(*,'(A,ES12.4)') 'PyTorch LNO import max abs diff: ', max_abs_diff

  if(max_abs_diff .gt. tol)then
     write(*,*) 'FAIL: imported PyTorch LNO output diverges from reference.'
     stop 1
  end if

  write(*,*) 'PASS: PyTorch-style LNO ONNX import matches reference output.'

end program test_onnx_pytorch_nop_import
