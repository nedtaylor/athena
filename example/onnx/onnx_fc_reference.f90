program onnx_fc_reference
  !! Generate a reference FC network, export to ONNX binary, and run a forward
  !! pass with known input. Writes the output to a text file for comparison
  !! with external tools (Python/PyTorch/ONNX Runtime).
  use coreutils, only: real32
  use athena

  implicit none

  type(network_type) :: net
  real(real32), allocatable :: input_data(:,:), output_data(:,:)
  integer :: num_seed, i, j, unit_out
  integer, allocatable, dimension(:) :: seed

  ! Reproducible random seed
  call random_seed(size=num_seed)
  allocate(seed(num_seed))
  seed = (/ (i, i=1, num_seed) /)
  call random_setup(seed, restart=.false.)

  ! Build FC network: 10 -> 8 (relu) -> 4
  call net%add(input_layer_type(input_shape=[10]))
  call net%add(full_layer_type(num_outputs=8, activation='relu'))
  call net%add(full_layer_type(num_outputs=4))

  call net%compile( &
       optimiser = sgd_optimiser_type(learning_rate = 1.E-2_real32), &
       loss_method = "mse", &
       accuracy_method = "mse", &
       verbose = 0, &
       batch_size = 1 &
  )

  ! Export to binary ONNX
  call save_onnx('reference_fc.onnx', net)
  write(*,*) 'Exported reference_fc.onnx'

  ! Prepare input: all ones, shape (1, 10)
  allocate(input_data(10, 1))
  input_data = 1.0_real32

  ! Forward pass
  output_data = net%predict(input_data)

  ! Write reference output
  open(newunit=unit_out, file='reference_fc_output.txt', status='replace')
  write(unit_out, '(A)') '# Athena FC network forward pass output'
  write(unit_out, '(A)') '# Input: ones(1, 10)'
  write(unit_out, '(A)') '# Output (4 values):'
  do j = 1, 4
     write(unit_out, '(E20.12)') output_data(j, 1)
  end do
  close(unit_out)

  write(*,*) 'Athena output:'
  do j = 1, 4
     write(*,'(A,I0,A,E20.12)') '  output[', j-1, '] = ', output_data(j, 1)
  end do

end program onnx_fc_reference
