program onnx_nop_example
  !! ONNX round-trip test for Neural Operator (NOP) layers
  !!
  !! This example builds a small network with NOP layers, exports it to
  !! ONNX JSON, reimports it, and compares the forward-pass outputs to
  !! verify that the round-trip preserves both architecture and weights.
  use athena
  implicit none

  integer, parameter :: num_in = 48, num_out = 32, num_modes = 16
  !! Input size, output size and spectral mode count
  integer, parameter :: batch = 4
  !! Batch size for the round-trip comparison
  real, dimension(num_in, batch) :: x
  !! Random input batch used for both forward passes
  real, dimension(num_out, batch) :: y_orig, y_read
  !! Original and reloaded network outputs
  type(network_type) :: net_orig, net_read
  !! Original network and network reloaded from ONNX
  character(256) :: onnx_file
  !! ONNX JSON path for export/import
  real :: max_diff
  !! Maximum absolute difference between outputs
  integer :: i
  !! Loop index for debug printout


  !---------------------------------------------------------------------------
  ! Build a small dynamic-LNO network
  !---------------------------------------------------------------------------
  write(*,'(A)') 'Building dynamic LNO network ...'

  call net_orig%add(dynamic_lno_layer_type( &
       num_outputs = num_out, &
       num_modes   = num_modes, &
       num_inputs  = num_in, &
       use_bias    = .true., &
       activation  = 'relu'))
  call net_orig%add(fixed_lno_layer_type( &
       num_outputs = num_out, &
       num_modes   = num_modes, &
       use_bias    = .true., &
       activation  = 'relu'))

  call net_orig%compile( &
       optimiser   = base_optimiser_type(learning_rate=0.001), &
       loss_method = 'mse', &
       batch_size  = batch)

  call net_orig%print_summary()


  !---------------------------------------------------------------------------
  ! Forward pass with original network
  !---------------------------------------------------------------------------
  call random_number(x)

  y_orig = net_orig%predict(x)
  write(*,'(A,I0,A,I0)') 'Original output shape: ', &
       size(y_orig,1), ' x ', size(y_orig,2)


  !---------------------------------------------------------------------------
  ! Export to ONNX JSON
  !---------------------------------------------------------------------------
  onnx_file = 'example/onnx_nop/model.json'
  write(*,'(A,A)') 'Exporting to ONNX: ', trim(onnx_file)
  call write_onnx(onnx_file, net_orig, format=1)


  !---------------------------------------------------------------------------
  ! Re-import from ONNX JSON
  !---------------------------------------------------------------------------
  write(*,'(A)') 'Reimporting from ONNX ...'
  net_read = read_onnx(onnx_file, verbose=1)

  call net_read%compile( &
       optimiser   = base_optimiser_type(learning_rate=0.001), &
       loss_method = 'mse', &
       batch_size  = batch)

  call net_read%print_summary()


  !---------------------------------------------------------------------------
  ! Forward pass with reimported network and compare
  !---------------------------------------------------------------------------
  y_read = net_read%predict(x)

  max_diff = maxval(abs(y_orig - y_read))
  write(*,'(A,ES12.5)') 'Max absolute difference: ', max_diff

  if(max_diff .lt. 1.0e-5)then
     write(*,'(A)') 'PASS: Round-trip preserved network within tolerance.'
  else
     write(*,'(A)') 'FAIL: Round-trip mismatch exceeds tolerance!'
     do i = 1, min(5, batch)
        write(*,'(A,I0,A,ES12.5,A,ES12.5)') &
             '  sample ', i, &
             ':  orig=', y_orig(1,i), '  read=', y_read(1,i)
     end do
  end if

end program onnx_nop_example
