program test_onnx_binary
  !! Test program for binary ONNX (.onnx protobuf) export and import
  !! Tests round-trip: ATHENA -> binary .onnx -> ATHENA -> verify
  use coreutils, only: real32
  use athena

  implicit none

  type(network_type) :: network, net_text, net_binary, net_fc, net_fc_reimport
  integer :: num_seed, i, success
  integer, allocatable, dimension(:) :: seed
  logical :: test_passed
  integer :: num_layers_orig, num_layers_text, num_layers_binary, num_layers_fc

  success = 0

  ! Setup reproducible random seed
  call random_seed(size=num_seed)
  allocate(seed(num_seed))
  seed = (/ (i, i=1, num_seed) /)
  call random_setup(seed, restart=.false.)

  ! ============================================================
  ! Test 1: Simple network round-trip (binary ONNX)
  ! ============================================================
  write(*,*) '=== Test 1: Simple network binary ONNX round-trip ==='

  ! Create a simple network
  ! NOTE: Using 1-channel convolution input to avoid pre-existing ONNX
  ! reimport limitation with multi-channel conv2d initialisers
  call network%add(input_layer_type(input_shape=[10,10,1]))
  call network%add(conv2d_layer_type( &
       kernel_size=[3,3], &
       num_filters=1, &
       activation='swish' &
  ))
  call network%add(maxpool2d_layer_type( &
       pool_size=[2,2], &
       stride=[2,2] &
  ))
  call network%add(full_layer_type(num_outputs=16))
  call network%add(actv_layer_type('relu'))
  call network%add(full_layer_type(num_outputs=10))

  call network%compile( &
       optimiser = sgd_optimiser_type( &
            learning_rate = 1.E-2_real32 &
       ), &
       loss_method = "mse", &
       accuracy_method = "mse", &
       verbose = 0 &
  )

  num_layers_orig = network%num_layers

  ! Export to binary .onnx
  call write_onnx_binary('test_binary.onnx', network)
  write(*,*) '  Exported to test_binary.onnx'

  ! Export to text format for comparison
  call write_onnx('test_text.onnx.txt', network)
  write(*,*) '  Exported to test_text.onnx.txt'

  ! Import from binary
  net_binary = read_onnx_binary('test_binary.onnx', verbose=1)
  num_layers_binary = net_binary%num_layers
  write(*,*) '  Imported from binary: ', num_layers_binary, ' layers'

  ! Import from text
  net_text = read_onnx('test_text.onnx.txt', verbose=0)
  num_layers_text = net_text%num_layers
  write(*,*) '  Imported from text: ', num_layers_text, ' layers'

  ! Verify binary and text import produce same layer count
  test_passed = .true.
  if (num_layers_binary .ne. num_layers_text) then
     write(*,*) '  FAIL: Binary vs text layer count mismatch: ', &
          num_layers_binary, ' vs ', num_layers_text
     test_passed = .false.
     success = 1
  else
     write(*,*) '  PASS: Binary and text import layer counts match (', &
          num_layers_binary, ')'
  end if

  write(*,*) ''
  write(*,*) 'Test 1 result: ', merge('PASS', 'FAIL', test_passed)


  ! ============================================================
  ! Test 2: Unified save/load API with extension detection
  ! ============================================================
  write(*,*) ''
  write(*,*) '=== Test 2: Unified save_onnx / load_onnx API ==='

  ! Save as binary via extension detection
  call save_onnx('test_unified.onnx', network)
  write(*,*) '  save_onnx(.onnx) -> binary format'

  ! Save as text via extension detection
  call save_onnx('test_unified.onnx.txt', network)
  write(*,*) '  save_onnx(.onnx.txt) -> text format'

  ! Load binary via extension detection
  net_binary = load_onnx('test_unified.onnx', verbose=0)
  write(*,*) '  load_onnx(.onnx) -> binary reader, ', &
       net_binary%num_layers, ' layers'

  ! Load text via extension detection
  net_text = load_onnx('test_unified.onnx.txt', verbose=0)
  write(*,*) '  load_onnx(.onnx.txt) -> text reader, ', &
       net_text%num_layers, ' layers'

  test_passed = (net_binary%num_layers == net_text%num_layers)
  if (.not. test_passed) success = 1
  write(*,*) '  Test 2 result: ', merge('PASS', 'FAIL', test_passed)


  ! ============================================================
  ! Test 3: Fully-connected only network round-trip
  ! ============================================================
  write(*,*) ''
  write(*,*) '=== Test 3: FC-only network binary round-trip ==='

  ! Create a simple FC-only network (avoids conv reimport limitations)
  call net_fc%add(input_layer_type(input_shape=[10]))
  call net_fc%add(full_layer_type(num_outputs=8, activation='relu'))
  call net_fc%add(full_layer_type(num_outputs=4))

  call net_fc%compile( &
       optimiser = sgd_optimiser_type( &
            learning_rate = 1.E-2_real32 &
       ), &
       loss_method = "mse", &
       accuracy_method = "mse", &
       verbose = 0 &
  )
  num_layers_fc = net_fc%num_layers

  call write_onnx_binary('test_fc.onnx', net_fc)
  call write_onnx('test_fc.onnx.txt', net_fc)
  net_fc_reimport = read_onnx_binary('test_fc.onnx', verbose=0)
  net_text = read_onnx('test_fc.onnx.txt', verbose=0)

  test_passed = (net_fc_reimport%num_layers == net_text%num_layers)
  if (.not. test_passed) success = 1
  write(*,*) '  Binary reimport: ', net_fc_reimport%num_layers, &
       ' layers, Text reimport: ', net_text%num_layers, ' layers'
  write(*,*) '  Test 3 result: ', merge('PASS', 'FAIL', test_passed)


  ! ============================================================
  ! Summary
  ! ============================================================
  write(*,*) ''
  if (success == 0) then
     write(*,*) 'All binary ONNX tests PASSED!'
  else
     write(*,*) 'Some binary ONNX tests FAILED!'
     stop 1
  end if

end program test_onnx_binary
