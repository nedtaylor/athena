program test_onnx_nop
  use coreutils, only: real32
  use athena

  implicit none

  integer, parameter :: num_inputs = 24
  integer, parameter :: batch_size = 3
  logical :: success = .true.
  real(real32), parameter :: tol = 1.e-5_real32
  real(real32) :: input(num_inputs, batch_size)


!-------------------------------------------------------------------------------
! Exercise abstract and expanded ONNX round-trips for each NOP layer type.
!-------------------------------------------------------------------------------
  call random_setup(777, restart=.false.)
  call random_number(input)
  call check_dynamic_lno_roundtrip(input)
  call check_fixed_lno_roundtrip(input)
  call check_neural_operator_roundtrip(input)
  call check_orthogonal_nop_roundtrip(input)


!-------------------------------------------------------------------------------
! Check for failures.
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_onnx_nop passed all tests'
  else
     write(0,*) 'test_onnx_nop failed one or more tests'
     stop 1
  end if


contains

!-------------------------------------------------------------------------------
  subroutine check_dynamic_lno_roundtrip(input_batch)
    real(real32), intent(in) :: input_batch(:,:)
    type(network_type) :: network, imported_network
    real(real32), allocatable :: output_ref(:,:), output_new(:,:)

    call network%add(dynamic_lno_layer_type( &
         num_inputs = num_inputs, &
         num_outputs = 18, &
         num_modes = 6, &
         use_bias = .true., &
         activation = 'relu' &
    ))
    call compile_nop_network(network)
    output_ref = network%predict(input_batch)

    call write_onnx('test_dynamic_lno_model.json', network)
    imported_network = read_onnx('test_dynamic_lno_model.json', verbose=0)
    call compile_nop_network(imported_network)
    output_new = imported_network%predict(input_batch)
    call require_close(output_new, output_ref, &
         'ATHENA dynamic_lno round-trip changed predictions')

    call write_onnx( &
         'test_dynamic_lno_model_expanded.json', network, &
         format='onnx_expanded' &
    )
    imported_network = read_onnx( &
         'test_dynamic_lno_model_expanded.json', verbose=0)
    call compile_nop_network(imported_network)
    output_new = imported_network%predict(input_batch)
    call require_close(output_new, output_ref, &
         'expanded dynamic_lno round-trip changed predictions')
  end subroutine check_dynamic_lno_roundtrip

  !-------------------------------------------------------------------------------
  subroutine check_fixed_lno_roundtrip(input_batch)
    real(real32), intent(in) :: input_batch(:,:)
    type(network_type) :: network, imported_network
    real(real32), allocatable :: output_ref(:,:), output_new(:,:)

    call network%add(fixed_lno_layer_type( &
         num_inputs = num_inputs, &
         num_outputs = 12, &
         num_modes = 4, &
         use_bias = .true., &
         activation = 'relu' &
    ))
    call compile_nop_network(network)
    output_ref = network%predict(input_batch)

    call write_onnx('test_fixed_lno_model.json', network)
    imported_network = read_onnx('test_fixed_lno_model.json', verbose=0)
    call compile_nop_network(imported_network)
    output_new = imported_network%predict(input_batch)
    call require_close(output_new, output_ref, &
         'ATHENA fixed_lno round-trip changed predictions')

    call write_onnx( &
         'test_fixed_lno_model_expanded.json', network, &
         format='onnx_expanded' &
    )
    imported_network = read_onnx( &
         'test_fixed_lno_model_expanded.json', verbose=0)
    call compile_nop_network(imported_network)
    output_new = imported_network%predict(input_batch)
    call require_close(output_new, output_ref, &
         'expanded fixed_lno round-trip changed predictions')
  end subroutine check_fixed_lno_roundtrip

  !-------------------------------------------------------------------------------
  subroutine check_neural_operator_roundtrip(input_batch)
    real(real32), intent(in) :: input_batch(:,:)
    type(network_type) :: network, imported_network
    real(real32), allocatable :: output_ref(:,:), output_new(:,:)

    call network%add(neural_operator_layer_type( &
         num_inputs = num_inputs, &
         num_outputs = 8, &
         use_bias = .true., &
         activation = 'tanh' &
    ))
    call compile_nop_network(network)
    output_ref = network%predict(input_batch)

    call write_onnx('test_neural_operator_model.json', network)
    imported_network = read_onnx('test_neural_operator_model.json', verbose=0)
    call compile_nop_network(imported_network)
    output_new = imported_network%predict(input_batch)
    call require_close(output_new, output_ref, &
         'ATHENA neural_operator round-trip changed predictions')

    call write_onnx( &
         'test_neural_operator_model_expanded.json', network, &
         format='onnx_expanded' &
    )
    imported_network = read_onnx( &
         'test_neural_operator_model_expanded.json', verbose=0)
    call compile_nop_network(imported_network)
    output_new = imported_network%predict(input_batch)
    call require_close(output_new, output_ref, &
         'expanded neural_operator round-trip changed predictions')
  end subroutine check_neural_operator_roundtrip

  !-------------------------------------------------------------------------------
  subroutine check_orthogonal_nop_roundtrip(input_batch)
    real(real32), intent(in) :: input_batch(:,:)
    type(network_type) :: network, imported_network
    real(real32), allocatable :: output_ref(:,:), output_new(:,:)

    write(*,'("Checking orthogonal_nop ONNX round-trip")')

    call network%add(orthogonal_nop_block_type( &
         num_inputs = num_inputs, &
         num_outputs = 10, &
         num_basis = 5, &
         use_bias = .true., &
         activation = 'swish' &
    ))
    call compile_nop_network(network)
    output_ref = network%predict(input_batch)

    call write_onnx('test_orthogonal_nop_model.json', network)
    imported_network = read_onnx('test_orthogonal_nop_model.json', verbose=0)
    call compile_nop_network(imported_network)
    output_new = imported_network%predict(input_batch)
    call require_close(output_new, output_ref, &
         'ATHENA orthogonal_nop round-trip changed predictions')
  end subroutine check_orthogonal_nop_roundtrip

  !-------------------------------------------------------------------------------
  subroutine check_orthogonal_attention_roundtrip(input_batch)
    real(real32), intent(in) :: input_batch(:,:)
    type(network_type) :: network, imported_network
    real(real32), allocatable :: output_ref(:,:), output_new(:,:)

    write(*,'("Checking orthogonal_attention ONNX round-trip")')

    call network%add(orthogonal_attention_layer_type( &
         num_inputs = num_inputs, &
         num_outputs = 9, &
         num_basis = 4, &
         key_dim = 6, &
         use_bias = .true., &
         activation = 'leaky_relu' &
    ))
    call compile_nop_network(network)
    output_ref = network%predict(input_batch)

    call write_onnx('test_orthogonal_attention_model.json', network)
    imported_network = read_onnx( &
         'test_orthogonal_attention_model.json', verbose=0)
    call compile_nop_network(imported_network)
    output_new = imported_network%predict(input_batch)
    call require_close(output_new, output_ref, &
         'ATHENA orthogonal_attention round-trip changed predictions')
  end subroutine check_orthogonal_attention_roundtrip

  !-------------------------------------------------------------------------------
  subroutine check_spectral_filter_roundtrip(input_batch)
    real(real32), intent(in) :: input_batch(:,:)
    type(network_type) :: network, imported_network
    real(real32), allocatable :: output_ref(:,:), output_new(:,:)

    write(*,'("Checking spectral_filter ONNX round-trip")')

    call network%add(spectral_filter_layer_type( &
         num_inputs = num_inputs, &
         num_outputs = 11, &
         num_modes = 5, &
         use_bias = .true., &
         activation = 'sigmoid' &
    ))
    call compile_nop_network(network)
    output_ref = network%predict(input_batch)

    call write_onnx( &
         'test_spectral_filter_model_expanded.json', network, &
         format='onnx_expanded' &
    )
    imported_network = read_onnx( &
         'test_spectral_filter_model_expanded.json', verbose=0)
    call compile_nop_network(imported_network)
    output_new = imported_network%predict(input_batch)
    call require_close(output_new, output_ref, &
         'expanded spectral_filter round-trip changed predictions')
  end subroutine check_spectral_filter_roundtrip

  !-------------------------------------------------------------------------------
  subroutine build_nop_network(net)
    type(network_type), intent(out) :: net

    call net%add(dynamic_lno_layer_type( &
         num_inputs = num_inputs, &
         num_outputs = 18, &
         num_modes = 6, &
         use_bias = .true., &
         activation = 'relu' &
    ))
    call net%add(fixed_lno_layer_type( &
         num_outputs = 12, &
         num_modes = 4, &
         use_bias = .true., &
         activation = 'relu' &
    ))
    call net%add(neural_operator_layer_type( &
         num_outputs = 8, &
         use_bias = .true., &
         activation = 'tanh' &
    ))

    call compile_nop_network(net)
  end subroutine build_nop_network

!-------------------------------------------------------------------------------
  subroutine compile_nop_network(net)
    type(network_type), intent(inout) :: net

    call net%compile( &
         optimiser = base_optimiser_type(learning_rate=1.e-3_real32), &
         loss_method = 'mse', &
         batch_size = batch_size, &
         verbose = 0 &
    )
    call net%set_batch_size(batch_size)
    call net%set_inference_mode()
  end subroutine compile_nop_network

!-------------------------------------------------------------------------------
  subroutine require_close(actual, expected, message)
    real(real32), intent(in) :: actual(:,:), expected(:,:)
    character(*), intent(in) :: message

    call require( &
         all(shape(actual).eq.shape(expected)), &
         trim(message) // ' (shape mismatch)' &
    )
    if(all(shape(actual).eq.shape(expected)))then
       call require(maxval(abs(actual - expected)).lt.tol, message)
    end if
  end subroutine require_close

!-------------------------------------------------------------------------------
  subroutine require(condition, message)
    logical, intent(in) :: condition
    character(*), intent(in) :: message

    if(.not.condition)then
       success = .false.
       write(0,*) trim(message)
    end if
  end subroutine require

end program test_onnx_nop
