program test_onnx
  use coreutils, only: real32
  use athena
  use diffstruc, only: array_type

  implicit none

  logical :: success = .true.
  type(network_type) :: network, network_imported
  real(real32), allocatable :: input(:,:,:,:), output_ref(:,:), output_new(:,:)


!-------------------------------------------------------------------------------
! Build a deterministic network that exercises the standard ONNX import path.
!-------------------------------------------------------------------------------
  call random_setup(123, restart=.false.)
  call build_standard_network(network)
  call build_input_batch(input)

  output_ref = evaluate_standard_network(network, input)


!-------------------------------------------------------------------------------
! Round-trip through ONNX JSON and compare predictions.
!-------------------------------------------------------------------------------
  call write_onnx('test_model.json', network)
  network_imported = read_onnx('test_model.json', verbose=0)
  call compile_standard_network(network_imported)

  output_new = evaluate_standard_network(network_imported, input)

  call require( &
       network_imported%num_outputs.eq.network%num_outputs, &
       'ONNX import changed the number of network outputs' &
  )
  call require( &
       all(shape(output_new).eq.shape(output_ref)), &
       'ONNX import changed the prediction shape' &
  )

  call write_onnx('test_model_roundtrip.json', network_imported)


!-------------------------------------------------------------------------------
! Check for failures.
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_onnx passed all tests'
  else
     write(0,*) 'test_onnx failed one or more tests'
     stop 1
  end if


contains

!-------------------------------------------------------------------------------
  subroutine build_standard_network(net)
    implicit none
    type(network_type), intent(out) :: net

    call net%add(conv2d_layer_type( &
         input_shape = [8, 8, 1], &
         kernel_size = [3, 3], &
         num_filters = 4, &
         activation = 'relu' &
    ))
    call net%add(maxpool2d_layer_type(pool_size=2, stride=2))
    call net%add(full_layer_type(num_outputs=6, activation='relu'))
    call net%add(full_layer_type(num_outputs=3, activation='softmax'))

    call compile_standard_network(net)
  end subroutine build_standard_network

!-------------------------------------------------------------------------------
  subroutine compile_standard_network(net)
    implicit none
    type(network_type), intent(inout) :: net

    call net%compile( &
         optimiser = sgd_optimiser_type( &
              learning_rate = 1.e-2_real32 &
         ), &
         loss_method = 'mse', &
         metrics = ['loss'], &
         accuracy_method = 'mse', &
         batch_size = 1, &
         verbose = 0 &
    )
    call net%set_batch_size(1)
    call net%set_inference_mode()
  end subroutine compile_standard_network

!-------------------------------------------------------------------------------
  subroutine build_input_batch(input_batch)
    implicit none
    real(real32), allocatable, intent(out) :: input_batch(:,:,:,:)

    allocate(input_batch(8, 8, 1, 1))
    call random_number(input_batch)
  end subroutine build_input_batch

!-------------------------------------------------------------------------------
  function evaluate_standard_network(net, input_data) result(output)
    implicit none
    type(network_type), intent(inout) :: net
    real(real32), intent(in) :: input_data(:,:,:,:)
    real(real32), allocatable :: output(:,:)
#ifdef __INTEL_COMPILER
    type(array_type), pointer :: input_array(:,:)
#else
    type(array_type) :: input_array(1,1)
#endif
    integer :: leaf_id

#ifdef __INTEL_COMPILER
    allocate(input_array(1,1))
#endif
    call input_array(1,1)%allocate( &
         array_shape = shape(input_data), source = 0._real32 &
    )
    call input_array(1,1)%set(input_data)
    call net%forward(input_array)
    leaf_id = net%auto_graph%vertex(net%leaf_vertices(1))%id
    allocate(output, source=net%model(leaf_id)%layer%output(1,1)%val)
    call input_array(1,1)%deallocate()
#ifdef __INTEL_COMPILER
    deallocate(input_array)
#endif
  end function evaluate_standard_network

!-------------------------------------------------------------------------------
  subroutine require(condition, message)
    implicit none
    logical, intent(in) :: condition
    character(*), intent(in) :: message

    if(.not.condition)then
       success = .false.
       write(0,*) trim(message)
    end if
  end subroutine require

end program test_onnx
