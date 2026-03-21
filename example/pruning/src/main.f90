program pruning_example
  !! Post-training pruning and network compaction demonstration
  !!
  !! This example demonstrates how to:
  !! 1. Train an over-parameterised network on sin(x)
  !! 2. Apply fraction-based pruning to achieve ~50% sparsity
  !! 3. Print sparsity statistics
  !! 4. Build a compact network with dead neurons removed
  !! 5. Compare accuracy and parameter counts
  !!
  !! ## Workflow
  !!
  !! 1. Build a wide 1 -> 64 -> 32 -> 1 network
  !! 2. Train on sin(x) samples
  !! 3. Prune the smallest 50% of parameters
  !! 4. Report sparsity
  !! 5. Compact the network (remove dead neurons)
  !! 6. Compare predictions before and after
  use athena
  use coreutils, only: real32
  implicit none

  real(real32), parameter :: pi = 4.0_real32 * atan(1.0_real32)
  integer, parameter :: num_iterations = 5000
  integer, parameter :: test_size = 20

  ! Training variables
  type(network_type) :: network, compacted
  type(array_type), pointer :: loss
  real(real32), dimension(1,1) :: x, y
  type(array_type) :: y_array(1,1)
  real(real32), dimension(1,test_size) :: x_test, y_test
  real(real32), dimension(1,test_size) :: y_pred
  type(sparsity_info_type) :: info
  real(real32) :: mse_before, mse_after, mse_compact

  integer :: i, n
  integer :: seed_size
  integer, allocatable :: seed(:)


  !-----------------------------------------------------------------------------
  ! Set random seed for reproducibility
  !-----------------------------------------------------------------------------
  seed_size = 8
  call random_seed(size=seed_size)
  allocate(seed(seed_size))
  seed = 42
  call random_seed(put=seed)

  write(*,'(A)') "================================================="
  write(*,'(A)') "  Post-Training Pruning & Compaction Example"
  write(*,'(A)') "================================================="
  write(*,*)


  !-----------------------------------------------------------------------------
  ! Build wide network: 1 -> 64 -> 32 -> 1
  !-----------------------------------------------------------------------------
  call network%add(full_layer_type( &
       num_inputs=1, num_outputs=64, activation="tanh"))
  call network%add(full_layer_type( &
       num_outputs=32, activation="tanh"))
  call network%add(full_layer_type( &
       num_outputs=1, activation="sigmoid"))
  call network%compile( &
       optimiser=sgd_optimiser_type(learning_rate=0.1_real32), &
       loss_method="mse", metrics=["loss"], verbose=0)
  call network%set_batch_size(1)

  call y_array(1,1)%allocate(array_shape=[1,1])

  write(*,'(A,I0)') "Network parameters: ", network%num_params
  write(*,*)


  !-----------------------------------------------------------------------------
  ! Generate test data
  !-----------------------------------------------------------------------------
  do i = 1, test_size
     x_test(1, i) = real(i - 1, real32) * 2.0_real32 * pi / &
          real(test_size - 1, real32)
     y_test(1, i) = (sin(x_test(1, i)) + 1.0_real32) / 2.0_real32
  end do


  !-----------------------------------------------------------------------------
  ! Train network
  !-----------------------------------------------------------------------------
  write(*,'(A)') "Training on sin(x)..."
  do n = 1, num_iterations
     call random_number(x)
     x = x * 2.0_real32 * pi
     y(1,1) = (sin(x(1,1)) + 1.0_real32) / 2.0_real32

     y_array(1,1)%val = y
     network%expected_array = y_array

     call network%forward(x)
     loss => network%loss_eval(1, 1)
     call loss%grad_reverse()
     call network%update()
     call loss%nullify_graph()
     loss => null()

     if(mod(n, 1000) .eq. 0)then
        y_pred = network%predict(input=x_test)
        write(*,'(A,I6,A,F9.6)') "  Epoch ", n, &
             "  test MSE = ", &
             sum((y_pred - y_test)**2) / real(test_size, real32)
     end if
  end do


  !-----------------------------------------------------------------------------
  ! Evaluate before pruning
  !-----------------------------------------------------------------------------
  y_pred = network%predict(input=x_test)
  mse_before = sum((y_pred - y_test)**2) / real(test_size, real32)

  write(*,*)
  write(*,'(A,F9.6)') "Test MSE before pruning: ", mse_before


  !-----------------------------------------------------------------------------
  ! Sparsity before pruning
  !-----------------------------------------------------------------------------
  write(*,*)
  write(*,'(A)') "--- Before Pruning ---"
  call print_sparsity_info(network)


  !-----------------------------------------------------------------------------
  ! Apply fraction pruning (prune smallest 50%)
  !-----------------------------------------------------------------------------
  write(*,*)
  write(*,'(A)') "Applying fraction pruning (50%)..."
  call prune_fraction(network, 0.5_real32)


  !-----------------------------------------------------------------------------
  ! Sparsity after pruning
  !-----------------------------------------------------------------------------
  write(*,*)
  write(*,'(A)') "--- After Pruning ---"
  call print_sparsity_info(network)

  info = get_sparsity_info(network)
  write(*,*)
  write(*,'(A,I0,A,I0)') "Pruned ", info%pruned_params, &
       " of ", info%total_params
  write(*,'(A,F6.2,A)') "Sparsity: ", info%sparsity * 100.0_real32, "%"


  !-----------------------------------------------------------------------------
  ! Evaluate after pruning
  !-----------------------------------------------------------------------------
  y_pred = network%predict(input=x_test)
  mse_after = sum((y_pred - y_test)**2) / real(test_size, real32)

  write(*,*)
  write(*,'(A,F9.6)') "Test MSE after pruning:  ", mse_after
  write(*,'(A,F6.2,A)') "MSE change: ", &
       (mse_after - mse_before) / max(mse_before, 1.0E-10_real32) &
       * 100.0_real32, "%"


  !-----------------------------------------------------------------------------
  ! Build compact network with dead neurons removed
  !-----------------------------------------------------------------------------
  write(*,*)
  write(*,'(A)') "Building compact network..."
  call compact_network(network, compacted, batch_size=1)

  write(*,'(A,I0)') "Original parameters:  ", network%num_params
  write(*,'(A,I0)') "Compact parameters:   ", compacted%num_params
  write(*,'(A,F6.2,A)') "Size reduction:       ", &
       (1.0_real32 - real(compacted%num_params, real32) / &
            real(network%num_params, real32)) * 100.0_real32, "%"


  !-----------------------------------------------------------------------------
  ! Evaluate compact network
  !-----------------------------------------------------------------------------
  y_pred = compacted%predict(input=x_test)
  mse_compact = sum((y_pred - y_test)**2) / real(test_size, real32)

  write(*,*)
  write(*,'(A,F9.6)') "Test MSE compact:      ", mse_compact

  write(*,*)
  write(*,'(A)') "--- Compact Network Sparsity ---"
  call print_sparsity_info(compacted)

  write(*,*)
  write(*,'(A)') "================================================="
  write(*,'(A)') "  Done"
  write(*,'(A)') "================================================="

end program pruning_example
