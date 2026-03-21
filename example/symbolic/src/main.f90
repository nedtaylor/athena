program symbolic_example
  !! Symbolic regression extraction demonstration
  !!
  !! This example demonstrates how to:
  !! 1. Train a KAN network to learn sin(x)
  !! 2. Run symbolic extraction on the trained network
  !! 3. Print the recovered expression
  !! 4. Simplify the expression
  !!
  !! The extracted expression decomposes the learned function into a
  !! weighted sum of B-spline basis functions, providing an interpretable
  !! approximation of the network's learned mapping.
  use athena
  use coreutils, only: real32
  implicit none

  real(real32), parameter :: pi = 4.0_real32 * atan(1.0_real32)
  integer, parameter :: num_iterations = 2000
  integer, parameter :: test_size = 20

  ! Training variables
  type(network_type) :: network
  type(array_type), pointer :: loss
  real(real32), dimension(1,1) :: x, y
  type(array_type) :: y_array(1,1)
  real(real32), dimension(1,test_size) :: x_test, y_test, y_pred
  type(symbolic_expr_type), allocatable :: exprs(:), simplified(:)

  integer :: i, n, e
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
  write(*,'(A)') "  Symbolic Regression Extraction Example"
  write(*,'(A)') "================================================="
  write(*,*)


  !-----------------------------------------------------------------------------
  ! Build KAN network: 1 -> 1 with B-spline basis
  !-----------------------------------------------------------------------------
  write(*,'(A)') "Building KAN network for sin(x) approximation..."
  call network%add(kan_layer_type( &
       num_inputs=1, num_outputs=1, n_basis=10, spline_degree=3))
  call network%compile( &
       optimiser=sgd_optimiser_type(learning_rate=0.01_real32), &
       loss_method="mse", metrics=["loss"], verbose=0)
  call network%set_batch_size(1)

  call y_array(1,1)%allocate(array_shape=[1,1])

  write(*,'(A,I0)') "KAN parameters: ", network%num_params
  write(*,*)


  !-----------------------------------------------------------------------------
  ! Generate test data
  !-----------------------------------------------------------------------------
  do i = 1, test_size
     x_test(1, i) = -1.0_real32 + 2.0_real32 * &
          real(i - 1, real32) / real(test_size - 1, real32)
     y_test(1, i) = (sin(pi * x_test(1, i)) + 1.0_real32) / 2.0_real32
  end do


  !-----------------------------------------------------------------------------
  ! Train network on sin(pi*x) over [-1, 1]
  !-----------------------------------------------------------------------------
  write(*,'(A)') "Training on sin(pi*x)..."
  allocate(network%expected_array(1,1))
  call network%expected_array(1,1)%allocate(array_shape=[1,1])

  do n = 1, num_iterations
     call random_number(x)
     x = x * 2.0_real32 - 1.0_real32
     y(1,1) = (sin(pi * x(1,1)) + 1.0_real32) / 2.0_real32

     network%expected_array(1,1)%val = y

     call network%set_batch_size(1)
     call network%forward(x)
     loss => network%loss_eval(1, 1)
     call loss%grad_reverse()
     call network%update()
     call loss%nullify_graph()
     loss => null()

     if(mod(n, 500) .eq. 0)then
        y_pred = network%predict(input=x_test)
        write(*,'(A,I6,A,F9.6)') "  Epoch ", n, &
             "  test MSE = ", &
             sum((y_pred - y_test)**2) / real(test_size, real32)
     end if
  end do

  y_pred = network%predict(input=x_test)
  write(*,*)
  write(*,'(A,F9.6)') "Final test MSE: ", &
       sum((y_pred - y_test)**2) / real(test_size, real32)


  !-----------------------------------------------------------------------------
  ! Extract symbolic expression
  !-----------------------------------------------------------------------------
  write(*,*)
  write(*,'(A)') "================================================="
  write(*,'(A)') "  Extracted Symbolic Expression"
  write(*,'(A)') "================================================="
  write(*,*)

  exprs = extract_symbolic_kan(network, tolerance=0.01_real32)
  call print_symbolic_expr(exprs)

  write(*,'(A,I0,A)') "Found ", exprs(1)%num_terms, " significant terms"


  !-----------------------------------------------------------------------------
  ! Simplify expression
  !-----------------------------------------------------------------------------
  write(*,*)
  write(*,'(A)') "================================================="
  write(*,'(A)') "  Simplified Expression"
  write(*,'(A)') "================================================="
  write(*,*)

  allocate(simplified(size(exprs)))
  do e = 1, size(exprs)
     simplified(e) = simplify_expression(exprs(e))
  end do
  call print_symbolic_expr(simplified)

  write(*,'(A,I0,A)') "Simplified to ", simplified(1)%num_terms, " terms"

  write(*,*)
  write(*,'(A)') "================================================="
  write(*,'(A)') "  Extraction complete"
  write(*,'(A)') "================================================="

  deallocate(seed)

end program symbolic_example
