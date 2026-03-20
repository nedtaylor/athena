program kan_sine
  !! KAN layer benchmark: FastKAN (RBF) vs KAN (B-spline)
  !!
  !! This example benchmarks the KAN (B-spline) layer against the FastKAN (RBF)
  !! layer on sine and polynomial function approximation tasks.
  !!
  !! Training protocol (matches pykan_comparison.py):
  !!   - n_train fixed samples drawn uniformly from [-1, 1] before training
  !!   - n_epochs passes over the training set, batch size = 1
  !!   - Optimiser: Adam (lr = 0.01), matching PyKAN default
  !!   - Test set: 30 evenly-spaced points in [-1, 1]
  !!
  !! Both layers are trained on the same data with the same architecture
  !! (single layer, same number of basis functions) to compare:
  !! - training loss
  !! - runtime
  !! - number of parameters
  use athena
  use coreutils, only: real32

  implicit none

  integer, parameter :: n_epochs = 10
  integer, parameter :: n_train  = 1000
  integer, parameter :: test_size = 30
  integer, parameter :: n_basis = 10
  real(real32), parameter :: pi = 4.0_real32 * atan(1.0_real32)
  real(real32), parameter :: lr = 0.01_real32

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed


  !-----------------------------------------------------------------------------
  ! set random seed for reproducibility
  !-----------------------------------------------------------------------------
  seed_size = 8
  allocate(seed(seed_size))
  seed = 42
  call random_seed(put=seed)

  write(*,*) "============================================================"
  write(*,*) "  KAN Layer Benchmark: FastKAN (RBF) vs B-spline KAN"
  write(*,'(A,I0,A,I0,A)') "   Adam, lr=0.01, ", n_epochs, &
       " epochs x ", n_train, " samples = 10 000 gradient steps"
  write(*,*) "============================================================"
  write(*,*) ""

  call run_fastkan_benchmark()
  write(*,*) ""

  ! Reset seed for fair comparison
  call random_seed(put=seed)
  call run_kan_benchmark()
  write(*,*) ""

  ! Reset seed for polynomial test
  call random_seed(put=seed)
  call run_polynomial_benchmark()
  write(*,*) ""

  deallocate(seed)

  write(*,*) "============================================================"
  write(*,*) "  Benchmark complete"
  write(*,*) "============================================================"

contains


  !-----------------------------------------------------------------------------
  subroutine run_fastkan_benchmark()
    !! Train a FastKAN (RBF) layer on sin(pi*x) and report metrics
    implicit none

    type(network_type) :: network
    type(array_type), pointer :: loss
    real(real32), dimension(1,1) :: x, y
    real(real32), dimension(1, n_train) :: x_tr, y_tr
    real(real32), dimension(1,test_size) :: x_test, y_test, y_pred
    real(real32) :: test_mse
    real(real32) :: t_start, t_end
    integer :: i, n, epoch

    write(*,*) "------------------------------------------------------------"
    write(*,*) "  FastKAN (RBF) — sin(pi*x) approximation"
    write(*,*) "------------------------------------------------------------"

    call network%add(fastkan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=n_basis))
    call network%compile( &
         optimiser = adam_optimiser_type(learning_rate=lr), &
         loss_method="mse", metrics=["loss"], verbose=0)
    call network%set_batch_size(1)

    ! Create test data: x in [-1,1], y = (sin(pi*x) + 1) / 2
    do i = 1, test_size
       x_test(1,i) = -1.0_real32 + 2.0_real32 * real(i - 1, real32) / &
            real(test_size - 1, real32)
       y_test(1,i) = (sin(pi * x_test(1,i)) + 1.0_real32) / 2.0_real32
    end do

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(array_shape=[1,1])

    select type(layer => network%model(1)%layer)
    type is(fastkan_layer_type)
       write(*,'(A,I0)') "  Parameters: ", layer%get_num_params()
    end select

    ! Generate fixed training set (matches PyKAN dataset)
    call random_number(x_tr)
    x_tr = x_tr * 2.0_real32 - 1.0_real32
    y_tr = (sin(pi * x_tr) + 1.0_real32) / 2.0_real32

    write(*,'(A6,A12)') "Epoch", "Test MSE"

    call cpu_time(t_start)
    do epoch = 1, n_epochs
       do n = 1, n_train
          x(1,1) = x_tr(1,n)
          y(1,1) = y_tr(1,n)
          network%expected_array(1,1)%val = y

          call network%set_batch_size(1)
          call network%forward(x)
          loss => network%loss_eval(1, 1)
          call loss%grad_reverse()
          call network%update()
          call loss%nullify_graph()
          loss => null()
       end do

       y_pred(:,:) = network%predict(input=x_test(:,:))
       test_mse = sum((y_pred - y_test)**2) / size(y_pred)
       write(*,'(I6,F12.6)') epoch, test_mse
    end do
    call cpu_time(t_end)

    y_pred(:,:) = network%predict(input=x_test(:,:))
    test_mse = sum((y_pred - y_test)**2) / size(y_pred)
    write(*,'(A,F10.6)') "  Final test MSE:  ", test_mse
    write(*,'(A,F10.4,A)') "  Training time:   ", t_end - t_start, " s"

  end subroutine run_fastkan_benchmark


  !-----------------------------------------------------------------------------
  subroutine run_kan_benchmark()
    !! Train a KAN (B-spline) layer on sin(pi*x) and report metrics
    implicit none

    type(network_type) :: network
    type(array_type), pointer :: loss
    real(real32), dimension(1,1) :: x, y
    real(real32), dimension(1, n_train) :: x_tr, y_tr
    real(real32), dimension(1,test_size) :: x_test, y_test, y_pred
    real(real32) :: test_mse
    real(real32) :: t_start, t_end
    integer :: i, n, epoch

    write(*,*) "------------------------------------------------------------"
    write(*,*) "  KAN (B-spline) — sin(pi*x) approximation"
    write(*,*) "------------------------------------------------------------"

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=n_basis, spline_degree=3))
    call network%compile( &
         optimiser = adam_optimiser_type(learning_rate=lr), &
         loss_method="mse", metrics=["loss"], verbose=0)
    call network%set_batch_size(1)

    ! Create test data: x in [-1,1], y = (sin(pi*x) + 1) / 2
    do i = 1, test_size
       x_test(1,i) = -1.0_real32 + 2.0_real32 * real(i - 1, real32) / &
            real(test_size - 1, real32)
       y_test(1,i) = (sin(pi * x_test(1,i)) + 1.0_real32) / 2.0_real32
    end do

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(array_shape=[1,1])

    select type(layer => network%model(1)%layer)
    type is(kan_layer_type)
       write(*,'(A,I0)') "  Parameters: ", layer%get_num_params()
    end select

    ! Generate fixed training set (matches PyKAN dataset)
    call random_number(x_tr)
    x_tr = x_tr * 2.0_real32 - 1.0_real32
    y_tr = (sin(pi * x_tr) + 1.0_real32) / 2.0_real32

    write(*,'(A6,A12)') "Epoch", "Test MSE"

    call cpu_time(t_start)
    do epoch = 1, n_epochs
       do n = 1, n_train
          x(1,1) = x_tr(1,n)
          y(1,1) = y_tr(1,n)
          network%expected_array(1,1)%val = y

          call network%set_batch_size(1)
          call network%forward(x)
          loss => network%loss_eval(1, 1)
          call loss%grad_reverse()
          call network%update()
          call loss%nullify_graph()
          loss => null()
       end do

       y_pred(:,:) = network%predict(input=x_test(:,:))
       test_mse = sum((y_pred - y_test)**2) / size(y_pred)
       write(*,'(I6,F12.6)') epoch, test_mse
    end do
    call cpu_time(t_end)

    y_pred(:,:) = network%predict(input=x_test(:,:))
    test_mse = sum((y_pred - y_test)**2) / size(y_pred)
    write(*,'(A,F10.6)') "  Final test MSE:  ", test_mse
    write(*,'(A,F10.4,A)') "  Training time:   ", t_end - t_start, " s"

  end subroutine run_kan_benchmark


  !-----------------------------------------------------------------------------
  subroutine run_polynomial_benchmark()
    !! Train both FastKAN and KAN (B-spline) on a polynomial: y = x^3 - x
    !! to compare on a different function shape.
    !! Uses the same epoch-based Adam protocol as the sine benchmarks.
    implicit none

    type(network_type) :: net_rbf, net_bsp
    type(array_type), pointer :: loss
    real(real32), dimension(1,1) :: x, y
    real(real32), dimension(1, n_train) :: x_tr, y_tr
    real(real32), dimension(1,test_size) :: x_test, y_test
    real(real32), dimension(1,test_size) :: y_pred_rbf, y_pred_bsp
    real(real32) :: mse_rbf, mse_bsp
    real(real32) :: t_start, t_end, time_rbf, time_bsp
    integer :: i, n, epoch

    write(*,*) "------------------------------------------------------------"
    write(*,*) "  Polynomial benchmark: y = x^3 - x  on [-1, 1]"
    write(*,*) "------------------------------------------------------------"

    ! Setup FastKAN
    call net_rbf%add(fastkan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=n_basis))
    call net_rbf%compile( &
         optimiser = adam_optimiser_type(learning_rate=lr), &
         loss_method="mse", metrics=["loss"], verbose=0)
    call net_rbf%set_batch_size(1)
    allocate(net_rbf%expected_array(1,1))
    call net_rbf%expected_array(1,1)%allocate(array_shape=[1,1])

    ! Setup KAN (B-spline)
    call net_bsp%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=n_basis, spline_degree=3))
    call net_bsp%compile( &
         optimiser = adam_optimiser_type(learning_rate=lr), &
         loss_method="mse", metrics=["loss"], verbose=0)
    call net_bsp%set_batch_size(1)
    allocate(net_bsp%expected_array(1,1))
    call net_bsp%expected_array(1,1)%allocate(array_shape=[1,1])

    ! Create test data
    do i = 1, test_size
       x_test(1,i) = -1.0_real32 + 2.0_real32 * real(i - 1, real32) / &
            real(test_size - 1, real32)
       y_test(1,i) = x_test(1,i)**3 - x_test(1,i)
    end do

    ! Generate fixed training set (matches PyKAN dataset)
    call random_number(x_tr)
    x_tr = x_tr * 2.0_real32 - 1.0_real32
    y_tr(1,:) = x_tr(1,:)**3 - x_tr(1,:)

    ! Train FastKAN
    call cpu_time(t_start)
    do epoch = 1, n_epochs
       do n = 1, n_train
          x(1,1) = x_tr(1,n)
          y(1,1) = y_tr(1,n)
          net_rbf%expected_array(1,1)%val = y
          call net_rbf%set_batch_size(1)
          call net_rbf%forward(x)
          loss => net_rbf%loss_eval(1, 1)
          call loss%grad_reverse()
          call net_rbf%update()
          call loss%nullify_graph()
          loss => null()
       end do
    end do
    call cpu_time(t_end)
    time_rbf = t_end - t_start

    ! Train KAN (B-spline)
    call cpu_time(t_start)
    do epoch = 1, n_epochs
       do n = 1, n_train
          x(1,1) = x_tr(1,n)
          y(1,1) = y_tr(1,n)
          net_bsp%expected_array(1,1)%val = y
          call net_bsp%set_batch_size(1)
          call net_bsp%forward(x)
          loss => net_bsp%loss_eval(1, 1)
          call loss%grad_reverse()
          call net_bsp%update()
          call loss%nullify_graph()
          loss => null()
       end do
    end do
    call cpu_time(t_end)
    time_bsp = t_end - t_start

    ! Evaluate
    y_pred_rbf(:,:) = net_rbf%predict(input=x_test(:,:))
    y_pred_bsp(:,:) = net_bsp%predict(input=x_test(:,:))
    mse_rbf = sum((y_pred_rbf - y_test)**2) / size(y_test)
    mse_bsp = sum((y_pred_bsp - y_test)**2) / size(y_test)

    write(*,'(A)') ""
    write(*,'(A20,A15,A15)') "", "FastKAN (RBF)", "KAN (B-spline)"
    write(*,'(A20,F15.6,F15.6)') "Test MSE:", mse_rbf, mse_bsp
    write(*,'(A20,F12.4,A,F12.4,A)') "Time (s):", &
         time_rbf, "   ", time_bsp, ""

  end subroutine run_polynomial_benchmark


end program kan_sine
