program kan_sine
  !! KAN layer benchmark: FastKAN (RBF) vs KAN (B-spline) vs PyKAN-style KAN
  !!
  !! This example benchmarks the KAN (B-spline) layer against the FastKAN (RBF)
  !! layer and the PyKAN-style base+spline KAN on sine and polynomial function
  !! approximation tasks.
  !!
  !! Training protocol (matches pykan_comparison.py):
  !!   - n_train fixed samples drawn uniformly from [-1, 1] before training
  !!   - n_epochs passes over the training set, batch size = 1
  !!   - Optimiser: Adam (lr = 0.01), matching PyKAN default
  !!   - Test set: 30 evenly-spaced points in [-1, 1]
  !!
  !! All layers are trained on the same data with the same architecture
  !! (single layer, same number of basis functions) to compare:
  !! - training loss
  !! - runtime
  !! - number of parameters
  use athena
  use coreutils, only: real32

  implicit none

  integer, parameter :: n_epochs = 200
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

  write(*,*) "============================================================"
  write(*,*) "  KAN Layer Benchmark: B-spline vs PyKAN-style"
  write(*,'(A,I0,A,I0,A)') "   Adam, lr=0.01, ", n_epochs, &
       " epochs x ", n_train, " samples = 10 000 gradient steps"
  write(*,*) "============================================================"
  write(*,*) ""

!   ! Reset seed for fair comparison
!   call random_seed(put=seed)
!   call run_kan_benchmark()
!   write(*,*) ""

  ! Reset seed for PyKAN-style benchmark
  call random_seed(put=seed)
  call run_pykan_style_benchmark()
  write(*,*) ""

  deallocate(seed)

  write(*,*) "============================================================"
  write(*,*) "  Benchmark complete"
  write(*,*) "============================================================"

contains


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
    call network%set_batch_size(n_train)

    ! Create test data: x in [-1,1], y = (sin(pi*x) + 1) / 2
    do i = 1, test_size
       x_test(1,i) = -1.0_real32 + 2.0_real32 * real(i - 1, real32) / &
            real(test_size - 1, real32)
       y_test(1,i) = (sin(pi * x_test(1,i)) + 1.0_real32) / 2.0_real32
    end do

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(array_shape=[1,n_train])

    select type(layer => network%model(1)%layer)
    type is(kan_layer_type)
       write(*,'(A,I0)') "  Parameters: ", layer%get_num_params()
    end select

    ! Generate fixed training set (matches PyKAN dataset)
    call random_number(x_tr)
    x_tr = x_tr * 2.0_real32 - 1.0_real32
    y_tr = (sin(pi * x_tr) + 1.0_real32) / 2.0_real32

    write(*,'(A6,A12)') "Epoch", "Test MSE"

    call network%set_batch_size(n_train)
    call cpu_time(t_start)
    do epoch = 1, n_epochs
       call network%expected_array(1,1)%set(y_tr)

       call network%forward(x_tr)
       loss => network%loss_eval(1, n_train)
       call loss%grad_reverse()
       call network%update()
       call loss%nullify_graph()
       loss => null()

       if(mod(epoch, 20) .eq. 0) then
          y_pred(:,:) = network%predict(input=x_test(:,:))
          test_mse = sum((y_pred - y_test)**2) / size(y_pred)
          write(*,'(I6,F12.6)') epoch, test_mse
       end if
    end do
    call cpu_time(t_end)

    y_pred(:,:) = network%predict(input=x_test(:,:))
    test_mse = sum((y_pred - y_test)**2) / size(y_pred)
    write(*,'(A,F10.6)') "  Final test MSE:  ", test_mse
    write(*,'(A,F10.4,A)') "  Training time:   ", t_end - t_start, " s"

  end subroutine run_kan_benchmark


  !-----------------------------------------------------------------------------
  subroutine run_pykan_style_benchmark()
    !! Train a PyKAN-style KAN (base activation + spline) on sin(pi*x)
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
    write(*,*) "  PyKAN-style KAN (base+spline) — sin(pi*x) approximation"
    write(*,*) "------------------------------------------------------------"

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=n_basis, spline_degree=3, &
         use_base_activation=.true.))
    call network%compile( &
         optimiser = adam_optimiser_type(learning_rate=lr), &
         loss_method="mse", metrics=["loss"], verbose=0)
    call network%set_batch_size(n_train)

    ! Create test data
    do i = 1, test_size
       x_test(1,i) = -1.0_real32 + 2.0_real32 * real(i - 1, real32) / &
            real(test_size - 1, real32)
       y_test(1,i) = (sin(pi * x_test(1,i)) + 1.0_real32) / 2.0_real32
    end do

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(array_shape=[1,n_train])

    select type(layer => network%model(1)%layer)
    type is(kan_layer_type)
       write(*,'(A,I0)') "  Parameters: ", layer%get_num_params()
    end select

    ! Generate fixed training set
    call random_number(x_tr)
    x_tr = x_tr * 2.0_real32 - 1.0_real32
    y_tr = (sin(pi * x_tr) + 1.0_real32) / 2.0_real32

    write(*,'(A6,A12)') "Epoch", "Test MSE"

    call network%set_batch_size(n_train)
    call cpu_time(t_start)
    do epoch = 1, n_epochs
       call network%expected_array(1,1)%set(y_tr)

       call network%forward(x_tr)
       loss => network%loss_eval(1, n_train)
       call loss%grad_reverse()
       call network%update()
       call loss%nullify_graph()
       loss => null()

       if(mod(epoch, 20) .eq. 0) then
          y_pred(:,:) = network%predict(input=x_test(:,:))
          test_mse = sum((y_pred - y_test)**2) / size(y_pred)
          write(*,'(I6,F12.6)') epoch, test_mse
       end if
    end do
    call cpu_time(t_end)

    y_pred(:,:) = network%predict(input=x_test(:,:))
    test_mse = sum((y_pred - y_test)**2) / size(y_pred)
    write(*,'(A,F10.6)') "  Final test MSE:  ", test_mse
    write(*,'(A,F10.4,A)') "  Training time:   ", t_end - t_start, " s"

  end subroutine run_pykan_style_benchmark

end program kan_sine
