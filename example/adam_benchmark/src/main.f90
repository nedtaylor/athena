program adam_benchmark
  !! Benchmark the ATHENA Adam optimiser on two analytic test problems.
  !!
  !! Writes per-step data to CSV files for comparison with PyTorch:
  !!   fortran_adam_scalar.csv  — scalar  f(x)      = (x-3)^2
  !!   fortran_adam_multi.csv   — 2-param g(x1,x2)  = (x1-3)^2 + (x2+1)^2
  !!
  !! Columns: step, param_1 [, param_2], grad_1 [, grad_2], m_1 [, m_2],
  !!          v_1 [, v_2], loss
  !!
  !! The iter counter is pre-incremented (as in athena_network_sub.f90) before
  !! each minimise call so that bias corrections align with t=1 on the first step.
  use athena__optimiser, only: adam_optimiser_type
  use athena__learning_rate_decay, only: base_lr_decay_type
  use coreutils, only: real32
  implicit none

  ! ── Hyperparameters (must match the Python notebook) ──────────────────────
  real(real32), parameter :: LR    = 0.01_real32
  real(real32), parameter :: BETA1 = 0.9_real32
  real(real32), parameter :: BETA2 = 0.999_real32
  real(real32), parameter :: EPS   = 1.E-8_real32
  integer,      parameter :: N_STEPS = 20

  ! ── Scalar problem ─────────────────────────────────────────────────────────
  ! f(x) = (x-3)^2,  df/dx = 2(x-3)
  integer, parameter :: N_SCALAR = 1
  real(real32), dimension(N_SCALAR) :: x_scalar, g_scalar
  type(adam_optimiser_type)         :: opt_scalar

  ! ── Multi-parameter problem ────────────────────────────────────────────────
  ! g(x1,x2) = (x1-3)^2 + (x2+1)^2,  dg/dx1 = 2(x1-3),  dg/dx2 = 2(x2+1)
  integer, parameter :: N_MULTI = 2
  real(real32), dimension(N_MULTI)  :: x_multi, g_multi
  type(adam_optimiser_type)         :: opt_multi

  integer :: step, unit_scalar, unit_multi
  real(real32) :: loss_scalar, loss_multi

  ! ── Initialise scalar optimiser and starting point ─────────────────────────
  opt_scalar = adam_optimiser_type( &
       learning_rate = LR,          &
       beta1         = BETA1,       &
       beta2         = BETA2,       &
       epsilon       = EPS,         &
       num_params    = N_SCALAR,    &
       lr_decay      = base_lr_decay_type())
  x_scalar = 0.0_real32   ! identical start to Python notebook: X0_SCALAR = 0.0

  ! ── Initialise multi-param optimiser and starting point ───────────────────
  opt_multi = adam_optimiser_type( &
       learning_rate = LR,         &
       beta1         = BETA1,      &
       beta2         = BETA2,      &
       epsilon       = EPS,        &
       num_params    = N_MULTI,    &
       lr_decay      = base_lr_decay_type())
  x_multi = [0.0_real32, 2.0_real32]   ! X0_MULTI = [0, 2]

  ! ── Open CSV output files ──────────────────────────────────────────────────
  open(newunit=unit_scalar, file="fortran_adam_scalar.csv", &
       status="replace", action="write")
  write(unit_scalar, '(A)') "step,param_1,grad_1,m_1,v_1,loss"

  open(newunit=unit_multi, file="fortran_adam_multi.csv", &
       status="replace", action="write")
  write(unit_multi, '(A)') "step,param_1,param_2,grad_1,grad_2,m_1,m_2,v_1,v_2,loss"

  ! ── Training loops ─────────────────────────────────────────────────────────
  do step = 1, N_STEPS

     ! ── Scalar: compute loss and gradient BEFORE update ─────────────────────
     loss_scalar = (x_scalar(1) - 3.0_real32)**2
     g_scalar(1) = 2.0_real32 * (x_scalar(1) - 3.0_real32)

     ! Pre-increment iter (mirrors athena_network_sub.f90 behaviour)
     opt_scalar%iter = opt_scalar%iter + 1
     call opt_scalar%minimise(x_scalar, g_scalar)

     write(unit_scalar, '(I4,5(",",ES20.10E3))') &
          step, x_scalar(1), g_scalar(1), &
          opt_scalar%m(1), opt_scalar%v(1), loss_scalar

     ! ── Multi-param: compute loss and gradient BEFORE update ────────────────
     loss_multi = (x_multi(1) - 3.0_real32)**2 + (x_multi(2) + 1.0_real32)**2
     g_multi(1) = 2.0_real32 * (x_multi(1) - 3.0_real32)
     g_multi(2) = 2.0_real32 * (x_multi(2) + 1.0_real32)

     opt_multi%iter = opt_multi%iter + 1
     call opt_multi%minimise(x_multi, g_multi)

     write(unit_multi, '(I4,9(",",ES20.10E3))') &
          step, x_multi(1), x_multi(2), g_multi(1), g_multi(2), &
          opt_multi%m(1), opt_multi%m(2), &
          opt_multi%v(1), opt_multi%v(2), loss_multi

  end do

  close(unit_scalar)
  close(unit_multi)

  write(*,'(A,I0,A)') "fortran_adam: wrote ", N_STEPS, &
       " steps to fortran_adam_scalar.csv and fortran_adam_multi.csv"

end program adam_benchmark
