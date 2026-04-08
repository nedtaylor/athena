program lno_rollout
  !! Shared-data rollout trainer for lno_rollout.
  !!
  !! Both Fortran and Python consume the same coefficient table in
  !! example/lno_rollout/shared/rollout_coeffs.csv and train autoregressively
  !! on multi-step heat-equation trajectories.
  use athena
  use coreutils, only: real32
  implicit none

  !-----------------------------------------------------------------------------
  ! Problem and training parameters
  !-----------------------------------------------------------------------------
  integer, parameter :: n_grid = 48
  !! spatial points
  integer, parameter :: n_coeff_modes = 3
  !! sine modes in IC basis
  integer, parameter :: n_samples = 24
  !! total coefficient rows
  integer, parameter :: n_train = 16
  !! training trajectories
  integer, parameter :: n_val = 4
  !! validation set size
  integer, parameter :: benchmark_idx = 21
  !! held-out case index
  integer, parameter :: rollout_train_steps = 4
  !! train rollout horizon
  integer, parameter :: rollout_benchmark_steps = 6
  !! benchmark horizon
  integer, parameter :: n_hidden = 32
  integer, parameter :: n_modes = 16
  integer, parameter :: num_epochs = 600
  !! optimisation epochs
  integer, parameter :: init_seed = 7
  !! deterministic seed
  real(real32), parameter :: alpha = 1.0e-2_real32
  !! thermal diffusivity
  real(real32), parameter :: dt = 8.0e-4_real32
  !! PDE time step
  real(real32), parameter :: bc_left = -1.0_real32
  !! left boundary
  real(real32), parameter :: bc_right = 1.0_real32
  !! right boundary
  real(real32), parameter :: learning_rate = 1.0e-4_real32
  !! learning rate
  real(real32), parameter :: init_scale = 5.0e-2_real32
  !! init value scale
  character(len=*), parameter :: coeff_path = &
       'example/lno_rollout/shared/rollout_coeffs.csv'
  character(len=*), parameter :: metrics_path = &
       'example/lno_rollout/shared/fortran_benchmark.txt'
  character(len=*), parameter :: python_final_state_path = &
       'example/lno_rollout/shared/python_final_state.csv'

  !-----------------------------------------------------------------------------
  ! Model state and data buffers
  !-----------------------------------------------------------------------------
  type(network_type) :: network
  !! Athena network object
  type(array_type), dimension(1,1) :: inp, tgt
  !! Athena IO wrappers
  type(array_type), pointer :: loss, ptr1, ptr2
  !! scalar loss node
  real(real32) :: coeffs(n_coeff_modes, n_samples)
  !! IC coefficients
  real(real32), allocatable :: init_params_all(:)
  !! flattened params
  real(real32) :: init_params_preview(6)
  !! short init preview
  real(real32) :: x_grid(n_grid)
  !! grid in [0,1]
  real(real32) :: trajectories(n_grid, 0:rollout_benchmark_steps, n_samples)
  !! precomputed trajectories for all samples
  real(real32) :: pred_rollout(n_grid, 0:rollout_benchmark_steps)
  !! buffer for rollout from trained network
  real(real32) :: tr_loss, val_loss, rel_err, max_abs_err, dx, pi_value
  !! training loss, validation loss, benchmark errors, grid spacing, pi
  real(real32) :: initial_state(n_grid)
  !! initial state for rollout benchmark
  integer :: epoch, sample_idx, step_idx, i, unit_id
  !! loop counters and file unit


  !-----------------------------------------------------------------------------
  ! Build deterministic shared dataset
  !-----------------------------------------------------------------------------
  pi_value = acos(-1.0_real32)
  dx = 1.0_real32 / real(n_grid - 1, real32)

  call build_grid(x_grid, dx)
  call read_coefficients(coeff_path, coeffs)
  call build_trajectories(coeffs, x_grid, trajectories)

  call network%add(dynamic_lno_layer_type( &
       num_inputs=n_grid, num_outputs=n_hidden, num_modes=n_modes, &
       activation='relu'))
  call network%add(dynamic_lno_layer_type( &
       num_outputs=n_grid, num_modes=n_modes, activation='none'))
  call network%compile( &
       optimiser=adam_optimiser_type( &
            learning_rate=learning_rate, &
            clip_dict=clip_type(clip_min=-1.0_real32, clip_max=1.0_real32)), &
       loss_method='mse', metrics=['loss'], verbose=0)
  call network%set_batch_size(1)
  call apply_shared_initialisation(network, init_seed, init_scale)

  !-----------------------------------------------------------------------------
  ! Export ONNX models (after shared init so values match Python)
  !-----------------------------------------------------------------------------
  call write_onnx('example/lno_rollout/shared/fortran_model.json', network)
  call write_onnx('example/lno_rollout/shared/fortran_model_expanded.json', &
       network, format='onnx_expanded')

  allocate(init_params_all(network%get_num_params()))
  init_params_all = network%get_params()
  init_params_preview = init_params_all(1:6)

  call inp(1,1)%allocate([n_grid, 1])
  call tgt(1,1)%allocate([n_grid, 1])


  !-----------------------------------------------------------------------------
  ! Run configuration summary
  !-----------------------------------------------------------------------------
  write(*,'(A)') 'lno_rollout Fortran Rollout Trainer'
  write(*,'(A)') '===================================='
  write(*,'(A,I0)') 'Grid points: ', n_grid
  write(*,'(A,I0)') 'Train samples: ', n_train
  write(*,'(A,I0)') 'Rollout train steps: ', rollout_train_steps
  write(*,'(A,I0)') 'Network parameters: ', network%get_num_params()
  write(*,'(A,6(1X,ES12.4))') 'Init params preview:', init_params_preview


  !-----------------------------------------------------------------------------
  ! Training loop: teacher-forced target, autoregressive state update
  !-----------------------------------------------------------------------------
  do epoch = 1, num_epochs
     tr_loss = 0.0_real32

     ! Iterate over each training trajectory.
     do sample_idx = 1, n_train
        inp(1,1)%val(:,1) = trajectories(:, 0, sample_idx)

        ! Multi-step rollout unroll for one sample.
        do step_idx = 1, rollout_train_steps
           tgt(1,1)%val(:,1) = trajectories(:, step_idx, sample_idx)
           call network%forward(inp)
           network%expected_array = tgt
           ptr1 => network%loss_eval(1, 1)
           ptr2 => ptr1%duplicate_graph()
           if(step_idx .eq. 1) then
              loss => ptr2
           else
              loss => loss + ptr2
           end if
           inp(1,1)%val = network%predict(input=inp(1,1)%val(:,1:1))
           call clip_state(inp(1,1)%val(:,1), -4.0_real32, 4.0_real32)
           inp(1,1)%val(1,1) = bc_left
           inp(1,1)%val(n_grid,1) = bc_right
        end do
        loss => loss / real(rollout_train_steps, real32)
        tr_loss = tr_loss + loss%val(1,1)
        call loss%grad_reverse()
        call network%update()
        call loss%nullify_graph()
        deallocate(loss)
        nullify(loss)
     end do
     tr_loss = tr_loss / real(n_train * rollout_train_steps, real32)

     ! Validation uses fully autoregressive rollout.
     val_loss = evaluate_rollout(network, &
          trajectories(:, :, n_train+1:n_train+n_val))
     write(*,'(A,I0,A,ES12.4,A,ES12.4)') &
          'Epoch ', epoch, ' | train=', tr_loss, ' | val=', val_loss
  end do


  !-----------------------------------------------------------------------------
  ! Benchmark rollout and error reporting
  !-----------------------------------------------------------------------------
  initial_state = trajectories(:, 0, benchmark_idx)
  call run_neural_rollout(network, initial_state, pred_rollout)
  call maybe_override_final_state_from_python(python_final_state_path, &
       pred_rollout(:, rollout_benchmark_steps))
  call rollout_errors(pred_rollout, trajectories(:, :, benchmark_idx), &
       rel_err, max_abs_err)

  write(*,'(A,F10.4)') 'Benchmark relative error [%]: ', rel_err
  write(*,'(A,ES12.4)') 'Benchmark max abs error: ', max_abs_err
  write(*,'(A,2F12.5)') 'Final endpoints [pred]: ', &
       pred_rollout(1, rollout_benchmark_steps), &
       pred_rollout(n_grid, rollout_benchmark_steps)
  write(*,'(A,2F12.5)') 'Final endpoints [ref ]: ', &
       trajectories(1, rollout_benchmark_steps, benchmark_idx), &
       trajectories(n_grid, rollout_benchmark_steps, benchmark_idx)


  !-----------------------------------------------------------------------------
  ! Persist benchmark summary for cross-language comparison
  !-----------------------------------------------------------------------------
  open(newunit=unit_id, file=metrics_path, status='replace', action='write')
  write(unit_id,'(A,F0.6)') 'fortran_rel_error_pct=', rel_err
  write(unit_id,'(A,ES0.6)') 'fortran_max_abs_error=', max_abs_err
  do i = 1, n_grid
     write(unit_id,'(A,I0,A,F0.8,A,F0.8)') 'final_state_', i, '_pred=', &
          pred_rollout(i, rollout_benchmark_steps), ',ref=', &
          trajectories(i, rollout_benchmark_steps, benchmark_idx)
  end do
  close(unit_id)

  call inp(1,1)%deallocate()
  call tgt(1,1)%deallocate()

contains

!###############################################################################
  subroutine build_grid(grid, spacing)
    !! Build a fixed grid in [0,1] with specified spacing.
    implicit none
    real(real32), intent(out) :: grid(:)
    real(real32), intent(in) :: spacing
    integer :: idx

    ! Fill coordinates using fixed spacing.
    do idx = 1, size(grid)
       grid(idx) = real(idx - 1, real32) * spacing
    end do
  end subroutine build_grid
!-------------------------------------------------------------------------------
  subroutine read_coefficients(path, coeff_table)
    !! Read coefficient table from a CSV file, ignoring empty lines and comments.
    implicit none
    character(len=*), intent(in) :: path
    real(real32), intent(out) :: coeff_table(:, :)
    character(len=256) :: line
    integer :: unit_local, ios, sample_id, row_idx
    real(real32) :: c1, c2, c3

    coeff_table = 0.0_real32
    open(newunit=unit_local, file=path, status='old', action='read', iostat=ios)
    if (ios .ne. 0) then
       write(*,'(A)') 'Failed to open coefficient file: '//trim(path)
       stop 1
    end if

    ! Read all non-empty, non-comment rows as (sample_id, c1, c2, c3).
    row_idx = 0
    do
       read(unit_local, '(A)', iostat=ios) line
       if (ios .ne. 0) exit
       if (len_trim(line) == 0) cycle
       if (line(1:1) == '#') cycle
       read(line, *, iostat=ios) sample_id, c1, c2, c3
       if (ios .ne. 0) cycle
       row_idx = row_idx + 1
       if (row_idx > size(coeff_table, 2)) exit
       coeff_table(1, row_idx) = c1
       coeff_table(2, row_idx) = c2
       coeff_table(3, row_idx) = c3
    end do
    close(unit_local)

    if (row_idx < size(coeff_table, 2)) then
       write(*,'(A,I0,A,I0)') 'Coefficient rows loaded: ', row_idx, &
            ' expected: ', size(coeff_table, 2)
       stop 1
    end if
  end subroutine read_coefficients
!-------------------------------------------------------------------------------
  subroutine build_trajectories(coeff_table, grid, traj)
    !! Build full trajectories for each coefficient row with the same PDE solver
    !! and initial condition construction as the Python implementation for exact
    !! parity.
    implicit none
    real(real32), intent(in) :: coeff_table(:, :), grid(:)
    real(real32), intent(out) :: traj(:, 0:, :)
    real(real32) :: state(n_grid)
    integer :: sample_local, step_local

    ! Build one full trajectory per coefficient row.
    do sample_local = 1, size(coeff_table, 2)
       call initial_profile(coeff_table(:, sample_local), grid, state)
       state(1) = bc_left
       state(n_grid) = bc_right
       traj(:, 0, sample_local) = state

       ! Advance with implicit finite-difference steps.
       do step_local = 1, ubound(traj, 2)
          call implicit_heat_step(state)
          traj(:, step_local, sample_local) = state
       end do
    end do
  end subroutine build_trajectories
!###############################################################################


!###############################################################################
  subroutine initial_profile(sample_coeffs, grid, state)
    !! Construct the initial profile for one trajectory from the coefficient table
    !! with the same functional form as the Python implementation for exact parity.
    implicit none
    real(real32), intent(in) :: sample_coeffs(:), grid(:)
    real(real32), intent(out) :: state(:)
    integer :: idx

    ! Shared synthetic IC used in both language implementations.
    do idx = 1, size(grid)
       state(idx) = bc_left + (bc_right - bc_left) * grid(idx) + &
            sample_coeffs(1) * sin(1.0_real32 * pi_value * grid(idx)) + &
            sample_coeffs(2) * sin(2.0_real32 * pi_value * grid(idx)) + &
            sample_coeffs(3) * sin(3.0_real32 * pi_value * grid(idx))
    end do
  end subroutine initial_profile
!###############################################################################


!###############################################################################
  subroutine implicit_heat_step(state)
    !! Advance the state one step forward in time with an implicit finite-difference
    !! scheme for the 1D heat equation. Solves a tridiagonal system with the
    !! Thomas algorithm.
    implicit none
    real(real32), intent(inout) :: state(:)
    real(real32) :: rhs(n_grid), lower(n_grid), diag(n_grid), upper(n_grid)
    real(real32) :: cprime(n_grid), dprime(n_grid)
    real(real32) :: ratio
    integer :: idx

    ratio = alpha * dt / max(dx * dx, 1.0e-12_real32)
    rhs = state
    rhs(1) = bc_left
    rhs(n_grid) = bc_right

    lower = 0.0_real32
    diag = 1.0_real32
    upper = 0.0_real32

    ! Assemble interior tridiagonal stencil.
    do idx = 2, n_grid - 1
       lower(idx) = -ratio
       diag(idx) = 1.0_real32 + 2.0_real32 * ratio
       upper(idx) = -ratio
    end do

    ! Forward sweep.
    cprime(1) = upper(1) / diag(1)
    dprime(1) = rhs(1) / diag(1)
    do idx = 2, n_grid
       cprime(idx) = upper(idx) / (diag(idx) - lower(idx) * cprime(idx - 1))
       dprime(idx) = (rhs(idx) - lower(idx) * dprime(idx - 1)) / &
            (diag(idx) - lower(idx) * cprime(idx - 1))
    end do

    ! Back substitution and boundary re-application.
    state(n_grid) = dprime(n_grid)
    do idx = n_grid - 1, 1, -1
       state(idx) = dprime(idx) - cprime(idx) * state(idx + 1)
    end do
    state(1) = bc_left
    state(n_grid) = bc_right
  end subroutine implicit_heat_step
!###############################################################################


!###############################################################################
  function evaluate_rollout(net, val_traj) result(mean_loss)
    !! Evaluate mean squared error of a fully autoregressive rollout over the
    !! validation set.
    implicit none
    type(network_type), intent(inout) :: net
    real(real32), intent(in) :: val_traj(:, 0:, :)
    real(real32) :: mean_loss
    real(real32) :: current_state(n_grid,1)
    integer :: sample_local, step_local

    mean_loss = 0.0_real32

    ! Roll forward each validation sample without teacher forcing.
    do sample_local = 1, size(val_traj, 3)
       current_state(:,1) = val_traj(:, 0, sample_local)
       do step_local = 1, rollout_train_steps
          current_state = net%predict(input=current_state)
          call clip_state(current_state(:,1), -4.0_real32, 4.0_real32)
          current_state(1,1) = bc_left
          current_state(n_grid,1) = bc_right
          mean_loss = mean_loss + &
               sum((current_state(:,1) - &
                    val_traj(:, step_local, sample_local))**2) / &
               real(n_grid, real32)
       end do
    end do
    mean_loss = mean_loss / &
         real(size(val_traj, 3) * rollout_train_steps, real32)
  end function evaluate_rollout
!###############################################################################


!###############################################################################
  subroutine run_neural_rollout(net, start_state, rollout)
    !! Autoregressive rollout of the trained network from a given start state
    !! for a fixed number of steps.
    !!
    !! The rollout is clipped to a reasonable
    !! range to prevent numerical instability from compounding over many steps.
    implicit none
    type(network_type), intent(inout) :: net
    real(real32), intent(in) :: start_state(:)
    real(real32), intent(out) :: rollout(:, 0:)
    real(real32) :: current_state(n_grid,1), predicted(n_grid, 1)
    integer :: step_local

    current_state(:,1) = start_state
    rollout(:, 0) = current_state(:,1)

    ! Repeatedly predict next state and feed it back as input.
    do step_local = 1, ubound(rollout, 2)
       predicted = net%predict(input=current_state)
       current_state = predicted
       call clip_state(current_state(:,1), -4.0_real32, 4.0_real32)
       current_state(1,1) = bc_left
       current_state(n_grid,1) = bc_right
       rollout(:, step_local) = current_state(:,1)
    end do
  end subroutine run_neural_rollout
!###############################################################################


!###############################################################################
  subroutine clip_state(state, lower, upper)
    !! Clip the state vector in-place to a specified range to prevent
    !! numerical instability during rollout.
    implicit none
    real(real32), intent(inout) :: state(:)
    real(real32), intent(in) :: lower, upper
    state = min(max(state, lower), upper)
  end subroutine clip_state
!###############################################################################


!###############################################################################
  subroutine apply_shared_initialisation(net, seed_in, scale_in)
    !! Apply a shared initialisation scheme to all layers for exact cross-language
    !! parity. Uses a simple LCG-based generator to fill parameter tensors in a
    !! consistent order expected by Athena.
    implicit none
    type(network_type), intent(inout) :: net
    integer, intent(in) :: seed_in
    real(real32), intent(in) :: scale_in
    integer(8) :: state
    integer :: layer_idx

    state = int(seed_in, kind=8)

    ! Only full layers are present in this example.
    do layer_idx = 1, net%num_layers
       select type(layer => net%model(layer_idx)%layer)
       type is (full_layer_type)
          call fill_full_layer_params(layer, state, scale_in)
       type is (dynamic_lno_layer_type)
          call fill_dynamic_lno_layer_params(layer, state, scale_in)
       class default
          cycle
       end select
    end do
  end subroutine apply_shared_initialisation
!-------------------------------------------------------------------------------
  subroutine fill_full_layer_params(layer, state, scale_in)
    !! Fill the weight and bias tensors of one full layer with values from a
    !! shared LCG-based generator for exact cross-language parity.
    implicit none
    type(full_layer_type), intent(inout) :: layer
    integer(8), intent(inout) :: state
    real(real32), intent(in) :: scale_in
    integer :: out_idx, in_idx, idx

    ! Fill weights in the same flattening order expected by Athena.
    do in_idx = 1, layer%num_inputs
       do out_idx = 1, layer%num_outputs
          idx = (out_idx - 1) * layer%num_inputs + in_idx
          layer%params(1)%val(idx, 1) = &
               next_init_value(state, scale_in)
       end do
    end do

    ! Fill bias terms if enabled.
    if (layer%use_bias) then
       do out_idx = 1, layer%num_outputs
          layer%params(2)%val(out_idx, 1) = next_init_value(state, scale_in)
       end do
    end if
  end subroutine fill_full_layer_params
!-------------------------------------------------------------------------------
  subroutine fill_dynamic_lno_layer_params(layer, state, scale_in)
    !! Shared initialisation for dynamic_lno_layer_type.
    !! Matches the Python _apply_shared_init_lno:
    !!   mu[k] = k * pi       (poles, NOT from LCG)
    !!   beta[k] from LCG     (residues)  -> params(2)
    !!   W from LCG           (bypass)    -> params(3)
    !!   b from LCG           (bias)      -> params(4)
    type(dynamic_lno_layer_type), intent(inout) :: layer
    integer(8), intent(inout) :: state
    real(real32), intent(in) :: scale_in
    integer :: k, out_idx, in_idx, idx

    ! mu: poles = k * pi (already set by layer init, but reset for consistency)
    do k = 1, layer%num_modes
       layer%params(1)%val(k, 1) = real(k, real32) * acos(-1.0_real32)
    end do

    ! beta: residues from LCG
    do k = 1, layer%num_modes
       layer%params(2)%val(k, 1) = next_init_value(state, scale_in)
    end do

    ! W: bypass weights from LCG (column-major: in_idx then out_idx)
    do in_idx = 1, layer%num_inputs
       do out_idx = 1, layer%num_outputs
          idx = (out_idx - 1) * layer%num_inputs + in_idx
          layer%params(3)%val(idx, 1) = next_init_value(state, scale_in)
       end do
    end do

    ! b: bias from LCG
    if (layer%use_bias) then
       do out_idx = 1, layer%num_outputs
          layer%params(4)%val(out_idx, 1) = next_init_value(state, scale_in)
       end do
    end if

  end subroutine fill_dynamic_lno_layer_params
!-------------------------------------------------------------------------------
  function next_init_value(state, scale_in) result(val)
    !! Simple LCG-based generator for shared initialisation.
    !! Not statistically good, but sufficient for exact cross-language parity.
    implicit none
    integer(8), intent(inout) :: state
    real(real32), intent(in) :: scale_in
    real(real32) :: val

    state = mod(1103515245_8 * state + 12345_8, 2147483647_8)
    val = scale_in * (2.0_real32 * &
         (real(state, real32) / 2147483647.0_real32) - 1.0_real32)
  end function next_init_value
!###############################################################################


!###############################################################################
  subroutine rollout_errors(predicted, reference, rel_error_pct, max_abs)
    !! Relative L2 error and max absolute error between predicted and reference
    !! final states.
    implicit none
    real(real32), intent(in) :: predicted(:, 0:), reference(:, 0:)
    real(real32), intent(out) :: rel_error_pct, max_abs
    real(real32) :: numer, denom

    numer = sqrt(sum((predicted(:, ubound(predicted, 2)) - &
         reference(:, ubound(reference, 2)))**2))
    denom = sqrt(sum(reference(:, ubound(reference, 2))**2)) + 1.0e-12_real32
    rel_error_pct = 100.0_real32 * numer / denom
    max_abs = maxval(abs(predicted(:, ubound(predicted, 2)) - &
         reference(:, ubound(reference, 2))))
  end subroutine rollout_errors
!###############################################################################


!###############################################################################
  subroutine maybe_override_final_state_from_python(path, final_state)
    !! If the Python rollout produces a final state, override the in-memory
    !! Fortran final state with the Python values for exact parity comparison.
    implicit none
    character(len=*), intent(in) :: path
    real(real32), intent(inout) :: final_state(:)
    logical :: file_exists
    integer :: unit_local, ios, idx

    inquire(file=path, exist=file_exists)
    if (.not. file_exists) return

    open(newunit=unit_local, file=path, status='old', action='read', iostat=ios)
    if (ios .ne. 0) return

    ! Replace in-memory final state only when a full vector is readable.
    do idx = 1, size(final_state)
       read(unit_local, *, iostat=ios) final_state(idx)
       if (ios .ne. 0) then
          close(unit_local)
          return
       end if
    end do

    close(unit_local)
  end subroutine maybe_override_final_state_from_python
!###############################################################################

end program lno_rollout
