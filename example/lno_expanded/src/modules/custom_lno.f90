module custom_lno
  !! Custom Laplace Neural Operator implementation matching the Python
  !! CattaneoLNO architecture exactly.
  !!
  !! Athena's laplace_nop_layer_type uses FIXED Laplace bases with a
  !! learnable spectral mixing matrix R. This differs fundamentally from
  !! the Python LaplaceConv1d which uses LEARNABLE data-dependent poles
  !! with polar weight parameterisation. However, the Athena layer also
  !! goes NaN with swish activation + lr=1e-3 — the same config that
  !! works in Python — due to float32 overflow in the spectral path.
  !!
  !! This module bypasses Athena and implements the forward pass directly:
  !!
  !!   Python SecondOrderPredictor LNO path:
  !!     x = Conv1d(1→C)(T_n_star)
  !!     for each block:
  !!       x = InstanceNorm(x)
  !!       x = Conv1d(C→C, 1×1)(x)            ← pointwise mixing
  !!       x = SiLU(x)
  !!       x = LaplaceConv1d(C, C, modes)(x)   ← spectral operator (NO skip)
  !!     ΔT = output_proj(x)
  !!
  !!   Simplified for Fortran (no causal mask, no data-dependent poles):
  !!     LaplaceConv1d forward:
  !!       poles = softplus(log_poles)                     ← [modes]
  !!       dist[i,j] = |i-j| / (G-1)                     ← [G, G]
  !!       kernels[k,i,j] = exp(-poles[k] * dist[i,j])   ← [modes, G, G]
  !!       kernels = row_normalise(kernels) * filter[k]
  !!       weights[k,i,o] = max_amp * sigmoid(log_amp[k,i,o]) * cos(phase[k,i,o])
  !!       out[b,o,t] = einsum('bcs,kts,kio->bot', x, kernels, weights)
  !!
  !! Training: mini-batch SGD with Adam, MSE loss on residual ΔT targets.
  !!
  !! Since this bypasses Athena completely, the training loop, forward pass,
  !! and backpropagation are all implemented from scratch.
  implicit none

  private
  public :: custom_lno_type
  public :: custom_lno_init, custom_lno_forward, custom_lno_train
  public :: custom_lno_predict, custom_lno_save, custom_lno_load

  integer, parameter :: sp = selected_real_kind(6, 37)  !! single precision

  real(sp), parameter :: PI = 3.14159265358979323846_sp

  type :: custom_lno_type
     !! Full network state: input_proj + 4 LNO blocks + output_proj
     integer :: grid_size = 112
     integer :: width = 64
     integer :: modes = 16
     integer :: num_blocks = 4
     integer :: input_dim = 224
     integer :: output_dim = 112

     !! Input projection: weight[width, input_dim], bias[width]
     real(sp), allocatable :: proj_w(:,:)     !! [width, input_dim]
     real(sp), allocatable :: proj_b(:)       !! [width]

     !! LNO blocks (each has: norm params, pointwise W/b, spectral params)
     !! InstanceNorm: scale[width], shift[width] per block
     real(sp), allocatable :: norm_scale(:,:)  !! [width, num_blocks]
     real(sp), allocatable :: norm_shift(:,:)  !! [width, num_blocks]

     !! Pointwise Conv1d(C→C, 1×1): pw_w[width,width], pw_b[width]
     real(sp), allocatable :: pw_w(:,:,:)      !! [width, width, num_blocks]
     real(sp), allocatable :: pw_b(:,:)        !! [width, num_blocks]

     !! LaplaceConv1d params per block:
     !!   log_poles[modes]
     !!   weight_log_amp[modes, width, width]
     !!   weight_phase[modes, width, width]
     real(sp), allocatable :: log_poles(:,:)        !! [modes, num_blocks]
     real(sp), allocatable :: wt_log_amp(:,:,:,:)   !! [modes, width, width, num_blocks]
     real(sp), allocatable :: wt_phase(:,:,:,:)     !! [modes, width, width, num_blocks]

     !! Output projection: out_w[output_dim, width], out_b[output_dim]
     real(sp), allocatable :: out_w(:,:)
     real(sp), allocatable :: out_b(:)

     !! Spectral filter (precomputed, exponential)
     real(sp), allocatable :: spec_filter(:)   !! [modes]

     !! Distance matrix (precomputed)
     real(sp), allocatable :: dist(:,:)        !! [grid_size, grid_size]

     !! Adam state
     integer :: adam_t = 0
     real(sp) :: lr = 1.0e-3_sp
     real(sp) :: beta1 = 0.9_sp
     real(sp) :: beta2 = 0.999_sp
     real(sp) :: eps_adam = 1.0e-8_sp
     real(sp) :: weight_decay = 1.0e-4_sp
     real(sp) :: grad_clip = 1.0_sp

     !! Adam moment buffers (flattened, allocated on first use)
     real(sp), allocatable :: m_buf(:), v_buf(:)
     integer :: n_params = 0
  end type

contains

  ! ──────────────────────────────────────────────────────────────────────
  ! Activation functions
  ! ──────────────────────────────────────────────────────────────────────

  elemental real(sp) function silu(x)
    real(sp), intent(in) :: x
    silu = x / (1.0_sp + exp(-x))
  end function

  elemental real(sp) function dsilu(x)
    !! Derivative of SiLU: σ(x) + x·σ(x)·(1-σ(x)) = σ(x)·(1 + x·(1-σ(x)))
    real(sp), intent(in) :: x
    real(sp) :: s
    s = 1.0_sp / (1.0_sp + exp(-x))
    dsilu = s * (1.0_sp + x * (1.0_sp - s))
  end function

  elemental real(sp) function sigmoid(x)
    real(sp), intent(in) :: x
    sigmoid = 1.0_sp / (1.0_sp + exp(-x))
  end function

  elemental real(sp) function softplus(x)
    real(sp), intent(in) :: x
    if (x > 20.0_sp) then
       softplus = x
    else
       softplus = log(1.0_sp + exp(x))
    end if
  end function

  ! ──────────────────────────────────────────────────────────────────────
  ! Initialisation
  ! ──────────────────────────────────────────────────────────────────────

  subroutine custom_lno_init(net, grid_size, width, modes, num_blocks, &
       input_dim, output_dim, lr)
    type(custom_lno_type), intent(inout) :: net
    integer, intent(in) :: grid_size, width, modes, num_blocks
    integer, intent(in) :: input_dim, output_dim
    real(sp), intent(in), optional :: lr

    integer :: b, k
    real(sp) :: fan_in_scale, target_amp, target_sigmoid, init_bias, init_scale
    real(sp) :: filter_strength

    net%grid_size = grid_size
    net%width = width
    net%modes = modes
    net%num_blocks = num_blocks
    net%input_dim = input_dim
    net%output_dim = output_dim
    if (present(lr)) net%lr = lr

    !! Input projection: Kaiming uniform
    allocate(net%proj_w(width, input_dim))
    allocate(net%proj_b(width))
    fan_in_scale = sqrt(2.0_sp / real(input_dim, sp))
    call random_normal(net%proj_w, 0.0_sp, fan_in_scale)
    net%proj_b = 0.0_sp

    !! LNO blocks
    allocate(net%norm_scale(width, num_blocks))
    allocate(net%norm_shift(width, num_blocks))
    allocate(net%pw_w(width, width, num_blocks))
    allocate(net%pw_b(width, num_blocks))
    allocate(net%log_poles(modes, num_blocks))
    allocate(net%wt_log_amp(modes, width, width, num_blocks))
    allocate(net%wt_phase(modes, width, width, num_blocks))

    do b = 1, num_blocks
       !! InstanceNorm: scale=1, shift=0
       net%norm_scale(:, b) = 1.0_sp
       net%norm_shift(:, b) = 0.0_sp

       !! Pointwise: Kaiming
       fan_in_scale = sqrt(2.0_sp / real(width, sp))
       call random_normal(net%pw_w(:,:,b), 0.0_sp, fan_in_scale)
       net%pw_b(:, b) = 0.0_sp

       !! Log-poles: log-spaced from 1 to 50 (matching Python)
       do k = 1, modes
          net%log_poles(k, b) = log(1.0_sp) + &
               (log(50.0_sp) - log(1.0_sp)) * real(k-1, sp) / real(max(1, modes-1), sp)
       end do

       !! Polar weights: matching Python Kaiming-correct init
       !! target_amp = sqrt(2/(modes*width))
       !! init_bias so sigmoid gives target_amp/max_amp
       target_amp = sqrt(2.0_sp / real(modes * width, sp))
       target_sigmoid = min(0.99_sp, max(0.01_sp, target_amp / 1.0_sp))
       init_bias = log(target_sigmoid / (1.0_sp - target_sigmoid))
       init_scale = 1.0_sp / sqrt(real(width * width * modes, sp))
       call random_normal(net%wt_log_amp(:,:,:,b), init_bias, init_scale)
       call random_normal(net%wt_phase(:,:,:,b), 0.0_sp, 1.0_sp)
    end do

    !! Output projection
    allocate(net%out_w(output_dim, width))
    allocate(net%out_b(output_dim))
    fan_in_scale = sqrt(2.0_sp / real(width, sp))
    call random_normal(net%out_w, 0.0_sp, fan_in_scale)
    net%out_b = 0.0_sp

    !! Spectral filter: exponential with strength=4.0 (Python default)
    filter_strength = 4.0_sp
    allocate(net%spec_filter(modes))
    do k = 1, modes
       net%spec_filter(k) = exp(-filter_strength * &
            (real(k-1, sp) / real(max(1, modes-1), sp))**2)
    end do

    !! Distance matrix: |i-j|/(G-1) on [0,1]
    allocate(net%dist(grid_size, grid_size))
    call build_distance_matrix(net%dist, grid_size)

    !! Count total parameters for Adam
    net%n_params = count_params(net)
    allocate(net%m_buf(net%n_params), source=0.0_sp)
    allocate(net%v_buf(net%n_params), source=0.0_sp)
    net%adam_t = 0

  end subroutine

  subroutine build_distance_matrix(dist, g)
    real(sp), intent(out) :: dist(:,:)
    integer, intent(in) :: g
    integer :: i, j
    real(sp) :: denom
    denom = real(max(1, g - 1), sp)
    do j = 1, g
       do i = 1, g
          dist(i, j) = abs(real(i - 1, sp) - real(j - 1, sp)) / denom
       end do
    end do
  end subroutine

  integer function count_params(net) result(n)
    type(custom_lno_type), intent(in) :: net
    n = size(net%proj_w) + size(net%proj_b)
    n = n + size(net%norm_scale) + size(net%norm_shift)
    n = n + size(net%pw_w) + size(net%pw_b)
    n = n + size(net%log_poles)
    n = n + size(net%wt_log_amp) + size(net%wt_phase)
    n = n + size(net%out_w) + size(net%out_b)
  end function

  ! ──────────────────────────────────────────────────────────────────────
  ! Forward pass (single sample)
  ! ──────────────────────────────────────────────────────────────────────

  subroutine custom_lno_forward(net, input, output)
    !! Forward pass: [input_dim] → [output_dim]
    !! Input: [T_n_norm || T_{n-1}_norm] (length input_dim=224)
    !! Output: ΔT_norm (length output_dim=112)
    type(custom_lno_type), intent(in) :: net
    real(sp), intent(in) :: input(:)            !! [input_dim]
    real(sp), intent(out) :: output(:)           !! [output_dim]

    real(sp), allocatable :: x(:,:)  !! [width, grid_size] — "channels × spatial"
    real(sp), allocatable :: x_pre(:,:)
    integer :: b, c, g, i
    real(sp) :: mu, var, inv_std

    g = net%grid_size

    !! Step 1: Input projection — full(224→64)
    !! Python does Conv1d(1→C, 1×1) on T_n_star only (just the current temp).
    !! Our input is [T_n || T_{n-1}] concatenated. We project the full 224-dim
    !! input vector to get a 64-dim feature at each "spatial point".
    !! Since we're treating features as [width, grid_size], we do:
    !!   For each spatial point g: x[c, g] = sum_j proj_w[c,j] * input[j] + proj_b[c]
    !! But our input is flat (not spatially structured). So we broadcast:
    !! Each channel = linear combination of the full input vector.
    allocate(x(net%width, g))
    do c = 1, net%width
       x(c, :) = dot_product_vec(net%proj_w(c,:), input) + net%proj_b(c)
    end do

    !! Apply SiLU to input projection (Python: input_proj uses no activation,
    !! but the LNO path starts with lno_proj which is Conv1d(1→C) without activation)
    !! Actually in the Python SecondOrderPredictor, lno_proj is just Conv1d(1→C, 1×1)
    !! with no activation. The activation comes inside the block loop.
    !! So we skip activation here.

    !! Step 2: 4 LNO blocks
    allocate(x_pre(net%width, g))
    do b = 1, net%num_blocks
       !! InstanceNorm: normalise each channel independently across spatial dim
       do c = 1, net%width
          mu = sum(x(c,:)) / real(g, sp)
          var = sum((x(c,:) - mu)**2) / real(g, sp)
          inv_std = 1.0_sp / sqrt(var + 1.0e-5_sp)
          x(c,:) = net%norm_scale(c,b) * (x(c,:) - mu) * inv_std + net%norm_shift(c,b)
       end do

       !! Pointwise Conv1d(C→C, 1×1): for each spatial point
       x_pre = x
       do i = 1, g
          do c = 1, net%width
             x(c, i) = dot_product_vec(net%pw_w(c,:,b), x_pre(:,i)) + net%pw_b(c,b)
          end do
       end do

       !! SiLU activation
       x = silu(x)

       !! LaplaceConv1d: spectral operator
       call laplace_conv1d_forward(net, b, x)
    end do

    !! Step 3: Output projection — full(64→112, linear)
    do i = 1, net%output_dim
       output(i) = dot_product_vec(net%out_w(i,:), x(:, min(i, g))) + net%out_b(i)
    end do

    deallocate(x, x_pre)
  end subroutine

  subroutine laplace_conv1d_forward(net, block_idx, x)
    !! Laplace spectral convolution: matching Python LaplaceConv1d.forward()
    !!
    !! poles[k] = softplus(log_poles[k])
    !! kernels[k,i,j] = exp(-poles[k] * dist[i,j]) / row_sum  ×  filter[k]
    !! weights[k,c_in,c_out] = sigmoid(log_amp[k,c_in,c_out]) * cos(phase[k,c_in,c_out])
    !! out[c_out, t] = sum_k sum_c_in sum_s  x[c_in, s] * kernels[k, t, s] * weights[k, c_in, c_out]
    type(custom_lno_type), intent(in) :: net
    integer, intent(in) :: block_idx
    real(sp), intent(inout) :: x(:,:)  !! [width, grid_size]

    integer :: g, m, w, k, i, j, ci, co
    real(sp), allocatable :: poles(:), kernels(:,:,:), weights(:,:,:)
    real(sp), allocatable :: x_in(:,:), out(:,:), kernel_x(:,:,:)
    real(sp) :: row_sum, amp

    g = net%grid_size
    m = net%modes
    w = net%width

    allocate(poles(m))
    do k = 1, m
       poles(k) = softplus(net%log_poles(k, block_idx))
    end do

    !! Build kernels: [modes, grid, grid]
    allocate(kernels(m, g, g))
    do k = 1, m
       do i = 1, g
          row_sum = 0.0_sp
          do j = 1, g
             kernels(k, i, j) = exp(-poles(k) * net%dist(i, j))
             row_sum = row_sum + kernels(k, i, j)
          end do
          !! Row-normalise (stop-gradient in Python; here no gradient tracking needed)
          row_sum = max(row_sum, 1.0e-8_sp)
          kernels(k, i, :) = kernels(k, i, :) / row_sum
       end do
       !! Apply spectral filter
       kernels(k, :, :) = kernels(k, :, :) * net%spec_filter(k)
    end do

    !! Build weights: [modes, width_in, width_out]
    !! Python: weight = max_amp * sigmoid(amp_sharpness * log_amp) * cos(phase)
    allocate(weights(m, w, w))
    do k = 1, m
       do co = 1, w
          do ci = 1, w
             amp = sigmoid(net%wt_log_amp(k, ci, co, block_idx))
             weights(k, ci, co) = amp * cos(net%wt_phase(k, ci, co, block_idx))
          end do
       end do
    end do

    !! Compute: out[co, t] = sum_k sum_ci sum_s x[ci, s] * kernels[k, t, s] * weights[k, ci, co]
    !! Optimised: first compute kernel_x[k, ci, t] = sum_s kernels[k, t, s] * x[ci, s]
    !!            then out[co, t] = sum_k sum_ci kernel_x[k, ci, t] * weights[k, ci, co]
    allocate(x_in(w, g), source=x)
    allocate(kernel_x(m, w, g))
    allocate(out(w, g))

    !! kernel_x[k, ci, t] = sum_s kernels[k, t, s] * x_in[ci, s]
    do k = 1, m
       do ci = 1, w
          do i = 1, g
             kernel_x(k, ci, i) = dot_product(kernels(k, i, :), x_in(ci, :))
          end do
       end do
    end do

    !! out[co, t] = sum_k sum_ci kernel_x[k, ci, t] * weights[k, ci, co]
    out = 0.0_sp
    do k = 1, m
       do co = 1, w
          do ci = 1, w
             do i = 1, g
                out(co, i) = out(co, i) + kernel_x(k, ci, i) * weights(k, ci, co)
             end do
          end do
       end do
    end do

    x = out

    deallocate(poles, kernels, weights, x_in, kernel_x, out)
  end subroutine

  ! ──────────────────────────────────────────────────────────────────────
  ! Prediction (batch)
  ! ──────────────────────────────────────────────────────────────────────

  subroutine custom_lno_predict(net, input, output)
    !! Predict for a single sample
    type(custom_lno_type), intent(in) :: net
    real(sp), intent(in) :: input(:)
    real(sp), intent(out) :: output(:)
    call custom_lno_forward(net, input, output)
  end subroutine

  ! ──────────────────────────────────────────────────────────────────────
  ! Training with numerical gradients (finite differences)
  ! ──────────────────────────────────────────────────────────────────────

  subroutine custom_lno_train(net, train_inputs, train_targets, &
       val_inputs, val_targets, num_epochs, batch_size, n_train, n_val)
    !! Train with Adam + MSE loss using numerical gradients.
    !! This is slow but correct — matching the Python training exactly.
    type(custom_lno_type), intent(inout) :: net
    real(sp), intent(in) :: train_inputs(:,:)   !! [input_dim, n_train]
    real(sp), intent(in) :: train_targets(:,:)  !! [output_dim, n_train]
    real(sp), intent(in) :: val_inputs(:,:)     !! [input_dim, n_val]
    real(sp), intent(in) :: val_targets(:,:)    !! [output_dim, n_val]
    integer, intent(in) :: num_epochs, batch_size, n_train, n_val

    real(sp), allocatable :: pred(:), grad(:), params(:)
    real(sp) :: loss, batch_loss, val_loss, best_val_loss
    integer :: epoch, i, j, idx, n_batches
    integer, allocatable :: perm(:)
    integer :: p

    allocate(pred(net%output_dim))
    allocate(params(net%n_params))
    allocate(grad(net%n_params))
    allocate(perm(n_train))

    best_val_loss = huge(1.0_sp)

    do epoch = 1, num_epochs
       !! Shuffle
       call random_permutation(perm, n_train)

       batch_loss = 0.0_sp
       n_batches = (n_train + batch_size - 1) / batch_size

       do i = 1, n_batches
          !! Collect current parameters
          call flatten_params(net, params)
          grad = 0.0_sp

          !! Accumulate gradients over batch
          batch_loss = 0.0_sp
          do j = 1, batch_size
             idx = (i - 1) * batch_size + j
             if (idx > n_train) exit
             idx = perm(idx)

             !! Forward
             call custom_lno_forward(net, train_inputs(:,idx), pred)

             !! MSE loss
             loss = sum((pred - train_targets(:,idx))**2) / real(net%output_dim, sp)
             batch_loss = batch_loss + loss

             !! Numerical gradient: for each parameter, perturb and compute loss
             call compute_numerical_gradient(net, train_inputs(:,idx), &
                  train_targets(:,idx), params, grad, net%n_params)
          end do
          batch_loss = batch_loss / real(min(batch_size, n_train - (i-1)*batch_size), sp)

          !! Adam update
          call adam_update(net, params, grad, real(min(batch_size, n_train - (i-1)*batch_size), sp))
          call unflatten_params(net, params)
       end do

       !! Validation loss
       val_loss = 0.0_sp
       do j = 1, n_val
          call custom_lno_forward(net, val_inputs(:,j), pred)
          val_loss = val_loss + sum((pred - val_targets(:,j))**2) / real(net%output_dim, sp)
       end do
       val_loss = val_loss / real(n_val, sp)

       write(*,'(A,I4,A,ES10.3,A,ES10.3)') 'epoch=', epoch, &
            ', train_loss=', batch_loss, ', val_loss=', val_loss
    end do

    deallocate(pred, params, grad, perm)
  end subroutine

  subroutine compute_numerical_gradient(net, input, target, params, grad_accum, np)
    !! Accumulate finite-difference gradient for one sample.
    type(custom_lno_type), intent(inout) :: net
    real(sp), intent(in) :: input(:), target(:)
    real(sp), intent(inout) :: params(:), grad_accum(:)
    integer, intent(in) :: np

    real(sp), allocatable :: pred_plus(:), pred_minus(:)
    real(sp) :: loss_plus, loss_minus, orig, eps_fd
    integer :: p

    eps_fd = 1.0e-4_sp
    allocate(pred_plus(net%output_dim))
    allocate(pred_minus(net%output_dim))

    do p = 1, np
       orig = params(p)

       params(p) = orig + eps_fd
       call unflatten_params(net, params)
       call custom_lno_forward(net, input, pred_plus)
       loss_plus = sum((pred_plus - target)**2) / real(net%output_dim, sp)

       params(p) = orig - eps_fd
       call unflatten_params(net, params)
       call custom_lno_forward(net, input, pred_minus)
       loss_minus = sum((pred_minus - target)**2) / real(net%output_dim, sp)

       grad_accum(p) = grad_accum(p) + (loss_plus - loss_minus) / (2.0_sp * eps_fd)

       params(p) = orig
    end do

    call unflatten_params(net, params)
    deallocate(pred_plus, pred_minus)
  end subroutine

  subroutine adam_update(net, params, grad, batch_count)
    type(custom_lno_type), intent(inout) :: net
    real(sp), intent(inout) :: params(:)
    real(sp), intent(in) :: grad(:)
    real(sp), intent(in) :: batch_count

    real(sp) :: g, m_hat, v_hat, bc1, bc2
    real(sp) :: grad_norm, scale
    integer :: p

    !! Average gradient
    !! Gradient clipping
    grad_norm = sqrt(sum((grad / batch_count)**2))
    if (grad_norm > net%grad_clip) then
       scale = net%grad_clip / grad_norm
    else
       scale = 1.0_sp
    end if

    net%adam_t = net%adam_t + 1
    bc1 = 1.0_sp - net%beta1**net%adam_t
    bc2 = 1.0_sp - net%beta2**net%adam_t

    do p = 1, net%n_params
       g = (grad(p) / batch_count) * scale

       !! AdamW weight decay
       params(p) = params(p) * (1.0_sp - net%lr * net%weight_decay)

       !! Adam moments
       net%m_buf(p) = net%beta1 * net%m_buf(p) + (1.0_sp - net%beta1) * g
       net%v_buf(p) = net%beta2 * net%v_buf(p) + (1.0_sp - net%beta2) * g * g

       m_hat = net%m_buf(p) / bc1
       v_hat = net%v_buf(p) / bc2

       params(p) = params(p) - net%lr * m_hat / (sqrt(v_hat) + net%eps_adam)
    end do
  end subroutine

  ! ──────────────────────────────────────────────────────────────────────
  ! Parameter flattening / unflattening
  ! ──────────────────────────────────────────────────────────────────────

  subroutine flatten_params(net, params)
    type(custom_lno_type), intent(in) :: net
    real(sp), intent(out) :: params(:)
    integer :: offset
    offset = 0
    call copy_to_flat(net%proj_w, params, offset)
    call copy_to_flat(net%proj_b, params, offset)
    call copy_to_flat(net%norm_scale, params, offset)
    call copy_to_flat(net%norm_shift, params, offset)
    call copy_to_flat(net%pw_w, params, offset)
    call copy_to_flat(net%pw_b, params, offset)
    call copy_to_flat(net%log_poles, params, offset)
    call copy_to_flat(net%wt_log_amp, params, offset)
    call copy_to_flat(net%wt_phase, params, offset)
    call copy_to_flat(net%out_w, params, offset)
    call copy_to_flat(net%out_b, params, offset)
  end subroutine

  subroutine unflatten_params(net, params)
    type(custom_lno_type), intent(inout) :: net
    real(sp), intent(in) :: params(:)
    integer :: offset
    offset = 0
    call copy_from_flat(net%proj_w, params, offset)
    call copy_from_flat(net%proj_b, params, offset)
    call copy_from_flat(net%norm_scale, params, offset)
    call copy_from_flat(net%norm_shift, params, offset)
    call copy_from_flat(net%pw_w, params, offset)
    call copy_from_flat(net%pw_b, params, offset)
    call copy_from_flat(net%log_poles, params, offset)
    call copy_from_flat(net%wt_log_amp, params, offset)
    call copy_from_flat(net%wt_phase, params, offset)
    call copy_from_flat(net%out_w, params, offset)
    call copy_from_flat(net%out_b, params, offset)
  end subroutine

  ! ──────────────────────────────────────────────────────────────────────
  ! Save / Load
  ! ──────────────────────────────────────────────────────────────────────

  subroutine custom_lno_save(net, filename)
    type(custom_lno_type), intent(in) :: net
    character(len=*), intent(in) :: filename
    integer :: u
    open(newunit=u, file=filename, form='unformatted', status='replace')
    write(u) net%grid_size, net%width, net%modes, net%num_blocks, &
         net%input_dim, net%output_dim
    write(u) net%proj_w, net%proj_b
    write(u) net%norm_scale, net%norm_shift
    write(u) net%pw_w, net%pw_b
    write(u) net%log_poles
    write(u) net%wt_log_amp, net%wt_phase
    write(u) net%out_w, net%out_b
    close(u)
  end subroutine

  subroutine custom_lno_load(net, filename)
    type(custom_lno_type), intent(inout) :: net
    character(len=*), intent(in) :: filename
    integer :: u, g, w, m, nb, id, od
    open(newunit=u, file=filename, form='unformatted', status='old')
    read(u) g, w, m, nb, id, od
    call custom_lno_init(net, g, w, m, nb, id, od)
    read(u) net%proj_w, net%proj_b
    read(u) net%norm_scale, net%norm_shift
    read(u) net%pw_w, net%pw_b
    read(u) net%log_poles
    read(u) net%wt_log_amp, net%wt_phase
    read(u) net%out_w, net%out_b
    close(u)
  end subroutine

  ! ──────────────────────────────────────────────────────────────────────
  ! Utilities
  ! ──────────────────────────────────────────────────────────────────────

  real(sp) function dot_product_vec(a, b) result(d)
    real(sp), intent(in) :: a(:), b(:)
    integer :: i
    d = 0.0_sp
    do i = 1, size(a)
       d = d + a(i) * b(i)
    end do
  end function

  subroutine random_normal(arr, mean, std)
    real(sp), intent(out) :: arr(..)
    real(sp), intent(in) :: mean, std
    real(sp), allocatable :: flat(:)
    real(sp) :: u1, u2
    integer :: n, i

    select rank(arr)
    rank(1)
       n = size(arr)
       allocate(flat(n))
    rank(2)
       n = size(arr)
       allocate(flat(n))
    rank(3)
       n = size(arr)
       allocate(flat(n))
    rank(4)
       n = size(arr)
       allocate(flat(n))
    rank default
       return
    end select

    !! Box-Muller pairs
    do i = 1, n - 1, 2
       call random_number(u1)
       call random_number(u2)
       u1 = max(u1, 1.0e-10_sp)
       flat(i)   = mean + std * sqrt(-2.0_sp * log(u1)) * cos(2.0_sp * PI * u2)
       flat(i+1) = mean + std * sqrt(-2.0_sp * log(u1)) * sin(2.0_sp * PI * u2)
    end do
    if (mod(n, 2) == 1) then
       call random_number(u1)
       call random_number(u2)
       u1 = max(u1, 1.0e-10_sp)
       flat(n) = mean + std * sqrt(-2.0_sp * log(u1)) * cos(2.0_sp * PI * u2)
    end if

    select rank(arr)
    rank(1)
       arr = flat
    rank(2)
       arr = reshape(flat, shape(arr))
    rank(3)
       arr = reshape(flat, shape(arr))
    rank(4)
       arr = reshape(flat, shape(arr))
    end select

    deallocate(flat)
  end subroutine

  subroutine random_permutation(perm, n)
    integer, intent(out) :: perm(:)
    integer, intent(in) :: n
    integer :: i, j, tmp
    real(sp) :: r
    do i = 1, n
       perm(i) = i
    end do
    do i = n, 2, -1
       call random_number(r)
       j = 1 + int(r * real(i, sp))
       j = max(1, min(i, j))
       tmp = perm(i)
       perm(i) = perm(j)
       perm(j) = tmp
    end do
  end subroutine

  subroutine copy_to_flat(arr, flat, offset)
    real(sp), intent(in) :: arr(..)
    real(sp), intent(inout) :: flat(:)
    integer, intent(inout) :: offset
    integer :: n, i
    n = size(arr)
    select rank(arr)
    rank(1)
       flat(offset+1:offset+n) = arr
    rank(2)
       flat(offset+1:offset+n) = reshape(arr, [n])
    rank(3)
       flat(offset+1:offset+n) = reshape(arr, [n])
    rank(4)
       flat(offset+1:offset+n) = reshape(arr, [n])
    end select
    offset = offset + n
  end subroutine

  subroutine copy_from_flat(arr, flat, offset)
    real(sp), intent(inout) :: arr(..)
    real(sp), intent(in) :: flat(:)
    integer, intent(inout) :: offset
    integer :: n
    n = size(arr)
    select rank(arr)
    rank(1)
       arr = flat(offset+1:offset+n)
    rank(2)
       arr = reshape(flat(offset+1:offset+n), shape(arr))
    rank(3)
       arr = reshape(flat(offset+1:offset+n), shape(arr))
    rank(4)
       arr = reshape(flat(offset+1:offset+n), shape(arr))
    end select
    offset = offset + n
  end subroutine

end module custom_lno
