submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_nop
  !! Submodule containing autodiff operations for the Graph Neural Operator
  !!
  !! Provides two differentiable operations:
  !!
  !! 1. `gno_kernel_eval` — evaluates the kernel MLP on each edge:
  !!    \(\kappa(\Delta x) = V \, \mathrm{relu}(U \Delta x + b_u) + b_v\)
  !!    left_operand  → edge_features [d, num_edges]
  !!    right_operand → packed kernel params [H*d + H + F*H + F, 1]
  !!       where F = F_out * F_in
  !!    output → [F_out * F_in, num_edges] per-edge kernel values
  !!
  !! 2. `gno_aggregate` — aggregates messages using per-edge kernels:
  !!    \(m_i = \sum_{j \in \mathcal{N}(i)} \kappa_{ij} \, h_j\)
  !!    left_operand  → features [F_in, num_vertices]
  !!    right_operand → edge_kernels [F_out * F_in, num_edges]
  !!    output → [F_out, num_vertices]
  !!
  !! `gno_aggregate` stores `adj_ia` and `adj_ja` on the result for
  !! use in the backward pass. Metadata (d, H, F_in, F_out) is stored
  !! in `indices` of the kernel evaluation result.

contains

!###############################################################################
  module function gno_kernel_eval( &
       coords, kernel_params, adj_ia, adj_ja, &
       coord_dim, kernel_hidden, F_in, F_out &
  ) result(c)
    !! Evaluate the GNO kernel MLP on every directed edge in the graph.
    !!
    !! For each edge feature column e, compute:
    !!   dx      = edge_features(:,e)                 [d]
    !!   hidden  = relu( U @ dx + b_u )                [H]
    !!   kappa_e = V @ hidden + b_v                    [F_out*F_in]
    !!
    !! Kernel params layout (flat column, size H*d + H + F*H + F):
    !!   U   : params(1 : H*d)
    !!   b_u : params(H*d+1 : H*d+H)
    !!   V   : params(H*d+H+1 : H*d+H+F*H)
    !!   b_v : params(H*d+H+F*H+1 : end)
    implicit none

    class(array_type), intent(in), target :: coords
    !! Edge features / relative coordinates [d, num_edges]
    class(array_type), intent(in), target :: kernel_params
    !! Packed kernel parameters [H*d + H + F*H + F, 1]
    integer, dimension(:), intent(in)  :: adj_ia
    !! CSR row pointers (size num_vertices + 1)
    integer, dimension(:,:), intent(in) :: adj_ja
    !! CSR column indices (adj_ja(1,:) = neighbour index)
    integer, intent(in) :: coord_dim, kernel_hidden, F_in, F_out
    !! Metadata for unpacking kernel_params
    type(array_type), pointer :: c

    ! locals
    integer :: num_e, d, H, F, e
    integer :: off_U, off_bu, off_V, off_bv
    real(real32), allocatable :: U(:,:), b_u(:), V(:,:), b_v(:)
    real(real32), allocatable :: dx(:), hidden(:)

    d = coord_dim
    H = kernel_hidden
    F = F_out * F_in        ! kernel output width
    num_e = size(coords%val, 2)

    ! ---- Unpack kernel params ------------------------------------------------
    off_U  = 0
    off_bu = H * d
    off_V  = off_bu + H
    off_bv = off_V + F * H

    allocate(U(H, d)); U = reshape(kernel_params%val(off_U+1:off_bu, 1), [H, d])
    allocate(b_u(H));  b_u = kernel_params%val(off_bu+1:off_V, 1)
    allocate(V(F, H)); V = reshape(kernel_params%val(off_V+1:off_bv, 1), [F, H])
    allocate(b_v(F));  b_v = kernel_params%val(off_bv+1:, 1)

    ! ---- Forward: evaluate kernel on every edge ------------------------------
    c => coords%create_result(array_shape=[F, num_e])
    allocate(dx(d), hidden(H))

    do e = 1, num_e
       dx = coords%val(:, e)
       hidden = matmul(U, dx) + b_u
       hidden = max(hidden, 0.0_real32)          ! ReLU
       c%val(:, e) = matmul(V, hidden) + b_v
    end do

    deallocate(dx, hidden, U, b_u, V, b_v)

    ! ---- Store metadata for backward -----------------------------------------
    allocate(c%indices(4))
    c%indices = [d, H, F_in, F_out]

    c%get_partial_left     => get_partial_gno_kernel_coords
    c%get_partial_right    => get_partial_gno_kernel_params
    c%get_partial_left_val => get_partial_gno_kernel_coords_val
    c%get_partial_right_val => get_partial_gno_kernel_params_val
    if(coords%requires_grad .or. kernel_params%requires_grad) then
       c%requires_grad    = .true.
       c%is_forward       = coords%is_forward .or. kernel_params%is_forward
       c%operation        = 'gno_kernel_eval'
       c%left_operand     => coords
       c%right_operand    => kernel_params
       c%owns_left_operand  = coords%is_temporary
       c%owns_right_operand = kernel_params%is_temporary
    end if

  end function gno_kernel_eval
!-------------------------------------------------------------------------------
  function get_partial_gno_kernel_coords(this, upstream_grad) result(output)
    !! Gradient of gno_kernel_eval w.r.t. edge features (left operand)
    !!
    !! upstream_grad has shape [F, num_edges]
    !! output has shape [d, num_edges]
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%left_operand%val))
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_gno_kernel_coords
!-------------------------------------------------------------------------------
  pure subroutine get_partial_gno_kernel_coords_val( &
       this, upstream_grad, output)
    !! In-place gradient w.r.t. edge features
    !!
    !! Chain rule through kernel:
    !!   kappa_e = V @ relu(U @ dx_e + b_u) + b_v
    !!   d(kappa_e)/d(dx_e) = V @ diag(relu'(U dx_e + b_u)) @ U
    !! Since the left operand already stores edge features directly,
    !! gradients accumulate independently for each edge column.
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: d, H, F, num_e, e, k
    integer :: off_U, off_bu, off_V
    real(real32), allocatable :: U(:,:), b_u(:), V(:,:)
    real(real32), allocatable :: dx(:), pre_act(:), relu_mask(:)
    real(real32), allocatable :: dkappa_ddx(:,:)   ! [F, d]
    real(real32), allocatable :: grad_dx(:)         ! [d]

    d = this%indices(1)
    H = this%indices(2)
    F = this%indices(3) * this%indices(4)
    num_e = size(this%left_operand%val, 2)

    off_U  = 0
    off_bu = H * d
    off_V  = off_bu + H

    allocate(U(H, d))
    U = reshape(this%right_operand%val(off_U+1:off_bu, 1), [H, d])
    allocate(b_u(H))
    b_u = this%right_operand%val(off_bu+1:off_V, 1)
    allocate(V(F, H))
    V = reshape(this%right_operand%val(off_V+1:off_V+F*H, 1), [F, H])

    allocate(dx(d), pre_act(H), relu_mask(H))
    allocate(dkappa_ddx(F, d), grad_dx(d))

    output = 0.0_real32

    do e = 1, num_e
       dx = this%left_operand%val(:, e)
       pre_act = matmul(U, dx) + b_u
       do k = 1, H
          if(pre_act(k) > 0.0_real32) then
             relu_mask(k) = 1.0_real32
          else
             relu_mask(k) = 0.0_real32
          end if
       end do

       dkappa_ddx = 0.0_real32
       do k = 1, H
          if(relu_mask(k) > 0.0_real32) then
             dkappa_ddx = dkappa_ddx + &
                  spread(V(:, k), 2, d) * spread(U(k, :), 1, F)
          end if
       end do

       grad_dx = matmul(upstream_grad(:, e), dkappa_ddx)
       output(:, e) = output(:, e) + grad_dx
    end do

    deallocate(U, b_u, V, dx, pre_act, relu_mask, dkappa_ddx, grad_dx)

  end subroutine get_partial_gno_kernel_coords_val
!-------------------------------------------------------------------------------
  function get_partial_gno_kernel_params(this, upstream_grad) result(output)
    !! Gradient of gno_kernel_eval w.r.t. kernel_params (right operand)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%right_operand%val))
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_gno_kernel_params
!-------------------------------------------------------------------------------
  pure subroutine get_partial_gno_kernel_params_val( &
       this, upstream_grad, output)
    !! In-place gradient w.r.t. packed kernel params
    !!
    !! Accumulate gradients over all edges:
    !!   d(kappa_e)/dU   = V^T @ diag(relu_mask) outer dx
    !!   d(kappa_e)/db_u = V^T @ diag(relu_mask) dot upstream
    !!   d(kappa_e)/dV   = upstream outer relu(U dx + b_u)
    !!   d(kappa_e)/db_v = upstream directly
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: d, H, F, num_e, e, k, f_idx
    integer :: off_U, off_bu, off_V, off_bv
    real(real32), allocatable :: U(:,:), b_u(:), V(:,:)
    real(real32), allocatable :: dx(:), pre_act(:), hidden(:)
    real(real32), allocatable :: grad_hidden(:)  ! [H]

    d = this%indices(1)
    H = this%indices(2)
    F = this%indices(3) * this%indices(4)
    num_e = size(this%left_operand%val, 2)

    off_U  = 0
    off_bu = H * d
    off_V  = off_bu + H
    off_bv = off_V + F * H

    allocate(U(H, d))
    U = reshape(this%right_operand%val(off_U+1:off_bu, 1), [H, d])
    allocate(b_u(H))
    b_u = this%right_operand%val(off_bu+1:off_V, 1)
    allocate(V(F, H))
    V = reshape(this%right_operand%val(off_V+1:off_bv, 1), [F, H])

    allocate(dx(d), pre_act(H), hidden(H), grad_hidden(H))

    output = 0.0_real32

    do e = 1, num_e
       dx = this%left_operand%val(:, e)
       pre_act = matmul(U, dx) + b_u
       hidden = max(pre_act, 0.0_real32)

       ! --- d/d(b_v): upstream_grad(:,e) directly ---
       output(off_bv+1:, 1) = output(off_bv+1:, 1) + upstream_grad(:, e)

       ! --- d/dV: upstream outer hidden => grad_V(f,h) += upstream(f,e)*hidden(h) ---
       do k = 1, H
          do f_idx = 1, F
             output(off_V + (k-1)*F + f_idx, 1) = &
                  output(off_V + (k-1)*F + f_idx, 1) + &
                  upstream_grad(f_idx, e) * hidden(k)
          end do
       end do

       ! --- Backprop through relu: grad_hidden = V^T @ upstream(:,e) * relu' ---
       grad_hidden = matmul(transpose(V), upstream_grad(:, e))
       do k = 1, H
          if(pre_act(k) <= 0.0_real32) grad_hidden(k) = 0.0_real32
       end do

       ! --- d/d(b_u): grad_hidden directly ---
       output(off_bu+1:off_V, 1) = output(off_bu+1:off_V, 1) + grad_hidden

       ! --- d/dU: grad_hidden outer dx => grad_U(h,dd) += grad_hidden(h)*dx(dd) ---
       do k = 1, d
          do f_idx = 1, H
             output(off_U + (k-1)*H + f_idx, 1) = &
                  output(off_U + (k-1)*H + f_idx, 1) + &
                  grad_hidden(f_idx) * dx(k)
          end do
       end do
    end do

    deallocate(U, b_u, V, dx, pre_act, hidden, grad_hidden)

  end subroutine get_partial_gno_kernel_params_val
!###############################################################################


!###############################################################################
  module function gno_aggregate( &
       features, edge_kernels, adj_ia, adj_ja, F_in, F_out &
  ) result(c)
    !! Aggregate neighbour messages using pre-computed per-edge kernels.
    !!
    !! For each node i:
    !!   m_i = sum_{j in N(i)} reshape(kappa_e, [F_out, F_in]) @ h_j
    !!
    !! where e is the edge index corresponding to (i, j).
    !!
    !! left_operand  → features      [F_in, num_vertices]
    !! right_operand → edge_kernels  [F_out*F_in, num_edges]
    !! output        → [F_out, num_vertices]
    implicit none

    class(array_type), intent(in), target :: features
    !! Node features [F_in, num_vertices]
    class(array_type), intent(in), target :: edge_kernels
    !! Per-edge kernel values [F_out*F_in, num_edges]
    integer, dimension(:), intent(in)  :: adj_ia
    !! CSR row pointers
    integer, dimension(:,:), intent(in) :: adj_ja
    !! CSR column indices
    integer, intent(in) :: F_in, F_out
    !! Feature dimensions
    type(array_type), pointer :: c

    integer :: num_v, i, j, jj, edge_idx

    num_v = size(features%val, 2)
    c => features%create_result(array_shape=[F_out, num_v])
    c%val = 0.0_real32

    do i = 1, num_v
       do jj = adj_ia(i), adj_ia(i+1) - 1
          j = adj_ja(1, jj)
          edge_idx = adj_ja(2, jj)
          ! kappa_e reshaped to [F_out, F_in], multiplied by h_j [F_in]
          c%val(:, i) = c%val(:, i) + &
               matmul( &
                    reshape(edge_kernels%val(:, edge_idx), [F_out, F_in]), &
                    features%val(:, j) &
               )
       end do
    end do

    c%indices = adj_ia
    c%adj_ja  = adj_ja

    c%get_partial_left     => get_partial_gno_agg_features
    c%get_partial_right    => get_partial_gno_agg_kernels
    c%get_partial_left_val => get_partial_gno_agg_features_val
    c%get_partial_right_val => get_partial_gno_agg_kernels_val
    if(features%requires_grad .or. edge_kernels%requires_grad) then
       c%requires_grad    = .true.
       c%is_forward       = features%is_forward .or. edge_kernels%is_forward
       c%operation        = 'gno_aggregate'
       c%left_operand     => features
       c%right_operand    => edge_kernels
       c%owns_left_operand  = features%is_temporary
       c%owns_right_operand = edge_kernels%is_temporary
    end if

  end function gno_aggregate
!-------------------------------------------------------------------------------
  function get_partial_gno_agg_features(this, upstream_grad) result(output)
    !! Gradient of gno_aggregate w.r.t. features (left operand)
    !!
    !! d(m_i)/d(h_j) = kappa_{ij}^T  (the [F_in, F_out] transpose)
    !! So: grad_h(j) += kappa_{ij}^T @ upstream(:,i)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%left_operand%val))
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_gno_agg_features
!-------------------------------------------------------------------------------
  pure subroutine get_partial_gno_agg_features_val( &
       this, upstream_grad, output)
    !! In-place gradient w.r.t. features
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: F_in, F_out, num_v, i, j, jj, edge_idx

    ! Infer dimensions from operands
    F_in  = size(this%left_operand%val, 1)
    F_out = size(upstream_grad, 1)
    num_v = size(this%left_operand%val, 2)

    output = 0.0_real32
    do i = 1, num_v
       do jj = this%indices(i), this%indices(i+1) - 1
          j = this%adj_ja(1, jj)
          edge_idx = this%adj_ja(2, jj)
          ! grad_h(j) += kappa_e^T @ upstream(:,i)
          ! kappa_e is [F_out*F_in] → reshape to [F_out, F_in]
          ! kappa_e^T is [F_in, F_out]
          output(:, j) = output(:, j) + &
               matmul( &
                    transpose(reshape( &
                         this%right_operand%val(:, edge_idx), [F_out, F_in])), &
                    upstream_grad(:, i) &
               )
       end do
    end do

  end subroutine get_partial_gno_agg_features_val
!-------------------------------------------------------------------------------
  function get_partial_gno_agg_kernels(this, upstream_grad) result(output)
    !! Gradient of gno_aggregate w.r.t. edge_kernels (right operand)
    !!
    !! d(m_i)/d(kappa_e) = h_j (Kronecker-product structure)
    !! For vectorised kappa: grad_kappa(e) = upstream(:,i) ⊗ h_j
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%right_operand%val))
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_gno_agg_kernels
!-------------------------------------------------------------------------------
  pure subroutine get_partial_gno_agg_kernels_val( &
       this, upstream_grad, output)
    !! In-place gradient w.r.t. edge_kernels
    !!
    !! The aggregation is: m_i += reshape(kappa_e,[F_out,F_in]) @ h_j
    !! So d(m_i)/d(kappa_e) viewed as reshape:
    !!   grad_kappa_e = vec( upstream(:,i) @ h_j^T )
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: F_in, F_out, num_v, i, j, jj, edge_idx
    integer :: fo, fi

    ! Infer dimensions from operands
    F_in  = size(this%left_operand%val, 1)
    F_out = size(upstream_grad, 1)
    num_v = size(this%left_operand%val, 2)

    output = 0.0_real32
    do i = 1, num_v
       do jj = this%indices(i), this%indices(i+1) - 1
          j = this%adj_ja(1, jj)
          edge_idx = this%adj_ja(2, jj)
          ! kappa_e is stored as vec(K) where K = reshape(kappa_e, [F_out, F_in])
          ! d(m_i)/d(K(fo,fi)) = upstream(fo, i) * h(fi, j)
          ! vec index: (fi-1)*F_out + fo
          do fi = 1, F_in
             do fo = 1, F_out
                output((fi-1)*F_out + fo, edge_idx) = &
                     output((fi-1)*F_out + fo, edge_idx) + &
                     upstream_grad(fo, i) * this%left_operand%val(fi, j)
             end do
          end do
       end do
    end do

  end subroutine get_partial_gno_agg_kernels_val
!###############################################################################


!###############################################################################
! Laplace Neural Operator — encode and decode with differentiable poles
!###############################################################################

!###############################################################################
  module function lno_encode( &
       input, poles, num_inputs, num_modes &
  ) result(c)
    !! Encode input through the Laplace basis built from learnable poles.
    !!
    !! Forward:  y = E(mu) @ u   [M, batch]
    !!   E[m,j] = exp(-mu_m * t_j),  t_j = (j-1)/(n_in-1)
    !!
    !! left_operand  → input u  [n_in, batch]
    !! right_operand → poles mu [M, 1]
    !! output        → encoded  [M, batch]
    implicit none

    class(array_type), intent(in), target :: input
    class(array_type), intent(in), target :: poles
    integer, intent(in) :: num_inputs, num_modes
    type(array_type), pointer :: c

    integer :: num_samples, m, j
    real(real32) :: t, s
    real(real32), allocatable :: E(:,:)  ! [M, n_in]

    num_samples = size(input%val, 2)

    ! Build encoder basis E [M x n_in]
    allocate(E(num_modes, num_inputs))
    do j = 1, num_inputs
       if(num_inputs .gt. 1) then
          t = real(j-1, real32) / real(num_inputs-1, real32)
       else
          t = 0.0_real32
       end if
       do m = 1, num_modes
          s = poles%val(m, 1)
          E(m, j) = exp(-s * t)
       end do
    end do

    ! Forward: y = E @ u
    c => input%create_result(array_shape=[num_modes, num_samples])
    c%val = matmul(E, input%val)

    deallocate(E)

    ! Store metadata for backward
    allocate(c%indices(2))
    c%indices = [num_inputs, num_modes]

    c%get_partial_left     => get_partial_lno_encode_input
    c%get_partial_right    => get_partial_lno_encode_poles
    c%get_partial_left_val => get_partial_lno_encode_input_val
    c%get_partial_right_val => get_partial_lno_encode_poles_val
    if(input%requires_grad .or. poles%requires_grad) then
       c%requires_grad    = .true.
       c%is_forward       = input%is_forward .or. poles%is_forward
       c%operation        = 'lno_encode'
       c%left_operand     => input
       c%right_operand    => poles
       c%owns_left_operand  = input%is_temporary
       c%owns_right_operand = poles%is_temporary
    end if

  end function lno_encode
!-------------------------------------------------------------------------------
  function get_partial_lno_encode_input(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%left_operand%val))
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_lno_encode_input
!-------------------------------------------------------------------------------
  pure subroutine get_partial_lno_encode_input_val( &
       this, upstream_grad, output)
    !! dL/du = E^T @ upstream  [n_in, batch]
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: n_in, M, m, j, s, num_samples
    real(real32) :: t, mu_m
    real(real32), allocatable :: ET(:,:)  ! [n_in, M]

    n_in = this%indices(1)
    M    = this%indices(2)
    num_samples = size(upstream_grad, 2)

    allocate(ET(n_in, M))
    do m = 1, M
       mu_m = this%right_operand%val(m, 1)
       do j = 1, n_in
          if(n_in .gt. 1) then
             t = real(j-1, real32) / real(n_in-1, real32)
          else
             t = 0.0_real32
          end if
          ET(j, m) = exp(-mu_m * t)
       end do
    end do

    output = matmul(ET, upstream_grad)

    deallocate(ET)

  end subroutine get_partial_lno_encode_input_val
!-------------------------------------------------------------------------------
  function get_partial_lno_encode_poles(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%right_operand%val))
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_lno_encode_poles
!-------------------------------------------------------------------------------
  pure subroutine get_partial_lno_encode_poles_val( &
       this, upstream_grad, output)
    !! dL/dmu_m per sample:
    !!   output[m,s] = upstream[m,s] * sum_j (-t_j) * exp(-mu_m*t_j) * u[j,s]
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: n_in, M, m, j, s, num_samples
    real(real32) :: t, mu_m, accum

    n_in = this%indices(1)
    M    = this%indices(2)
    num_samples = size(upstream_grad, 2)

    output = 0.0_real32
    do s = 1, num_samples
       do m = 1, M
          mu_m = this%right_operand%val(m, 1)
          accum = 0.0_real32
          do j = 1, n_in
             if(n_in .gt. 1) then
                t = real(j-1, real32) / real(n_in-1, real32)
             else
                t = 0.0_real32
             end if
             accum = accum + (-t) * exp(-mu_m * t) * &
                  this%left_operand%val(j, s)
          end do
          output(m, s) = upstream_grad(m, s) * accum
       end do
    end do

  end subroutine get_partial_lno_encode_poles_val
!###############################################################################


!###############################################################################
  module function lno_decode( &
       spectral, poles, num_outputs, num_modes &
  ) result(c)
    !! Decode through the Laplace basis built from learnable poles.
    !!
    !! Forward:  y = D(mu) @ x   [n_out, batch]
    !!   D[i,m] = exp(-mu_m * tau_i),  tau_i = (i-1)/(n_out-1)
    !!
    !! left_operand  → spectral x  [M, batch]
    !! right_operand → poles mu    [M, 1]
    !! output        → decoded     [n_out, batch]
    implicit none

    class(array_type), intent(in), target :: spectral
    class(array_type), intent(in), target :: poles
    integer, intent(in) :: num_outputs, num_modes
    type(array_type), pointer :: c

    integer :: num_samples, m, i
    real(real32) :: t, s
    real(real32), allocatable :: D(:,:)  ! [n_out, M]

    num_samples = size(spectral%val, 2)

    ! Build decoder basis D [n_out x M]
    allocate(D(num_outputs, num_modes))
    do m = 1, num_modes
       s = poles%val(m, 1)
       do i = 1, num_outputs
          if(num_outputs .gt. 1) then
             t = real(i-1, real32) / real(num_outputs-1, real32)
          else
             t = 0.0_real32
          end if
          D(i, m) = exp(-s * t)
       end do
    end do

    ! Forward: y = D @ x
    c => spectral%create_result(array_shape=[num_outputs, num_samples])
    c%val = matmul(D, spectral%val)

    deallocate(D)

    ! Store metadata for backward
    allocate(c%indices(2))
    c%indices = [num_outputs, num_modes]

    c%get_partial_left     => get_partial_lno_decode_spectral
    c%get_partial_right    => get_partial_lno_decode_poles
    c%get_partial_left_val => get_partial_lno_decode_spectral_val
    c%get_partial_right_val => get_partial_lno_decode_poles_val
    if(spectral%requires_grad .or. poles%requires_grad) then
       c%requires_grad    = .true.
       c%is_forward       = spectral%is_forward .or. poles%is_forward
       c%operation        = 'lno_decode'
       c%left_operand     => spectral
       c%right_operand    => poles
       c%owns_left_operand  = spectral%is_temporary
       c%owns_right_operand = poles%is_temporary
    end if

  end function lno_decode
!-------------------------------------------------------------------------------
  function get_partial_lno_decode_spectral(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%left_operand%val))
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_lno_decode_spectral
!-------------------------------------------------------------------------------
  pure subroutine get_partial_lno_decode_spectral_val( &
       this, upstream_grad, output)
    !! dL/dx = D^T @ upstream  [M, batch]
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: n_out, M, m, i, num_samples
    real(real32) :: t, mu_m
    real(real32), allocatable :: DT(:,:)  ! [M, n_out]

    n_out = this%indices(1)
    M     = this%indices(2)
    num_samples = size(upstream_grad, 2)

    allocate(DT(M, n_out))
    do m = 1, M
       mu_m = this%right_operand%val(m, 1)
       do i = 1, n_out
          if(n_out .gt. 1) then
             t = real(i-1, real32) / real(n_out-1, real32)
          else
             t = 0.0_real32
          end if
          DT(m, i) = exp(-mu_m * t)
       end do
    end do

    output = matmul(DT, upstream_grad)

    deallocate(DT)

  end subroutine get_partial_lno_decode_spectral_val
!-------------------------------------------------------------------------------
  function get_partial_lno_decode_poles(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%right_operand%val))
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_lno_decode_poles
!-------------------------------------------------------------------------------
  pure subroutine get_partial_lno_decode_poles_val( &
       this, upstream_grad, output)
    !! dL/dmu_m per sample:
    !!   output[m,s] = sum_i upstream[i,s]*(-tau_i)*exp(-mu_m*tau_i)*x[m,s]
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: n_out, M, m, i, s, num_samples
    real(real32) :: t, mu_m, accum

    n_out = this%indices(1)
    M     = this%indices(2)
    num_samples = size(upstream_grad, 2)

    output = 0.0_real32
    do s = 1, num_samples
       do m = 1, M
          mu_m = this%right_operand%val(m, 1)
          accum = 0.0_real32
          do i = 1, n_out
             if(n_out .gt. 1) then
                t = real(i-1, real32) / real(n_out-1, real32)
             else
                t = 0.0_real32
             end if
             accum = accum + upstream_grad(i, s) * (-t) * exp(-mu_m * t)
          end do
          output(m, s) = accum * this%left_operand%val(m, s)
       end do
    end do

  end subroutine get_partial_lno_decode_poles_val
!###############################################################################


!###############################################################################
! Element-wise scale: out[i,s] = input[i,s] * scale[i,1]
! Handles non-sample-dependent scale vectors correctly (unlike built-in *)
!###############################################################################

!###############################################################################
  module function elem_scale(input, scale) result(c)
    implicit none

    class(array_type), intent(in), target :: input
    class(array_type), intent(in), target :: scale
    type(array_type), pointer :: c

    integer :: i, s, n, ns

    n  = size(input%val, 1)
    ns = size(input%val, 2)

    c => input%create_result(array_shape=[n, ns])
    do concurrent(s = 1:ns, i = 1:n)
       c%val(i, s) = input%val(i, s) * scale%val(i, 1)
    end do

    c%get_partial_left     => null()
    c%get_partial_right    => null()
    c%get_partial_left_val  => get_partial_elem_scale_input_val
    c%get_partial_right_val => get_partial_elem_scale_scale_val
    if(input%requires_grad .or. scale%requires_grad) then
       c%requires_grad    = .true.
       c%is_forward       = input%is_forward .or. scale%is_forward
       c%operation        = 'elem_scale'
       c%left_operand     => input
       c%right_operand    => scale
       c%owns_left_operand  = input%is_temporary
       c%owns_right_operand = scale%is_temporary
    end if

  end function elem_scale
!-------------------------------------------------------------------------------


!-------------------------------------------------------------------------------
  pure subroutine get_partial_elem_scale_input_val(this, upstream_grad, output)
    !! d(out)/d(input): upstream * scale (broadcast scale along samples)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output
    integer :: i, s

    do concurrent(s = 1:size(output,2), i = 1:size(output,1))
       output(i, s) = upstream_grad(i, s) * this%right_operand%val(i, 1)
    end do

  end subroutine get_partial_elem_scale_input_val
!-------------------------------------------------------------------------------


!-------------------------------------------------------------------------------
  pure subroutine get_partial_elem_scale_scale_val(this, upstream_grad, output)
    !! d(out)/d(scale): upstream * input (element-wise, per sample)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output
    integer :: i, s

    do concurrent(s = 1:size(output,2), i = 1:size(output,1))
       output(i, s) = upstream_grad(i, s) * this%left_operand%val(i, s)
    end do

  end subroutine get_partial_elem_scale_scale_val
!###############################################################################


!###############################################################################
! Orthogonal Neural Operator — encode and decode with differentiable basis
!###############################################################################

!###############################################################################
  module function ono_encode( &
       input, basis_weights, num_inputs, num_basis &
  ) result(c)
    !! Encode input through an orthogonalised basis.
    !!
    !! Forward: y = Q(B)^T @ u   [k, batch]
    !!   Q = modified_gram_schmidt(B), B [n x k] from basis_weights
    !!
    !! left_operand  → input u         [n, batch]
    !! right_operand → basis weights B  [n*k, 1]
    !! output        → encoded          [k, batch]
    implicit none

    class(array_type), intent(in), target :: input
    class(array_type), intent(in), target :: basis_weights
    integer, intent(in) :: num_inputs, num_basis
    type(array_type), pointer :: c

    integer :: num_samples, n, k, i, j, s
    real(real32), allocatable :: B(:,:), Q(:,:), QT(:,:)
    real(real32) :: norm_val, proj

    n = num_inputs
    k = num_basis
    num_samples = size(input%val, 2)

    ! Modified Gram-Schmidt: B -> Q
    allocate(B(n, k), Q(n, k), QT(k, n))
    B = reshape(basis_weights%val(:, 1), [n, k])
    Q = B
    do j = 1, k
       do i = 1, j - 1
          proj = dot_product(Q(:,i), Q(:,j))
          Q(:,j) = Q(:,j) - proj * Q(:,i)
       end do
       norm_val = sqrt(dot_product(Q(:,j), Q(:,j)))
       if(norm_val .gt. 1.0e-12_real32)then
          Q(:,j) = Q(:,j) / norm_val
       else
          Q(:,j) = 0.0_real32
       end if
    end do

    ! Transpose
    do j = 1, n
       do i = 1, k
          QT(i, j) = Q(j, i)
       end do
    end do

    ! Forward: y = Q^T @ u
    c => input%create_result(array_shape=[k, num_samples])
    c%val = matmul(QT, input%val)

    deallocate(B, Q, QT)

    ! Store metadata
    allocate(c%indices(2))
    c%indices = [n, k]

    c%get_partial_left     => get_partial_ono_encode_input
    c%get_partial_right    => get_partial_ono_encode_basis
    c%get_partial_left_val => get_partial_ono_encode_input_val
    c%get_partial_right_val => get_partial_ono_encode_basis_val
    if(input%requires_grad .or. basis_weights%requires_grad) then
       c%requires_grad    = .true.
       c%is_forward       = input%is_forward .or. basis_weights%is_forward
       c%operation        = 'ono_encode'
       c%left_operand     => input
       c%right_operand    => basis_weights
       c%owns_left_operand  = input%is_temporary
       c%owns_right_operand = basis_weights%is_temporary
    end if

  end function ono_encode
!-------------------------------------------------------------------------------
  function get_partial_ono_encode_input(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%left_operand%val))
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_ono_encode_input
!-------------------------------------------------------------------------------
  pure subroutine get_partial_ono_encode_input_val( &
       this, upstream_grad, output)
    !! dL/du = Q @ upstream [n, batch]
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: n, k, i, j
    real(real32), allocatable :: B(:,:), Q(:,:)
    real(real32) :: norm_val, proj

    n = this%indices(1)
    k = this%indices(2)

    ! Recompute Q from B
    allocate(B(n, k), Q(n, k))
    B = reshape(this%right_operand%val(:,1), [n, k])
    Q = B
    do j = 1, k
       do i = 1, j - 1
          proj = dot_product(Q(:,i), Q(:,j))
          Q(:,j) = Q(:,j) - proj * Q(:,i)
       end do
       norm_val = sqrt(dot_product(Q(:,j), Q(:,j)))
       if(norm_val .gt. 1.0e-12_real32) then
          Q(:,j) = Q(:,j) / norm_val
       else
          Q(:,j) = 0.0_real32
       end if
    end do

    ! dL/du = Q @ upstream
    output = matmul(Q, upstream_grad)

    deallocate(B, Q)

  end subroutine get_partial_ono_encode_input_val
!-------------------------------------------------------------------------------
  function get_partial_ono_encode_basis(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%right_operand%val))
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_ono_encode_basis
!-------------------------------------------------------------------------------
  pure subroutine get_partial_ono_encode_basis_val( &
       this, upstream_grad, output)
    !! dL/dB per sample through Gram-Schmidt backward.
    !!
    !! For encode y = Q^T @ u:
    !!   dL/dQ from sample s: u(:,s) @ upstream(:,s)^T  → [n, k]
    !!   dL/dB from sample s: gs_backward(B, dL/dQ_s)   → [n, k]
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: n, k, s, i, j, num_samples
    real(real32), allocatable :: B(:,:), Q(:,:), R(:,:)
    real(real32), allocatable :: dQ(:,:), dQ_work(:,:), dB(:,:)
    real(real32), allocatable :: dv(:), v_recon(:)
    real(real32) :: norm_j, dprod, dR_ij, proj

    n = this%indices(1)
    k = this%indices(2)
    num_samples = size(upstream_grad, 2)

    ! Recompute Q and R from B via modified Gram-Schmidt
    allocate(B(n, k), Q(n, k), R(k, k))
    B = reshape(this%right_operand%val(:,1), [n, k])
    Q = B
    R = 0.0_real32
    do j = 1, k
       do i = 1, j - 1
          R(i,j) = dot_product(Q(:,i), Q(:,j))
          Q(:,j) = Q(:,j) - R(i,j) * Q(:,i)
       end do
       R(j,j) = sqrt(dot_product(Q(:,j), Q(:,j)))
       if(R(j,j) .gt. 1.0e-12_real32) then
          Q(:,j) = Q(:,j) / R(j,j)
       else
          Q(:,j) = 0.0_real32
       end if
    end do

    allocate(dQ(n, k), dQ_work(n, k), dB(n, k))
    allocate(dv(n), v_recon(n))

    output = 0.0_real32

    do s = 1, num_samples
       ! dL/dQ for this sample: u(:,s) outer upstream(:,s)
       ! dQ[j_n, i_k] = u(j_n, s) * upstream(i_k, s)
       do j = 1, k
          do i = 1, n
             dQ(i, j) = this%left_operand%val(i, s) * upstream_grad(j, s)
          end do
       end do

       ! Gram-Schmidt backward: dQ -> dB
       dQ_work = dQ
       dB = 0.0_real32

       do j = k, 1, -1
          norm_j = R(j, j)
          if(norm_j .le. 1.0e-12_real32) then
             dB(:,j) = 0.0_real32
             cycle
          end if

          ! Backward through normalization
          dprod = dot_product(dQ_work(:,j), Q(:,j))
          dv = (dQ_work(:,j) - dprod * Q(:,j)) / norm_j

          ! Reconstruct v before normalization
          v_recon = norm_j * Q(:,j)

          ! Backward through projections (reverse order)
          do i = j-1, 1, -1
             v_recon = v_recon + R(i,j) * Q(:,i)
             dR_ij = -dot_product(dv, Q(:,i))
             dQ_work(:,i) = dQ_work(:,i) - R(i,j) * dv
             dQ_work(:,i) = dQ_work(:,i) + dR_ij * v_recon
             dv = dv + dR_ij * Q(:,i)
          end do

          dB(:,j) = dv
       end do

       output(:, s) = reshape(dB, [n*k])
    end do

    deallocate(B, Q, R, dQ, dQ_work, dB, dv, v_recon)

  end subroutine get_partial_ono_encode_basis_val
!###############################################################################


!###############################################################################
  module function ono_decode( &
       mixed, basis_weights, num_inputs, num_basis &
  ) result(c)
    !! Decode through an orthogonalised basis.
    !!
    !! Forward: y = Q(B) @ x   [n, batch]
    !!   Q = modified_gram_schmidt(B), B [n x k] from basis_weights
    !!
    !! left_operand  → mixed x          [k, batch]
    !! right_operand → basis weights B   [n*k, 1]
    !! output        → decoded           [n, batch]
    implicit none

    class(array_type), intent(in), target :: mixed
    class(array_type), intent(in), target :: basis_weights
    integer, intent(in) :: num_inputs, num_basis
    type(array_type), pointer :: c

    integer :: num_samples, n, k, i, j
    real(real32), allocatable :: B(:,:), Q(:,:)
    real(real32) :: norm_val, proj

    n = num_inputs
    k = num_basis
    num_samples = size(mixed%val, 2)

    ! Modified Gram-Schmidt: B -> Q
    allocate(B(n, k), Q(n, k))
    B = reshape(basis_weights%val(:, 1), [n, k])
    Q = B
    do j = 1, k
       do i = 1, j - 1
          proj = dot_product(Q(:,i), Q(:,j))
          Q(:,j) = Q(:,j) - proj * Q(:,i)
       end do
       norm_val = sqrt(dot_product(Q(:,j), Q(:,j)))
       if(norm_val .gt. 1.0e-12_real32)then
          Q(:,j) = Q(:,j) / norm_val
       else
          Q(:,j) = 0.0_real32
       end if
    end do

    ! Forward: y = Q @ x
    c => mixed%create_result(array_shape=[n, num_samples])
    c%val = matmul(Q, mixed%val)

    deallocate(B, Q)

    ! Store metadata
    allocate(c%indices(2))
    c%indices = [n, k]

    c%get_partial_left     => get_partial_ono_decode_mixed
    c%get_partial_right    => get_partial_ono_decode_basis
    c%get_partial_left_val => get_partial_ono_decode_mixed_val
    c%get_partial_right_val => get_partial_ono_decode_basis_val
    if(mixed%requires_grad .or. basis_weights%requires_grad) then
       c%requires_grad    = .true.
       c%is_forward       = mixed%is_forward .or. basis_weights%is_forward
       c%operation        = 'ono_decode'
       c%left_operand     => mixed
       c%right_operand    => basis_weights
       c%owns_left_operand  = mixed%is_temporary
       c%owns_right_operand = basis_weights%is_temporary
    end if

  end function ono_decode
!-------------------------------------------------------------------------------
  function get_partial_ono_decode_mixed(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%left_operand%val))
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_ono_decode_mixed
!-------------------------------------------------------------------------------
  pure subroutine get_partial_ono_decode_mixed_val( &
       this, upstream_grad, output)
    !! dL/dx = Q^T @ upstream  [k, batch]
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: n, k, i, j
    real(real32), allocatable :: B(:,:), Q(:,:), QT(:,:)
    real(real32) :: norm_val, proj

    n = this%indices(1)
    k = this%indices(2)

    ! Recompute Q from B
    allocate(B(n, k), Q(n, k), QT(k, n))
    B = reshape(this%right_operand%val(:,1), [n, k])
    Q = B
    do j = 1, k
       do i = 1, j - 1
          proj = dot_product(Q(:,i), Q(:,j))
          Q(:,j) = Q(:,j) - proj * Q(:,i)
       end do
       norm_val = sqrt(dot_product(Q(:,j), Q(:,j)))
       if(norm_val .gt. 1.0e-12_real32) then
          Q(:,j) = Q(:,j) / norm_val
       else
          Q(:,j) = 0.0_real32
       end if
    end do

    ! Transpose
    do j = 1, n
       do i = 1, k
          QT(i, j) = Q(j, i)
       end do
    end do

    output = matmul(QT, upstream_grad)

    deallocate(B, Q, QT)

  end subroutine get_partial_ono_decode_mixed_val
!-------------------------------------------------------------------------------
  function get_partial_ono_decode_basis(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape=shape(this%right_operand%val))
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_ono_decode_basis
!-------------------------------------------------------------------------------
  pure subroutine get_partial_ono_decode_basis_val( &
       this, upstream_grad, output)
    !! dL/dB per sample through Gram-Schmidt backward.
    !!
    !! For decode y = Q @ x:
    !!   dL/dQ from sample s: upstream(:,s) @ x(:,s)^T  → [n, k]
    !!   dL/dB from sample s: gs_backward(B, dL/dQ_s)   → [n, k]
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in)  :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: n, k, s, i, j, num_samples
    real(real32), allocatable :: B(:,:), Q(:,:), R(:,:)
    real(real32), allocatable :: dQ(:,:), dQ_work(:,:), dB(:,:)
    real(real32), allocatable :: dv(:), v_recon(:)
    real(real32) :: norm_j, dprod, dR_ij, proj

    n = this%indices(1)
    k = this%indices(2)
    num_samples = size(upstream_grad, 2)

    ! Recompute Q and R from B
    allocate(B(n, k), Q(n, k), R(k, k))
    B = reshape(this%right_operand%val(:,1), [n, k])
    Q = B
    R = 0.0_real32
    do j = 1, k
       do i = 1, j - 1
          R(i,j) = dot_product(Q(:,i), Q(:,j))
          Q(:,j) = Q(:,j) - R(i,j) * Q(:,i)
       end do
       R(j,j) = sqrt(dot_product(Q(:,j), Q(:,j)))
       if(R(j,j) .gt. 1.0e-12_real32) then
          Q(:,j) = Q(:,j) / R(j,j)
       else
          Q(:,j) = 0.0_real32
       end if
    end do

    allocate(dQ(n, k), dQ_work(n, k), dB(n, k))
    allocate(dv(n), v_recon(n))

    output = 0.0_real32

    do s = 1, num_samples
       ! dL/dQ for this sample: upstream(:,s) outer x(:,s)
       ! dQ[i_n, j_k] = upstream(i_n, s) * x(j_k, s)
       do j = 1, k
          do i = 1, n
             dQ(i, j) = upstream_grad(i, s) * this%left_operand%val(j, s)
          end do
       end do

       ! Gram-Schmidt backward: dQ -> dB
       dQ_work = dQ
       dB = 0.0_real32

       do j = k, 1, -1
          norm_j = R(j, j)
          if(norm_j .le. 1.0e-12_real32) then
             dB(:,j) = 0.0_real32
             cycle
          end if

          ! Backward through normalization
          dprod = dot_product(dQ_work(:,j), Q(:,j))
          dv = (dQ_work(:,j) - dprod * Q(:,j)) / norm_j

          ! Reconstruct v before normalization
          v_recon = norm_j * Q(:,j)

          ! Backward through projections (reverse order)
          do i = j-1, 1, -1
             v_recon = v_recon + R(i,j) * Q(:,i)
             dR_ij = -dot_product(dv, Q(:,i))
             dQ_work(:,i) = dQ_work(:,i) - R(i,j) * dv
             dQ_work(:,i) = dQ_work(:,i) + dR_ij * v_recon
             dv = dv + dR_ij * Q(:,i)
          end do

          dB(:,j) = dv
       end do

       output(:, s) = reshape(dB, [n*k])
    end do

    deallocate(B, Q, R, dQ, dQ_work, dB, dv, v_recon)

  end subroutine get_partial_ono_decode_basis_val
!###############################################################################

end submodule athena__diffstruc_extd_submodule_nop
