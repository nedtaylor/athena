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

end submodule athena__diffstruc_extd_submodule_nop
