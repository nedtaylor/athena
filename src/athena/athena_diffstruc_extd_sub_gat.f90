submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_msgpass_gat
  !! Submodule containing GAT (Graph Attention Network) propagation operations
  !!
  !! Implements multi-head attention-weighted message passing:
  !! For each head k and node i:
  !!   e_ij^k = LeakyReLU(a_left_k^T Wh_i^k + a_right_k^T Wh_j^k)
  !!   alpha_ij^k = softmax_j(e_ij^k)
  !!   h'_i^k = sum_j alpha_ij^k * Wh_j^k
  !!
  !! Heads are concatenated or averaged in the output.
  !! Both projected_features (left_operand) and attn_params (right_operand)
  !! participate in the autodiff graph with gradient computation.
  !!
  !! direction layout: [negative_slope, num_heads_real, concat_flag]

contains

!###############################################################################
  module function gat_propagate( &
       projected_features, attn_params, &
       adj_ia, adj_ja, negative_slope, &
       num_heads, concat_heads &
  ) result(c)
    class(array_type), intent(in), target :: projected_features
    !! Projected features: shape (f_per_head * num_heads, N)
    class(array_type), intent(in), target :: attn_params
    !! Attention parameters flattened: val(f_per_head * 2 * num_heads, 1)
    !! Layout: [a_l^1, ..., a_l^K, a_r^1, ..., a_r^K] (column-major)
    integer, dimension(:), intent(in) :: adj_ia
    integer, dimension(:,:), intent(in) :: adj_ja
    real(real32), intent(in) :: negative_slope
    integer, intent(in) :: num_heads
    logical, intent(in) :: concat_heads
    type(array_type), pointer :: c

    integer :: num_nodes, f_per_head, f_total_in, f_out
    integer :: v, w, j_node, edge_start, edge_end, k, f_offset
    real(real32) :: e_ij, max_e, sum_exp
    real(real32), dimension(:), allocatable :: attn_src, attn_dst
    real(real32), dimension(:,:), allocatable :: attn_left, attn_right

    f_total_in = size(projected_features%val, 1)
    num_nodes = size(projected_features%val, 2)
    f_per_head = f_total_in / num_heads

    if(concat_heads) then
       f_out = f_total_in
    else
       f_out = f_per_head
    end if

    ! Extract left/right attention from flat params
    allocate(attn_left(f_per_head, num_heads))
    allocate(attn_right(f_per_head, num_heads))
    attn_left = reshape( &
         attn_params%val(1:f_per_head*num_heads, 1), &
         [f_per_head, num_heads])
    attn_right = reshape( &
         attn_params%val(f_per_head*num_heads+1:, 1), &
         [f_per_head, num_heads])

    ! Create result using create_result for autodiff graph linkage
    c => projected_features%create_result()

    ! If averaging, we need to resize val
    if(.not. concat_heads) then
       deallocate(c%val)
       allocate(c%val(f_per_head, num_nodes))
       if(allocated(c%shape)) deallocate(c%shape)
       allocate(c%shape(1))
       c%shape = [f_per_head]
       c%size = f_per_head * num_nodes
    end if
    c%val = 0._real32

    ! Process each head
    allocate(attn_src(num_nodes), attn_dst(num_nodes))

    do k = 1, num_heads
       f_offset = (k-1) * f_per_head

       ! Compute per-node attention scores for head k
       do v = 1, num_nodes
          attn_src(v) = dot_product( &
               attn_left(:,k), &
               projected_features%val(f_offset+1:f_offset+f_per_head, v))
          attn_dst(v) = dot_product( &
               attn_right(:,k), &
               projected_features%val(f_offset+1:f_offset+f_per_head, v))
       end do

       ! Attention-weighted aggregation for each node
       do v = 1, num_nodes
          edge_start = adj_ia(v)
          edge_end = adj_ia(v+1) - 1
          if(edge_end < edge_start) cycle

          ! Find max for numerical stability
          max_e = -huge(1._real32)
          do w = edge_start, edge_end
             j_node = adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             max_e = max(max_e, e_ij)
          end do

          ! Softmax denominator
          sum_exp = 0._real32
          do w = edge_start, edge_end
             j_node = adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             sum_exp = sum_exp + exp(e_ij - max_e)
          end do

          ! Weighted sum
          if(concat_heads) then
             do w = edge_start, edge_end
                j_node = adj_ja(1, w)
                e_ij = attn_src(v) + attn_dst(j_node)
                if(e_ij < 0._real32) e_ij = e_ij * negative_slope
                c%val(f_offset+1:f_offset+f_per_head, v) = &
                     c%val(f_offset+1:f_offset+f_per_head, v) + &
                     (exp(e_ij - max_e) / sum_exp) * &
                     projected_features%val( &
                          f_offset+1:f_offset+f_per_head, j_node)
             end do
          else
             do w = edge_start, edge_end
                j_node = adj_ja(1, w)
                e_ij = attn_src(v) + attn_dst(j_node)
                if(e_ij < 0._real32) e_ij = e_ij * negative_slope
                c%val(:, v) = c%val(:, v) + &
                     (exp(e_ij - max_e) / sum_exp) * &
                     projected_features%val( &
                          f_offset+1:f_offset+f_per_head, j_node)
             end do
          end if
       end do
    end do

    deallocate(attn_src, attn_dst, attn_left, attn_right)

    ! Average if not concatenating
    if(.not. concat_heads) then
       c%val = c%val / real(num_heads, real32)
    end if

    ! Store graph topology and metadata for backward pass
    ! indices layout: [neg_slope_bits, num_heads, concat_flag, adj_ia...]
    if(allocated(c%indices)) deallocate(c%indices)
    allocate(c%indices(3 + size(adj_ia)))
    c%indices(1) = transfer(negative_slope, 1)
    c%indices(2) = num_heads
    if(concat_heads) then
       c%indices(3) = 1
    else
       c%indices(3) = 0
    end if
    c%indices(4:) = adj_ia
    c%adj_ja = adj_ja

    ! Set gradient computation functions
    c%get_partial_left => get_partial_gat_propagate_left
    c%get_partial_left_val => get_partial_gat_propagate_left_val
    c%get_partial_right => get_partial_gat_propagate_right
    c%get_partial_right_val => get_partial_gat_propagate_right_val

    if(projected_features%requires_grad .or. attn_params%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = projected_features%is_forward .or. attn_params%is_forward
       c%operation = 'gat_propagate'
       c%left_operand => projected_features
       c%right_operand => attn_params
       c%owns_left_operand = projected_features%is_temporary
       c%owns_right_operand = attn_params%is_temporary
    end if
  end function gat_propagate
!###############################################################################


!###############################################################################
  function get_partial_gat_propagate_left(this, upstream_grad) result(output)
    !! Backward pass for gat_propagate wrt projected_features
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    integer :: num_nodes, f_per_head, num_heads, f_total_in
    integer :: v, w, j_node, edge_start, edge_end, f, k, f_offset
    real(real32) :: e_ij, max_e, sum_exp, alpha_ij, negative_slope
    logical :: is_concat
    real(real32), dimension(:), allocatable :: attn_src, attn_dst
    real(real32), dimension(:,:), allocatable :: attn_left, attn_right

    f_total_in = size(this%left_operand%val, 1)
    num_nodes = size(this%left_operand%val, 2)

    ! Read metadata from indices prefix
    negative_slope = transfer(this%indices(1), 1._real32)
    num_heads = this%indices(2)
    is_concat = (this%indices(3) == 1)
    f_per_head = f_total_in / num_heads

    ! Get attention from right_operand
    allocate(attn_left(f_per_head, num_heads))
    allocate(attn_right(f_per_head, num_heads))
    attn_left = reshape( &
         this%right_operand%val(1:f_per_head*num_heads, 1), &
         [f_per_head, num_heads])
    attn_right = reshape( &
         this%right_operand%val(f_per_head*num_heads+1:, 1), &
         [f_per_head, num_heads])

    ! Output gradient has same shape as projected_features
    allocate(output%val(f_total_in, num_nodes))
    allocate(output%shape(2))
    output%shape = [f_total_in, num_nodes]
    output%rank = 2
    output%size = f_total_in * num_nodes
    output%allocated = .true.
    output%val = 0._real32

    allocate(attn_src(num_nodes), attn_dst(num_nodes))

    do k = 1, num_heads
       f_offset = (k-1) * f_per_head

       ! Recompute attention scores for head k
       do v = 1, num_nodes
          attn_src(v) = dot_product( &
               attn_left(:,k), &
               this%left_operand%val(f_offset+1:f_offset+f_per_head, v))
          attn_dst(v) = dot_product( &
               attn_right(:,k), &
               this%left_operand%val(f_offset+1:f_offset+f_per_head, v))
       end do

       do v = 1, num_nodes
          edge_start = this%indices(3 + v)
          edge_end = this%indices(3 + v + 1) - 1
          if(edge_end < edge_start) cycle

          max_e = -huge(1._real32)
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             max_e = max(max_e, e_ij)
          end do

          sum_exp = 0._real32
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             sum_exp = sum_exp + exp(e_ij - max_e)
          end do

          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             alpha_ij = exp(e_ij - max_e) / sum_exp

             if(is_concat) then
                do f = 1, f_per_head
                   output%val(f_offset+f, j_node) = &
                        output%val(f_offset+f, j_node) + &
                        alpha_ij * upstream_grad%val(f_offset+f, v)
                end do
             else
                do f = 1, f_per_head
                   output%val(f_offset+f, j_node) = &
                        output%val(f_offset+f, j_node) + &
                        alpha_ij * upstream_grad%val(f, v) / &
                        real(num_heads, real32)
                end do
             end if
          end do
       end do
    end do

    deallocate(attn_src, attn_dst, attn_left, attn_right)

  end function get_partial_gat_propagate_left
!###############################################################################


!###############################################################################
  pure subroutine get_partial_gat_propagate_left_val( &
       this, upstream_grad, output &
  )
    !! Value-only backward pass for gat_propagate wrt projected_features
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: num_nodes, f_per_head, num_heads, f_total_in
    integer :: v, w, j_node, edge_start, edge_end, f, k, f_offset
    real(real32) :: e_ij, max_e, sum_exp, alpha_ij, negative_slope
    logical :: is_concat
    real(real32), dimension(:), allocatable :: attn_src, attn_dst
    real(real32), dimension(:,:), allocatable :: attn_left, attn_right

    f_total_in = size(this%left_operand%val, 1)
    num_nodes = size(this%left_operand%val, 2)

    ! Read metadata from indices prefix
    negative_slope = transfer(this%indices(1), 1._real32)
    num_heads = this%indices(2)
    is_concat = (this%indices(3) == 1)
    f_per_head = f_total_in / num_heads

    ! Get attention from right_operand
    allocate(attn_left(f_per_head, num_heads))
    allocate(attn_right(f_per_head, num_heads))
    attn_left = reshape( &
         this%right_operand%val(1:f_per_head*num_heads, 1), &
         [f_per_head, num_heads])
    attn_right = reshape( &
         this%right_operand%val(f_per_head*num_heads+1:, 1), &
         [f_per_head, num_heads])

    output = 0._real32

    allocate(attn_src(num_nodes), attn_dst(num_nodes))

    do k = 1, num_heads
       f_offset = (k-1) * f_per_head

       do v = 1, num_nodes
          attn_src(v) = dot_product( &
               attn_left(:,k), &
               this%left_operand%val(f_offset+1:f_offset+f_per_head, v))
          attn_dst(v) = dot_product( &
               attn_right(:,k), &
               this%left_operand%val(f_offset+1:f_offset+f_per_head, v))
       end do

       do v = 1, num_nodes
          edge_start = this%indices(3 + v)
          edge_end = this%indices(3 + v + 1) - 1
          if(edge_end < edge_start) cycle

          max_e = -huge(1._real32)
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             max_e = max(max_e, e_ij)
          end do

          sum_exp = 0._real32
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             sum_exp = sum_exp + exp(e_ij - max_e)
          end do

          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             alpha_ij = exp(e_ij - max_e) / sum_exp

             if(is_concat) then
                do f = 1, f_per_head
                   output(f_offset+f, j_node) = &
                        output(f_offset+f, j_node) + &
                        alpha_ij * upstream_grad(f_offset+f, v)
                end do
             else
                do f = 1, f_per_head
                   output(f_offset+f, j_node) = &
                        output(f_offset+f, j_node) + &
                        alpha_ij * upstream_grad(f, v) / &
                        real(num_heads, real32)
                end do
             end if
          end do
       end do
    end do

    deallocate(attn_src, attn_dst, attn_left, attn_right)

  end subroutine get_partial_gat_propagate_left_val
!###############################################################################


!###############################################################################
  function get_partial_gat_propagate_right(this, upstream_grad) result(output)
    !! Backward pass for gat_propagate wrt attention parameters
    !!
    !! Computes dL/d(attn_params) using the softmax-attention chain rule:
    !!   dL/d(alpha_ij^k) = <g_i^k, Wh_j^k>
    !!   dL/d(e_ij^k) = alpha_ij * (dL/d(alpha_ij) - delta_i)
    !!   where delta_i = sum_j alpha_ij * dL/d(alpha_ij)
    !!   dL/d(a_l^k) = sum_ij beta_ij * Wh_i^k
    !!   dL/d(a_r^k) = sum_ij beta_ij * Wh_j^k
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    integer :: num_nodes, f_per_head, num_heads, f_total_in, n_attn
    integer :: v, w, j_node, edge_start, edge_end, f, k, f_offset
    integer :: left_base, right_base
    real(real32) :: e_ij, z_ij, max_e, sum_exp, alpha_ij, negative_slope
    real(real32) :: dl_dalpha, delta_v, beta_ij, sum_beta
    logical :: is_concat
    real(real32), dimension(:), allocatable :: attn_src, attn_dst
    real(real32), dimension(:,:), allocatable :: attn_left, attn_right

    f_total_in = size(this%left_operand%val, 1)
    num_nodes = size(this%left_operand%val, 2)

    negative_slope = transfer(this%indices(1), 1._real32)
    num_heads = this%indices(2)
    is_concat = (this%indices(3) == 1)
    f_per_head = f_total_in / num_heads
    n_attn = f_per_head * 2 * num_heads

    ! Get attention from right_operand
    allocate(attn_left(f_per_head, num_heads))
    allocate(attn_right(f_per_head, num_heads))
    attn_left = reshape( &
         this%right_operand%val(1:f_per_head*num_heads, 1), &
         [f_per_head, num_heads])
    attn_right = reshape( &
         this%right_operand%val(f_per_head*num_heads+1:, 1), &
         [f_per_head, num_heads])

    ! Output gradient has same shape as attn_params: (n_attn, num_nodes)
    ! Each column is the per-node contribution; summed across dim 2 gives total
    allocate(output%val(n_attn, num_nodes))
    allocate(output%shape(1))
    output%shape = [n_attn]
    output%rank = 1
    output%size = n_attn * num_nodes
    output%allocated = .true.
    output%val = 0._real32

    allocate(attn_src(num_nodes), attn_dst(num_nodes))

    do k = 1, num_heads
       f_offset = (k-1) * f_per_head
       left_base = (k-1) * f_per_head
       right_base = f_per_head * num_heads + (k-1) * f_per_head

       ! Recompute attention scores for head k
       do v = 1, num_nodes
          attn_src(v) = dot_product( &
               attn_left(:,k), &
               this%left_operand%val(f_offset+1:f_offset+f_per_head, v))
          attn_dst(v) = dot_product( &
               attn_right(:,k), &
               this%left_operand%val(f_offset+1:f_offset+f_per_head, v))
       end do

       do v = 1, num_nodes
          edge_start = this%indices(3 + v)
          edge_end = this%indices(3 + v + 1) - 1
          if(edge_end < edge_start) cycle

          ! Softmax: max for stability
          max_e = -huge(1._real32)
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             max_e = max(max_e, e_ij)
          end do

          sum_exp = 0._real32
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             sum_exp = sum_exp + exp(e_ij - max_e)
          end do

          ! Compute delta_v = sum_j alpha_vj * dL/d(alpha_vj)
          delta_v = 0._real32
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             alpha_ij = exp(e_ij - max_e) / sum_exp

             ! dL/d(alpha_vj^k) = <g_v^k, Wh_j^k>
             if(is_concat) then
                dl_dalpha = dot_product( &
                     upstream_grad%val(f_offset+1:f_offset+f_per_head, v), &
                     this%left_operand%val( &
                          f_offset+1:f_offset+f_per_head, j_node))
             else
                dl_dalpha = dot_product( &
                     upstream_grad%val(1:f_per_head, v), &
                     this%left_operand%val( &
                          f_offset+1:f_offset+f_per_head, j_node)) / &
                     real(num_heads, real32)
             end if
             delta_v = delta_v + alpha_ij * dl_dalpha
          end do

          ! Compute per-edge beta and accumulate attention gradients
          sum_beta = 0._real32
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             z_ij = attn_src(v) + attn_dst(j_node)
             e_ij = z_ij
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             alpha_ij = exp(e_ij - max_e) / sum_exp

             if(is_concat) then
                dl_dalpha = dot_product( &
                     upstream_grad%val(f_offset+1:f_offset+f_per_head, v), &
                     this%left_operand%val( &
                          f_offset+1:f_offset+f_per_head, j_node))
             else
                dl_dalpha = dot_product( &
                     upstream_grad%val(1:f_per_head, v), &
                     this%left_operand%val( &
                          f_offset+1:f_offset+f_per_head, j_node)) / &
                     real(num_heads, real32)
             end if

             ! dL/d(e_ij) * leaky_relu'(z_ij)
             beta_ij = alpha_ij * (dl_dalpha - delta_v)
             if(z_ij < 0._real32) then
                beta_ij = beta_ij * negative_slope
             end if

             sum_beta = sum_beta + beta_ij

             ! Right attention: dL/d(a_r^k) += beta_ij * Wh_j^k
             do f = 1, f_per_head
                output%val(right_base+f, v) = &
                     output%val(right_base+f, v) + &
                     beta_ij * this%left_operand%val(f_offset+f, j_node)
             end do
          end do

          ! Left attention: dL/d(a_l^k) += sum_beta * Wh_v^k
          do f = 1, f_per_head
             output%val(left_base+f, v) = &
                  output%val(left_base+f, v) + &
                  sum_beta * this%left_operand%val(f_offset+f, v)
          end do
       end do
    end do

    deallocate(attn_src, attn_dst, attn_left, attn_right)

  end function get_partial_gat_propagate_right
!###############################################################################


!###############################################################################
  pure subroutine get_partial_gat_propagate_right_val( &
       this, upstream_grad, output &
  )
    !! Value-only backward pass for gat_propagate wrt attention parameters
    !!
    !! output shape: (f_per_head * 2 * num_heads, num_nodes)
    !! Each column is the per-node contribution to the attention gradient.
    !! The caller sums across dim 2 for the total gradient.
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: num_nodes, f_per_head, num_heads, f_total_in
    integer :: v, w, j_node, edge_start, edge_end, f, k, f_offset
    integer :: left_base, right_base
    real(real32) :: e_ij, z_ij, max_e, sum_exp, alpha_ij, negative_slope
    real(real32) :: dl_dalpha, delta_v, beta_ij, sum_beta
    logical :: is_concat
    real(real32), dimension(:), allocatable :: attn_src, attn_dst
    real(real32), dimension(:,:), allocatable :: attn_left, attn_right

    f_total_in = size(this%left_operand%val, 1)
    num_nodes = size(this%left_operand%val, 2)

    negative_slope = transfer(this%indices(1), 1._real32)
    num_heads = this%indices(2)
    is_concat = (this%indices(3) == 1)
    f_per_head = f_total_in / num_heads

    ! Get attention from right_operand
    allocate(attn_left(f_per_head, num_heads))
    allocate(attn_right(f_per_head, num_heads))
    attn_left = reshape( &
         this%right_operand%val(1:f_per_head*num_heads, 1), &
         [f_per_head, num_heads])
    attn_right = reshape( &
         this%right_operand%val(f_per_head*num_heads+1:, 1), &
         [f_per_head, num_heads])

    output = 0._real32

    allocate(attn_src(num_nodes), attn_dst(num_nodes))

    do k = 1, num_heads
       f_offset = (k-1) * f_per_head
       left_base = (k-1) * f_per_head
       right_base = f_per_head * num_heads + (k-1) * f_per_head

       do v = 1, num_nodes
          attn_src(v) = dot_product( &
               attn_left(:,k), &
               this%left_operand%val(f_offset+1:f_offset+f_per_head, v))
          attn_dst(v) = dot_product( &
               attn_right(:,k), &
               this%left_operand%val(f_offset+1:f_offset+f_per_head, v))
       end do

       do v = 1, num_nodes
          edge_start = this%indices(3 + v)
          edge_end = this%indices(3 + v + 1) - 1
          if(edge_end < edge_start) cycle

          max_e = -huge(1._real32)
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             max_e = max(max_e, e_ij)
          end do

          sum_exp = 0._real32
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             sum_exp = sum_exp + exp(e_ij - max_e)
          end do

          ! delta_v = sum_j alpha_vj * dL/d(alpha_vj)
          delta_v = 0._real32
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             e_ij = attn_src(v) + attn_dst(j_node)
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             alpha_ij = exp(e_ij - max_e) / sum_exp

             if(is_concat) then
                dl_dalpha = dot_product( &
                     upstream_grad(f_offset+1:f_offset+f_per_head, v), &
                     this%left_operand%val( &
                          f_offset+1:f_offset+f_per_head, j_node))
             else
                dl_dalpha = dot_product( &
                     upstream_grad(1:f_per_head, v), &
                     this%left_operand%val( &
                          f_offset+1:f_offset+f_per_head, j_node)) / &
                     real(num_heads, real32)
             end if
             delta_v = delta_v + alpha_ij * dl_dalpha
          end do

          ! Compute beta and accumulate
          sum_beta = 0._real32
          do w = edge_start, edge_end
             j_node = this%adj_ja(1, w)
             z_ij = attn_src(v) + attn_dst(j_node)
             e_ij = z_ij
             if(e_ij < 0._real32) e_ij = e_ij * negative_slope
             alpha_ij = exp(e_ij - max_e) / sum_exp

             if(is_concat) then
                dl_dalpha = dot_product( &
                     upstream_grad(f_offset+1:f_offset+f_per_head, v), &
                     this%left_operand%val( &
                          f_offset+1:f_offset+f_per_head, j_node))
             else
                dl_dalpha = dot_product( &
                     upstream_grad(1:f_per_head, v), &
                     this%left_operand%val( &
                          f_offset+1:f_offset+f_per_head, j_node)) / &
                     real(num_heads, real32)
             end if

             beta_ij = alpha_ij * (dl_dalpha - delta_v)
             if(z_ij < 0._real32) then
                beta_ij = beta_ij * negative_slope
             end if

             sum_beta = sum_beta + beta_ij

             ! Right attention gradient
             do f = 1, f_per_head
                output(right_base+f, v) = &
                     output(right_base+f, v) + &
                     beta_ij * this%left_operand%val(f_offset+f, j_node)
             end do
          end do

          ! Left attention gradient
          do f = 1, f_per_head
             output(left_base+f, v) = &
                  output(left_base+f, v) + &
                  sum_beta * this%left_operand%val(f_offset+f, v)
          end do
       end do
    end do

    deallocate(attn_src, attn_dst, attn_left, attn_right)

  end subroutine get_partial_gat_propagate_right_val
!###############################################################################

end submodule athena__diffstruc_extd_submodule_msgpass_gat
