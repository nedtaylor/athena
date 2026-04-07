submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_msgpass_kipf
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function kipf_propagate(vertex_features, adj_ia, adj_ja) result(c)
    !! Propagate values from one autodiff array to another
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: vertex_features
    !! Vertex feature tensor
    integer, dimension(:), intent(in) :: adj_ia
    !! CSR row pointers
    integer, dimension(:,:), intent(in) :: adj_ja
    !! CSR neighbour and edge lookup indices
    type(array_type), pointer :: c
    !! Propagated node feature tensor

    ! Local variables
    integer :: v, w
    !! Vertex and adjacency traversal indices
    real(real32) :: coeff
    !! Symmetric normalisation coefficient per edge

    c => vertex_features%create_result()
    ! propagate 1D array by using shape to swap dimensions
    do concurrent(v = 1:size(vertex_features%val,2))
       c%val(:,v) = 0._real32
       do w = adj_ia(v), adj_ia(v+1)-1

          ! if( adj_ja(2,w) .eq. 0 )then
          !    coeff = 1._real32
          ! !else
          ! !   coeff = edge_weights(adj_ja(2,w))
          ! end if
          !coeff = coeff * ( &
          coeff = ( &
               ( adj_ia(v+1) - adj_ia(v) ) * &
               ( adj_ia( adj_ja(1,w) + 1 ) - adj_ia( adj_ja(1,w) ) ) &
          ) ** ( -0.5_real32 )

          c%val(:,v) = c%val(:,v) + coeff * [ vertex_features%val(:, adj_ja(1, w)) ]
       end do
    end do

    c%indices = adj_ia
    c%adj_ja = adj_ja
    c%get_partial_left => get_partial_kipf_propagate_left
    c%get_partial_left_val => get_partial_kipf_propagate_left_val
    if(vertex_features%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = vertex_features%is_forward
       c%operation = 'kipf_propagate'
       c%left_operand => vertex_features
       c%owns_left_operand = vertex_features%is_temporary
    end if
  end function kipf_propagate
!-------------------------------------------------------------------------------
  function get_partial_kipf_propagate_left(this, upstream_grad) result(output)
    !! Gradient of kipf_propagate with respect to vertex features.
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Forward result node containing saved operands
    type(array_type), intent(in) :: upstream_grad
    !! Upstream gradient tensor
    type(array_type) :: output
    !! Gradient tensor for left operand

    output = reverse_kipf_propagate( upstream_grad, &
         this%indices, this%adj_ja, &
         num_features = [ &
              this%left_operand%shape(1), this%right_operand%shape(1) &
         ], &
         num_elements = [ &
              size(this%left_operand%val,2), size(this%right_operand%val,2) &
         ] &
    )

  end function get_partial_kipf_propagate_left
!-------------------------------------------------------------------------------
  pure subroutine get_partial_kipf_propagate_left_val(this, upstream_grad, output)
    !! In-place value gradient for kipf_propagate left operand.
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    !! Forward result node containing saved operands
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    !! Upstream gradient values
    real(real32), dimension(:,:), intent(out) :: output
    !! Output gradient values for left operand

    ! Local variables
    integer :: v, w, i
    !! Loop indices

    output = 0._real32
    do concurrent(v=1:size(upstream_grad,2))
       do w = this%indices(v), this%indices(v+1)-1
          do concurrent(i = 1:size(upstream_grad,1))
             output(i,this%adj_ja(1,w)) = output(i,this%adj_ja(1,w)) + &
                  upstream_grad(i, v)
          end do
       end do
    end do

  end subroutine get_partial_kipf_propagate_left_val
!###############################################################################


!###############################################################################
  function reverse_kipf_propagate( &
       a, adj_ia, adj_ja, num_features, num_elements &
  ) result(c)
    !! Reverse propagate values from one autodiff array to another
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: a
    !! Upstream tensor to reverse-propagate
    integer, dimension(:), intent(in) :: adj_ia
    !! CSR row pointers
    integer, dimension(:,:), intent(in) :: adj_ja
    !! CSR neighbour and edge lookup indices
    integer, dimension(2), intent(in) :: num_features, num_elements
    !! Output feature and element counts
    type(array_type), pointer :: c
    !! Reverse-propagated tensor

    ! Local variables
    integer :: v, w
    !! Loop indices

    c => a%create_result(array_shape=[ &
         num_features(1), num_elements(1) &
    ])
    c%val = 0.0_real32
    do concurrent(v=1:num_elements(1))
       do w = adj_ia(v), adj_ia(v+1)-1
          c%val(:,adj_ja(1,w)) = c%val(:,adj_ja(1,w)) + &
               [ a%val(:, v) ]
       end do
    end do

    c%indices = adj_ia
    c%adj_ja = adj_ja
    c%get_partial_left => get_partial_left_reverse_kipf_propagate
    c%get_partial_left_val => get_partial_left_reverse_kipf_propagate_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'reverse_kipf_propagate'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function reverse_kipf_propagate
!-------------------------------------------------------------------------------
  function get_partial_left_reverse_kipf_propagate( &
       this, upstream_grad &
  ) result(output)
    implicit none
    ! Arguments
    class(array_type), intent(inout) :: this
    !! Forward result node containing saved operands
    type(array_type), intent(in) :: upstream_grad
    !! Upstream gradient tensor
    type(array_type) :: output
    !! Gradient tensor for left operand

    output = kipf_propagate( upstream_grad, &
         this%indices, this%adj_ja &
    )

  end function get_partial_left_reverse_kipf_propagate
!-------------------------------------------------------------------------------
  pure subroutine get_partial_left_reverse_kipf_propagate_val( &
       this, upstream_grad, output &
  )
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    !! Forward result node containing saved operands
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    !! Upstream gradient values
    real(real32), dimension(:,:), intent(out) :: output
    !! Output gradient values for left operand

    ! Local variables
    integer :: v, w, i
    !! Loop indices
    output = 0._real32
    do concurrent(v=1:size(upstream_grad,2))
       do w = this%indices(v), this%indices(v+1)-1
          do concurrent(i = 1:size(upstream_grad,1))
             output(i,this%adj_ja(1,w)) = output(i,this%adj_ja(1,w)) + &
                  upstream_grad(i, v)
          end do
       end do
    end do

  end subroutine get_partial_left_reverse_kipf_propagate_val
!###############################################################################

end submodule athena__diffstruc_extd_submodule_msgpass_kipf
