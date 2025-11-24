submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_msgpass_kipf
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function kipf_propagate(vertex_features, adj_ia, adj_ja) result(c)
    !! Propagate values from one autodiff array to another
    class(array_type), intent(in), target :: vertex_features
    integer, dimension(:), intent(in) :: adj_ia
    integer, dimension(:,:), intent(in) :: adj_ja
    type(array_type), pointer :: c

    integer :: v, w
    real(real32) :: coeff

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
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

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
  subroutine get_partial_kipf_propagate_left_val(this, upstream_grad, output)
    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: v, w

    output = 0._real32
    do concurrent(v=1:size(upstream_grad,2))
       do w = this%indices(v), this%indices(v+1)-1
          output(:,this%adj_ja(1,w)) = output(:,this%adj_ja(1,w)) + &
               [ upstream_grad(:, v) ]
       end do
    end do

  end subroutine get_partial_kipf_propagate_left_val
!###############################################################################


!   module function kipf_update(a, weight, adj_ia) result(c)
!     !! Update the message passing layer
!     class(array_type), intent(in), target :: a
!     class(array_type), intent(in), target :: weight
!     integer, dimension(:), intent(in) :: adj_ia
!     type(array_type), pointer :: c

!     integer :: v, i, d
!     integer :: interval
!     real(real32), pointer :: w_ptr(:,:)

!     c => a%create_result(array_shape=[weight%shape(1), size(a%val,2)])
!     interval = weight%shape(1) * weight%shape(2)
!     do v = 1, size(a%val,2)
!        c%val(:,v) = matmul(w_ptr, a%val(:,v) / real(d, real32))
!     end do

!     c%indices = adj_ia
!     ! c%get_partial_left => get_partial_kipf_update_weight
!     ! c%get_partial_right => get_partial_kipf_update
!     if(a%requires_grad) then
!        c%requires_grad = .true.
!        c%operation = 'kipf_update'
!        c%right_operand => a
!        c%left_operand => weight
!     end if
!   end function kipf_update


! !###############################################################################
!   function reverse_kipf_update(a, weight, adj_ia) result(c)
!     !! Reverse update the message passing layer
!     class(array_type), intent(in), target :: a, weight
!     integer, dimension(:), intent(in) :: adj_ia
!     type(array_type), pointer :: c

!     integer :: v, d
!     integer :: interval
!     real(real32), pointer :: w_ptr(:,:)

!     c => a%create_result(array_shape=[weight%shape(2), size(a%val,2)])
!     interval = weight%shape(1) * weight%shape(2)
!     do v = 1, size(a%val,2)
!        d = max( weight%indices(1), &
!             min( adj_ia(v+1) - adj_ia(v), weight%indices(2) ) ) - weight%indices(1) + 1
!        w_ptr(1:weight%shape(1), 1:weight%shape(2)) => &
!             weight%val(interval*(d-1)+1:interval*d,1)
!        c%val(:,v) = matmul(transpose(w_ptr), a%val(:,v))
!     end do

!     ! if(a%requires_grad) then
!     !    c%requires_grad = .true.
!     !    c%operation = 'reverse_duvenaud_update'
!     !    c%left_operand => a
!     ! end if

!   end function reverse_kipf_update


!###############################################################################
  function reverse_kipf_propagate( &
       a, adj_ia, adj_ja, num_features, num_elements &
  ) result(c)
    !! Reverse propagate values from one autodiff array to another
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: adj_ia
    integer, dimension(:,:), intent(in) :: adj_ja
    integer, dimension(2), intent(in) :: num_features, num_elements
    type(array_type), pointer :: c

    integer :: v, w

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
    ! c%get_partial_left => get_partial_reverse_kipf_propagate_left
    ! c%get_partial_left_val => get_partial_reverse_kipf_propagate_left_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'reverse_kipf_propagate'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function reverse_kipf_propagate
!###############################################################################

end submodule athena__diffstruc_extd_submodule_msgpass_kipf
