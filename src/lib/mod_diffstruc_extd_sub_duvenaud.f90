submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_msgpass_duvenaud
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  function duvenaud_propagate(vertex_features, edge_features, adj_ia, adj_ja) result(c)
    !! Propagate values from one autodiff array to another
    implicit none
    class(array_type), intent(in), target :: vertex_features, edge_features
    integer, dimension(:), intent(in) :: adj_ia
    integer, dimension(:,:), intent(in) :: adj_ja
    type(array_type), pointer :: c

    integer :: v, w

    c => vertex_features%create_result( &
         array_shape = [ &
              size(vertex_features%val,1) + size(edge_features%val,1), &
              size(vertex_features%val,2) &
         ] &
    )
    ! propagate 1D array by using shape to swap dimensions
    do concurrent(v=1:size(vertex_features%val,2))
       c%val(:,v) = 0.0_real32
       do w = adj_ia(v), adj_ia(v+1)-1
          c%val(:,v) = c%val(:,v) + [ &
               vertex_features%val(:, adj_ja(1, w)), &
               edge_features%val(:, adj_ja(2, w)) &
          ]
       end do
    end do

    c%indices = adj_ia
    c%adj_ja = adj_ja
    c%get_partial_left => get_partial_duvenaud_propagate_left
    c%get_partial_right => get_partial_duvenaud_propagate_right
    c%get_partial_left_val => get_partial_duvenaud_propagate_left_val
    c%get_partial_right_val => get_partial_duvenaud_propagate_right_val
    if(vertex_features%requires_grad .or. edge_features%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = vertex_features%is_forward .or. edge_features%is_forward
       c%operation = 'duvenaud_propagate'
       c%left_operand => vertex_features
       c%right_operand => edge_features
       c%owns_left_operand = vertex_features%is_temporary
       c%owns_right_operand = edge_features%is_temporary
    end if
  end function duvenaud_propagate
!-------------------------------------------------------------------------------
  function get_partial_duvenaud_propagate_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = reverse_duvenaud_propagate( upstream_grad, &
         this%indices, this%adj_ja, &
         num_features = [ &
              this%left_operand%shape(1), this%right_operand%shape(1) &
         ], &
         num_elements = [ &
              size(this%left_operand%val,2), size(this%right_operand%val,2) &
         ], &
         left = .true. )

  end function get_partial_duvenaud_propagate_left
!-------------------------------------------------------------------------------
  function get_partial_duvenaud_propagate_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = reverse_duvenaud_propagate( upstream_grad, &
         this%indices, this%adj_ja, &
         num_features = [ &
              this%left_operand%shape(1), this%right_operand%shape(1) &
         ], &
         num_elements = [ &
              size(this%left_operand%val,2), size(this%right_operand%val,2) &
         ], &
         left = .false. )

  end function get_partial_duvenaud_propagate_right
!-------------------------------------------------------------------------------
  pure subroutine get_partial_duvenaud_propagate_left_val( &
       this, upstream_grad, output &
  )
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: v, w, num_features, num_elements

    num_features = size(this%left_operand%val,1)
    num_elements = size(this%left_operand%val,2)
    output = 0._real32
    do concurrent(v=1:num_elements)
       do w = this%indices(v), this%indices(v+1)-1
          output(:,this%adj_ja(1,w)) = output(:,this%adj_ja(1,w)) + &
               [ upstream_grad(1:num_features, v) ]
       end do
    end do
  end subroutine get_partial_duvenaud_propagate_left_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_duvenaud_propagate_right_val( &
       this, upstream_grad, output &
  )
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: v, w, num_features, num_elements

    num_features = size(this%left_operand%val,1)
    num_elements = size(this%left_operand%val,2)
    output = 0._real32
    do concurrent(v=1:num_elements)
       do w = this%indices(v), this%indices(v+1)-1
          output(:,this%adj_ja(2,w)) = output(:,this%adj_ja(2,w)) + &
               [ upstream_grad(num_features+1:, v) ]
       end do
    end do
  end subroutine get_partial_duvenaud_propagate_right_val
!###############################################################################


!###############################################################################
  function reverse_duvenaud_propagate( &
       a, adj_ia, adj_ja, num_features, num_elements, left &
  ) result(c)
    !! Reverse propagate values from one autodiff array to another
    implicit none
    class(array_type), intent(in), target :: a
    logical, intent(in) :: left
    integer, dimension(:), intent(in) :: adj_ia
    integer, dimension(:,:), intent(in) :: adj_ja
    integer, dimension(2), intent(in) :: num_features, num_elements
    type(array_type), pointer :: c

    integer :: v, w

    allocate(c)
    if(left)then
       call c%allocate(array_shape=[num_features(1), num_elements(1)])
       c%val = 0.0_real32
       do concurrent(v=1:num_elements(1))
          do w = adj_ia(v), adj_ia(v+1)-1
             c%val(:,adj_ja(1,w)) = c%val(:,adj_ja(1,w)) + &
                  [ a%val(1:num_features(1), v) ]
          end do
       end do
    else
       call c%allocate(array_shape=[num_features(2), num_elements(2)])
       c%val = 0.0_real32
       do concurrent(v=1:num_elements(1))
          do w = adj_ia(v), adj_ia(v+1)-1
             c%val(:,adj_ja(2,w)) = c%val(:,adj_ja(2,w)) + &
                  [ a%val(num_features(1)+1:, v) ]
          end do
       end do
    end if

  end function reverse_duvenaud_propagate
!###############################################################################


!###############################################################################
  function duvenaud_update(a, weight, adj_ia, min_degree, max_degree) result(c)
    !! Update the message passing layer
    implicit none
    class(array_type), intent(in), target :: a
    class(array_type), intent(in), target :: weight
    ! real(real32), dimension(:,:,:), intent(in) :: weight
    integer, dimension(:), intent(in) :: adj_ia
    integer, intent(in) :: min_degree, max_degree
    type(array_type), pointer :: c
    type(array_type), pointer :: weight_array

    integer :: v, i, d
    integer :: interval
    real(real32), pointer :: w_ptr(:,:)

    c => a%create_result(array_shape=[weight%shape(1), size(a%val,2)])
    interval = weight%shape(1) * weight%shape(2)
    do v = 1, size(a%val,2)
       d = max( min_degree, min( adj_ia(v+1) - adj_ia(v), max_degree ) ) - &
            min_degree + 1
       w_ptr(1:weight%shape(1), 1:weight%shape(2)) => &
            weight%val(interval*(d-1)+1:interval*d,1)
       c%val(:,v) = matmul(w_ptr, a%val(:,v) / real(d, real32))
    end do
    c%indices = adj_ia

    c%get_partial_left => get_partial_duvenaud_update_weight
    c%get_partial_right => get_partial_duvenaud_update
    c%get_partial_left_val => get_partial_duvenaud_update_weight_val
    c%get_partial_right_val => get_partial_duvenaud_update_val
    if(a%requires_grad .or. weight%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. weight%is_forward
       c%operation = 'duvenaud_update'
       c%right_operand => a
       c%left_operand => weight
       c%owns_right_operand = a%is_temporary
       c%owns_left_operand = weight%is_temporary
    end if

  end function duvenaud_update
!-------------------------------------------------------------------------------
  function get_partial_duvenaud_update(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = reverse_duvenaud_update( upstream_grad, this%left_operand, &
         this%indices )

  end function get_partial_duvenaud_update
!-------------------------------------------------------------------------------
  function get_partial_duvenaud_update_weight(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = reverse_duvenaud_update_weight( upstream_grad, this%right_operand, &
         this%left_operand%indices, this%indices )

  end function get_partial_duvenaud_update_weight
!-------------------------------------------------------------------------------
  pure subroutine get_partial_duvenaud_update_val( &
       this, upstream_grad, output &
  )
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: v, d
    integer :: interval, num_output_features, num_input_features
    integer :: min_degree, max_degree
    real(real32), dimension(size(upstream_grad,1), this%right_operand%shape(1)) :: tmp

    output = 0._real32
    num_output_features = size(upstream_grad,1)
    num_input_features = this%right_operand%shape(1)
    interval = num_output_features * num_input_features
    min_degree = this%left_operand%indices(1)
    max_degree = this%left_operand%indices(2)
    do concurrent(v=1:size(upstream_grad,2))
       d = max( &
            min_degree, &
            min(this%indices(v+1) - this%indices(v), max_degree ) &
       ) - min_degree + 1
       tmp = reshape(this%left_operand%val((d-1)*interval+1:d*interval,1), &
            [num_output_features, num_input_features] )
       output(:,v) = matmul(transpose(tmp), upstream_grad(:,v))
    end do

  end subroutine get_partial_duvenaud_update_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_duvenaud_update_weight_val( &
       this, upstream_grad, output &
  )
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: v, i, j, d
    integer :: interval, num_output_features, num_input_features
    integer :: min_degree, max_degree

    output = 0._real32
    num_output_features = size(upstream_grad,1)
    num_input_features = this%right_operand%shape(1)
    interval = num_output_features * num_input_features
    min_degree = this%left_operand%indices(1)
    max_degree = this%left_operand%indices(2)
    do concurrent(v=1:size(upstream_grad,2))
       d = ( max( &
            min_degree, &
            min(this%indices(v+1) - this%indices(v), max_degree ) &
       ) - min_degree ) * interval
       do concurrent(i = 1:num_output_features, j=1:num_input_features)
          output(d+i+num_output_features*(j-1),1) = &
              output(d+i+num_output_features*(j-1),1) + &
              upstream_grad(i,v) * this%right_operand%val(j,v)
       end do
    end do
    !upstream = nfeat x nelem
    !right = nfeat(-1) x nelem
    !weight = nfeat x nfeat(-1) x ndegree

  end subroutine get_partial_duvenaud_update_weight_val
!###############################################################################


!###############################################################################
  function reverse_duvenaud_update(a, weight, adj_ia) result(c)
    !! Reverse update the message passing layer
    class(array_type), intent(in), target :: a, weight
    integer, dimension(:), intent(in) :: adj_ia
    type(array_type), pointer :: c

    integer :: v, d
    integer :: interval
    real(real32), pointer :: w_ptr(:,:)

    allocate(c)
    call c%allocate( array_shape = [weight%shape(2), size(a%val,2)] )
    interval = weight%shape(1) * weight%shape(2)
    do v = 1, size(a%val,2)
       d = max( weight%indices(1), &
            min( adj_ia(v+1) - adj_ia(v), weight%indices(2) ) ) - weight%indices(1) + 1
       w_ptr(1:weight%shape(1), 1:weight%shape(2)) => &
            weight%val(interval*(d-1)+1:interval*d,1)
       c%val(:,v) = matmul(transpose(w_ptr), a%val(:,v))
    end do

  end function reverse_duvenaud_update
!-------------------------------------------------------------------------------
  function reverse_duvenaud_update_weight(a, b, indices, adj_ia) result(c)
    !! Reverse update the message passing layer for weights
    class(array_type), intent(in), target :: a, b
    integer, dimension(2), intent(in) :: indices
    integer, dimension(:), intent(in) :: adj_ia
    type(array_type), pointer :: c

    integer :: v, d, i
    integer :: interval
    real(real32), pointer :: c_ptr(:,:)

    allocate(c)
    call c%allocate( array_shape = &
         [size(a%val,1), size(b%val,1), indices(2)-indices(1)+1,1] &
    )
    interval = c%shape(1) * c%shape(2)
    c%val = 0.0_real32
    do v = 1, size(a%val,2)
       d = max( indices(1), &
            min( adj_ia(v+1) - adj_ia(v), indices(2) ) ) - indices(1) + 1
       c_ptr(1:c%shape(1), 1:c%shape(2)) => c%val(interval*(d-1)+1:interval*d,1)
       ! do the outer product of a and b, c is the weight matrix
       do i = 1, size(a%val,1)
          c_ptr(i,:) = c_ptr(i,:) + a%val(i,v) * b%val(:,v)
       end do
    end do

  end function reverse_duvenaud_update_weight
!###############################################################################

end submodule athena__diffstruc_extd_submodule_msgpass_duvenaud
