submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_msgpass_duvenaud
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function duvenaud_propagate( &
       vertex_features, edge_features, adj_ia, adj_ja &
  ) result(c)
    !! Propagate values from one autodiff array to another
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: vertex_features, edge_features
    !! Vertex and edge feature tensors
    integer, dimension(:), intent(in) :: adj_ia
    !! CSR row pointers
    integer, dimension(:,:), intent(in) :: adj_ja
    !! CSR neighbour and edge lookup indices
    type(array_type), pointer :: c
    !! Propagated concatenated feature tensor

    ! Local variables
    integer :: v, w
    !! Vertex and adjacency traversal indices

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
    if(vertex_features%requires_grad .or. edge_features%requires_grad)then
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
    !! Gradient of duvenaud_propagate with respect to vertex_features.
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Forward result node containing saved operands
    type(array_type), intent(in) :: upstream_grad
    !! Upstream gradient tensor
    type(array_type) :: output
    !! Gradient tensor for left operand

    ! Local variables
    logical :: right_is_temporary_local
    !! Saved temporary-ownership flag for right operand
    type(array_type), pointer :: ptr
    !! Intermediate gradient tensor pointer

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    ptr => duvenaud_propagate( upstream_grad, this%right_operand, &
         this%indices, this%adj_ja )
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_duvenaud_propagate_left
!-------------------------------------------------------------------------------
  function get_partial_duvenaud_propagate_right(this, upstream_grad) result(output)
    !! Gradient of duvenaud_propagate with respect to edge_features.
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Forward result node containing saved operands
    type(array_type), intent(in) :: upstream_grad
    !! Upstream gradient tensor
    type(array_type) :: output
    !! Gradient tensor for right operand

    ! Local variables
    logical :: left_is_temporary_local
    !! Saved temporary-ownership flag for left operand
    type(array_type), pointer :: ptr
    !! Intermediate gradient tensor pointer

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    ptr => duvenaud_propagate( this%left_operand, upstream_grad, &
         this%indices, this%adj_ja )
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_duvenaud_propagate_right
!-------------------------------------------------------------------------------
  pure subroutine get_partial_duvenaud_propagate_left_val( &
       this, upstream_grad, output &
  )
    !! In-place value gradient for duvenaud_propagate left operand.
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    !! Forward result node containing saved operands
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    !! Upstream gradient values
    real(real32), dimension(:,:), intent(out) :: output
    !! Output gradient values for left operand

    ! Local variables
    integer :: v, w, num_features, num_elements
    !! Loop indices and operand shape values

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
    !! In-place value gradient for duvenaud_propagate right operand.
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    !! Forward result node containing saved operands
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    !! Upstream gradient values
    real(real32), dimension(:,:), intent(out) :: output
    !! Output gradient values for right operand

    ! Local variables
    integer :: v, w, num_features, num_elements
    !! Loop indices and operand shape values

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
  module function duvenaud_update(a, weight, adj_ia, min_degree, max_degree) result(c)
    !! Update the message passing layer
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: a
    !! Aggregated neighbour features
    class(array_type), intent(in), target :: weight
    !! Packed degree-conditioned weight tensor
    ! real(real32), dimension(:,:,:), intent(in) :: weight
    integer, dimension(:), intent(in) :: adj_ia
    !! CSR row pointers
    integer, intent(in) :: min_degree, max_degree
    !! Minimum and maximum degree buckets
    type(array_type), pointer :: c
    !! Degree-conditioned updated feature tensor
    type(array_type), pointer :: weight_array
    !! Reserved pointer for weight reshaping operations

    ! Local variables
    integer :: v, i, d
    !! Loop indices and degree bucket index
    integer :: interval
    !! Flat parameter interval for one degree bucket
    real(real32), pointer :: w_ptr(:,:)
    !! 2D view over selected degree-specific weight matrix

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
    if(a%requires_grad .or. weight%requires_grad)then
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
    !! Gradient of duvenaud_update with respect to input features.
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Forward result node containing saved operands
    type(array_type), intent(in) :: upstream_grad
    !! Upstream gradient tensor
    type(array_type) :: output
    !! Gradient tensor for right operand (input features)

    ! Local variables
    logical :: left_is_temporary_local
    !! Saved temporary-ownership flag for left operand
    type(array_type), pointer :: ptr
    !! Intermediate gradient tensor pointer

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    ptr => duvenaud_update( upstream_grad, this%left_operand, &
         this%indices, this%left_operand%indices(1), this%left_operand%indices(2) )
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_duvenaud_update
!-------------------------------------------------------------------------------
  function get_partial_duvenaud_update_weight(this, upstream_grad) result(output)
    !! Gradient of duvenaud_update with respect to packed weights.
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Forward result node containing saved operands
    type(array_type), intent(in) :: upstream_grad
    !! Upstream gradient tensor
    type(array_type) :: output
    !! Gradient tensor for left operand (weights)

    ! Local variables
    logical :: right_is_temporary_local
    !! Saved temporary-ownership flag for right operand
    type(array_type), pointer :: ptr
    !! Intermediate gradient tensor pointer

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    ptr => duvenaud_update( this%right_operand, upstream_grad, &
         this%indices, this%left_operand%indices(1), this%left_operand%indices(2) )
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_duvenaud_update_weight
!-------------------------------------------------------------------------------
  pure subroutine get_partial_duvenaud_update_val( &
       this, upstream_grad, output &
  )
    !! In-place value gradient for duvenaud_update input features.
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    !! Forward result node containing saved operands
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    !! Upstream gradient values
    real(real32), dimension(:,:), intent(out) :: output
    !! Output gradient values for input features

    ! Local variables
    integer :: v, d
    !! Loop index and degree bucket index
    integer :: interval, num_output_features, num_input_features
    !! Flattening interval and matrix dimensions
    integer :: min_degree, max_degree
    !! Degree bucket limits
    real(real32), dimension(size(upstream_grad,1), this%right_operand%shape(1)) :: tmp
    !! Temporary reshaped weight matrix for one degree bucket

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
       output(:,v) = matmul(upstream_grad(:,v), tmp) / real(d, real32)
    end do

  end subroutine get_partial_duvenaud_update_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_duvenaud_update_weight_val( &
       this, upstream_grad, output &
  )
    !! In-place value gradient for duvenaud_update packed weights.
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    !! Forward result node containing saved operands
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    !! Upstream gradient values
    real(real32), dimension(:,:), intent(out) :: output
    !! Output gradient values for packed weights

    ! Local variables
    integer :: v, i, j, d_offset, d_val
    !! Loop indices, degree offset and degree bucket index
    integer :: interval, num_output_features, num_input_features
    !! Flattening interval and matrix dimensions
    integer :: min_degree, max_degree
    !! Degree bucket limits

    output = 0._real32
    num_output_features = size(upstream_grad,1)
    num_input_features = this%right_operand%shape(1)
    interval = num_output_features * num_input_features
    min_degree = this%left_operand%indices(1)
    max_degree = this%left_operand%indices(2)
    do concurrent(v=1:size(upstream_grad,2))
       d_val = max( &
            min_degree, &
            min(this%indices(v+1) - this%indices(v), max_degree ) &
       ) - min_degree + 1
       d_offset = (d_val - 1) * interval
       do concurrent(i = 1:num_output_features, j=1:num_input_features)
          output(d_offset+i+num_output_features*(j-1),1) = &
               output(d_offset+i+num_output_features*(j-1),1) + &
               upstream_grad(i,v) * this%right_operand%val(j,v) / &
               real(d_val, real32)
       end do
    end do

  end subroutine get_partial_duvenaud_update_weight_val
!###############################################################################

end submodule athena__diffstruc_extd_submodule_msgpass_duvenaud
