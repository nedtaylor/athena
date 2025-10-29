submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_pad
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  subroutine fill_edge_region(input, output)
    !! Fill an edge region based on padding method
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output

    ! Local variables
    integer :: i, m, s, f
    integer :: step, idx_in, idx_out

    do f = 1, output%indices(3)
       do concurrent( s = 1:output%shape(3), m = 1:output%shape(2) )
          select case(output%indices(1))
          case(3, 4) ! circular or reflection
             step = merge(1, -1, output%indices(1) .eq. 3)
             do i = 1, output%indices(2)
                idx_in = output%adj_ja(1,(f-1)*2 + 1) + step * (i - 1) + (m-1)*input%shape(1)
                idx_out = output%adj_ja(2,(f-1)*2 + 1) + i - 1 + (m-1)*input%shape(1)
                output%val(idx_out, s) = input%val(idx_in, s)
             end do
          case(5) ! replication
             idx_in = output%adj_ja(1,(f-1)*2 + 1) + (m-1)*input%shape(1)
             do i = 1, output%indices(2)
                idx_out = output%adj_ja(2,(f-1)*2 + 1) + i - 1 + (m-1)*input%shape(1)
                output%val(idx_out, s) = input%val(idx_in, s)
             end do
          end select
       end do
    end do

  end subroutine fill_edge_region
!-------------------------------------------------------------------------------
  subroutine accumulate_edge_gradients(input, output)
    !! Accumulate gradients for edge regions based on padding method
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output

    ! Local variables
    integer :: i, m, s, f
    integer :: step, idx_in, idx_out

    do f = 1, output%indices(3)
       do concurrent( s = 1:output%shape(3), m = 1:output%shape(2) )

          select case(output%indices(1))
          case(3, 4) ! circular or reflection
             step = merge(1, -1, output%indices(1) .eq. 3)
             do i = 1, output%indices(2)
                idx_in = output%adj_ja(1,(f-1)*2 + 1) + step * (i - 1) + (m-1)*input%shape(1)
                idx_out = output%adj_ja(2,(f-1)*2 + 1) + i - 1 + (m-1)*input%shape(1)
                output%val(idx_in, s) = output%val(idx_in, s) + input%val(idx_out, s)
             end do
          case(5) ! replication
             idx_in = output%adj_ja(1,(f-1)*2 + 1) + (m-1)*input%shape(1)
             do i = 1, output%indices(2)
                idx_out = output%adj_ja(2,(f-1)*2 + 1) + i - 1 + (m-1)*input%shape(1)
                output%val(idx_in, s) = output%val(idx_in, s) + input%val(idx_out, s)
             end do
          end select
       end do
    end do

  end subroutine accumulate_edge_gradients
!###############################################################################


!###############################################################################
  module function pad1d(input, facets, pad_size, imethod) result(output)
    !! 1D padding operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    type(facets_type), intent(in) :: facets
    integer, intent(in) :: pad_size
    integer, intent(in) :: imethod
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, m, s
    integer :: idx_in, idx_out
    integer :: input_size, output_size
    integer, dimension(3) :: output_shape

    input_size = input%shape(1)
    output_size = input_size +  2 * pad_size

    output_shape = [ output_size, input%shape(2), size(input%val, dim=2) ]
    output => input%create_result(array_shape = output_shape)

    ! save the facet values to indices and adj_ja
    allocate(output%indices(2 + facets%num))
    output%indices(1) = imethod
    output%indices(2) = pad_size
    output%indices(3) = facets%num
    allocate(output%adj_ja(2, 2 * facets%num))
    do i = 1, facets%num
       output%adj_ja(1,(i-1)*2 + 1) = facets%orig_bound(1,1,i)
       output%adj_ja(2,(i-1)*2 + 1) = facets%dest_bound(1,1,i)
       output%adj_ja(1,(i-1)*2 + 2) = facets%orig_bound(2,1,i)
       output%adj_ja(2,(i-1)*2 + 2) = facets%dest_bound(2,1,i)
    end do

    ! Initialize with pad_value
    output%val = 0._real32

    ! Copy input into the correct location in output
    do concurrent( &
         s = 1:output_shape(3), &
         m = 1:output_shape(2), &
         i = 1:input_size)
       idx_in = i + (m-1) * input_size
       idx_out = i + pad_size + (m-1) * output_size
       output%val(idx_out, s) = input%val(idx_in, s)
    end do

    if(output%indices(1) .gt. 3 .and. output%indices(1) .le. 5)then
       call fill_edge_region( input, output )
    end if


    output%get_partial_left => get_partial_pad1d
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'pad'
       output%left_operand = input
    end if

  end function pad1d
!-------------------------------------------------------------------------------
  function get_partial_pad1d(this, upstream_grad) result(output)
    !! Get the partial derivative for the pad1d operation
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, m, s, f
    integer :: idx_in, idx_out
    integer, dimension(3) :: input_shape


    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    call output%allocate(array_shape = input_shape)
    output%val = 0._real32

    do concurrent( &
         s = 1:input_shape(3), &
         m = 1:input_shape(2), &
         i = 1:input_shape(1))
       idx_in = i + (m-1) * input_shape(1)
       idx_out = i + this%indices(2) + (m-1) * input_shape(1)
       output%val(idx_in, s) = upstream_grad%val(idx_out, s)
    end do

    if(this%indices(1) .gt. 3 .and. this%indices(1) .le. 5)then
       call accumulate_edge_gradients( upstream_grad, output )
    end if

  end function get_partial_pad1d
!###############################################################################

end submodule athena__diffstruc_extd_submodule_pad
