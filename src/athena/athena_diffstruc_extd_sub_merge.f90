submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_merge
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function merge_scalar_over_channels(tsource, fsource, mask) result(output)
    !! Merge a scalar into masked elements, broadcasting over channels
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: tsource
    real(real32), intent(in) :: fsource
    logical, dimension(:,:), intent(in) :: mask
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, m, s
    integer :: num_elements, num_dims, num_channels


    output => tsource%create_result()
    num_dims = size(tsource%shape)
    if(num_dims.eq.1)then
       ! Rank-1 arrays have a single dimension containing all elements
       num_elements = tsource%shape(1)
       num_channels = 1
    else
       ! The final dimension is treated as channels
       num_elements = product(tsource%shape(1:num_dims - 1))
       num_channels = tsource%shape(num_dims)
    end if
    do concurrent(s = 1:size(tsource%val,2), m = 1:num_channels)
       do concurrent(i=1:num_elements)
          if(mask(i,1))then
             output%val(i + (m-1) * num_elements,s) = &
                  tsource%val(i + (m-1) * num_elements,s)
          else
             output%val(i + (m-1) * num_elements,s) = fsource
          end if
       end do
    end do
    output%mask = mask

    output%value_partial_ok = .true.
    output%get_partial_left => get_partial_merge_scalar_over_channels
    output%get_partial_left_val => get_partial_merge_scalar_over_channels_val
    if(tsource%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = tsource%is_forward
       output%operation = 'merge_over_channels'
       output%left_operand => canonical(tsource)
    end if

  end function merge_scalar_over_channels
!-------------------------------------------------------------------------------
  function get_partial_merge_scalar_over_channels(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: up

    ! The upstream gradient passed to a graph partial is a flat storage view,
    ! so rebuild a handle with the operand's logical shape before the merge.
    up => this%left_operand%create_result()
    up%val = upstream_grad%val
    output = merge_scalar_over_channels(up, 0._real32, this%mask)

  end function get_partial_merge_scalar_over_channels
!-------------------------------------------------------------------------------
  pure subroutine get_partial_merge_scalar_over_channels_val( &
       this, upstream_grad, output &
  )
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, m, s
    integer :: num_elements, num_dims, num_channels

    num_dims = size(this%left_operand%shape)
    if(num_dims.eq.1)then
       ! Rank-1 arrays have a single dimension containing all elements
       num_elements = this%left_operand%shape(1)
       num_channels = 1
    else
       ! The final dimension is treated as channels
       num_elements = product(this%left_operand%shape(1:num_dims - 1))
       num_channels = this%left_operand%shape(num_dims)
    end if

    do concurrent(s = 1:size(upstream_grad,2), m = 1: num_channels)
       do concurrent(i=1:num_elements)
          if(this%mask(i,1))then
             output(i + (m-1) * num_elements,s) = &
                  upstream_grad(i + (m-1) * num_elements,s)
          else
             output(i + (m-1) * num_elements,s) = 0._real32
          end if
       end do
    end do

  end subroutine get_partial_merge_scalar_over_channels_val
!###############################################################################

end submodule athena__diffstruc_extd_submodule_merge
