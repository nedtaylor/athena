submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_merge
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function merge_scalar_over_channels(tsource, fsource, mask) result(output)
    !! 1D average pooling operation
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: tsource
    real(real32), intent(in) :: fsource
    logical, dimension(:,:), intent(in) :: mask
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, m, s
    integer :: num_elements, num_dims


    output => tsource%create_result()
    num_dims = size(tsource%shape)
    num_elements = product(tsource%shape(1:num_dims - 1))
    do concurrent(s = 1:size(tsource%val,2), m = 1: tsource%shape(num_dims))
       do concurrent(i=1:num_elements)
          if(mask(i,1)) then
             output%val(i + (m-1) * num_elements,s) = tsource%val(i,s)
          else
             output%val(i + (m-1) * num_elements,s) = fsource
          end if
       end do
    end do
    output%mask = mask

    output%get_partial_left => get_partial_merge_scalar_over_channels
    if(tsource%requires_grad) then
       output%requires_grad = .true.
       output%is_forward = tsource%is_forward
       output%operation = 'merge_over_channels'
       output%left_operand => tsource
    end if

  end function merge_scalar_over_channels
!-------------------------------------------------------------------------------
  function get_partial_merge_scalar_over_channels(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = merge_scalar_over_channels(upstream_grad, 0._real32, this%mask)

  end function get_partial_merge_scalar_over_channels
!###############################################################################

end submodule athena__diffstruc_extd_submodule_merge
