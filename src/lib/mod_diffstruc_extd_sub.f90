submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function add_array_ptr(a, idx1, idx2) result(c)
    !! Add two autodiff arrays
    implicit none

    ! Arguments
    type(array_ptr_type), dimension(:), intent(in) :: a
    integer, intent(in) :: idx1, idx2
    type(array_type), pointer :: c

    ! Local variables
    integer :: i

    c => a(1)%array(idx1, idx2) + a(2)%array(idx1, idx2)
    do i = 2, size(a)
       c => c + a(i)%array(idx1, idx2)
    end do
  end function add_array_ptr
!###############################################################################


!###############################################################################
  module function concat_array_ptr(a, idx1, idx2, dim) result(c)
    !! Concatenate two autodiff arrays along a specified dimension
    implicit none

    ! Arguments
    type(array_ptr_type), dimension(:), intent(in) :: a
    integer, intent(in) :: idx1, idx2, dim
    type(array_type), pointer :: c

    ! Local variables
    integer :: i

    allocate(c)
    c => a(1)%array(idx1, idx2) .concat. a(2)%array(idx1, idx2)
    do i = 3, size(a)
       c => c .concat. a(i)%array(idx1, idx2)
    end do
  end function concat_array_ptr
!###############################################################################

end submodule athena__diffstruc_extd_submodule
