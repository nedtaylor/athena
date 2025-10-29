module athena__diffstruc_extd
  !! Module for extended differential structure types for Athena
  use diffstruc, only: array_type, operator(+), operator(.concat.)
  implicit none


  private

  public :: array_container_type, array_ptr_type
  public :: add, concat



!-------------------------------------------------------------------------------
! Array container types
!-------------------------------------------------------------------------------
  type :: array_container_type
     class(array_type), allocatable :: array
  end type array_container_type

  type :: array_ptr_type
     type(array_type), pointer :: array(:,:)
  end type array_ptr_type

  ! Operator interfaces
  !-----------------------------------------------------------------------------
  interface add
     module function add_array_ptr(a, idx1, idx2) result(c)
       type(array_ptr_type), dimension(:), intent(in) :: a
       integer, intent(in) :: idx1, idx2
       type(array_type), pointer :: c
     end function add_array_ptr
  end interface

  interface concat
     module function concat_array_ptr(a, idx1, idx2, dim) result(c)
       type(array_ptr_type), dimension(:), intent(in) :: a
       integer, intent(in) :: idx1, idx2, dim
       type(array_type), pointer :: c
     end function concat_array_ptr
  end interface
!-------------------------------------------------------------------------------

  interface
    module function avgpool(input, pool_size, stride) result(output)
      type(array_type), intent(in), target :: input
      integer, intent(in) :: pool_size
      integer, intent(in) :: stride
      type(array_type), pointer :: output
    end function avgpool
  end interface

end module athena__diffstruc_extd
