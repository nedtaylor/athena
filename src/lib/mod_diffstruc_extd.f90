module athena__diffstruc_extd
  !! Module for extended differential structure types for Athena
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(+), operator(.concat.)
  use athena__misc_types, only: facets_type
  implicit none


  private

  public :: array_container_type, array_ptr_type
  public :: add, concat
  public :: avgpool1d, avgpool2d, avgpool3d
  public :: maxpool1d, maxpool2d, maxpool3d
  public :: pad1d, pad2d, pad3d
  public :: merge_over_channels



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
     module function avgpool1d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, intent(in) :: pool_size
       integer, intent(in) :: stride
       type(array_type), pointer :: output
     end function avgpool1d

     module function avgpool2d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, dimension(2), intent(in) :: pool_size
       integer, dimension(2), intent(in) :: stride
       type(array_type), pointer :: output
     end function avgpool2d

     module function avgpool3d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, dimension(3), intent(in) :: pool_size
       integer, dimension(3), intent(in) :: stride
       type(array_type), pointer :: output
     end function avgpool3d
  end interface

  interface
     module function maxpool1d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, intent(in) :: pool_size
       integer, intent(in) :: stride
       type(array_type), pointer :: output
     end function maxpool1d

     module function maxpool2d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, dimension(2), intent(in) :: pool_size
       integer, dimension(2), intent(in) :: stride
       type(array_type), pointer :: output
     end function maxpool2d

     module function maxpool3d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, dimension(3), intent(in) :: pool_size
       integer, dimension(3), intent(in) :: stride
       type(array_type), pointer :: output
     end function maxpool3d
  end interface

  interface
     module function pad1d(input, facets, pad_size, imethod) result(output)
       type(array_type), intent(in), target :: input
       type(facets_type), intent(in) :: facets
       integer, intent(in) :: pad_size
       integer, intent(in) :: imethod
       type(array_type), pointer :: output
     end function pad1d

     module function pad2d(input, facets, pad_size, imethod) result(output)
       type(array_type), intent(in), target :: input
       type(facets_type), dimension(2), intent(in) :: facets
       integer, dimension(2), intent(in) :: pad_size
       integer, intent(in) :: imethod
       type(array_type), pointer :: output
     end function pad2d

     module function pad3d(input, facets, pad_size, imethod) result(output)
       type(array_type), intent(in), target :: input
       type(facets_type), dimension(3), intent(in) :: facets
       integer, dimension(3), intent(in) :: pad_size
       integer, intent(in) :: imethod
       type(array_type), pointer :: output
     end function pad3d
  end interface

  interface merge_over_channels
     module function merge_scalar_over_channels(tsource, fsource, mask) result(output)
       class(array_type), intent(in), target :: tsource
       real(real32), intent(in) :: fsource
       logical, dimension(:,:), intent(in) :: mask
       type(array_type), pointer :: output
     end function merge_scalar_over_channels
  end interface

end module athena__diffstruc_extd
