submodule(athena__misc_types) athena__misc_types_submodule
  !! Submodule containing implementations for derived types
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32



contains

!###############################################################################
  module subroutine setup_bounds(this, length, pad, imethod)
    !! Set up replication bounds for facets
    implicit none

    ! Arguments
    class(facets_type), intent(inout) :: this
    !! Instance of the facets type
    integer, dimension(this%rank), intent(in) :: length, pad
    !! Length of the shape and padding
    integer, intent(in) :: imethod
    !! Method for padding:
    !! 3 - circular, 4 - reflection, 5 - replication

    ! Local variables
    integer :: i, j, k, facet_idx, idim
    !! Loop indices and facet index
    logical :: btest_k0, btest_k1
    !! Binary test variables for edge cases


    ! Calculate number of facets based on rank and number of fixed dimensions
    !---------------------------------------------------------------------------
    ! For rank n, we have:
    ! nfixed_dims = 1: n choose 1 * 2 facets (faces, 2 per dimension)
    ! nfixed_dims = 2: n choose 2 * 4 facets (edges, 4 per dimension pair)
    ! nfixed_dims = 3: n choose 3 * 8 facets (corners, 8 for 3D)
    select case(this%nfixed_dims)
    case(1)
       this%type = "face"
       this%num = 2 * this%rank
    case(2)
       this%type = "edge"
       this%num = 4 * nint( &
            gamma(real(this%rank + 1)) / ( &
                 gamma(2.0 + 1.0) * gamma(real(this%rank - 2 + 1)) &
            ) &
       )
    case(3)
       this%type = "corner"
       this%num = 8
    case default
       call stop_program("Invalid number of fixed dimensions")
       return
    end select
    if(this%rank .lt. this%nfixed_dims) then
       call stop_program("Number of fixed dimensions exceeds rank")
       return
    end if


    ! Allocate arrays
    !---------------------------------------------------------------------------
    if (allocated(this%dim)) deallocate(this%dim)
    if (allocated(this%orig_bound)) deallocate(this%orig_bound)
    if (allocated(this%dest_bound)) deallocate(this%dest_bound)

    allocate(this%dim(this%num))
    allocate(this%orig_bound(2, this%rank, this%num))
    allocate(this%dest_bound(2, this%rank, this%num))


    ! Initialise all bounds to 1
    !---------------------------------------------------------------------------
    this%orig_bound = 1

    ! Set up replication bounds
    !---------------------------------------------------------------------------
    select case(this%nfixed_dims)
    case(1)  ! Faces
       facet_idx = 0
       do i = 1, this%rank
          do j = 1, 2  ! Two faces per dimension
             facet_idx = facet_idx + 1
             this%dim(facet_idx) = i

             ! Set origin bounds
             select case(imethod)
             case(3) ! circular
                if(j .eq. 1) then
                   this%orig_bound(:,i,facet_idx) = &
                        [ length(i) - pad(i) + 1, length(i) ]
                else
                   this%orig_bound(:,i,facet_idx) = [ 1, pad(i) ]
                end if
             case(4) ! reflection
                if(j .eq. 1) then
                   this%orig_bound(:,i,facet_idx) = [ pad(i) + 1, 2 ]
                else
                   this%orig_bound(:,i,facet_idx) = &
                        [ length(i) - 1, length(i) - pad(i) ]
                end if
             case(5) ! replication
                if(j .ne. 1) this%orig_bound(:,i,facet_idx) = length(i)
             end select

             ! Set destination bounds
             if(j .eq. 1) then
                this%dest_bound(:,i,facet_idx) = [1, pad(i)]
             else
                this%dest_bound(:,i,facet_idx) = &
                     [length(i) + pad(i) + 1, length(i) + pad(i) * 2]
             end if
          end do
       end do
    case(2)  ! Edges
       facet_idx = 0
       idim = 0
       do j = this%rank, 2, -1
          do i = j-1, 1, -1
             idim = idim + 1
             do k = 0, 3  ! Four combinations per dimension pair
                facet_idx = facet_idx + 1
                this%dim(facet_idx) = idim
                btest_k0 = btest(k,0)
                btest_k1 = btest(k,1)

                ! Set original bounds using binary pattern
                select case(imethod)
                case(3) ! circular
                     if(btest_k1) then
                        this%orig_bound(:,i,facet_idx) = &
                             [ 1, pad(i) ]
                     else
                        this%orig_bound(:,i,facet_idx) = &
                             [ length(i) - pad(i) + 1, length(i) ]
                     end if
                     if(btest_k0) then
                        this%orig_bound(:,j,facet_idx) = &
                             [ 1, pad(j) ]
                     else
                        this%orig_bound(:,j,facet_idx) = &
                             [ length(j) - pad(j) + 1, length(j) ]
                     end if
                case(4) ! reflection
                   this%orig_bound(:,i,facet_idx) = &
                        [ length(i) - 1, length(i) - pad(i) ]
                   this%orig_bound(:,j,facet_idx) = &
                        [ length(j) - 1, length(j) - pad(j) ]
                  !  this%orig_bound(:,j,facet_idx) = [2, pad(j) + 1]
                  !  if (btest_k1) then
                  !     this%orig_bound(:,i,facet_idx) = [2, pad(i) + 1]
                  !  else
                  !     this%orig_bound(:,i,facet_idx) = &
                  !          [length(i) - 1, length(i) - pad(i)]
                  !  end if
                  !  if (btest_k0) then
                  !     this%orig_bound(:,j,facet_idx) = [2, pad(j) + 1]
                  !  else
                  !     this%orig_bound(:,j,facet_idx) = &
                  !          [length(j) - 1, length(j) - pad(j)]
                  !  end if
                case(5) ! replication
                   if(btest_k1) this%orig_bound(:,i,facet_idx) = length(i)
                   if(btest_k0) this%orig_bound(:,j,facet_idx) = length(j)
                end select

                ! Set destination bounds
                this%dest_bound(:,i,facet_idx) = &
                     merge(&
                          [ length(i) + pad(i) + 1, length(i) + pad(i) * 2 ], &
                          [ 1, pad(i) ], &
                          btest_k1 &
                     )
                this%dest_bound(:,j,facet_idx) = &
                     merge( &
                          [ length(j) + pad(j) + 1, length(j) + pad(j) * 2 ], &
                          [ 1, pad(j) ], &
                          btest_k0 &
                     )
               !  write(*,*) "i,j,k:", i, j, k
               !  write(*,*) btest_k0, btest_k1
               !  write(*,*) this%dest_bound(:,i,facet_idx), &
               !       this%dest_bound(:,j,facet_idx)
               !  write(*,*) this%orig_bound(:,i,facet_idx), &
               !       this%orig_bound(:,j,facet_idx)
               !       write(*,*)
               !       write(*,*)
             end do
          end do
       end do
    case(3)  ! Corners (3D only)
       do i = 1, 8
          this%dim(i) = 0  ! All dimensions are fixed
          ! Use binary pattern for all three dimensions
          do j = 1, this%rank
             if(btest(i-1, this%rank-j)) then
                this%orig_bound(:,j,i) = length(j)
                this%dest_bound(:,j,i) = &
                     [ length(j) + pad(j) + 1, length(j) + pad(j) * 2 ]
             else
                this%orig_bound(:,j,i) = 1
                this%dest_bound(:,j,i) = [1, pad(j)]
             end if
          end do
       end do
    end select

  end subroutine setup_bounds
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure module function add_array(a, b) result(output)
    !! Add two arrays
    implicit none

    ! Arguments
    class(array_type), intent(in) :: a, b
    !! Input arrays
    class(array_type), allocatable :: output
    !! Output array

    output = a
    if(.not.allocated(a%val).or..not.allocated(b%val))then
       return
    elseif(a%size.ne.b%size) then
       return
    end if

    output%val = a%val + b%val

  end function add_array
!-------------------------------------------------------------------------------
  pure module function multiply_array(a, b) result(output)
    !! Add two arrays
    implicit none

    ! Arguments
    class(array_type), intent(in) :: a, b
    !! Input arrays
    class(array_type), allocatable :: output
    !! Output array

    output = a
    if(.not.allocated(a%val).or..not.allocated(b%val))then
       return
    elseif(a%size.ne.b%size) then
       return
    end if

    output%val = a%val * b%val

  end function multiply_array
!###############################################################################


!###############################################################################
  pure module function flatten_array(this) result(output)
    !! Flatten the array
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    !! Instance of the array type
    real(real32), dimension(this%size) :: output
    !! Flattened array

    output = reshape(this%val, [this%size])
  end function flatten_array
!###############################################################################


!###############################################################################
  pure module subroutine get_array(this, output)
    !! Get the array
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    !! Instance of the array type
    real(real32), dimension(..), allocatable, intent(out) :: output
    !! Output array

    select rank(output)
    rank(1)
       output = this%flatten()
    rank(2)
       output = this%val
    rank(3)
       select type(this)
       type is(array3d_type)
          output = this%val_ptr
       end select
    rank(4)
       select type(this)
       type is(array4d_type)
          output = this%val_ptr
       end select
    rank(5)
       select type(this)
       type is(array5d_type)
          output = this%val_ptr
       end select
    rank default
       return
    end select
  end subroutine get_array
!###############################################################################


!###############################################################################
  pure module subroutine set_array(this, input)
    !! Set the array
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Instance of the array type
    real(real32), dimension(..), intent(in) :: input
    !! Input array

    select rank(input)
    rank(1)
       this%val(:,1) = input
    rank(2)
       this%val(:,:) = input
    rank default
       return
    end select
  end subroutine set_array
!###############################################################################


!###############################################################################
  module subroutine assign_array(this, input)
    !! Assign the array
    implicit none

    ! Arguments
    class(array_type), intent(out), target :: this
    !! Instance of the array type
    class(array_type), intent(in) :: input
    !! Input array

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
    select type(input)
    type is(array1d_type)
       select type(this)
       type is(array1d_type)
          this%val_ptr( &
               1:this%shape(1) &
          ) => this%val
       end select
    type is(array2d_type)
       select type(this)
       type is(array2d_type)
          this%val_ptr( &
               1:this%shape(1), &
               1:size(this%val, dim=2) &
          ) => this%val
       end select
    type is(array3d_type)
       select type(this)
       type is(array3d_type)
          this%val_ptr( &
               1:this%shape(1), &
               1:this%shape(2),  &
               1:size(this%val, dim=2) &
          ) => this%val
       end select
    type is(array4d_type)
       select type(this)
       type is(array4d_type)
          this%val_ptr( &
               1:this%shape(1), &
               1:this%shape(2), &
               1:this%shape(3),  &
               1:size(this%val, dim=2) &
          ) => this%val
       end select
    type is(array5d_type)
       select type(this)
       type is(array5d_type)
          this%val_ptr( &
               1:this%shape(1), &
               1:this%shape(2), &
               1:this%shape(3), &
               1:this%shape(4),  &
               1:size(this%val, dim=2) &
          ) => this%val
       end select
    class default
       call stop_program('Incompatible types')
       return
    end select

  end subroutine assign_array
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function init_array1d(array_shape) result(output)
    !! Initialise 1D array
    implicit none

    ! Arguments
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    type(array1d_type) :: output
    !! Output array

    output%rank = 1
    allocate(output%shape(1))
    if(present(array_shape)) call output%allocate(array_shape)

  end function init_array1d
!###############################################################################


!###############################################################################
  module subroutine allocate_array1d(this, array_shape, source)
    !! Allocate 1D array
    implicit none

    ! Arguments
    class(array1d_type), intent(inout), target :: this
    !! Instance of the 1D array type
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    class(*), dimension(..), intent(in), optional :: source
    !! Source array

    if(allocated(this%val).or.this%allocated)then
       call stop_program("Trying to allocate already allocated array values")
       return
    end if
    this%rank = 1
    this%allocated = .true.
    if(present(array_shape))then
       this%shape = array_shape
       allocate( this%val( array_shape(1), 1 ) )
       this%val_ptr( 1:array_shape(1) ) => this%val
    end if
    if(present(source))then
       select rank(source)
       rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape))then
                call stop_program('Source shape not provided')
                return
             end if
             this%val(:,:) = source
          type is (array1d_type)
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source%val)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             end if
             this = source
          class default
             call stop_program('Incompatible source type for rank 0')
             return
          end select
       rank(1)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source))) &
                     call stop_program( &
                          'Source shape does not match array shape' &
                     )
                return
             else
                allocate( this%val( size(source, dim=1), 1 ) )
                this%val_ptr( 1:size(source, dim=1) ) => this%val
             end if
             this%val_ptr = source
          class default
             call stop_program('Incompatible source type for rank 1')
             return
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             this%val(:,:) = source
          class default
             call stop_program('Incompatible source type for rank 2')
             return
          end select
       rank(3)
       rank default
          call stop_program('Unrecognised source rank')
          return
       end select
    end if
    if(.not.present(source).and..not.present(array_shape))then
       call stop_program('No shape or source provided')
       return
    end if
    this%shape = shape(this%val_ptr)
    this%size = product(this%shape)

  end subroutine allocate_array1d
!###############################################################################


!###############################################################################
  pure module subroutine deallocate_array1d(this, keep_shape)
    !! Deallocate 1D array
    implicit none

    ! Arguments
    class(array1d_type), intent(inout) :: this
    !! Instance of the 1D array type
    logical, intent(in), optional :: keep_shape
    !! Boolean whether to keep shape

    ! Local variables
    logical :: keep_shape_
    !! Boolean whether to keep shape

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    deallocate(this%val)
    this%val_ptr => null()
    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array1d
!###############################################################################


!###############################################################################
  module subroutine assign_array1d(this, input)
    !! Assign 1D array
    implicit none

    ! Arguments
    type(array1d_type), intent(out), target :: this
    !! Instance of the 1D array type
    type(array1d_type), intent(in) :: input
    !! Input array

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
    this%val_ptr( &
         1:this%shape(1) &
    ) => this%val
  end subroutine assign_array1d
!###############################################################################


!###############################################################################
  module subroutine set_ptr_array1d(this)
    !! Set pointer for 1D array
    implicit none

    ! Arguments
    class(array1d_type), intent(inout), target :: this
    !! Instance of the 1D array type

    if(.not. this%allocated)then
       call stop_program('Array not allocated')
       return
    end if
    this%val_ptr(1:this%shape(1)) => this%val
  end subroutine set_ptr_array1d
!###############################################################################


!###############################################################################
  module subroutine finalise_array1d(this)
    !! Finalise 1D array
    implicit none

    ! Arguments
    type(array1d_type), intent(inout) :: this
    !! Instance of the 1D array type

    if(associated(this%val_ptr)) nullify(this%val_ptr)
    if(allocated(this%val)) deallocate(this%val)
    if(allocated(this%shape)) deallocate(this%shape)
  end subroutine finalise_array1d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function init_array2d(array_shape) result(output)
    !! Initialise 2D array
    implicit none

    ! Arguments
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    type(array2d_type) :: output
    !! Output array

    output%rank = 1
    allocate(output%shape(1))
    if(present(array_shape)) call output%allocate(array_shape)

  end function init_array2d
!###############################################################################


!###############################################################################
  module subroutine allocate_array2d(this, array_shape, source)
    !! Allocate 2D array
    implicit none

    ! Arguments
    class(array2d_type), intent(inout), target :: this
    !! Instance of the 2D array type
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    class(*), dimension(..), intent(in), optional :: source
    !! Source array

    if(allocated(this%val).or.this%allocated)then
       call stop_program("Trying to allocate already allocated array values")
       return
    end if
    this%rank = 1
    this%allocated = .true.
    if(present(array_shape)) allocate(this%val(array_shape(1), array_shape(2)))
    if(present(source))then
       select rank(source)
       rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape))then
                call stop_program('Source shape not provided')
                return
             end if
             this%val(:,:) = source
          type is (array2d_type)
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source%val)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             end if
             this = source
             this%val_ptr( &
                  1:source%shape(1), &
                  1:size(source%val, dim=2) &
             ) => this%val
          class default
             call stop_program('Incompatible source type for rank 0')
             return
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             end if
             this%val = source
          class default
             call stop_program('Incompatible source type for rank 2')
             return
          end select
       rank default
          call stop_program('Unrecognised source rank')
          return
       end select
    end if
    if(.not.present(source).and..not.present(array_shape))then
       call stop_program('No shape or source provided')
       return
    end if
    this%val_ptr(1:size(this%val, dim=1), 1:size(this%val, dim=2)) => this%val
    this%shape = [ size(this%val, dim=1) ]
    this%size = product(this%shape)

  end subroutine allocate_array2d
!###############################################################################


!###############################################################################
  pure module subroutine deallocate_array2d(this, keep_shape)
    !! Deallocate 2D array
    implicit none

    ! Arguments
    class(array2d_type), intent(inout) :: this
    !! Instance of the 2D array type
    logical, intent(in), optional :: keep_shape
    !! Boolean whether to keep shape

    ! Local variables
    logical :: keep_shape_
    !! Boolean whether to keep shape

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    deallocate(this%val)
    this%val_ptr => null()
    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array2d
!###############################################################################


!###############################################################################
  module subroutine assign_array2d(this, input)
    !! Assign 2D array
    implicit none

    ! Arguments
    type(array2d_type), intent(out), target :: this
    !! Instance of the 2D array type
    type(array2d_type), intent(in) :: input
    !! Input array

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
    this%val_ptr( &
         1:this%shape(1), &
         1:size(this%val, dim=2) &
    ) => this%val
  end subroutine assign_array2d
!###############################################################################


!###############################################################################
  module subroutine set_ptr_array2d(this)
    !! Set pointer for 2D array
    implicit none

    ! Arguments
    class(array2d_type), intent(inout), target :: this
    !! Instance of the 2D array type

    if(.not. this%allocated)then
       call stop_program('Array not allocated')
       return
    end if
    this%val_ptr( &
         1:this%shape(1), &
         1:size(this%val, dim=2) &
    ) => this%val
  end subroutine set_ptr_array2d
!###############################################################################


!###############################################################################
  module subroutine finalise_array2d(this)
    !! Finalise 2D array
    implicit none

    ! Arguments
    type(array2d_type), intent(inout) :: this
    !! Instance of the 2D array type

    if(associated(this%val_ptr)) nullify(this%val_ptr)
    if(allocated(this%val)) deallocate(this%val)
    if(allocated(this%shape)) deallocate(this%shape)
  end subroutine finalise_array2d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function init_array3d(array_shape) result(output)
    !! Initialise 3D array
    implicit none

    ! Arguments
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    type(array3d_type) :: output
    !! Output array

    output%rank = 2
    allocate(output%shape(2))
    if(present(array_shape)) call output%allocate(array_shape)

  end function init_array3d
!###############################################################################


!###############################################################################
  module subroutine allocate_array3d(this, array_shape, source)
    !! Allocate 3D array
    implicit none

    ! Arguments
    class(array3d_type), intent(inout), target :: this
    !! Instance of the 3D array type
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    class(*), dimension(..), intent(in), optional :: source
    !! Source array

    if(allocated(this%val).or.this%allocated)then
       call stop_program("Trying to allocate already allocated array values")
       return
    end if
    this%rank = 2
    this%allocated = .true.
    if(present(array_shape))then
       this%shape = array_shape
       allocate(this%val(&
            product(array_shape(1:2)),&
            array_shape(3) &
       ) )
       this%val_ptr( &
            1:array_shape(1), &
            1:array_shape(2), &
            1:array_shape(3) &
       ) => this%val
    end if
    if(present(source))then
       select rank(source)
       rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape))then
                call stop_program('Source shape not provided')
                return
             end if
             this%val(:,:) = source
          type is (array3d_type)
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source%val)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             end if
             this = source
             this%val_ptr( &
                  1:source%shape(1), &
                  1:source%shape(2), &
                  1:size(source%val, dim=2) &
             ) => this%val
          class default
             call stop_program('Incompatible source type for rank 0')
             return
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             this%val(:,:) = source
          class default
             call stop_program('Incompatible source type for rank 2')
             return
          end select
       rank(3)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             else
                allocate(this%val(size(source(:,:,1)), size(source,3)))
                this%val_ptr( &
                     1:size(source, dim=1), &
                     1:size(source, dim=2), &
                     1:size(source, dim=3) &
                ) => this%val
             end if
             this%val_ptr = source
          class default
             call stop_program('Incompatible source type for rank 3')
             return
          end select
       rank default
          call stop_program('Unrecognised source rank')
          return
       end select
    end if
    if(.not.present(source).and..not.present(array_shape))then
       call stop_program('No shape or source provided')
       return
    end if
    this%shape = shape(this%val_ptr(:,:,1))
    this%size = product(this%shape)

  end subroutine allocate_array3d
!###############################################################################


!###############################################################################
  pure module subroutine deallocate_array3d(this, keep_shape)
    !! Deallocate 3D array
    implicit none

    ! Arguments
    class(array3d_type), intent(inout) :: this
    !! Instance of the 3D array type
    logical, intent(in), optional :: keep_shape
    !! Boolean whether to keep shape

    ! Local variables
    logical :: keep_shape_
    !! Boolean whether to keep shape

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    deallocate(this%val)
    this%val_ptr => null()
    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array3d
!###############################################################################


!###############################################################################
  pure module subroutine set_array3d(this, input)
    !! Set 3D array
    implicit none

    ! Arguments
    class(array3d_type), intent(inout) :: this
    !! Instance of the 3D array type
    real(real32), dimension(..), intent(in) :: input
    !! Input array

    select rank(input)
    rank(1)
       this%val(:,1) = input
    rank(2)
       this%val(:,:) = input
    rank(3)
       this%val_ptr(:,:,:) = input
    rank default
       return
    end select
  end subroutine set_array3d
!###############################################################################


!###############################################################################
  module subroutine assign_array3d(this, input)
    !! Assign 3D array
    implicit none

    ! Arguments
    type(array3d_type), intent(out), target :: this
    !! Instance of the 3D array type
    type(array3d_type), intent(in) :: input
    !! Input array

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
    this%val_ptr( &
         1:this%shape(1), &
         1:this%shape(2), &
         1:size(this%val, dim=2) &
    ) => this%val
  end subroutine assign_array3d
!###############################################################################


!###############################################################################
  module subroutine set_ptr_array3d(this)
    !! Set pointer for 3D array
    implicit none

    ! Arguments
    class(array3d_type), intent(inout), target :: this
    !! Instance of the 3D array type

    if(.not. this%allocated)then
       call stop_program('Array not allocated')
       return
    end if
    this%val_ptr( &
         1:this%shape(1), &
         1:this%shape(2), &
         1:size(this%val, dim=2) &
    ) => this%val
  end subroutine set_ptr_array3d
!###############################################################################


!###############################################################################
  module subroutine finalise_array3d(this)
    !! Finalise 3D array
    implicit none

    ! Arguments
    type(array3d_type), intent(inout) :: this
    !! Instance of the 3D array type

    if(associated(this%val_ptr)) nullify(this%val_ptr)
    if(allocated(this%val)) deallocate(this%val)
    if(allocated(this%shape)) deallocate(this%shape)
  end subroutine finalise_array3d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function init_array4d(array_shape) result(output)
    !! Initialise 4D array
    implicit none

    ! Arguments
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    type(array4d_type) :: output
    !! Output array

    output%rank = 3
    allocate(output%shape(3))
    if(present(array_shape)) call output%allocate(array_shape)

  end function init_array4d
!###############################################################################


!###############################################################################
  module subroutine allocate_array4d(this, array_shape, source)
    !! Allocate 4D array
    implicit none

    ! Arguments
    class(array4d_type), intent(inout), target :: this
    !! Instance of the 4D array type
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    class(*), dimension(..), intent(in), optional :: source
    !! Source array

    if(allocated(this%val).or.this%allocated)then
       call stop_program("Trying to allocate already allocated array values")
       return
    end if
    this%rank = 3
    this%allocated = .true.
    if(present(array_shape))then
       this%shape = array_shape
       allocate(this%val(&
            product(array_shape(1:3)),&
            array_shape(4) &
       ) )
       this%val_ptr( &
            1:array_shape(1), &
            1:array_shape(2), &
            1:array_shape(3), &
            1:array_shape(4) &
       ) => this%val
    end if
    if(present(source))then
       select rank(source)
       rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape))then
                call stop_program('Source shape not provided')
                return
             end if
             this%val(:,:) = source
          type is (array4d_type)
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source%val)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             end if
             this = source
             this%val_ptr( &
                  1:source%shape(1), &
                  1:source%shape(2), &
                  1:source%shape(3), &
                  1:size(source%val, dim=2) &
             ) => this%val
          class default
             call stop_program('Incompatible source type for rank 0')
             return
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             this%val(:,:) = source
          class default
             call stop_program('Incompatible source type for rank 2')
             return
          end select
       rank(4)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             else
                allocate(this%val(size(source(:,:,:,1)), size(source,4)))
                this%val_ptr( &
                     1:size(source, dim=1), &
                     1:size(source, dim=2), &
                     1:size(source, dim=3), &
                     1:size(source, dim=4) &
                ) => this%val
             end if
             this%val_ptr = source
          class default
             call stop_program('Incompatible source type for rank 4')
             return
          end select
       rank default
          call stop_program('Unrecognised source rank')
          return
       end select
    end if
    if(.not.present(source).and..not.present(array_shape))then
       call stop_program('No shape or source provided')
       return
    end if
    this%shape = shape(this%val_ptr(:,:,:,1))
    this%size = product(this%shape)

  end subroutine allocate_array4d
!###############################################################################


!###############################################################################
  pure module subroutine deallocate_array4d(this, keep_shape)
    !! Deallocate 4D array
    implicit none

    ! Arguments
    class(array4d_type), intent(inout) :: this
    !! Instance of the 4D array type
    logical, intent(in), optional :: keep_shape
    !! Boolean whether to keep shape

    ! Local variables
    logical :: keep_shape_
    !! Boolean whether to keep shape

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    deallocate(this%val)
    this%val_ptr => null()
    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array4d
!###############################################################################


!###############################################################################
  pure module subroutine set_array4d(this, input)
    !! Set 4D array
    implicit none

    ! Arguments
    class(array4d_type), intent(inout) :: this
    !! Instance of the 4D array type
    real(real32), dimension(..), intent(in) :: input
    !! Input array

    select rank(input)
    rank(1)
       this%val(:,1) = input
    rank(2)
       this%val(:,:) = input
    rank(4)
       this%val_ptr(:,:,:,:) = input
    rank default
       return
    end select
  end subroutine set_array4d
!###############################################################################


!###############################################################################
  module subroutine assign_array4d(this, input)
    !! Assign 4D array
    implicit none

    ! Arguments
    type(array4d_type), intent(out), target :: this
    !! Instance of the 4D array type
    type(array4d_type), intent(in) :: input
    !! Input array

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
    this%val_ptr( &
         1:this%shape(1), &
         1:this%shape(2), &
         1:this%shape(3), &
         1:size(this%val, dim=2) &
    ) => this%val
  end subroutine assign_array4d
!###############################################################################


!###############################################################################
  module subroutine set_ptr_array4d(this)
    !! Set pointer for 4D array
    implicit none

    ! Arguments
    class(array4d_type), intent(inout), target :: this
    !! Instance of the 4D array type

    if(.not. this%allocated)then
       call stop_program('Array not allocated')
       return
    end if
    this%val_ptr( &
         1:this%shape(1), &
         1:this%shape(2), &
         1:this%shape(3), &
         1:size(this%val, dim=2) &
    ) => this%val
  end subroutine set_ptr_array4d
!###############################################################################


!###############################################################################
  module subroutine finalise_array4d(this)
    !! Finalise 4D array
    implicit none

    ! Arguments
    type(array4d_type), intent(inout) :: this
    !! Instance of the 4D array type

    if(associated(this%val_ptr)) nullify(this%val_ptr)
    if(allocated(this%val)) deallocate(this%val)
    if(allocated(this%shape)) deallocate(this%shape)
  end subroutine finalise_array4d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function init_array5d(array_shape) result(output)
    !! Initialise 5D array
    implicit none

    ! Arguments
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    type(array5d_type) :: output
    !! Output array

    output%rank = 4
    allocate(output%shape(4))
    if(present(array_shape)) call output%allocate(array_shape)

  end function init_array5d
!###############################################################################


!###############################################################################
  module subroutine allocate_array5d(this, array_shape, source)
    !! Allocate 5D array
    implicit none

    ! Arguments
    class(array5d_type), intent(inout), target :: this
    !! Instance of the 5D array type
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    class(*), dimension(..), intent(in), optional :: source
    !! Source array

    if(allocated(this%val).or.this%allocated)then
       call stop_program("Trying to allocate already allocated array values")
       return
    end if
    this%rank = 4
    this%allocated = .true.
    if(present(array_shape))then
       this%shape = array_shape
       allocate(this%val(&
            product(array_shape(1:4)),&
            array_shape(5) &
       ) )
       this%val_ptr( &
            1:array_shape(1), &
            1:array_shape(2), &
            1:array_shape(3), &
            1:array_shape(4), &
            1:array_shape(5) &
       ) => this%val
    end if
    if(present(source))then
       select rank(source)
       rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape))then
                call stop_program('Source shape not provided')
                return
             end if
             this%val(:,:) = source
          type is (array5d_type)
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source%val)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             end if
             this = source
             this%val_ptr( &
                  1:source%shape(1), &
                  1:source%shape(2), &
                  1:source%shape(3), &
                  1:source%shape(4), &
                  1:size(source%val, dim=2) &
             ) => this%val
          class default
             call stop_program('Incompatible source type for rank 0')
             return
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             this%val(:,:) = source
          class default
             call stop_program('Incompatible source type for rank 2')
             return
          end select
       rank(5)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             else
                allocate(this%val(size(source(:,:,:,:,1)), size(source,5)))
                this%val_ptr( &
                     1:size(source, dim=1), &
                     1:size(source, dim=2), &
                     1:size(source, dim=3), &
                     1:size(source, dim=4), &
                     1:size(source, dim=5) &
                ) => this%val
             end if
             this%val_ptr = source
          class default
             call stop_program('Incompatible source type for rank 5')
             return
          end select
       rank default
          call stop_program('Unrecognised source rank')
          return
       end select
    end if
    if(.not.present(source).and..not.present(array_shape))then
       call stop_program('No shape or source provided')
       return
    end if
    this%shape = shape(this%val_ptr(:,:,:,:,1))
    this%size = product(this%shape)

  end subroutine allocate_array5d
!###############################################################################


!###############################################################################
  pure module subroutine deallocate_array5d(this, keep_shape)
    !! Deallocate 5D array
    implicit none

    ! Arguments
    class(array5d_type), intent(inout) :: this
    !! Instance of the 5D array type
    logical, intent(in), optional :: keep_shape
    !! Boolean whether to keep shape

    ! Local variables
    logical :: keep_shape_
    !! Boolean whether to keep shape

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    deallocate(this%val)
    this%val_ptr => null()
    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array5d
!###############################################################################


!###############################################################################
  pure module subroutine set_array5d(this, input)
    !! Set 5D array
    implicit none

    ! Arguments
    class(array5d_type), intent(inout) :: this
    !! Instance of the 5D array type
    real(real32), dimension(..), intent(in) :: input
    !! Input array

    select rank(input)
    rank(1)
       this%val(:,1) = input
    rank(2)
       this%val(:,:) = input
    rank(5)
       this%val_ptr(:,:,:,:,:) = input
    rank default
       return
    end select
  end subroutine set_array5d
!###############################################################################


!###############################################################################
  module subroutine assign_array5d(this, input)
    !! Assign 5D array
    implicit none

    ! Arguments
    type(array5d_type), intent(out), target :: this
    !! Instance of the 5D array type
    type(array5d_type), intent(in) :: input
    !! Input array

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
    this%val_ptr( &
         1:this%shape(1), &
         1:this%shape(2), &
         1:this%shape(3), &
         1:this%shape(4), &
         1:size(this%val, dim=2) &
    ) => this%val
  end subroutine assign_array5d
!###############################################################################


!###############################################################################
  module subroutine set_ptr_array5d(this)
    !! Set pointer for 5D array
    implicit none

    ! Arguments
    class(array5d_type), intent(inout), target :: this
    !! Instance of the 5D array type

    if(.not. this%allocated)then
       call stop_program('Array not allocated')
       return
    end if
    this%val_ptr( &
         1:this%shape(1), &
         1:this%shape(2), &
         1:this%shape(3), &
         1:this%shape(4), &
         1:size(this%val, dim=2) &
    ) => this%val
  end subroutine set_ptr_array5d
!###############################################################################


!###############################################################################
  module subroutine finalise_array5d(this)
    !! Finalise 5D array
    implicit none

    ! Arguments
    type(array5d_type), intent(inout) :: this
    !! Instance of the 5D array type

    if(associated(this%val_ptr)) nullify(this%val_ptr)
    if(allocated(this%val)) deallocate(this%val)
    if(allocated(this%shape)) deallocate(this%shape)
  end subroutine finalise_array5d
!###############################################################################

end submodule athena__misc_types_submodule
