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
    integer :: i, j, k, l, facet_idx, idim
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
             do l = 1, this%rank
                this%orig_bound(:,l,facet_idx) = [ 1, length(l) ]
                this%dest_bound(:,l,facet_idx) = [ pad(l) + 1, pad(l) + length(l) ]
             end do

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
                do l = 1, this%rank
                   this%orig_bound(:,l,facet_idx) = [ 1, length(l) ]
                   this%dest_bound(:,l,facet_idx) = [ pad(l) + 1, pad(l) + length(l) ]
                end do

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
  module recursive subroutine deallocate_array(this, keep_shape)
    !! Deallocate array
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Instance of the array type
    logical, intent(in), optional :: keep_shape
    !! Boolean whether to keep shape

    ! Local variables
    logical :: keep_shape_
    !! Boolean whether to keep shape

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    if(allocated(this%val)) deallocate(this%val)
    if(allocated(this%indices)) deallocate(this%indices)
    if(allocated(this%adj_ja)) deallocate(this%adj_ja)
    if(allocated(this%mask)) deallocate(this%mask)

    ! Clean up gradients
    if(associated(this%grad) .and. this%owns_gradient) then
       call this%grad%deallocate()
       deallocate(this%grad)
    end if
    this%grad => null()
    this%owns_gradient = .false.

    ! Nullify computation graph pointers
    this%left_operand => null()
    this%right_operand => null()

    this%get_partial_left => null()
    this%get_partial_right => null()

    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array
!###############################################################################


!###############################################################################
  module subroutine set_ptr_array(this)
    !! Set pointer for array
    implicit none

    ! Arguments
    class(array_type), intent(inout), target :: this
    !! Instance of the array type

    if(.not. this%allocated)then
       call stop_program('Array not allocated')
       return
    end if
  end subroutine set_ptr_array
!###############################################################################


!###############################################################################
  module subroutine allocate_array(this, array_shape, source)
    !! Allocate array
    implicit none

    ! Arguments
    class(array_type), intent(inout), target :: this
    !! Instance of the array type
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    class(*), dimension(..), intent(in), optional :: source
    !! Source array

    if(allocated(this%val).or.this%allocated)then
       call stop_program("Trying to allocate already allocated array values")
       return
    end if
    if(present(array_shape))then
       allocate(this%val( &
            product(array_shape(1:size(array_shape)-1)),  &
            array_shape(size(array_shape)) &
       ))
       this%shape = array_shape(1:size(array_shape)-1)
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
          type is (array_type)
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source%val)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             end if
             this = source
             !  this%val_ptr( &
             !       1:source%shape(1), &
             !       1:size(source%val, dim=2) &
             !  ) => this%val
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
             !  this%val_ptr( &
             !       1:size(source, dim=1), &
             !       1:size(source, dim=2) &
             !  ) => this%val
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
    this%rank = 1
    this%allocated = .true.
    !  this%val_ptr(1:size(this%val, dim=1), 1:size(this%val, dim=2)) => this%val
    if(.not.allocated(this%shape)) this%shape = [ size(this%val, dim=1) ]
    this%size = product(this%shape)


  end subroutine allocate_array
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
    type(array_type), intent(in) :: input
    !! Input array

    this%rank = input%rank
    this%size = input%size
    this%is_sample_dependent = input%is_sample_dependent
    this%is_scalar = input%is_scalar
    this%allocated = input%allocated
    if(allocated(input%shape)) this%shape = input%shape
    if(allocated(input%val)) this%val = input%val
    this%requires_grad = input%requires_grad
    this%is_leaf = input%is_leaf
    if(associated(input%grad)) this%grad => input%grad
    if(associated(input%left_operand)) this%left_operand => input%left_operand
    if(associated(input%right_operand)) this%right_operand => input%right_operand
    this%operation = input%operation
    this%owns_gradient = .false.  ! Dont copy gradient ownership
    if(allocated(input%indices)) this%indices = input%indices
    if(allocated(input%adj_ja)) this%adj_ja = input%adj_ja
    if(allocated(input%mask)) this%mask = input%mask

    if(associated(input%get_partial_left)) &
         this%get_partial_left => input%get_partial_left
    if(associated(input%get_partial_right)) &
         this%get_partial_right => input%get_partial_right

    !  ! Don't copy pointers to avoid aliasing issues
    !  this%left_operand => null()
    !  this%right_operand => null()
    !  this%grad => null()

  end subroutine assign_array
!###############################################################################


!###############################################################################
  module recursive subroutine finalise_array(this)
    !! Finalise array - clean up memory safely
    type(array_type), intent(inout) :: this

    ! Clean up gradient if we own it
    if(associated(this%grad) .and. this%owns_gradient) then
       call this%grad%deallocate()
       deallocate(this%grad)
    end if

    ! Nullify pointers but don't deallocate targets (they may be used elsewhere)
    this%left_operand => null()
    this%right_operand => null()
    this%grad => null()
    this%owns_gradient = .false.
    this%get_partial_left => null()
    this%get_partial_right => null()
  end subroutine finalise_array
!###############################################################################

  !-----------------------------------------------------------------------------
  ! Core autodiff procedures
  !-----------------------------------------------------------------------------

  module recursive function forward_over_reverse(this, variable, itmp) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    class(array_type), intent(inout) :: variable
    type(array_type) :: output
    integer :: itmp

    integer :: s
    logical :: is_right_a_variable, is_left_a_variable
    type(array_type) :: left_deriv, right_deriv

    itmp = itmp + 1
    if(itmp.gt.50)then
       write(0,*) "MAX RECURSION DEPTH REACHED"
       return
    end if
    ! write(*,*) "Performing forward-over-reverse operation for: ", trim(this%operation)
    if(loc(this).eq.loc(variable))then
       call output%allocate(array_shape=[this%shape, size(this%val,2)])
       if(allocated(this%direction))then
          do s = 1, size(output%val,2)
             output%val(:,s) = this%direction
          end do
       else
          output%val(:,:) = 1._real32
       end if

    elseif(associated(this%left_operand).or.associated(this%right_operand))then
       ! if(associated(this%grad))then
       !    output = this%grad
       ! else
       is_left_a_variable = .false.
       if(associated(this%left_operand)) then
          if(associated(this%get_partial_left))then
             is_left_a_variable = .true.
             !if(associated(this%left_operand%grad))then
             !    left_deriv = this%left_operand%grad
             !else
             left_deriv = forward_over_reverse(this%left_operand, variable, itmp)
             call left_deriv%set_requires_grad(.false.)
             if( associated(this%get_partial_right) .and. &
                  .not.associated(this%right_operand) &
             )then
                left_deriv = this%get_partial_right(left_deriv)
             else
                left_deriv = this%get_partial_left(left_deriv)
             end if
             !end if
          end if
       end if

       is_right_a_variable = .false.
       if(associated(this%right_operand)) then
          if(associated(this%get_partial_right))then
             is_right_a_variable = .true.
             !if(associated(this%right_operand%grad))then
             !  right_deriv = this%right_operand%grad
             !else
             right_deriv = forward_over_reverse(this%right_operand, variable, itmp)
             call right_deriv%set_requires_grad(.false.)
             right_deriv = this%get_partial_right(right_deriv)
             !end if
          end if
       end if

       if(is_left_a_variable.and.is_right_a_variable)then
          output = left_deriv + right_deriv
       elseif(is_left_a_variable)then
          output = left_deriv
       elseif(is_right_a_variable) then
          output = right_deriv
       else
          write(*,*) "!!!SHOULDN'T!!!"
       end if

       ! end if
    else
       call output%allocate(array_shape=[this%shape, size(this%val,2)])
       output%val(:,:) = 0._real32
    end if

  end function forward_over_reverse



  subroutine backward_autodiff(this)
    !! Perform backward pass starting from this array
    class(array_type), intent(inout) :: this

    ! Initialize gradient if not allocated
    if(.not. associated(this%grad)) then
       allocate(this%grad)
       ! Safely initialize gradient without copying computation graph
       call this%grad%allocate(array_shape=[size(this%val,1), size(this%val,2)])
       this%grad%is_sample_dependent = this%is_sample_dependent
       this%grad%requires_grad = .true.
       this%grad%is_leaf = .true.
       this%grad%operation = 'none'
       this%grad%left_operand => null()
       this%grad%right_operand => null()
       this%grad%get_partial_left => null()
       this%grad%get_partial_right => null()
       this%grad%grad => null()
       this%grad%owns_gradient = .false.
       this%owns_gradient = .true.
       if(allocated(this%indices)) this%grad%indices = this%indices
       call this%grad%zero_grad()
       ! Set gradient to ones for starting node
       this%grad%val = 1.0_real32
    end if

    ! Recursively compute gradients
    call this%backward_op(this%grad)
  end subroutine backward_autodiff

  subroutine zero_grad_autodiff(this)
    !! Zero the gradients of this array
    class(array_type), intent(inout) :: this

    if(associated(this%grad)) then
       if(allocated(this%grad%val)) this%grad%val = 0.0_real32
    end if
  end subroutine zero_grad_autodiff

  subroutine detach_autodiff(this)
    !! Detach this array from the computation graph
    class(array_type), intent(inout) :: this

    this%requires_grad = .false.
    this%is_leaf = .true.
    this%operation = 'none'
    this%left_operand => null()
    this%right_operand => null()
  end subroutine detach_autodiff

  subroutine set_requires_grad_autodiff(this, requires_grad)
    !! Set the requires_grad flag
    class(array_type), intent(inout) :: this
    logical, intent(in) :: requires_grad

    this%requires_grad = requires_grad
  end subroutine set_requires_grad_autodiff

  module recursive subroutine backward_op_array(this, upstream_grad)
    !! Backward operation for arrays
    class(array_type), intent(inout) :: this
    class(array_type), intent(in) :: upstream_grad

    type(array_type), pointer :: left_partial, right_partial

    ! write(*,*) "Performing backward operation for: ", trim(this%operation)
    if(associated(this%left_operand))then
       if(this%left_operand%requires_grad) then
          allocate(left_partial)
          left_partial = this%get_partial_left(upstream_grad)
          call accumulate_gradient(this%left_operand, left_partial)
       end if
    end if
    if(associated(this%right_operand))then
       if(this%right_operand%requires_grad)then
          allocate(right_partial)
          right_partial = this%get_partial_right(upstream_grad)
          call accumulate_gradient(this%right_operand, right_partial)
       end if
    end if
  end subroutine backward_op_array

  recursive subroutine accumulate_gradient(array, grad)
    !! Accumulate gradient for array with safe memory management
    class(array_type), intent(inout) :: array
    class(array_type), intent(in), pointer :: grad

    integer :: s
    class(array_type), pointer :: directed_grad

    if(allocated(array%direction))then
       allocate(directed_grad)
       directed_grad = grad
       do s = 1, size(grad%val, 2)
          directed_grad%val(:, s) = grad%val(:, s) * array%direction
       end do
    else
       directed_grad => grad
    end if

    if(.not. associated(array%grad)) then
       if(array%is_sample_dependent)then
          array%grad => directed_grad
       else
          allocate(array%grad)
          ! Safely initialize gradient without copying computation graph
          call array%grad%allocate(array_shape=[size(array%val,1), &
               size(array%val,2)])
          array%grad%val = 0.0_real32
          array%grad%is_scalar = array%is_scalar
          array%grad%is_sample_dependent = array%is_sample_dependent
          if(.not.array%is_scalar)then
             array%grad%requires_grad = .true.
          else
             array%grad%requires_grad = .false.
          end if
          array%grad%is_leaf = .true.
          array%grad%operation = 'none'
          array%grad%left_operand => null()
          array%grad%right_operand => null()
          array%grad%get_partial_left => null()
          array%grad%get_partial_right => null()
          array%grad%grad => null()
          array%grad%owns_gradient = .false.
          array%owns_gradient = .true.
          !  if(allocated(array%indices)) array%grad%indices = array%indices
          call array%grad%zero_grad()

          do s = 1, size(grad%val, 2)
             array%grad%val(:,1) = directed_grad%val(:,s)
          end do
       end if
    else

       if(array%is_sample_dependent)then
          ! array%grad%val = array%grad%val + grad%val
          array%grad => array%grad + directed_grad
       else
          !! NEED TO FIX THIS ONE
          do s = 1, size(grad%val, 2)
             array%grad%val(:,1) = array%grad%val(:,1) + directed_grad%val(:,s)
          end do
       end if

    end if

    if(.not. array%is_leaf) then
       call array%backward_op(directed_grad)
    end if
  end subroutine accumulate_gradient

  module function create_result_array(this, shape_arr) result(result_ptr)
    !! Helper function to safely create result arrays with proper initialization
    class(array_type), intent(in) :: this
    integer, dimension(:), intent(in), optional :: shape_arr
    type(array_type), pointer :: result_ptr

    allocate(result_ptr)

    if(present(shape_arr)) then
       call result_ptr%allocate(array_shape=shape_arr)
    else
       if(allocated(this%shape))then
          call result_ptr%allocate(array_shape=[this%shape, &
               size(this%val,2)])
       else
          call result_ptr%allocate(array_shape=shape(this%val))
       end if
    end if

    ! Initialize autodiff fields
    result_ptr%requires_grad = .false.
    result_ptr%is_leaf = .true.
    result_ptr%is_scalar = this%is_scalar
    result_ptr%is_sample_dependent = this%is_sample_dependent
    result_ptr%operation = 'none'
    result_ptr%owns_gradient = .false.
    result_ptr%left_operand => null()
    result_ptr%right_operand => null()
    result_ptr%get_partial_left => null()
    result_ptr%get_partial_right => null()
  end function create_result_array


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
    if(present(array_shape))then
       if(size(array_shape) .ne. 1) then
          call stop_program('Array shape must be of size 1 for 1D array')
          return
       end if
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
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             end if
             this%val = source
             this%val_ptr( 1:size(source, dim=1) ) => this%val
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
    this%rank = 1
    this%allocated = .true.
    this%shape = shape(this%val_ptr)
    this%size = product(this%shape)

  end subroutine allocate_array1d
!###############################################################################


!###############################################################################
  module subroutine deallocate_array1d(this, keep_shape)
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
    if(allocated(this%val)) deallocate(this%val)
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
    if(present(array_shape))then
       if(size(array_shape) .ne. 2) then
          call stop_program('Array shape must be of size 2 for 2D array')
          return
       end if
       allocate(this%val(array_shape(1), array_shape(2)))
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
             this%val_ptr( &
                  1:size(source, dim=1), &
                  1:size(source, dim=2) &
             ) => this%val
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
    this%rank = 1
    this%allocated = .true.
    this%val_ptr(1:size(this%val, dim=1), 1:size(this%val, dim=2)) => this%val
    this%shape = [ size(this%val, dim=1) ]
    this%size = product(this%shape)

  end subroutine allocate_array2d
!###############################################################################


!###############################################################################
  module subroutine deallocate_array2d(this, keep_shape)
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

    this%val_ptr => null()
    call this%array_type%deallocate()
    !  keep_shape_ = .false.
    !  if(present(keep_shape)) keep_shape_ = keep_shape
    !  if(.not.keep_shape_) this%shape = 0
    !  if(allocated(this%val)) deallocate(this%val)
    !  this%allocated = .false.
    !  this%size = 0

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
    if(present(array_shape))then
       if(size(array_shape) .ne. 3) then
          call stop_program('Array shape must be of size 3 for 3D array')
          return
       end if
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
    this%rank = 2
    this%allocated = .true.
    this%shape = shape(this%val_ptr(:,:,1))
    this%size = product(this%shape)

  end subroutine allocate_array3d
!###############################################################################


!###############################################################################
  module subroutine deallocate_array3d(this, keep_shape)
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
    if(allocated(this%val)) deallocate(this%val)
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
    if(present(array_shape))then
       if(size(array_shape) .ne. 4) then
          call stop_program('Array shape must be of size 4 for 4D array')
          return
       end if
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
    this%rank = 3
    this%allocated = .true.
    this%shape = shape(this%val_ptr(:,:,:,1))
    this%size = product(this%shape)

  end subroutine allocate_array4d
!###############################################################################


!###############################################################################
  module subroutine deallocate_array4d(this, keep_shape)
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
    if(allocated(this%val)) deallocate(this%val)
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
    if(present(array_shape))then
       if(size(array_shape) .ne. 5) then
          call stop_program('Array shape must be of size 5 for 5D array')
          return
       end if
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
    this%rank = 4
    this%allocated = .true.
    this%shape = shape(this%val_ptr(:,:,:,:,1))
    this%size = product(this%shape)

  end subroutine allocate_array5d
!###############################################################################


!###############################################################################
  module subroutine deallocate_array5d(this, keep_shape)
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
    if(allocated(this%val)) deallocate(this%val)
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
