submodule(athena__misc_types) athena__misc_types_submodule
  !! Submodule containing implementations for derived types
  use coreutils, only: stop_program, print_warning



contains

!###############################################################################
  pure module function create_attribute(name, type, val) result(attribute)
    !! Function to create an ONNX attribute
    implicit none

    ! Arguments
    character(*), intent(in) :: name
    !! Name of the attribute
    character(*), intent(in) :: type
    !! Type of the attribute
    character(*), intent(in) :: val
    !! Value of the attribute as a string

    type(onnx_attribute_type) :: attribute
    !! Resulting ONNX attribute

    if(len_trim(name) .gt. 64)then
       attribute%name = name(1:64)
    else
       attribute%name = trim(name)
    end if

    if(len_trim(type) .gt. 10)then
       attribute%type = type(1:10)
    else
       attribute%type = trim(type)
    end if

    attribute%val = trim(val)
  end function create_attribute
!###############################################################################


!###############################################################################
  module subroutine print_to_unit_actv(this, unit, identifier)
    !! Interface for printing activation function details
    implicit none

    ! Arguments
    class(base_actv_type), intent(in) :: this
    !! Instance of the activation type
    integer, intent(in) :: unit
    !! Unit number for output
    character(*), intent(in), optional :: identifier
    !! Optional identifier for the activation function

    ! Local variables
    integer :: i
    !! Loop index
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes


    attributes = this%export_attributes()

    if(present(identifier))then
       write(unit,'(3X,"ACTIVATION: ",A)') trim(identifier)
    else
       write(unit,'(3X,"ACTIVATION")')
    end if
    do i = 1, size(attributes)
       write(unit,'(6X,A," = ",A)') &
            trim(attributes(i)%name), trim(attributes(i)%val)
    end do
    write(unit,'(3X,"END ACTIVATION")')

  end subroutine print_to_unit_actv
!###############################################################################


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
    if(this%rank .lt. this%nfixed_dims)then
       call stop_program("Number of fixed dimensions exceeds rank")
       return
    end if


    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%dim)) deallocate(this%dim)
    if(allocated(this%orig_bound)) deallocate(this%orig_bound)
    if(allocated(this%dest_bound)) deallocate(this%dest_bound)

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
                if(j .eq. 1)then
                   this%orig_bound(:,i,facet_idx) = &
                        [ length(i) - pad(i) + 1, length(i) ]
                else
                   this%orig_bound(:,i,facet_idx) = [ 1, pad(i) ]
                end if
             case(4) ! reflection
                if(j .eq. 1)then
                   this%orig_bound(:,i,facet_idx) = [ pad(i) + 1, 2 ]
                else
                   this%orig_bound(:,i,facet_idx) = &
                        [ length(i) - 1, length(i) - pad(i) ]
                end if
             case(5) ! replication
                if(j .ne. 1) this%orig_bound(:,i,facet_idx) = length(i)
             end select

             ! Set destination bounds
             if(j .eq. 1)then
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
                   if(btest_k1)then
                      this%orig_bound(:,i,facet_idx) = &
                           [ 1, pad(i) ]
                   else
                      this%orig_bound(:,i,facet_idx) = &
                           [ length(i) - pad(i) + 1, length(i) ]
                   end if
                   if(btest_k0)then
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
             if(btest(i-1, this%rank-j))then
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

end submodule athena__misc_types_submodule
