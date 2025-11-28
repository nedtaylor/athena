program test_misc_types
  !! Unit tests for the misc_types module
  use coreutils, only: real32, test_error_handling
  use athena__misc_types, only: facets_type
  implicit none

  logical :: success = .true.
  real(real32), parameter :: tol = 1.E-6_real32
  integer :: i ! Loop index for array initialization

  ! Test instances
  type(facets_type) :: facets


!-------------------------------------------------------------------------------
! Test facets type
!-------------------------------------------------------------------------------
  write(*,*) "Testing facets type..."

  facets%rank = 2
  facets%nfixed_dims = 1
  call facets%setup_bounds([3, 4], [1, 2], 5) ! replication method

  if(facets%num .ne. 4)then ! 2D has 4 faces
     success = .false.
     write(0,*) 'facets type face count incorrect'
  end if

  if(facets%type .ne. "face")then
     success = .false.
     write(0,*) 'facets type string incorrect'
  end if


!-------------------------------------------------------------------------------
! Test facets functionality
!-------------------------------------------------------------------------------
  write(*,*) "Testing facets functionality..."

  block
    type(facets_type) :: facets_test

    ! Test basic facets initialization
    facets_test%num = 5
    facets_test%rank = 3
    facets_test%nfixed_dims = 2
    facets_test%type = 'face'

    if(facets_test%num .ne. 5)then
       success = .false.
       write(0,*) 'facets num assignment failed'
    end if

    if(facets_test%rank .ne. 3)then
       success = .false.
       write(0,*) 'facets rank assignment failed'
    end if

    if(facets_test%nfixed_dims .ne. 2)then
       success = .false.
       write(0,*) 'facets nfixed_dims assignment failed'
    end if

    if(trim(facets_test%type) .ne. 'face')then
       success = .false.
       write(0,*) 'facets type assignment failed'
    end if

    ! Test allocation of arrays
    allocate(facets_test%dim(3))
    allocate(facets_test%orig_bound(2, 2, 5))
    allocate(facets_test%dest_bound(2, 2, 5))

    facets_test%dim = [1, 2, 3]

    if(size(facets_test%dim) .ne. 3)then
       success = .false.
       write(0,*) 'facets dim allocation failed'
    end if

    if(any(facets_test%dim .ne. [1, 2, 3]))then
       success = .false.
       write(0,*) 'facets dim assignment failed'
    end if

    deallocate(facets_test%dim)
    deallocate(facets_test%orig_bound)
    deallocate(facets_test%dest_bound)
  end block


!-------------------------------------------------------------------------------
! Check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_misc_types passed all tests'
  else
     write(0,*) 'test_misc_types failed one or more tests'
     stop 1
  end if

end program test_misc_types
