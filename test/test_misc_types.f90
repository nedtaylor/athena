program test_misc_types
  !! Unit tests for the misc_types module
  use athena__constants, only: real32
  use athena__misc_types, only: array1d_type, array2d_type, array3d_type, &
       array4d_type, array5d_type, &
       array_container_type, facets_type, array_type
  use athena__io_utils, only: test_error_handling
  implicit none

  logical :: success = .true.
  real(real32), parameter :: tol = 1.0e-6_real32
  integer :: i ! Loop index for array initialization

  ! Test arrays
  class(array_type), allocatable :: arr_allocatable
  type(array1d_type) :: arr1d
  type(array2d_type) :: arr2d
  type(array3d_type) :: arr3d
  type(array4d_type) :: arr4d
  type(array5d_type) :: arr5d
  type(array_container_type) :: container
  type(facets_type) :: facets

  ! Test data
  integer, allocatable :: test_idata_2d(:,:)
  real(real32), allocatable :: test_data_1d(:)
  real(real32), allocatable :: test_data_2d(:,:)
  real(real32), allocatable :: test_data_3d(:,:,:)
  real(real32), allocatable :: test_data_4d(:,:,:,:)
  real(real32), allocatable :: test_data_5d(:,:,:,:,:)


!-------------------------------------------------------------------------------
! Test 1D array type
!-------------------------------------------------------------------------------
  write(*,*) "Testing 1D array type..."

  ! Test allocation
  call arr1d%allocate([5])
  if(.not. arr1d%allocated)then
     success = .false.
     write(0,*) 'array1d allocation failed'
  end if

  if(arr1d%rank .ne. 1)then
     success = .false.
     write(0,*) 'array1d rank incorrect'
  end if

  if(arr1d%size .ne. 5)then
     success = .false.
     write(0,*) 'array1d size incorrect'
  end if

  if(any(arr1d%shape .ne. [5]))then
     success = .false.
     write(0,*) 'array1d shape incorrect'
  end if

  ! Test setting values
  allocate(test_data_1d(5))
  test_data_1d = [1.0_real32, 2.0_real32, 3.0_real32, 4.0_real32, 5.0_real32]
  call arr1d%set(test_data_1d)

  if(any(abs(arr1d%val_ptr - test_data_1d) .gt. tol))then
     success = .false.
     write(0,*) 'array1d set values failed'
  end if

  ! Test deallocation
  call arr1d%deallocate()
  if(arr1d%allocated)then
     success = .false.
     write(0,*) 'array1d deallocation failed'
  end if

  deallocate(test_data_1d)


!-------------------------------------------------------------------------------
! Test 2D array type
!-------------------------------------------------------------------------------
  write(*,*) "Testing 2D array type..."

  ! Test allocation
  call arr2d%allocate([3, 4])
  if(.not. arr2d%allocated)then
     success = .false.
     write(0,*) 'array2d allocation failed'
  end if

  if(arr2d%rank .ne. 1)then
     success = .false.
     write(0,*) 'array2d rank incorrect', arr2d%rank
  end if

  if(arr2d%size .ne. 3)then
     success = .false.
     write(0,*) 'array2d size incorrect', arr2d%size
  end if

  if(any(arr2d%shape .ne. [3]))then
     success = .false.
     write(0,*) 'array2d shape incorrect', arr2d%shape
  end if

  ! Test setting values
  allocate(test_data_2d(3, 4))
  test_data_2d = reshape([1.0_real32, 2.0_real32, 3.0_real32, &
       4.0_real32, 5.0_real32, 6.0_real32, &
       7.0_real32, 8.0_real32, 9.0_real32, &
       10.0_real32, 11.0_real32, 12.0_real32], [3, 4])
  call arr2d%set(test_data_2d)

  if(any(abs(arr2d%val_ptr - test_data_2d) .gt. tol))then
     success = .false.
     write(0,*) 'array2d set values failed'
  end if

  ! Test deallocation
  call arr2d%deallocate()
  if(arr2d%allocated)then
     success = .false.
     write(0,*) 'array2d deallocation failed'
  end if

  deallocate(test_data_2d)


!-------------------------------------------------------------------------------
! Test 3D array type
!-------------------------------------------------------------------------------
  write(*,*) "Testing 3D array type..."

  ! Test allocation
  call arr3d%allocate([2, 3, 2])
  if(.not. arr3d%allocated)then
     success = .false.
     write(0,*) 'array3d allocation failed'
  end if

  if(arr3d%rank .ne. 2)then
     success = .false.
     write(0,*) 'array3d rank incorrect', arr3d%rank
  end if

  if(arr3d%size .ne. 6)then
     success = .false.
     write(0,*) 'array3d size incorrect', arr3d%size
  end if

  if(any(arr3d%shape .ne. [2, 3]))then
     success = .false.
     write(0,*) 'array3d shape incorrect', arr3d%shape
  end if

  ! Test setting values
  allocate(test_data_3d(2, 3, 2))
  test_data_3d = reshape([1.0_real32, 2.0_real32, 3.0_real32, 4.0_real32, &
       5.0_real32, 6.0_real32, 7.0_real32, 8.0_real32, &
       9.0_real32, 10.0_real32, 11.0_real32, 12.0_real32], &
  [2, 3, 2])
  call arr3d%set(test_data_3d)

  if(any(abs(arr3d%val_ptr - test_data_3d) .gt. tol))then
     success = .false.
     write(0,*) 'array3d set values failed'
  end if

  ! Test deallocation
  call arr3d%deallocate()
  if(arr3d%allocated)then
     success = .false.
     write(0,*) 'array3d deallocation failed'
  end if

  deallocate(test_data_3d)


!-------------------------------------------------------------------------------
! Test 4D array type
!-------------------------------------------------------------------------------
  write(*,*) "Testing 4D array type..."

  ! Test allocation
  call arr4d%allocate([2, 2, 2, 2])
  if(.not. arr4d%allocated)then
     success = .false.
     write(0,*) 'array4d allocation failed'
  end if

  if(arr4d%rank .ne. 3)then
     success = .false.
     write(0,*) 'array4d rank incorrect', arr4d%rank
  end if

  if(arr4d%size .ne. 8)then
     success = .false.
     write(0,*) 'array4d size incorrect', arr4d%size
  end if

  if(any(arr4d%shape .ne. [2, 2, 2]))then
     success = .false.
     write(0,*) 'array4d shape incorrect', arr4d%shape
  end if

  ! Test setting values
  allocate(test_data_4d(2, 2, 2, 2))
  test_data_4d = reshape([(real(i, real32), i = 1, 16)], [2, 2, 2, 2])
  call arr4d%set(test_data_4d)

  if(any(abs(arr4d%val_ptr - test_data_4d) .gt. tol))then
     success = .false.
     write(0,*) 'array4d set values failed'
  end if

  ! Test deallocation
  call arr4d%deallocate()
  if(arr4d%allocated)then
     success = .false.
     write(0,*) 'array4d deallocation failed'
  end if

  deallocate(test_data_4d)


!-------------------------------------------------------------------------------
! Test 5D array type
!-------------------------------------------------------------------------------
  write(*,*) "Testing 5D array type..."

  ! Test allocation
  call arr5d%allocate([2, 2, 2, 2, 1])
  if(.not. arr5d%allocated)then
     success = .false.
     write(0,*) 'array5d allocation failed'
  end if

  if(arr5d%rank .ne. 4)then
     success = .false.
     write(0,*) 'array5d rank incorrect', arr5d%rank
  end if

  if(arr5d%size .ne. 16)then
     success = .false.
     write(0,*) 'array5d size incorrect', arr5d%size
  end if

  if(any(arr5d%shape .ne. [2, 2, 2, 2]))then
     success = .false.
     write(0,*) 'array5d shape incorrect', arr5d%shape
  end if

  ! Test setting values
  allocate(test_data_5d(2, 2, 2, 2, 1))
  test_data_5d = reshape([(real(i, real32), i = 1, 16)], [2, 2, 2, 2, 1])
  call arr5d%set(test_data_5d)

  if(any(abs(arr5d%val_ptr - test_data_5d) .gt. tol))then
     success = .false.
     write(0,*) 'array5d set values failed'
  end if

  ! Test deallocation
  call arr5d%deallocate()
  if(arr5d%allocated)then
     success = .false.
     write(0,*) 'array5d deallocation failed'
  end if

  deallocate(test_data_5d)


!-------------------------------------------------------------------------------
! Test array assignment interface
!-------------------------------------------------------------------------------
  ! write(*,*) "Testing array assignment interface..."

  ! ! Test 2D array assignment
  ! call arr2d%allocate([2, 3])
  ! allocate(test_data_2d(2, 3))
  ! test_data_2d = reshape([1.0_real32, 2.0_real32, 3.0_real32, &
  !                         4.0_real32, 5.0_real32, 6.0_real32], [2, 3])
  ! call arr2d%set(test_data_2d)

  ! block
  !   type(array2d_type) :: arr2d_copy

  !   ! Test assignment operator
  !   arr2d_copy = arr2d

  !   if(.not. arr2d_copy%allocated)then
  !      success = .false.
  !      write(0,*) 'array2d assignment failed - not allocated'
  !   end if

  !   if(any(arr2d_copy%shape .ne. [2, 3]))then
  !      success = .false.
  !      write(0,*) 'array2d assignment failed - wrong shape'
  !   end if

  !   if(any(abs(arr2d_copy%val_ptr - test_data_2d) .gt. tol))then
  !      success = .false.
  !      write(0,*) 'array2d assignment failed - wrong values'
  !   end if

  !   call arr2d_copy%deallocate()
  ! end block

  ! call arr2d%deallocate()
  ! deallocate(test_data_2d)


!-------------------------------------------------------------------------------
! Test array container type
!-------------------------------------------------------------------------------
  write(*,*) "Testing array container type..."

  allocate(container%array, source=array2d_type())
  select type(arr => container%array)
  type is (array2d_type)
     call arr%allocate([2, 2])
     allocate(test_data_2d(2, 2))
     test_data_2d = reshape([1.0_real32, 2.0_real32, &
          3.0_real32, 4.0_real32], [2, 2])
     call arr%set(test_data_2d)

     if(any(abs(arr%val_ptr - test_data_2d) .gt. tol))then
        success = .false.
        write(0,*) 'array container test failed'
     end if

     call arr%deallocate()
     deallocate(test_data_2d)
  class default
     success = .false.
     write(0,*) 'array container type allocation failed'
  end select


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
! Test edge case: zero-size arrays
!-------------------------------------------------------------------------------
  write(*,*) "Testing edge cases..."
  test_error_handling = .true. ! Enable error handling for tests

  ! Test allocation with zero size (should handle gracefully)
  call arr1d%allocate([0, 0])
  if(arr1d%size .ne. 0)then
     success = .false.
     write(0,*) 'zero-size array1d failed'
  end if
  call arr1d%deallocate()

  call arr2d%allocate([0])
  if(arr2d%size .ne. 0)then
     success = .false.
     write(0,*) 'zero-size array2d failed'
  end if
  call arr2d%deallocate()

  call arr3d%allocate([0])
  if(arr3d%size .ne. 0)then
     success = .false.
     write(0,*) 'zero-size array3d failed'
  end if
  call arr3d%deallocate()

  call arr4d%allocate([0])
  if(arr4d%size .ne. 0)then
     success = .false.
     write(0,*) 'zero-size array4d failed'
  end if
  call arr4d%deallocate()

  call arr5d%allocate([0])
  if(arr5d%size .ne. 0)then
     success = .false.
     write(0,*) 'zero-size array5d failed'
  end if
  call arr5d%deallocate()
  test_error_handling = .false. ! Enable error handling for tests


!-------------------------------------------------------------------------------
! Test keep_shape deallocation
!-------------------------------------------------------------------------------
  write(*,*) "Testing keep_shape deallocation..."

  call arr2d%allocate([3, 3])
  call arr2d%deallocate(keep_shape=.true.)

  if(arr2d%allocated)then
     success = .false.
     write(0,*) 'keep_shape deallocation still allocated'
  end if

  if(.not. allocated(arr2d%shape))then
     success = .false.
     write(0,*) 'keep_shape deallocation lost shape'
  end if

  if(any(arr2d%shape .ne. [3]))then
     success = .false.
     write(0,*) 'keep_shape deallocation wrong shape'
  end if


!-------------------------------------------------------------------------------
! Test array_container functionality
!-------------------------------------------------------------------------------
  write(*,*) "Testing array_container functionality..."

  block
    type(array_container_type) :: container1

    call arr1d%allocate([5])
    ! Test assigning arrays to container
    allocate(container1%array, source=arr1d)

    select type(contained_array => container1%array)
    type is (array1d_type)
       if(.not. associated(contained_array%val_ptr))then
          success = .false.
          write(0,*) 'container assignment failed - array not associated'
       end if
       if(size(contained_array%val_ptr) .ne. 5)then
          success = .false.
          write(0,*) 'container assignment failed - wrong size'
       end if
    class default
       success = .false.
       write(0,*) 'container assignment failed - wrong type'
    end select

    call arr1d%deallocate(); call arr1d%allocate([3]); container1%array = arr1d
    select type(contained_array => container1%array)
    type is (array1d_type)
       if(size(contained_array%val_ptr) .ne. 3)then
          success = .false.
          write(0,*) 'container assignment failed - wrong size'
       end if
    end select
    deallocate(container1%array)


    call arr2d%allocate([5, 5])
    allocate(container1%array, source=arr2d)
    select type(contained_array => container1%array)
    type is (array2d_type)
       if(.not. associated(contained_array%val_ptr))then
          success = .false.
          write(0,*) 'container assignment failed - array not associated'
       end if
       if(size(contained_array%val_ptr) .ne. 25)then
          success = .false.
          write(0,*) 'container assignment failed - wrong size'
       end if
    class default
       success = .false.
       write(0,*) 'container assignment failed - wrong type'
    end select

    call arr2d%deallocate(); call arr2d%allocate([3, 4]); container1%array = arr2d
    select type(contained_array => container1%array)
    type is (array2d_type)
       if(size(contained_array%val_ptr) .ne. 12)then
          success = .false.
          write(0,*) 'container assignment failed - wrong size'
       end if
    end select
    deallocate(container1%array)


    call arr3d%allocate([2, 3, 4])
    allocate(container1%array, source=arr3d)
    select type(contained_array => container1%array)
    type is (array3d_type)
       if(.not. associated(contained_array%val_ptr))then
          success = .false.
          write(0,*) 'container assignment failed - array not associated'
       end if
       if(size(contained_array%val_ptr) .ne. 24)then
          success = .false.
          write(0,*) 'container assignment failed - wrong size'
       end if
    class default
       success = .false.
       write(0,*) 'container assignment failed - wrong type'
    end select

    call arr3d%deallocate(); call arr3d%allocate([2, 2, 2]); container1%array = arr3d
    select type(contained_array => container1%array)
    type is (array3d_type)
       if(size(contained_array%val_ptr) .ne. 8)then
          success = .false.
          write(0,*) 'container assignment failed - wrong size'
       end if
    end select
    deallocate(container1%array)


    call arr4d%allocate([2, 2, 2, 2])
    allocate(container1%array, source=arr4d)
    select type(contained_array => container1%array)
    type is (array4d_type)
       if(.not. associated(contained_array%val_ptr))then
          success = .false.
          write(0,*) 'container assignment failed - array not associated'
       end if
       if(size(contained_array%val_ptr) .ne. 16)then
          success = .false.
          write(0,*) 'container assignment failed - wrong size'
       end if
    class default
       success = .false.
       write(0,*) 'container assignment failed - wrong type'
    end select

    call arr4d%deallocate(); call arr4d%allocate([2, 2, 2, 1])
    container1%array = arr4d
    select type(contained_array => container1%array)
    type is (array4d_type)
       if(size(contained_array%val_ptr) .ne. 8)then
          success = .false.
          write(0,*) 'container assignment failed - wrong size'
       end if
    end select
    deallocate(container1%array)


    call arr5d%allocate([2, 2, 2, 2, 1])
    allocate(container1%array, source=arr5d)
    select type(contained_array => container1%array)
    type is (array5d_type)
       if(.not. associated(contained_array%val_ptr))then
          success = .false.
          write(0,*) 'container assignment failed - array not associated'
       end if
       if(size(contained_array%val_ptr) .ne. 16)then
          success = .false.
          write(0,*) 'container assignment failed - wrong size'
       end if
    class default
       success = .false.
       write(0,*) 'container assignment failed - wrong type'
    end select

    call arr5d%deallocate(); call arr5d%allocate([2, 2, 2, 2, 1])
    container1%array = arr5d
    select type(contained_array => container1%array)
    type is (array5d_type)
       if(size(contained_array%val_ptr) .ne. 16)then
          success = .false.
          write(0,*) 'container assignment failed - wrong size'
       end if
    end select
    deallocate(container1%array)

  end block


!-------------------------------------------------------------------------------
! Test array initialisation
!-------------------------------------------------------------------------------
  write(*,*) "Testing array initialisation..."

  arr1d = array1d_type([5])
  if(.not. arr1d%allocated)then
     success = .false.
     write(0,*) 'array1d initialization failed'
  end if
  call arr1d%deallocate()

  arr2d = array2d_type([3, 4])
  if(.not. arr2d%allocated)then
     success = .false.
     write(0,*) 'array2d initialization failed'
  end if
  call arr2d%deallocate()

  arr3d = array3d_type([2, 3, 4])
  if(.not. arr3d%allocated)then
     success = .false.
     write(0,*) 'array3d initialization failed'
  end if
  call arr3d%deallocate()

  arr4d = array4d_type([2, 2, 2, 2])
  if(.not. arr4d%allocated)then
     success = .false.
     write(0,*) 'array4d initialization failed'
  end if
  call arr4d%deallocate()

  arr5d = array5d_type([2, 2, 2, 2, 1])
  if(.not. arr5d%allocated)then
     success = .false.
     write(0,*) 'array5d initialization failed'
  end if
  call arr5d%deallocate()


!-------------------------------------------------------------------------------
! Test incorrect allocation through wrong source type
!-------------------------------------------------------------------------------
  test_error_handling = .true. ! Enable error handling for tests
  write(*,*) "Testing incorrect allocation through wrong source type..."

  call arr1d%allocate(source=array2d_type([3]))
  if(arr1d%allocated)then
     success = .false.
     write(0,*) 'array1d allocation with array1d source should have failed'
  end if

  call arr2d%allocate(source=array1d_type([3]))
  if(arr2d%allocated)then
     success = .false.
     write(0,*) 'array2d allocation with array1d source should have failed'
  end if

  call arr3d%allocate(source=array4d_type([2, 2, 2, 2]))
  if(arr3d%allocated)then
     success = .false.
     write(0,*) 'array3d allocation with array4d source should have failed'
  end if

  call arr4d%allocate(source=array3d_type([2, 2, 2]))
  if(arr4d%allocated)then
     success = .false.
     write(0,*) 'array4d allocation with array3d source should have failed'
  end if

  call arr5d%allocate(source=array2d_type([3, 4]))
  if(arr5d%allocated)then
     success = .false.
     write(0,*) 'array5d allocation with array2d source should have failed'
  end if

  call arr1d%allocate(source=[1])
  call arr2d%allocate(source=[1])
  call arr3d%allocate(source=[1])
  call arr4d%allocate(source=[1])
  call arr5d%allocate(source=[1])

  if(arr1d%allocated .or. arr2d%allocated .or. arr3d%allocated .or. &
       arr4d%allocated .or. arr5d%allocated)then
     success = .false.
     write(0,*) 'array allocation with scalar source should have failed'
  end if

  allocate(test_data_1d(5), test_data_2d(3, 4), test_data_3d(2, 3, 2), &
       test_data_4d(2, 2, 2, 2), test_data_5d(2, 2, 2, 2, 1))
  call arr1d%allocate(source=test_data_4d)
  call arr2d%allocate(source=test_data_4d)
  call arr3d%allocate(source=test_data_4d)
  call arr4d%allocate(source=test_data_3d)
  call arr5d%allocate(source=test_data_4d)
  if(arr1d%allocated .or. arr2d%allocated .or. arr3d%allocated .or. &
       arr4d%allocated .or. arr5d%allocated)then
     success = .false.
     write(0,*) 'array allocation with incompatible source should have failed'
  end if

  allocate(test_idata_2d(3, 4))
  call arr1d%allocate(source=test_idata_2d)
  call arr2d%allocate(source=test_idata_2d)
  call arr3d%allocate(source=test_idata_2d)
  call arr4d%allocate(source=test_idata_2d)
  call arr5d%allocate(source=test_idata_2d)
  if(arr1d%allocated .or. arr2d%allocated .or. arr3d%allocated .or. &
       arr4d%allocated .or. arr5d%allocated)then
     success = .false.
     write(0,*) 'array allocation with integer source should have failed'
  end if


  test_error_handling = .false. ! Enable error handling for tests

  call arr1d%allocate(source=test_data_2d)
  call arr2d%allocate(source=test_data_2d)
  call arr3d%allocate(source=test_data_2d)
  call arr4d%allocate(source=test_data_2d)
  call arr5d%allocate(source=test_data_2d)
  if(.not. arr1d%allocated .or. &
       .not. arr2d%allocated .or. .not. arr3d%allocated .or. &
       .not. arr4d%allocated .or. .not. arr5d%allocated)then
     success = .false.
     write(0,*) 'array allocation with compatible source should have succeeded'
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

  ! Clean up all arrays
  call arr1d%deallocate()
  call arr2d%deallocate()


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
