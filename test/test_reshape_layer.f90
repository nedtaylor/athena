program test_reshape_layer
  use athena
  use coreutils, only: test_error_handling
  implicit none

  logical :: success = .true.
  real(real32), parameter :: tol = 1.E-5_real32


  ! Test 1: Flatten 2D to 1D
  call test_reshape_flatten_2d_to_1d(success)

  ! Test 2: Unflatten 1D to 2D
  call test_reshape_unflatten_1d_to_2d(success)

  ! Test 3: Reshape 3D to 2D
  call test_reshape_3d_to_2d(success)

  ! Test 4: Incompatible shapes
  call test_reshape_incompatible(success)

!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_reshape_layer passed all tests'
  else
     write(0,*) 'test_reshape_layer failed one or more tests'
     stop 1
  end if

contains

  !-----------------------------------------------------------------------------
  subroutine test_reshape_flatten_2d_to_1d(success)
    implicit none
    type(reshape_layer_type) :: layer
    integer, dimension(2) :: input_shape = [28, 28]
    integer, dimension(1) :: output_shape = [784]
    logical, intent(inout) :: success

    write(*,'("Testing reshape flatten 2D to 1D")')

    ! Create layer
    layer = reshape_layer_type( &
         output_shape=output_shape, &
         input_shape=input_shape, &
         verbose=0)

    ! Check output shape
    if(size(layer%output_shape) .ne. 1)then
       success = .false.
       write(*,'("  FAILED: Expected output rank 1, got ",I0)') size(layer%output_shape)
    end if
    if(layer%output_shape(1) .ne. 784)then
       success = .false.
       write(*,'("  FAILED: Expected output shape [784], got [",I0,"]")') &
            layer%output_shape(1)
    end if

  end subroutine test_reshape_flatten_2d_to_1d

  !-----------------------------------------------------------------------------
  subroutine test_reshape_unflatten_1d_to_2d(success)
    implicit none
    type(reshape_layer_type) :: layer
    integer, dimension(1) :: input_shape = [784]
    integer, dimension(2) :: output_shape = [28, 28]
    logical, intent(inout) :: success

    write(*,'("Testing reshape unflatten 1D to 2D")')

    layer = reshape_layer_type( &
         output_shape=output_shape, &
         input_shape=input_shape, &
         verbose=0)

    if(size(layer%output_shape) .ne. 2)then
       success = .false.
       write(*,'("  FAILED: Expected output rank 2, got ",I0)') size(layer%output_shape)
    end if
    if(any(layer%output_shape .ne. [28, 28]))then
       success = .false.
       write(*,'("  FAILED: Expected output shape [28,28], got [",2(I0,","),"]")') &
            layer%output_shape
    end if

  end subroutine test_reshape_unflatten_1d_to_2d

  !-----------------------------------------------------------------------------
  subroutine test_reshape_3d_to_2d(success)
    implicit none
    type(reshape_layer_type) :: layer
    integer, dimension(3) :: input_shape = [64, 32, 32]
    integer, dimension(2) :: output_shape = [64, 1024]
    logical, intent(inout) :: success

    write(*,'("Testing reshape 3D to 2D")')

    layer = reshape_layer_type( &
         output_shape=output_shape, &
         input_shape=input_shape, &
         verbose=0)

    if(size(layer%output_shape) .ne. 2)then
       success = .false.
       write(*,'("  FAILED: Expected output rank 2, got ",I0)') size(layer%output_shape)
    end if
    if(any(layer%output_shape .ne. [64, 1024]))then
       success = .false.
       write(*,'("  FAILED: Expected output shape [64,1024], got [",2(I0,","),"]")') &
            layer%output_shape
    end if

  end subroutine test_reshape_3d_to_2d

  !-----------------------------------------------------------------------------
  subroutine test_reshape_incompatible(success)
    implicit none
    type(reshape_layer_type) :: layer
    integer, dimension(2) :: input_shape = [28, 28]
    integer, dimension(1) :: output_shape = [22]
    logical, intent(inout) :: success

    test_error_handling = .true.
    write(*,'("Testing reshape incompatible shapes")')

    ! Create layer
    layer = reshape_layer_type( &
         output_shape=output_shape, &
         input_shape=input_shape, &
         verbose=0)

    test_error_handling = .false.
    write(*,'("Successfully detected incompatible reshape shapes")')

  end subroutine test_reshape_incompatible

end program test_reshape_layer
