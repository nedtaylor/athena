program test_maxpool1d_layer
  use coreutils, only: real32
  use athena, only: &
       maxpool1d_layer_type, &
       base_layer_type, &
       learnable_layer_type
  use athena__maxpool1d_layer, only: read_maxpool1d_layer
  use diffstruc, only: array_type, operator(+), operator(-)
  implicit none

  class(base_layer_type), allocatable, target :: pool_layer
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: num_channels = 3, pool = 3, stride = 2, width = 18
  integer :: unit
  real, parameter :: tol = 1.E-7
  logical :: success = .true.
  type(array_type) :: input(1,1), di_compare(1,1)
  type(array_type), pointer :: output, gradient

  integer :: i, c, output_width, max_loc
  integer :: ip1
  real, parameter :: max_value = 3.0


!-------------------------------------------------------------------------------
! set up layer
!-------------------------------------------------------------------------------
  pool_layer = maxpool1d_layer_type( &
       pool_size = pool, &
       stride = stride &
  )

  !! check layer name
  if(.not. pool_layer%name .eq. 'maxpool1d')then
     success = .false.
     write(0,*) 'maxpool1d layer has wrong name'
  end if

  !! check layer type
  select type(pool_layer)
  type is(maxpool1d_layer_type)
     !! check pool size
     if(any(pool_layer%pool .ne. pool))then
        success = .false.
        write(0,*) 'maxpool1d layer has wrong pool size'
     end if

     !! check stride size
     if(any(pool_layer%strd .ne. stride))then
        success = .false.
        write(0,*) 'maxpool1d layer has wrong stride size'
     end if

     !! check input shape allocated
     if(allocated(pool_layer%input_shape))then
        success = .false.
        write(0,*) 'maxpool1d layer shape should not be allocated yet'
     end if
  class default
     success = .false.
     write(0,*) 'maxpool1d layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! check layer input and output shape based on input layer
!-------------------------------------------------------------------------------
  !! initialise width and output width
  output_width = floor( (width - pool)/real(stride)) + 1
  max_loc = width / 2 + mod(width, 2)

  !! initialise sample input
  call input(1,1)%allocate(array_shape=[width, num_channels, 1], &
       source = 0._real32)
  call input(1,1)%set_requires_grad(.true.)
  do i = 1, num_channels
     input(1,1)%val(max_loc + (i-1)*width, 1) = max_value
  end do
  pool_layer = maxpool1d_layer_type( &
       pool_size = pool, &
       stride = stride &
  )

  !! check layer input and output shape based on input data
  call pool_layer%init(input(1,1)%shape)
  select type(pool_layer)
  type is(maxpool1d_layer_type)
     if(any(pool_layer%input_shape .ne. [width,num_channels]))then
        success = .false.
        write(0,*) 'maxpool1d layer has wrong input_shape'
     end if
     if(any(pool_layer%output_shape .ne. &
          [output_width,num_channels]))then
        success = .false.
        write(0,*) 'maxpool1d layer has wrong output_shape', &
             pool_layer%output_shape
        write(0,*) 'expected', [output_width,num_channels]
     end if
  end select


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! run forward pass
  call pool_layer%forward(input)
  output => pool_layer%output(1,1)

  !! check outputs have expected value
  do i = 1, output_width
     if(  max_loc .ge. (i-1)*stride + 1    .and. &
          max_loc .le. (i-1)*stride + pool )then
        if(abs(output%val(i, 1) - max_value) .gt. 1.E-6)then
           success = .false.
           write(*,*) 'maxpool1d layer forward pass failed'
        end if
     else if(abs(output%val(i, 1)) .gt. 1.E-6)then
        success = .false.
        write(*,*) 'maxpool1d layer forward pass failed'
     end if
  end do


!-------------------------------------------------------------------------------
! test backward pass and check expected output
!-------------------------------------------------------------------------------
  !! run backward pass
  gradient => output
  allocate(gradient%grad)
  gradient%grad = output
  call gradient%grad_reverse()
  call di_compare(1,1)%allocate(array_shape=[width,num_channels,1], source = 0.0)
  do c = 1, num_channels
     do i = 1, output_width
        do ip1 = (i-1) * stride + 1, (i-1) * stride + pool
           if(ip1.eq. max_loc)then
              di_compare(1,1)%val(ip1 + (c-1)*width,1) = &
                   di_compare(1,1)%val(ip1 + (c-1)*width,1) + gradient%val(i,1)
           end if
        end do
     end do
  end do

  !! check gradient has expected value
  if(any(abs(input(1,1)%grad%val(:,1) - di_compare(1,1)%val(:,1)) .gt. &
       tol))then
     success = .false.
     write(*,*) 'maxpool1d layer backward pass failed'
     write(*,*) di_compare(1,1)%val(:,1)
     write(*,*) input(1,1)%grad%val(:,1)
  end if


!-------------------------------------------------------------------------------
! check expected initialisation of pool and stride
!-------------------------------------------------------------------------------
  pool_layer = maxpool1d_layer_type( &
       pool_size = [2], &
       stride = [2] &
  )
  select type(pool_layer)
  type is (maxpool1d_layer_type)
     if(any(pool_layer%pool .ne. [2]))then
        success = .false.
        write(0,*) 'maxpool1d layer has wrong pool size'
     end if
     if(any(pool_layer%strd .ne. [2]))then
        success = .false.
        write(0,*) 'maxpool1d layer has wrong stride size'
     end if
  end select

  !! check expected initialisation of pool and stride
  pool_layer = maxpool1d_layer_type( &
       pool_size = [4], &
       stride = [4] &
  )
  select type(pool_layer)
  type is (maxpool1d_layer_type)
     if(any(pool_layer%pool .ne. 4))then
        success = .false.
        write(0,*) 'maxpool1d layer has wrong pool size'
     end if
     if(any(pool_layer%strd .ne. 4))then
        success = .false.
        write(0,*) 'maxpool1d layer has wrong stride size'
     end if
  end select


!-------------------------------------------------------------------------------
! Test padding functionality
!-------------------------------------------------------------------------------
  write(*,*) "Testing padding functionality..."

  ! Test with "constant" padding
  pool_layer = maxpool1d_layer_type( &
       pool_size = [3], &
       stride = [1], &
       padding = "constant" &
  )

  select type(pool_layer)
  type is(maxpool1d_layer_type)
     ! Check that pad_layer is allocated
     if(.not. allocated(pool_layer%pad_layer))then
        success = .false.
        write(0,*) 'maxpool1d layer pad_layer should be allocated ', &
             'with padding'
     end if
  end select

  ! Test with "valid" padding (no padding)
  pool_layer = maxpool1d_layer_type( &
       pool_size = [3], &
       stride = [1], &
       padding = "valid" &
  )

  select type(pool_layer)
  type is(maxpool1d_layer_type)
     ! Check that pad_layer is not allocated
     if(allocated(pool_layer%pad_layer))then
        success = .false.
        write(0,*) 'maxpool1d layer pad_layer should not be allocated ', &
             'with valid padding'
     end if
  end select


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."
  call pool_layer%init(input(1,1)%shape)

  ! Create a temporary file for testing
  open(newunit=unit, file='test_maxpool1d_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("MAXPOOL1D")')
  call pool_layer%print_to_unit(unit)
  write(unit,'("END MAXPOOL1D")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_maxpool1d_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_maxpool1d_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (maxpool1d_layer_type)
     if(.not. read_layer%name .eq. 'maxpool1d')then
        success = .false.
        write(0,*) 'read maxpool1d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not maxpool1d_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_maxpool1d_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_maxpool1d_layer passed all tests'
  else
     write(0,*) 'test_maxpool1d_layer failed one or more tests'
     stop 1
  end if

end program test_maxpool1d_layer
