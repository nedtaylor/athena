program test_maxpool3d_layer
  use coreutils, only: real32
  use athena, only: &
       maxpool3d_layer_type, &
       base_layer_type, &
       learnable_layer_type
  use athena__maxpool3d_layer, only: read_maxpool3d_layer
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

  integer :: i, j, k, c, output_width, max_loc
  integer :: ip1, jp1, kp1
  real, parameter :: max_value = 3.0


!-------------------------------------------------------------------------------
! set up layer
!-------------------------------------------------------------------------------
  pool_layer = maxpool3d_layer_type( &
       pool_size = pool, &
       stride = stride &
  )

  !! check layer name
  if(.not. pool_layer%name .eq. 'maxpool3d')then
     success = .false.
     write(0,*) 'maxpool3d layer has wrong name'
  end if

  !! check layer type
  select type(pool_layer)
  type is(maxpool3d_layer_type)
     !! check pool size
     if(any(pool_layer%pool .ne. pool))then
        success = .false.
        write(0,*) 'maxpool3d layer has wrong pool size'
     end if

     !! check stride size
     if(any(pool_layer%strd .ne. stride))then
        success = .false.
        write(0,*) 'maxpool3d layer has wrong stride size'
     end if

     !! check input shape allocated
     if(allocated(pool_layer%input_shape))then
        success = .false.
        write(0,*) 'maxpool3d layer shape should not be allocated yet'
     end if
  class default
     success = .false.
     write(0,*) 'maxpool3d layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! check layer input and output shape based on input layer
!-------------------------------------------------------------------------------
  !! initialise width and output width
  output_width = floor( (width - pool)/real(stride)) + 1
  max_loc = width / 2 + mod(width, 2)

  !! initialise sample input
  call input(1,1)%allocate(&
       array_shape=[width, width, width, num_channels, 1], &
       source = 0._real32)
  call input(1,1)%set_requires_grad(.true.)
  do c = 1, num_channels
     input(1,1)%val(&
          max_loc + (max_loc-1)*width + (max_loc-1)*width*width + &
          (c-1)*width*width*width, 1) = max_value
  end do
  pool_layer = maxpool3d_layer_type( &
       pool_size = pool, &
       stride = stride &
  )

  !! check layer input and output shape based on input layer
  call pool_layer%init(input(1,1)%shape)
  select type(pool_layer)
  type is(maxpool3d_layer_type)
     if(any(pool_layer%input_shape .ne. [width,width,width,num_channels]))then
        success = .false.
        write(0,*) 'maxpool3d layer has wrong input_shape'
     end if
     if(any(pool_layer%output_shape .ne. &
          [output_width,output_width,output_width,num_channels]))then
        success = .false.
        write(0,*) 'maxpool3d layer has wrong output shape', &
             pool_layer%output_shape
        write(0,*) 'expected', &
             [output_width,output_width,output_width,num_channels]
     end if
  end select


!-------------------------------------------------------------------------------
! test forward pass and check expected output
! use existing layer
!-------------------------------------------------------------------------------
  !! run forward pass
  call pool_layer%forward(input)
  output => pool_layer%output(1,1)

  !! check outputs have expected value
  do i = 1, output_width
     do j = 1, output_width
        do k = 1, output_width
           if(  max_loc .ge. (i-1)*stride + 1    .and. &
                max_loc .le. (i-1)*stride + pool .and. &
                max_loc .ge. (j-1)*stride + 1    .and. &
                max_loc .le. (j-1)*stride + pool .and. &
                max_loc .ge. (k-1)*stride + 1    .and. &
                max_loc .le. (k-1)*stride + pool )then
              if( abs( &
                   output%val(&
                        i + (j-1)*output_width + &
                        (k-1)*output_width*output_width, 1 &
                   ) - max_value &
              ) .gt. 1.E-6 )then
                 success = .false.
                 write(*,*) 'maxpool3d layer forward pass failed'
              end if
           else if( abs( output%val(&
                i + (j-1)*output_width + &
                (k-1)*output_width*output_width, 1) ) .gt. 1.E-6 ) then
              success = .false.
              write(*,*) 'maxpool3d layer forward pass failed'
           end if
        end do
     end do
  end do


!-------------------------------------------------------------------------------
! check output request using rank 1 and rank 2 arrays is consistent
!-------------------------------------------------------------------------------
!  call pool_layer%get_output(output_1d)
!  call pool_layer%get_output(output_2d)
!  if(any(abs(output_1d - reshape(output_2d, [size(output_2d)])) .gt. 1.E-6))then
!     success = .false.
!     write(0,*) 'output_1d and output_2d are not consistent'
!  end if


!-------------------------------------------------------------------------------
! test backward pass and check expected output
!-------------------------------------------------------------------------------
  !! run backward pass
  gradient => output
  allocate(gradient%grad)
  gradient%grad = output
  call gradient%grad_reverse()
  call di_compare(1,1)%allocate(&
       array_shape=[width,width,width,num_channels,1], source = 0.0)
  do c = 1, num_channels
     do i = 1, output_width
        do j = 1, output_width
           do k = 1, output_width
              do ip1 = (i-1) * stride + 1, (i-1) * stride + pool
                 do jp1 = (j-1) * stride + 1, (j-1) * stride + pool
                    do kp1 = (k-1) * stride + 1, (k-1) * stride + pool
                       if(all([ip1,jp1,kp1].eq. &
                            [max_loc,max_loc,max_loc]))then
                          di_compare(1,1)%val(&
                               ip1 + (jp1-1)*width + (kp1-1)*width*width + &
                               (c-1)*width*width*width, 1) = &
                               di_compare(1,1)%val(&
                                    ip1 + (jp1-1)*width + (kp1-1)*width*width + &
                                    (c-1)*width*width*width, 1 &
                               ) + &
                               gradient%val(&
                                    i + (j-1)*output_width + &
                                    (k-1)*output_width*output_width, 1)
                       end if
                    end do
                 end do
              end do
           end do
        end do
     end do
  end do

  !! check gradient has expected value
  if(any(abs(input(1,1)%grad%val(:,1) - di_compare(1,1)%val(:,1)) .gt. &
       tol))then
     success = .false.
     write(*,*) 'maxpool3d layer backward pass failed'
     write(*,*) 'Expected:', di_compare(1,1)%val(:,1)
     write(*,*) 'Got:', input(1,1)%grad%val(:,1)
  end if


!-------------------------------------------------------------------------------
! check expected initialisation of pool and stride
!-------------------------------------------------------------------------------
  pool_layer = maxpool3d_layer_type( &
       pool_size = [2, 2, 2], &
       stride = [2, 2, 2] &
  )
  select type(pool_layer)
  type is (maxpool3d_layer_type)
     if(any(pool_layer%pool .ne. [2, 2, 2]))then
        success = .false.
        write(0,*) 'maxpool3d layer has wrong pool size'
     end if
     if(any(pool_layer%strd .ne. [2, 2, 2]))then
        success = .false.
        write(0,*) 'maxpool3d layer has wrong stride size'
     end if
  end select

  !! check expected initialisation of pool and stride
  pool_layer = maxpool3d_layer_type( &
       pool_size = [4], &
       stride = [4] &
  )
  select type(pool_layer)
  type is (maxpool3d_layer_type)
     if(any(pool_layer%pool .ne. 4))then
        success = .false.
        write(0,*) 'maxpool3d layer has wrong pool size'
     end if
     if(any(pool_layer%strd .ne. 4))then
        success = .false.
        write(0,*) 'maxpool3d layer has wrong stride size'
     end if
  end select


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."
  call pool_layer%init(input(1,1)%shape)

  ! Create a temporary file for testing
  open(newunit=unit, file='test_maxpool3d_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("MAXPOOL3D")')
  call pool_layer%print_to_unit(unit)
  write(unit,'("END MAXPOOL3D")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_maxpool3d_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_maxpool3d_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (maxpool3d_layer_type)
     if (.not. read_layer%name .eq. 'maxpool3d') then
        success = .false.
        write(0,*) 'read maxpool3d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not maxpool3d_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_maxpool3d_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_maxpool3d_layer passed all tests'
  else
     write(0,*) 'test_maxpool3d_layer failed one or more tests'
     stop 1
  end if

end program test_maxpool3d_layer
