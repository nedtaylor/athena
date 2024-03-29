program test_maxpool3d_layer
  use athena, only: &
       maxpool3d_layer_type, &
       base_layer_type, &
       learnable_layer_type
  implicit none

  class(base_layer_type), allocatable :: pool_layer
  integer, parameter :: num_channels = 3, pool = 3, stride = 2, width = 18
  real, allocatable, dimension(:,:,:,:,:) :: input_data, output, gradient
  real, allocatable, dimension(:) :: output_1d
  real, allocatable, dimension(:,:) :: output_2d
  real, parameter :: tol = 1.E-7
  logical :: success = .true.

  integer :: i, j, k, output_width, max_loc
  integer :: num_windows_i, num_windows_j, num_windows_k, num_windows
  real, parameter :: max_value = 3.0


!!!-----------------------------------------------------------------------------
!!! set up layer
!!!-----------------------------------------------------------------------------
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


!!!-----------------------------------------------------------------------------
!!! check layer input and output shape based on input layer
!!!-----------------------------------------------------------------------------
  !! initialise width and output width
  output_width = floor( (width - pool)/real(stride)) + 1
  max_loc = width / 2 + mod(width, 2)

  !! initialise sample input
  allocate(input_data(width, width, width, num_channels, 1), source = 0.0)
  input_data(max_loc, max_loc, max_loc, :, 1) = max_value
  pool_layer = maxpool3d_layer_type( &
       pool_size = pool, &
       stride = stride &
       )

  !! check layer input and output shape based on input layer
  call pool_layer%init(shape(input_data(:,:,:,:,1)), batch_size=1)
  select type(pool_layer)
  type is(maxpool3d_layer_type)
    if(any(pool_layer%input_shape .ne. [width,width,width,num_channels]))then
       success = .false.
       write(0,*) 'maxpool3d layer has wrong input_shape'
    end if
    if(any(pool_layer%output_shape .ne. &
         [output_width,output_width,output_width,num_channels]))then
       success = .false.
       write(0,*) 'maxpool3d layer has wrong output_shape', &
            pool_layer%output_shape
       write(0,*) 'expected', &
            [output_width,output_width,output_width,num_channels]
    end if
  end select


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! run forward pass
  call pool_layer%forward(input_data)
  call pool_layer%get_output(output)

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
             if(output(i, j, k, 1, 1) .ne. max_value)then
                success = .false.
                write(*,*) 'maxpool3d layer forward pass failed'
             end if
           else if(output(i, j, k, 1, 1) .ne. 0.0) then
              success = .false.
              write(*,*) 'maxpool3d layer forward pass failed'
           end if
        end do
     end do
  end do


!!!-----------------------------------------------------------------------------
!!! check output request using rank 1 and rank 2 arrays is consistent
!!!-----------------------------------------------------------------------------
  call pool_layer%get_output(output_1d)
  call pool_layer%get_output(output_2d)
  if(any(abs(output_1d - reshape(output_2d, [size(output_2d)])) .gt. 1.E-6))then
     success = .false.
     write(0,*) 'output_1d and output_2d are not consistent'
  end if


!!!-----------------------------------------------------------------------------
!!! test backward pass and check expected output
!!!-----------------------------------------------------------------------------
  !! run backward pass
  allocate(gradient, source = output)
  call pool_layer%backward(input_data, gradient)

  !! check gradient has expected value
  select type(current => pool_layer)
  type is(maxpool3d_layer_type)
     do i = 1, width
      num_windows_i = pool - stride + 1 - mod((stride+1)*(i-1),2)
      do j = 1, width
         num_windows_j = pool - stride + 1 - mod((stride+1)*(j-1),2)
         do k = 1, width
            num_windows_k = pool - stride + 1 - mod((stride+1)*(k-1),2)
            num_windows = num_windows_i * num_windows_j * num_windows_k
            if(all([i,j,k].eq.maxloc(input_data(:,:,:,1,1))))then
              if(current%di(i, j, k, 1, 1) .ne. maxval(output)*num_windows)then
                success = .false.
                write(*,*) num_windows_i, num_windows_j, num_windows_k
                write(*,*) 'maxpool3d layer backward pass failed'
              end if
            else
              if(current%di(i, j, k, 1, 1) .ne. 0.0) then
                success = .false.
                write(*,*) 'maxpool3d layer backward pass failed'
              end if
            end if
         end do
       end do
     end do
  end select


!!!-----------------------------------------------------------------------------
!!! check expected initialisation of pool and stride
!!!-----------------------------------------------------------------------------
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


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_maxpool3d_layer passed all tests'
  else
     write(0,*) 'test_maxpool3d_layer failed one or more tests'
     stop 1
  end if

end program test_maxpool3d_layer