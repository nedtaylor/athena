program test_avgpool2d_layer
  use athena, only: &
       avgpool2d_layer_type, &
       base_layer_type, &
       learnable_layer_type
   use custom_types, only: array4d_type
  implicit none

  class(base_layer_type), allocatable :: pool_layer
  integer, parameter :: num_channels = 3, pool = 3, stride = 3, width = 9
  real, allocatable, dimension(:) :: output_1d
  real, allocatable, dimension(:,:) :: output_2d
  real, allocatable, dimension(:,:,:,:) :: input_data, output, gradient, &
       di_compare
  real, parameter :: tol = 1.E-7
  logical :: success = .true.

  integer :: i, j, output_width, max_loc
  integer :: ip1, ip2, jp1, jp2
  real, parameter :: max_value = 3.0


  !! set up avgpool2d layer
  pool_layer = avgpool2d_layer_type( &
       pool_size = pool, &
       stride = stride &
       )

  !! check layer name
  if(.not. pool_layer%name .eq. 'avgpool2d')then
     success = .false.
     write(0,*) 'avgpool2d layer has wrong name'
  end if

!!!-----------------------------------------------------------------------------

  !! check layer type
  select type(pool_layer)
  type is(avgpool2d_layer_type)
     !! check pool size
     if(any(pool_layer%pool .ne. pool))then
        success = .false.
        write(0,*) 'avgpool2d layer has wrong pool size'
     end if

     !! check stride size
     if(any(pool_layer%strd .ne. stride))then
        success = .false.
        write(0,*) 'avgpool2d layer has wrong stride size'
     end if

     !! check input shape allocated
     if(allocated(pool_layer%input_shape))then
        success = .false.
        write(0,*) 'avgpool2d layer shape should not be allocated yet'
     end if
  class default
     success = .false.
     write(0,*) 'avgpool2d layer has wrong type'
  end select

!!!-----------------------------------------------------------------------------

  !! initialise width and output width
  output_width = floor( (width - pool)/real(stride)) + 1
  max_loc = floor(width / 2.0) + mod(width, 2)

  !! initialise sample input
  allocate(input_data(width, width, num_channels, 1), source = 0.0)
  input_data(max_loc, max_loc, :, 1) = max_value
  pool_layer = avgpool2d_layer_type( &
       pool_size = pool, &
       stride = stride &
       )

!!!-----------------------------------------------------------------------------

  !! check layer input and output shape based on input layer
  call pool_layer%init(shape(input_data(:,:,:,1)), batch_size=1)
  select type(pool_layer)
  type is(avgpool2d_layer_type)
    if(any(pool_layer%input_shape .ne. [width,width,num_channels]))then
       success = .false.
       write(0,*) 'avgpool2d layer has wrong input_shape'
    end if
    if(any( &
         pool_layer%output%shape .ne. &
         [output_width,output_width,num_channels] &
    ))then
       success = .false.
       write(0,*) 'avgpool2d layer has wrong output shape', &
            pool_layer%output%shape
       write(0,*) 'expected', [output_width,output_width,num_channels]
    end if
  end select

  !! run forward pass
  call pool_layer%forward(input_data)
  call pool_layer%get_output(output)

  !! check outputs have expected value
  do i = 1, output_width
     do j = 1, output_width
       if(  max_loc .ge. (i-1)*stride + 1    .and. &
            max_loc .le. (i-1)*stride + pool .and. &
            max_loc .ge. (j-1)*stride + 1    .and. &
            max_loc .le. (j-1)*stride + pool )then
         if(output(i, j, 1, 1) .ne. max_value/(pool*pool))then
            success = .false.
            write(*,*) 'avgpool2d layer forward pass failed'
         end if
       else if(output(i, j, 1, 1) .ne. 0.0) then
          success = .false.
          write(*,*) 'avgpool2d layer forward pass failed'
       end if
     end do
  end do

  !! check 1d and 2d output are the same
  call pool_layer%get_output(output_1d)
  call pool_layer%get_output(output_2d)
  if(any(abs(output_1d - &
       reshape(output, [output_width*output_width*num_channels])) &
       .gt. 1.E-6))then
     success = .false.
     write(*,*) 'avgpool2d layer output pass failed'
  end if
  if(any(abs(&
       reshape(output_2d, [output_width*output_width*num_channels]) - &
       reshape(output, [output_width*output_width*num_channels])) &
       .gt. 1.E-6))then
     success = .false.
     write(*,*) 'avgpool2d layer output pass failed'
  end if

!!!-----------------------------------------------------------------------------

  !! run backward pass
  allocate(gradient, source = output)
  call pool_layer%backward(input_data, gradient)
  allocate(di_compare(width,width,num_channels,1), source = 0.0)
  do i = 1, output_width
     ip1 = (i-1) * stride + 1
     ip2 = (i-1) * stride + pool
     do j = 1, output_width
        jp1 = (j-1) * stride + 1
        jp2 = (j-1) * stride + pool
        di_compare(ip1:ip2,jp1:jp2,1,1) = &
             di_compare(ip1:ip2,jp1:jp2,1,1) + gradient(i,j,1,1)/pool**2
     end do
  end do

  !! check gradient has expected value
  select type(di => pool_layer%di)
  type is(array4d_type)
     if(any(abs(di%val(:,:,1,1) - di_compare(:,:,1,1)) .gt. tol))then
        success = .false.
        write(*,*) 'avgpool2d layer backward pass failed'
     end if
  class default
     success = .false.
     write(0,*) 'avgpool2d layer has not set di type correctly'
  end select

  !! check backward pass recovers input (with division by pool**2)
  !! https://stats.stackexchange.com/questions/565032/cnn-upsampling-backprop-gradients-across-average-pooling-layer
  call pool_layer%forward(di_compare)
  call pool_layer%get_output(output)

   !! check outputs have expected value
  if (any(abs(output(:,:,1,1) - (gradient(:,:,1,1))/real(pool**2)) .gt. tol)) then
     success = .false.
     write(*,*) 'avgpool2d layer forward pass failed'
     do i = 1, width
     write(*,'(18(1X,F7.3))') input_data(i,:,1,1)
     end do
     write(*,*) "----------------------------------------"
     do i = 1, width
     write(*,'(18(1X,F7.3))') di_compare(i,:,1,1)
     end do
     write(*,*) "----------------------------------------"
     write(*,*) "----------------------------------------"
     do i = 1, output_width
     write(*,'(3(1X,F7.5))') gradient(i,:,1,1)/pool**2
     end do
     write(*,*) "----------------------------------------"
     do i = 1, output_width
     write(*,'(3(1X,F7.5))') output(i,:,1,1)
     end do

  end if

  !! check expected initialisation of pool and stride
  pool_layer = avgpool2d_layer_type( &
       pool_size = [2, 2], &
       stride = [2, 2] &
       )
  select type(pool_layer)
  type is (avgpool2d_layer_type)
     if(any(pool_layer%pool .ne. [2, 2]))then
        success = .false.
        write(0,*) 'avgpool2d layer has wrong pool size'
     end if
     if(any(pool_layer%strd .ne. [2, 2]))then
        success = .false.
        write(0,*) 'avgpool2d layer has wrong stride size'
     end if
  end select

  !! check expected initialisation of pool and stride
  pool_layer = avgpool2d_layer_type( &
       pool_size = [4], &
       stride = [4] &
       )
  select type(pool_layer)
  type is (avgpool2d_layer_type)
     if(any(pool_layer%pool .ne. 4))then
        success = .false.
        write(0,*) 'avgpool2d layer has wrong pool size'
     end if
     if(any(pool_layer%strd .ne. 4))then
        success = .false.
        write(0,*) 'avgpool2d layer has wrong stride size'
     end if
  end select

!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_avgpool2d_layer passed all tests'
  else
     write(0,*) 'test_avgpool2d_layer failed one or more tests'
     stop 1
  end if

end program test_avgpool2d_layer