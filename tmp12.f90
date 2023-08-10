program main
  implicit none

  integer :: input_x=7, input_y=5, output_x=2, output_y=2
  integer :: stride_x = 2, stride_y = 2
  integer :: kernel_x = 3, kernel_y = 3
  integer :: num_channels = 1, num_filters = 1
  integer :: pad_x = 0, pad_y = 0
  
  integer :: i,j,l,m, a, b
  integer :: half_x, half_y, centre_x, centre_y
  integer :: iend_idx, jend_idx
  integer :: i_start,j_start,i_end,j_end,x_start,y_start,x_end,y_end
  integer :: ioffset, joffset, istride, jstride
  integer :: k_x, i_x, k_y, i_y, int1, int2
  
  half_x = kernel_x/2
  half_y = kernel_y/2
  centre_x = 2 - mod(kernel_x,2)
  centre_y = 2 - mod(kernel_y,2)
  iend_idx = half_x + (centre_x - 1)
  jend_idx = half_y + (centre_y - 1)
  ioffset  = 1 + half_x - pad_x
  joffset  = 1 + half_y - pad_y

  write(*,*) "LOOK HERE", mod(-5,3)

  k_x = kernel_x - 1
  i_x = input_x + stride_x
  int1 = kernel_x - input_x + (output_x -1)*stride_x
  
  write(*,*) "LOOK NED", int1
  !int1 = mod(input_x-1,2)
  !int2 = mod(input_x,2)

  
  !! do the 4 corners separately? This is kernel size dependent
  do concurrent( &
       i=1:input_x, &
       j=1:1,&!input_y, &
       m=1:num_channels, &
       l=1:num_filters &
       )

     istride = (i-ioffset)/stride_x + 1
     i_start = max(1,             istride)
     i_end   = min(output_x,      istride + (i-1)/stride_x )
     x_start = max(k_x-i,        -half_x + mod(abs(i_x-(i-int1)),stride_x))
     x_end   = min(input_x-i-k_x+1+int1, iend_idx - mod(abs(i_x+(i+int1)),stride_x))
     !x_start = max(k_x-i,        -half_x + mod(abs(i_x-(i+1)),stride_x))
     !x_end   = min(input_x-i-k_x, iend_idx - mod(abs(i_x+(i+1)),stride_x))
     !!x_start = max(1-i,     -half_x + mod(i-1,stride_x))
     !!x_end   = min(input_x - i-1, iend_idx - mod(i-1,stride_x))
     !if(x_start.gt.x_end) cycle


     jstride = (j-joffset)/stride_y + 1
     j_start = max(1,            jstride - half_y)
     j_end   = min(output_y,     jstride + half_y)
     y_start = max(1 - j,       -half_y) !check old vs new, was I use a rotated version?
     y_end   = min(output_y - j, jend_idx)    !check old vs new, was I use a rotated version?
     if(y_start.gt.y_end) cycle

     !! apply full convolution to compute input gradients
     write(*,*) i, j
     write(*,*) k_x-i,        -half_x + mod((i_x-(i-int1)),stride_x), "max"
     write(*,*) input_x-i-k_x, iend_idx - mod((i_x+(i+int1)),stride_x), "min",iend_idx
     write(*,*) x_end,x_start,-stride_x, "test", (/ (a, a=x_end,x_start,-stride_x) /)
     !write(*,*) y_end,y_start,-stride_y
     write(*,*) 
     !this%di(i,j,m) = &
     !     this%di(i,j,m) + &
     !     sum( &
     !     grad_dz(i_start:i_end,j_start:j_end,l) * &
     !     this%weight(x_end:x_start:-this%stride_x,y_end:y_start:-1,m,-this%stride_y) )

  end do


end program main
