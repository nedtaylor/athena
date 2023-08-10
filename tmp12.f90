program main
  implicit none

  integer :: input_x=5, input_y=5, output_x=2, output_y=2
  integer :: stride_x = 2, stride_y = 2
  integer :: kernel_x = 3, kernel_y = 3
  integer :: num_channels = 1, num_filters = 1
  integer :: pad_x = 0, pad_y = 0
  
  integer :: i,j,l,m, a, b
  integer :: half_x, half_y
  integer :: iend_idx, jend_idx
  integer :: i_start,j_start,i_end,j_end,x_start,y_start,x_end,y_end
  integer :: ioffset, joffset, istride, jstride
  integer :: k_x, k_y, int_x, int_y, n_stride_x, n_stride_y
  
  k_x = kernel_x - 1
  k_y = kernel_x - 1
  half_x = (k_x)/2
  half_y = (k_y)/2
  iend_idx = half_x + (1 - mod(kernel_x,2))
  jend_idx = half_y + (1 - mod(kernel_y,2))
  ioffset  = 1 + half_x - pad_x
  joffset  = 1 + half_y - pad_y

  !int_x = kernel_x - input_x + (output_x -1)*stride_x + (2 - kernel_x)
  int_x = (output_x - 1)*stride_x + 1 + iend_idx
  int_y = (output_y - 1)*stride_y + 1 + jend_idx
  n_stride_x = output_x * stride_x
  n_stride_y = output_y * stride_y
  

  
  !! do the 4 corners separately? This is kernel size dependent
  do concurrent( &
       i=1:input_x, &
       j=1:input_y, &
       m=1:num_channels, &
       l=1:num_filters &
       )

     istride = (i-ioffset)/stride_x + 1
     i_start = max(1,             istride)
     i_end   = min(output_x,      istride + (i-1)/stride_x )
     !! max( ...
     !!   ... 1. distance from first output to centre of scanning kernel
     !!   ... 2. current lowest output overlapping with left of kernel (this is a repeating pattern until 1. takes over as min)
     !!   ...)
     x_start = max(k_x-i,    -half_x  + mod(n_stride_x - i + kernel_x,stride_x))
     !! min( ...
     !!   ... 1. distance from final output to centre of scanning kernel
     !!   ... 2. current highest output overlapping with right of kernel (this is a repeating pattern until 1. takes over as max)
     !!   ...)
     x_end   = min(int_x-i,  iend_idx - mod(n_stride_x + i - 1,stride_x))
     if(x_start.gt.x_end) cycle


     jstride = (j-joffset)/stride_y + 1
     j_start = max(1,             jstride)
     j_end   = min(output_y,      jstride + (j-1)/stride_y )
     !! max( ...
     !!   ... 1. distance from first output to centre of scanning kernel
     !!   ... 2. current lowest output overlapping with left of kernel (this is a repeating pattern until 1. takes over as min)
     !!   ...)
     y_start = max(k_y-j,    -half_y  + mod(n_stride_y - j + kernel_y,stride_y))
     !! min( ...
     !!   ... 1. distance from final output to centre of scanning kernel
     !!   ... 2. current highest output overlapping with right of kernel (this is a repeating pattern until 1. takes over as max)
     !!   ...)
     y_end   = min(int_y-j,  jend_idx - mod(n_stride_y + j - 1,stride_y))
     if(y_start.gt.y_end) cycle

     !! apply full convolution to compute input gradients
     write(*,'(I0,2X,I0)') i, j
     write(*,*) (/ (a, a=x_end,x_start,-stride_x) /)
     write(*,*) 
     !this%di(i,j,m) = &
     !     this%di(i,j,m) + &
     !     sum( &
     !     grad_dz(i_start:i_end,j_start:j_end,l) * &
     !     this%weight(x_end:x_start:-this%stride_x,y_end:y_start:-1,m,-this%stride_y) )

  end do


end program main
