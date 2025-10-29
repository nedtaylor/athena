submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_pool
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function avgpool1d(input, pool_size, stride) result(output)
    !! 1D average pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, intent(in) :: pool_size
    integer, intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, m, s
    integer :: stride_idx, idx
    integer, dimension(3) :: output_shape

    output_shape = [ &
         (input%shape(1) - pool_size) / stride + 1, &
         input%shape(2), &
         size(input%val, dim=2)]
    output => input%create_result(array_shape = output_shape)
    do concurrent(&
         s = 1:output_shape(3), &
         m = 1:output_shape(2), &
         i = 1:output_shape(1))
       stride_idx = (i - 1) * stride + (m - 1) * input%shape(1)
       idx = i + (m - 1) * output_shape(1)
       output%val(idx, s) = sum( &
            input%val( stride_idx + 1 : stride_idx + pool_size, s ) &
       ) / pool_size
    end do
    allocate(output%adj_ja(1,2))
    output%adj_ja(1,1) = pool_size
    output%adj_ja(1,2) = stride

    output%get_partial_left => get_partial_avgpool1d
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'avgpool'
       output%left_operand = input
    end if

  end function avgpool1d
!-------------------------------------------------------------------------------
  function get_partial_avgpool1d(this, upstream_grad) result(output)
    !! Get the partial derivative for average pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, m, s
    integer :: stride_idx, idx
    real(real32) :: pool_norm
    integer, dimension(3) :: input_shape

    pool_norm = 1.0_real32 / real(this%adj_ja(1,1), real32)
    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    call output%allocate(array_shape = input_shape)
    output%val = 0._real32

    do concurrent(&
         s = 1:this%shape(3), &
         m = 1:this%shape(2), &
         i = 1:this%shape(1))
       stride_idx = (i - 1) * this%adj_ja(1,2) + (m - 1) * input_shape(1)
       idx = i + (m - 1) * this%shape(1)
       output%val( stride_idx + 1 : stride_idx + this%adj_ja(1,1), s ) = &
            output%val( stride_idx + 1 : stride_idx + this%adj_ja(1,1), s ) + &
            upstream_grad%val(idx, s) * pool_norm
    end do

  end function get_partial_avgpool1d
!###############################################################################


!###############################################################################
  module function avgpool2d(input, pool_size, stride) result(output)
    !! 2D average pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, dimension(2), intent(in) :: pool_size
    integer, dimension(2), intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, m, s, i_step, j_step
    integer :: stride_idx, idx, multiplier
    real(real32) :: pool_sum, pool_norm, channel_size_in, channel_size_out
    integer, dimension(4) :: output_shape

    output_shape = [ &
         (input%shape(1) - pool_size(1)) / stride(1) + 1, &
         (input%shape(2) - pool_size(2)) / stride(2) + 1, &
         input%shape(3), &
         size(input%val, dim=2)]
    output => input%create_result(array_shape = output_shape)
    pool_norm = 1.0_real32 / real(pool_size(1) * pool_size(2), real32)
    channel_size_in = real( input%shape(1) * input%shape(2), real32 )
    channel_size_out = real( output_shape(1) * output_shape(2), real32 )
    do concurrent(&
         s = 1:output_shape(4), &
         m = 1:output_shape(3), &
         j = 1:output_shape(2), &
         i = 1:output_shape(1))
       pool_sum = 0._real32
       do j_step = 0, pool_size(2)-1
          do i_step = 0, pool_size(1)-1
             stride_idx = (i-1)*stride(1) + i_step + &
                  ((j-1)*stride(2) + j_step) * input%shape(1) + &
                  (m-1) * channel_size_in
             pool_sum = pool_sum + input%val(stride_idx + 1, s)
          end do
       end do
       idx = i + (j - 1) * output_shape(1) + (m - 1) * channel_size_out
       output%val(idx, s) = pool_sum * pool_norm
    end do
    allocate(output%adj_ja(2,2))
    output%adj_ja(:,1) = pool_size
    output%adj_ja(:,2) = stride

    output%get_partial_left => get_partial_avgpool2d
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'avgpool'
       output%left_operand = input
    end if

  end function avgpool2d
!-------------------------------------------------------------------------------
  function get_partial_avgpool2d(this, upstream_grad) result(output)
    !! Get the partial derivative for average pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, j, m, s
    integer :: i_step, j_step
    integer :: in_idx, out_idx
    real(real32) :: pool_norm, channel_size_in, channel_size_out
    integer, dimension(4) :: input_shape, pool_size, stride

    ! Unpack parameters
    input_shape = this%left_operand%shape
    pool_size = this%adj_ja(:,1)
    stride    = this%adj_ja(:,2)
    channel_size_in = real( input_shape(1) * input_shape(2), real32 )
    channel_size_out = real( this%shape(1) * this%shape(2), real32 )

    call output%allocate(array_shape = input_shape)
    output%val = 0._real32

    pool_norm = 1.0_real32 / real(pool_size(1) * pool_size(2), real32)

    do concurrent( &
         s = 1:this%shape(4), &    ! batch
         m = 1:this%shape(3), &    ! channels
         j = 1:this%shape(2), &  ! pooled width
         i = 1:this%shape(1))    ! pooled height

       ! Distribute gradient over pooling window
       do j_step = 0, pool_size(2) - 1
          do i_step = 0, pool_size(1) - 1

             in_idx = ( (i-1) * stride(1) + i_step ) + &
                  ( (j-1) * stride(2) + j_step ) * input_shape(1) + &
                  (m-1) * channel_size_in

             out_idx = i + (j-1) * upstream_grad%shape(1) + &
                  (m-1) *  channel_size_out

             output%val(in_idx + 1, s) = output%val(in_idx + 1, s) + &
                  upstream_grad%val(out_idx, s) * pool_norm

          end do
       end do
    end do

  end function get_partial_avgpool2d
!###############################################################################


!###############################################################################
  module function avgpool3d(input, pool_size, stride) result(output)
    !! 3D average pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, dimension(3), intent(in) :: pool_size
    integer, dimension(3), intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, k, m, s
    integer :: i_step, j_step, k_step
    integer :: stride_idx, idx
    real(real32) :: pool_sum, pool_norm, channel_size_in, channel_size_out
    integer, dimension(5) :: output_shape

    ! output_shape = [H_out, W_out, D_out, C, B]
    output_shape = [ &
         (input%shape(1) - pool_size(1)) / stride(1) + 1, &
         (input%shape(2) - pool_size(2)) / stride(2) + 1, &
         (input%shape(3) - pool_size(3)) / stride(3) + 1, &
         input%shape(4), &
         size(input%val, dim=2) ]

    output => input%create_result(array_shape = output_shape)
    pool_norm = 1.0_real32 / real(product(pool_size), real32)
    channel_size_in = &
         real( input%shape(1) * input%shape(2) * input%shape(3), real32 )
    channel_size_out = &
         real( output_shape(1) * output_shape(2) * output_shape(3), real32 )

    do concurrent( &
         s = 1:output_shape(5), &
         m = 1:output_shape(4), &
         k = 1:output_shape(3), &
         j = 1:output_shape(2), &
         i = 1:output_shape(1))

       pool_sum = 0._real32

       do k_step = 0, pool_size(3)-1
          do j_step = 0, pool_size(2)-1
             do i_step = 0, pool_size(1)-1
                stride_idx = ((i-1)*stride(1) + i_step) + &
                     ((j-1)*stride(2) + j_step) * input%shape(1) + &
                     ((k-1)*stride(3) + k_step) * &
                     input%shape(1) * input%shape(2) + &
                     (m-1) * channel_size_in
                pool_sum = pool_sum + input%val(stride_idx + 1, s)
             end do
          end do
       end do

       idx = i + (j-1) * output_shape(1) + &
            (k-1) * output_shape(1)*output_shape(2) + &
            (m-1) * channel_size_out

       output%val(idx, s) = pool_sum * pool_norm
    end do

    allocate(output%adj_ja(3,2))
    output%adj_ja(:,1) = pool_size
    output%adj_ja(:,2) = stride

    output%get_partial_left => get_partial_avgpool3d
    if (input%requires_grad) then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'avgpool3d'
       output%left_operand = input
    end if

  end function avgpool3d
!-------------------------------------------------------------------------------
  function get_partial_avgpool3d(this, upstream_grad) result(output)
    !! Get the partial derivative for 3D average pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, j, k, m, s
    integer :: i_step, j_step, k_step
    integer :: in_idx, out_idx
    real(real32) :: pool_norm, channel_size_in, channel_size_out
    integer, dimension(5) :: input_shape
    integer, dimension(3) :: pool_size, stride

    ! Unpack parameters
    input_shape = this%left_operand%shape
    pool_size = this%adj_ja(:,1)
    stride    = this%adj_ja(:,2)

    call output%allocate(array_shape = input_shape)
    output%val = 0._real32

    pool_norm = 1.0_real32 / real(product(pool_size), real32)
    channel_size_in = &
         real( input_shape(1) * input_shape(2) * input_shape(3), real32 )
    channel_size_out = &
         real( this%shape(1) * this%shape(2) * this%shape(3), real32 )

    do concurrent( &
         s = 1:this%shape(5), &     ! batch
         m = 1:this%shape(4), &     ! channels
         k = 1:this%shape(3), &  ! pooled depth
         j = 1:this%shape(2), &  ! pooled width
         i = 1:this%shape(1))    ! pooled height

       ! Distribute gradient over pooling window
       do k_step = 0, pool_size(3)-1
          do j_step = 0, pool_size(2)-1
             do i_step = 0, pool_size(1)-1
                in_idx = ((i-1)*stride(1) + i_step) + &
                     ((j-1)*stride(2) + j_step) * input_shape(1) + &
                     ((k-1)*stride(3) + k_step) * input_shape(1)*input_shape(2) + &
                     (m-1) * channel_size_in

                out_idx = i + (j-1) * upstream_grad%shape(1) + &
                     (k-1) * upstream_grad%shape(1)*upstream_grad%shape(2) + &
                     (m-1) * channel_size_out

                output%val(in_idx + 1, s) = output%val(in_idx + 1, s) + &
                     upstream_grad%val(out_idx, s) * pool_norm
             end do
          end do
       end do
    end do

  end function get_partial_avgpool3d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function maxpool1d(input, pool_size, stride) result(output)
    !! 1D max pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, intent(in) :: pool_size
    integer, intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, m, s
    integer :: stride_idx, idx
    integer, dimension(3) :: output_shape

    output_shape = [ &
         (input%shape(1) - pool_size) / stride + 1, &
         input%shape(2), &
         size(input%val, dim=2)]
    output => input%create_result(array_shape = output_shape)
    do concurrent(&
         s = 1:output_shape(3), &
         m = 1:output_shape(2), &
         i = 1:output_shape(1))
       stride_idx = (i - 1) * stride + (m - 1) * input%shape(1)
       idx = i + (m - 1) * output_shape(1)
       output%val(idx, s) = maxval( &
            input%val( stride_idx + 1 : stride_idx + pool_size, s ) &
       )
    end do
    allocate(output%adj_ja(1,2))
    output%adj_ja(1,1) = pool_size
    output%adj_ja(1,2) = stride

    output%get_partial_left => get_partial_maxpool1d
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'maxpool'
       output%left_operand = input
    end if

  end function maxpool1d
!-------------------------------------------------------------------------------
  function get_partial_maxpool1d(this, upstream_grad) result(output)
    !! Get the partial derivative for max pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, m, s, max_idx
    integer :: stride_idx, idx, p
    integer, dimension(3) :: input_shape

    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    call output%allocate(array_shape = input_shape)
    output%val = 0._real32

    do concurrent(&
         s = 1:this%shape(3), &
         m = 1:this%shape(2), &
         i = 1:this%shape(1))
       stride_idx = (i - 1) * this%adj_ja(1,2) + (m - 1) * input_shape(1)
       idx = i + (m - 1) * this%shape(1)

       ! Find the max value index within the pooling window
       max_idx = maxloc( &
            this%left_operand%val( &
                 stride_idx + 1 : stride_idx + this%adj_ja(1,1), s &
            ), dim = 1 )

       output%val( stride_idx + max_idx, s ) = &
            output%val( stride_idx + max_idx, s ) + upstream_grad%val(idx, s)
    end do

  end function get_partial_maxpool1d
!###############################################################################


!###############################################################################
  module function maxpool2d(input, pool_size, stride) result(output)
    !! 2D max pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, dimension(2), intent(in) :: pool_size
    integer, dimension(2), intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, m, s, i_step, j_step
    integer :: stride_idx, idx
    real(real32) :: pool_max, channel_size_in, channel_size_out
    integer, dimension(4) :: output_shape

    output_shape = [ &
         (input%shape(1) - pool_size(1)) / stride(1) + 1, &
         (input%shape(2) - pool_size(2)) / stride(2) + 1, &
         input%shape(3), &
         size(input%val, dim=2)]
    output => input%create_result(array_shape = output_shape)
    channel_size_in = real( input%shape(1) * input%shape(2), real32 )
    channel_size_out = real( output_shape(1) * output_shape(2), real32 )
    do concurrent(&
         s = 1:output_shape(4), &
         m = 1:output_shape(3), &
         j = 1:output_shape(2), &
         i = 1:output_shape(1))
       pool_max = -huge(1.0_real32)
       do j_step = 0, pool_size(2)-1
          do i_step = 0, pool_size(1)-1
             stride_idx = (i-1)*stride(1) + i_step + &
                  ((j-1)*stride(2) + j_step) * input%shape(1) + &
                  (m-1) * channel_size_in
             pool_max = max(pool_max, input%val(stride_idx + 1, s))
          end do
       end do
       idx = i + (j - 1) * output_shape(1) + (m - 1) * channel_size_out
       output%val(idx, s) = pool_max
    end do
    allocate(output%adj_ja(2,2))
    output%adj_ja(:,1) = pool_size
    output%adj_ja(:,2) = stride

    output%get_partial_left => get_partial_maxpool2d
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'maxpool'
       output%left_operand = input
    end if

  end function maxpool2d
!-------------------------------------------------------------------------------
  function get_partial_maxpool2d(this, upstream_grad) result(output)
    !! Get the partial derivative for max pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, j, m, s
    integer :: i_step, j_step
    integer :: in_idx, out_idx, max_i, max_j
    real(real32) :: pool_max, channel_size_in, channel_size_out
    integer, dimension(4) :: input_shape, pool_size, stride

    ! Unpack parameters
    input_shape = this%left_operand%shape
    pool_size = this%adj_ja(:,1)
    stride    = this%adj_ja(:,2)
    channel_size_in = real( input_shape(1) * input_shape(2), real32 )
    channel_size_out = real( this%shape(1) * this%shape(2), real32 )

    call output%allocate(array_shape = input_shape)
    output%val = 0._real32

    do s = 1, this%shape(4)
       do m = 1, this%shape(3)
          do j = 1, this%shape(2)
             do i = 1, this%shape(1)
                ! Find max value location in pooling window
                pool_max = -huge(1.0_real32)
                max_i = 0
                max_j = 0

                do j_step = 0, pool_size(2) - 1
                   do i_step = 0, pool_size(1) - 1
                      in_idx = ( (i-1) * stride(1) + i_step ) + &
                           ( (j-1) * stride(2) + j_step ) * input_shape(1) + &
                           (m-1) * channel_size_in

                      if (this%left_operand%val(in_idx + 1, s) > pool_max) then
                         pool_max = this%left_operand%val(in_idx + 1, s)
                         max_i = i_step
                         max_j = j_step
                      end if
                   end do
                end do

                ! Assign gradient to max location
                in_idx = ( (i-1) * stride(1) + max_i ) + &
                     ( (j-1) * stride(2) + max_j ) * input_shape(1) + &
                     (m-1) * channel_size_in

                out_idx = i + (j-1) * upstream_grad%shape(1) + &
                     (m-1) * channel_size_out

                output%val(in_idx + 1, s) = output%val(in_idx + 1, s) + &
                     upstream_grad%val(out_idx, s)
             end do
          end do
       end do
    end do

  end function get_partial_maxpool2d
!###############################################################################


!###############################################################################
  module function maxpool3d(input, pool_size, stride) result(output)
    !! 3D max pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, dimension(3), intent(in) :: pool_size
    integer, dimension(3), intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, k, m, s
    integer :: i_step, j_step, k_step
    integer :: stride_idx, idx
    real(real32) :: pool_max, channel_size_in, channel_size_out
    integer, dimension(5) :: output_shape

    ! output_shape = [H_out, W_out, D_out, C, B]
    output_shape = [ &
         (input%shape(1) - pool_size(1)) / stride(1) + 1, &
         (input%shape(2) - pool_size(2)) / stride(2) + 1, &
         (input%shape(3) - pool_size(3)) / stride(3) + 1, &
         input%shape(4), &
         size(input%val, dim=2) ]

    output => input%create_result(array_shape = output_shape)
    channel_size_in = &
         real( input%shape(1) * input%shape(2) * input%shape(3), real32 )
    channel_size_out = &
         real( output_shape(1) * output_shape(2) * output_shape(3), real32 )

    do concurrent( &
         s = 1:output_shape(5), &
         m = 1:output_shape(4), &
         k = 1:output_shape(3), &
         j = 1:output_shape(2), &
         i = 1:output_shape(1))

       pool_max = -huge(1.0_real32)

       do k_step = 0, pool_size(3)-1
          do j_step = 0, pool_size(2)-1
             do i_step = 0, pool_size(1)-1
                stride_idx = ((i-1)*stride(1) + i_step) + &
                     ((j-1)*stride(2) + j_step) * input%shape(1) + &
                     ((k-1)*stride(3) + k_step) * &
                     input%shape(1) * input%shape(2) + &
                     (m-1) * channel_size_in
                pool_max = max(pool_max, input%val(stride_idx + 1, s))
             end do
          end do
       end do

       idx = i + (j-1) * output_shape(1) + &
            (k-1) * output_shape(1)*output_shape(2) + &
            (m-1) * channel_size_out

       output%val(idx, s) = pool_max
    end do

    allocate(output%adj_ja(3,2))
    output%adj_ja(:,1) = pool_size
    output%adj_ja(:,2) = stride

    output%get_partial_left => get_partial_maxpool3d
    if (input%requires_grad) then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'maxpool3d'
       output%left_operand = input
    end if

  end function maxpool3d
!-------------------------------------------------------------------------------
  function get_partial_maxpool3d(this, upstream_grad) result(output)
    !! Get the partial derivative for 3D max pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, j, k, m, s
    integer :: i_step, j_step, k_step
    integer :: in_idx, out_idx, max_i, max_j, max_k
    real(real32) :: pool_max, channel_size_in, channel_size_out, tmp_multi
    integer, dimension(5) :: input_shape
    integer, dimension(3) :: pool_size, stride

    ! Unpack parameters
    input_shape = this%left_operand%shape
    pool_size = this%adj_ja(:,1)
    stride    = this%adj_ja(:,2)

    call output%allocate(array_shape = input_shape)
    output%val = 0._real32

    channel_size_in = &
         real( input_shape(1) * input_shape(2) * input_shape(3), real32 )
    channel_size_out = &
         real( this%shape(1) * this%shape(2) * this%shape(3), real32 )
    tmp_multi = input_shape(1) * input_shape(2)

    do s = 1, this%shape(5)
       do m = 1, this%shape(4)
          do k = 1, this%shape(3)
             do j = 1, this%shape(2)
                do i = 1, this%shape(1)
                   ! Find max value location in pooling window
                   pool_max = -huge(1.0_real32)
                   max_i = 0
                   max_j = 0
                   max_k = 0

                   do k_step = 0, pool_size(3)-1
                      do j_step = 0, pool_size(2)-1
                         do i_step = 0, pool_size(1)-1
                            in_idx = ((i-1)*stride(1) + i_step) + &
                                 ((j-1)*stride(2) + j_step) * input_shape(1) + &
                                 ((k-1)*stride(3) + k_step) * tmp_multi + &
                                 (m-1) * channel_size_in

                            if (this%left_operand%val(in_idx + 1, s) > pool_max) then
                               pool_max = this%left_operand%val(in_idx + 1, s)
                               max_i = i_step
                               max_j = j_step
                               max_k = k_step
                            end if
                         end do
                      end do
                   end do

                   ! Assign gradient to max location
                   in_idx = ((i-1)*stride(1) + max_i) + &
                        ((j-1)*stride(2) + max_j) * input_shape(1) + &
                        ((k-1)*stride(3) + max_k) * tmp_multi + &
                        (m-1) * channel_size_in

                   out_idx = i + (j-1) * upstream_grad%shape(1) + &
                        (k-1) * upstream_grad%shape(1)*upstream_grad%shape(2) + &
                        (m-1) * channel_size_out

                   output%val(in_idx + 1, s) = output%val(in_idx + 1, s) + &
                        upstream_grad%val(out_idx, s)
                end do
             end do
          end do
       end do
    end do

  end function get_partial_maxpool3d
!###############################################################################

end submodule athena__diffstruc_extd_submodule_pool
