submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_pad
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  subroutine fill_edge_region_1d(input, output)
    !! Fill edge region for 1D padding
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output

    ! Local variables
    integer :: i, m, s, f
    integer :: step, idx_in, idx_out
    integer :: input_size, output_size, pad_size

    input_size = input%shape(1)
    output_size = output%shape(1)
    pad_size = output%indices(2)

    do f = 1, output%indices(3)
       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(2) )
          select case(output%indices(1))
          case(3, 4) ! circular or reflection
             step = merge(1, -1, output%indices(1) .eq. 3)
             do i = 1, pad_size
                idx_in = output%adj_ja(1,(f-1)*2 + 1) + step * (i - 1) + &
                     (m-1)*input_size
                idx_out = output%adj_ja(2,(f-1)*2 + 1) + i - 1 + &
                     (m-1)*output_size
                output%val(idx_out, s) = input%val(idx_in, s)
             end do
          case(5) ! replication
             idx_in = output%adj_ja(1,(f-1)*2 + 1) + (m-1)*input_size
             do i = 1, pad_size
                idx_out = output%adj_ja(2,(f-1)*2 + 1) + i - 1 + &
                     (m-1)*output_size
                output%val(idx_out, s) = input%val(idx_in, s)
             end do
          end select
       end do
    end do

  end subroutine fill_edge_region_1d
!-------------------------------------------------------------------------------
  subroutine accumulate_edge_gradients_1d(input, output, indices, adj_ja)
    !! Accumulate gradients for 1D edge regions
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output
    integer, dimension(3), intent(in) :: indices
    integer, dimension(2,4), intent(in) :: adj_ja

    ! Local variables
    integer :: i, m, s, f
    integer :: step, idx_in, idx_out
    integer :: input_size, output_size
    integer, dimension(2) :: orig, dest

    input_size = output%shape(1)
    output_size = input%shape(1)

    do f = 1, indices(3)
       orig(1) = adj_ja(1,(f-1)*2 + 1)
       orig(2) = adj_ja(1,(f-1)*2 + 2)
       dest(1) = adj_ja(2,(f-1)*2 + 1)
       dest(2) = adj_ja(2,(f-1)*2 + 2)
       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(2) )
          select case(indices(1))
          case(3, 4) ! circular or reflection
             step = merge(1, -1, indices(1) .eq. 3)
             do i = dest(1), dest(2)
                   idx_out = i + (m-1) * output_size
                   idx_in = orig(1) + step * (i - dest(1)) + (m-1) * input_size
                   output%val(idx_in, s) = output%val(idx_in, s) + &
                        input%val(idx_out, s)
             end do
          case(5) ! replication
             idx_in = orig(1) + (m-1) * input_size
             do i = dest(1), dest(2)
                idx_out = i + (m-1) * output_size
                output%val(idx_in, s) = output%val(idx_in, s) + &
                     input%val(idx_out, s)
             end do
          end select
       end do
    end do

  end subroutine accumulate_edge_gradients_1d
!###############################################################################


!###############################################################################
  subroutine fill_corner_region_2d(input, output)
    !! Fill corner region for 2D padding
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output

    ! Local variables
    integer :: i, j, m, s, f
    integer :: step, idx_in, idx_out, idx_shift
    integer :: input_h, input_w, output_h, output_w
    integer :: pad_h, pad_w
    integer, dimension(2,2) :: orig, dest

    input_h = input%shape(1)
    input_w = input%shape(2)
    output_h = output%shape(1)
    output_w = output%shape(2)
    pad_h = output%indices(2)
    pad_w = output%indices(3)

    idx_shift = output%indices(4) * 4
    do f = 1, output%indices(5)
       orig(1:2,1) = output%adj_ja(1,(f-1)*4 + 1 + idx_shift:(f-1)*4 + 2 + idx_shift)
       orig(1:2,2) = output%adj_ja(1,(f-1)*4 + 3 + idx_shift:(f-1)*4 + 4 + idx_shift)
       dest(1:2,1) = output%adj_ja(2,(f-1)*4 + 1 + idx_shift:(f-1)*4 + 2 + idx_shift)
       dest(1:2,2) = output%adj_ja(2,(f-1)*4 + 3 + idx_shift:(f-1)*4 + 4 + idx_shift)

       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(3) )
          select case(output%indices(1))
          case(3, 4) ! circular or reflection
             step = merge(1, -1, output%indices(1) .eq. 3)
             do j = dest(1,2), dest(2,2)
                do i = dest(1,1), dest(2,1)
                   idx_out = i + (j-1) * output_h + (m - 1) * output_h * output_w
                   idx_in = orig(1,1) + step * (i - dest(1,1)) + &
                           (orig(1,2) + step * (j - dest(1,2)) - 1) * input_h + &
                           (m - 1) * input_h * input_w
                   output%val(idx_out, s) = input%val(idx_in, s)
                end do
             end do
          case(5) ! replication
             idx_in = orig(1,1) + (orig(1,2) - 1) * input_h + &
                  (m - 1) * input_h * input_w
             do j = dest(1,2), dest(2,2)
                do i = dest(1,1), dest(2,1)
                   idx_out = i + (j-1) * output_h + (m - 1) * output_h * output_w
                   output%val(idx_out, s) = input%val(idx_in, s)
                end do
             end do
          end select
       end do
    end do

  end subroutine fill_corner_region_2d
!-------------------------------------------------------------------------------
  subroutine fill_edge_region_2d(input, output)
    !! Fill edge region for 2D padding
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output

    ! Local variables
    integer :: i, j, m, s, f, idim
    integer :: step1, step2, idx_in, idx_out
    integer :: input_h, input_w, output_h, output_w
    integer :: pad_h, pad_w
    integer, dimension(2,2) :: orig, dest

    input_h = input%shape(1)
    input_w = input%shape(2)
    output_h = output%shape(1)
    output_w = output%shape(2)
    pad_h = output%indices(2)
    pad_w = output%indices(3)

    do f = 1, output%indices(4)
       idim = output%indices(5 + f)
       orig(1:2,1) = output%adj_ja(1,(f-1)*4 + 1:(f-1)*4 + 2)
       orig(1:2,2) = output%adj_ja(1,(f-1)*4 + 3:(f-1)*4 + 4)
       dest(1:2,1) = output%adj_ja(2,(f-1)*4 + 1:(f-1)*4 + 2)
       dest(1:2,2) = output%adj_ja(2,(f-1)*4 + 3:(f-1)*4 + 4)

       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(3) )
          select case(output%indices(1))
          case(3, 4) ! circular or reflection
             step1 = merge(-1, 1, output%indices(1) .eq. 4 .and. idim .eq. 1)
             step2 = merge(-1, 1, output%indices(1) .eq. 4 .and. idim .eq. 2)
             do j = dest(1,2), dest(2,2)
                do i = dest(1,1), dest(2,1)
                   idx_out = i + (j-1) * output_h + (m - 1) * output_h * output_w
                   idx_in = orig(1,1) + step1 * (i - dest(1,1)) + &
                           (orig(1,2) + step2 * (j - dest(1,2)) - 1) * input_h + &
                           (m - 1) * input_h * input_w
                   output%val(idx_out, s) = input%val(idx_in, s)
                end do
             end do
          case(5) ! replication
             select case(idim)
             case(1)
                do j = dest(1,2), dest(2,2)
                   idx_in = orig(1,1) + (j - dest(1,2)) * input_h + &
                        (m - 1) * input_h * input_w
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j-1) * output_h + (m - 1) * output_h * output_w
                      output%val(idx_out, s) = input%val(idx_in, s)
                   end do
                end do
             case(2)
                idx_in = (orig(1,2) - 1) * input_h + (m - 1) * input_h * input_w
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j-1) * output_h + (m - 1) * output_h * output_w
                      output%val(idx_out, s) = &
                           input%val(idx_in + i - dest(1,1) + 1, s)
                   end do
                end do
             end select
          end select
       end do
    end do

  end subroutine fill_edge_region_2d
!-------------------------------------------------------------------------------
  subroutine accumulate_corner_gradients_2d(input, output, indices, adj_ja)
    !! Accumulate gradients for 2D corner regions
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output
    integer, dimension(13), intent(in) :: indices
    integer, dimension(2,32), intent(in) :: adj_ja

    ! Local variables
    integer :: i, j, m, s, f
    integer :: step, idx_in, idx_out, idx_shift
    integer :: input_h, input_w, output_h, output_w
    integer, dimension(2,2) :: orig, dest
    real(real32) :: grad_sum

    input_h = output%shape(1)
    input_w = output%shape(2)
    output_h = input%shape(1)
    output_w = input%shape(2)

    idx_shift = indices(4) * 4
    do f = 1, indices(5)
       orig(1:2,1) = adj_ja(1,(f-1)*4 + 1 + idx_shift:(f-1)*4 + 2 + idx_shift)
       orig(1:2,2) = adj_ja(1,(f-1)*4 + 3 + idx_shift:(f-1)*4 + 4 + idx_shift)
       dest(1:2,1) = adj_ja(2,(f-1)*4 + 1 + idx_shift:(f-1)*4 + 2 + idx_shift)
       dest(1:2,2) = adj_ja(2,(f-1)*4 + 3 + idx_shift:(f-1)*4 + 4 + idx_shift)

       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(3) )
          select case(indices(1))
          case(3, 4) ! circular or reflection
             step = merge(1, -1, indices(1) .eq. 3)
             do j = dest(1,2), dest(2,2)
                do i = dest(1,1), dest(2,1)
                   idx_out = i + (j-1) * output_h + (m - 1) * output_h * output_w
                   idx_in = orig(1,1) + step * (i - dest(1,1)) + &
                           (orig(1,2) + step * (j - dest(1,2)) - 1) * input_h + &
                           (m - 1) * input_h * input_w
                   output%val(idx_in, s) = output%val(idx_in, s) + &
                        input%val(idx_out, s)
                end do
             end do
          case(5) ! replication
             grad_sum = 0._real32
             do j = dest(1,2), dest(2,2)
                do i = dest(1,1), dest(2,1)
                   idx_out = i + (j-1) * output_h
                   grad_sum = grad_sum + input%val(idx_out, s)
                end do
             end do
             idx_in = orig(1,1) + (orig(1,2) - 1) * input_h + &
                  (m - 1) * input_h * input_w
             output%val(idx_in, s) = output%val(idx_in, s) + grad_sum
          end select
       end do
    end do

  end subroutine accumulate_corner_gradients_2d
!-------------------------------------------------------------------------------
  subroutine accumulate_edge_gradients_2d(input, output, indices, adj_ja)
    !! Accumulate gradients for 2D edge regions
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output
    integer, dimension(13), intent(in) :: indices
    integer, dimension(2,32), intent(in) :: adj_ja

    ! Local variables
    integer :: i, j, m, s, f, idim
    integer :: step1, step2, idx_in, idx_out
    integer :: input_h, input_w, output_h, output_w
    integer, dimension(2,2) :: orig, dest
    real(real32) :: grad_sum

    input_h = output%shape(1)
    input_w = output%shape(2)
    output_h = input%shape(1)
    output_w = input%shape(2)

    do f = 1, indices(4)
       idim = indices(5 + f)
       orig(1:2,1) = adj_ja(1,(f-1)*4 + 1:(f-1)*4 + 2)
       orig(1:2,2) = adj_ja(1,(f-1)*4 + 3:(f-1)*4 + 4)
       dest(1:2,1) = adj_ja(2,(f-1)*4 + 1:(f-1)*4 + 2)
       dest(1:2,2) = adj_ja(2,(f-1)*4 + 3:(f-1)*4 + 4)

       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(3) )
          select case(indices(1))
          case(3, 4) ! circular or reflection
             step1 = merge(-1, 1, indices(1) .eq. 4 .and. idim .eq. 1)
             step2 = merge(-1, 1, indices(1) .eq. 4 .and. idim .eq. 2)
             do j = dest(1,2), dest(2,2)
                do i = dest(1,1), dest(2,1)
                   idx_out = i + (j-1) * output_h + (m - 1) * output_h * output_w
                   idx_in = orig(1,1) + step1 * (i - dest(1,1)) + &
                           (orig(1,2) + step2 * (j - dest(1,2)) - 1) * input_h + &
                           (m - 1) * input_h * input_w
                   output%val(idx_in, s) = output%val(idx_in, s) + &
                        input%val(idx_out, s)
                end do
             end do
          case(5) ! replication
             select case(idim)
             case(1)
                do j = dest(1,2), dest(2,2)
                   grad_sum = 0._real32
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j-1) * output_h + (m - 1) * output_h * output_w
                      grad_sum = grad_sum + input%val(idx_out, s)
                   end do
                   idx_in = orig(1,1) + (j - dest(1,2)) * input_h + &
                        (m - 1) * input_h * input_w
                   output%val(idx_in, s) = output%val(idx_in, s) + grad_sum
                end do
             case(2)
                do i = dest(1,1), dest(2,1)
                   grad_sum = 0._real32
                   do j = dest(1,2), dest(2,2)
                     idx_out = i + (j-1) * output_h + (m - 1) * output_h * output_w
                      grad_sum = grad_sum + input%val(idx_out, s)
                   end do
                   idx_in = (orig(1,2) - 1) * input_h + (m - 1) * input_h * input_w
                   output%val(idx_in + i - dest(1,1) + 1, s) = &
                        output%val(idx_in + i - dest(1,1) + 1, s) + grad_sum
                end do
             end select
          end select
       end do
    end do

  end subroutine accumulate_edge_gradients_2d
!###############################################################################


!###############################################################################
  subroutine fill_corner_region_3d(input, output)
    !! Fill corner region for 3D padding
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output

    ! Local variables
    integer :: i, j, k, m, s, f
    integer :: step, idx_in, idx_out, idx_shift
    integer :: input_h, input_w, input_d
    integer :: output_h, output_w, output_d
    integer, dimension(2,3) :: orig, dest

    input_h = input%shape(1)
    input_w = input%shape(2)
    input_d = input%shape(3)
    output_h = output%shape(1)
    output_w = output%shape(2)
    output_d = output%shape(3)

    idx_shift = ( output%indices(5) + output%indices(6) ) * 6
    do f = 1, output%indices(7)
       orig(1:2,1) = output%adj_ja(1,(f-1)*6 + 1 + idx_shift:(f-1)*6 + 2 + idx_shift)
       orig(1:2,2) = output%adj_ja(1,(f-1)*6 + 3 + idx_shift:(f-1)*6 + 4 + idx_shift)
       orig(1:2,3) = output%adj_ja(1,(f-1)*6 + 5 + idx_shift:(f-1)*6 + 6 + idx_shift)
       dest(1:2,1) = output%adj_ja(2,(f-1)*6 + 1 + idx_shift:(f-1)*6 + 2 + idx_shift)
       dest(1:2,2) = output%adj_ja(2,(f-1)*6 + 3 + idx_shift:(f-1)*6 + 4 + idx_shift)
       dest(1:2,3) = output%adj_ja(2,(f-1)*6 + 5 + idx_shift:(f-1)*6 + 6 + idx_shift)

       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(4) )
          select case(output%indices(1))
          case(3, 4) ! circular or reflection
             step = merge(1, -1, output%indices(1) .eq. 3)
             do k = dest(1,3), dest(2,3)
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j-1) * output_h + &
                           (k-1) * output_h * output_w + &
                           (m - 1) * output_h * output_w * output_d
                      idx_in = orig(1,1) + step * (i - dest(1,1)) + &
                           (orig(1,2) + step * (j - dest(1,2)) - 1) * input_h + &
                           (orig(1,3) + step * (k - dest(1,3)) - 1) * &
                           input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      output%val(idx_out, s) = input%val(idx_in, s)
                   end do
                end do
             end do
          case(5) ! replication
             idx_in = orig(1,1) + &
                  (orig(1,2) - 1) * input_h + &
                  (orig(1,3) - 1) * input_h * input_w + &
                  (m - 1) * input_h * input_w * input_d
             do k = dest(1,3), dest(2,3)
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j - 1) * output_h + &
                           (k - 1) * output_h * output_w + &
                           (m - 1) * output_h * output_w * output_d
                      output%val(idx_out, s) = input%val(idx_in, s)
                   end do
                end do
             end do
          end select
       end do
    end do

  end subroutine fill_corner_region_3d
!-------------------------------------------------------------------------------
  subroutine fill_edge_region_3d(input, output)
    !! Fill edge region for 3D padding
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output

    ! Local variables
    integer :: i, j, k, m, s, f, idim
    integer :: step1, step2, step3, idx_in, idx_out, idx_shift
    integer :: input_h, input_w, input_d
    integer :: output_h, output_w, output_d
    integer :: pad_h, pad_w, pad_d
    integer, dimension(2,3) :: orig, dest

    input_h = input%shape(1)
    input_w = input%shape(2)
    input_d = input%shape(3)
    output_h = output%shape(1)
    output_w = output%shape(2)
    output_d = output%shape(3)
    pad_h = output%indices(2)
    pad_w = output%indices(3)
    pad_d = output%indices(4)

    idx_shift = output%indices(5) * 6
    do f = 1, output%indices(6)
       idim = output%indices(7 + output%indices(5) + f)
       orig(1:2,1) = output%adj_ja(1,(f-1)*6 + 1 + idx_shift:(f-1)*6 + 2 + idx_shift)
       orig(1:2,2) = output%adj_ja(1,(f-1)*6 + 3 + idx_shift:(f-1)*6 + 4 + idx_shift)
       orig(1:2,3) = output%adj_ja(1,(f-1)*6 + 5 + idx_shift:(f-1)*6 + 6 + idx_shift)
       dest(1:2,1) = output%adj_ja(2,(f-1)*6 + 1 + idx_shift:(f-1)*6 + 2 + idx_shift)
       dest(1:2,2) = output%adj_ja(2,(f-1)*6 + 3 + idx_shift:(f-1)*6 + 4 + idx_shift)
       dest(1:2,3) = output%adj_ja(2,(f-1)*6 + 5 + idx_shift:(f-1)*6 + 6 + idx_shift)

       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(4) )
          select case(output%indices(1))
          case(3, 4) ! circular or reflection
             step1 = merge(-1, 1, output%indices(1) .eq. 4 .and. idim .eq. 1)
             step2 = merge(-1, 1, output%indices(1) .eq. 4 .and. idim .eq. 2)
             step3 = merge(-1, 1, output%indices(1) .eq. 4 .and. idim .eq. 3)
             do k = dest(1,3), dest(2,3)
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j-1) * output_h + &
                           (k-1) * output_h * output_w + &
                           (m - 1) * output_h * output_w * output_d
                      idx_in = orig(1,1) + step1 * (i - dest(1,1)) + &
                           (orig(1,2) + step2 * (j - dest(1,2)) - 1) * &
                           input_h + &
                           (orig(1,3) + step3 * (k - dest(1,3)) - 1) * &
                           input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      output%val(idx_out, s) = input%val(idx_in, s)
                   end do
                end do
             end do
          case(5) ! replication
             select case(idim)
             case(1) ! Edge along dimension 1
                do i = dest(1,1), dest(2,1)
                   idx_in = i - dest(1,1) + 1 + &
                        (orig(1,2) - 1) * input_h + &
                        (orig(1,3) - 1) * input_h * input_w + &
                        (m - 1) * input_h * input_w * input_d
                   do k = dest(1,3), dest(2,3)
                      do j = dest(1,2), dest(2,2)
                         idx_out = i + (j - 1) * output_h + &
                              (k - 1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         output%val(idx_out, s) = input%val(idx_in, s)
                      end do
                   end do
                end do
             case(2) ! Edge along dimension 2
                do j = dest(1,2), dest(2,2)
                   idx_in = orig(1,1) + &
                        (j - dest(1,2)) * input_h + &
                        (orig(1,3) - 1) * input_h * input_w + &
                        (m - 1) * input_h * input_w * input_d
                   do k = dest(1,3), dest(2,3)
                      do i = dest(1,1), dest(2,1)
                         idx_out = i + (j - 1) * output_h + &
                              (k - 1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         output%val(idx_out, s) = input%val(idx_in, s)
                      end do
                   end do
                end do
             case(3) ! Edge along dimension 3
                do k = dest(1,3), dest(2,3)
                   idx_in = orig(1,1) + &
                        (orig(1,2) - 1) * input_h + &
                        (k - dest(1,3)) * input_h * input_w + &
                        (m - 1) * input_h * input_w * input_d
                   do j = dest(1,2), dest(2,2)
                      do i = dest(1,1), dest(2,1)
                         idx_out = i + (j - 1) * output_h + &
                              (k - 1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         output%val(idx_out, s) = input%val(idx_in, s)
                      end do
                   end do
                end do
             end select
          end select
       end do
    end do

  end subroutine fill_edge_region_3d
!-------------------------------------------------------------------------------
  subroutine fill_face_region_3d(input, output)
    !! Fill face region for 3D padding
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output

    ! Local variables
    integer :: i, j, k, m, s, f, idim
    integer :: step1, step2, step3, idx_in, idx_out
    integer :: input_h, input_w, input_d
    integer :: output_h, output_w, output_d
    integer, dimension(2,3) :: orig, dest

    input_h = input%shape(1)
    input_w = input%shape(2)
    input_d = input%shape(3)
    output_h = output%shape(1)
    output_w = output%shape(2)
    output_d = output%shape(3)

    do f = 1, output%indices(5)
       idim = output%indices(7 + f)
       orig(1:2,1) = output%adj_ja(1,(f-1)*6 + 1:(f-1)*6 + 2)
       orig(1:2,2) = output%adj_ja(1,(f-1)*6 + 3:(f-1)*6 + 4)
       orig(1:2,3) = output%adj_ja(1,(f-1)*6 + 5:(f-1)*6 + 6)
       dest(1:2,1) = output%adj_ja(2,(f-1)*6 + 1:(f-1)*6 + 2)
       dest(1:2,2) = output%adj_ja(2,(f-1)*6 + 3:(f-1)*6 + 4)
       dest(1:2,3) = output%adj_ja(2,(f-1)*6 + 5:(f-1)*6 + 6)

       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(4) )
          select case(output%indices(1))
          case(3, 4) ! circular or reflection
             step1 = merge(-1, 1, output%indices(1) .eq. 4 .and. idim .eq. 1)
             step2 = merge(-1, 1, output%indices(1) .eq. 4 .and. idim .eq. 2)
             step3 = merge(-1, 1, output%indices(1) .eq. 4 .and. idim .eq. 3)
             do k = dest(1,3), dest(2,3)
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j-1) * output_h + &
                           (k-1) * output_h * output_w + &
                           (m - 1) * output_h * output_w * output_d
                      idx_in = orig(1,1) + step1 * (i - dest(1,1)) + &
                           (orig(1,2) + step2 * (j - dest(1,2)) - 1) * &
                           input_h + &
                           (orig(1,3) + step3 * (k - dest(1,3)) - 1) * &
                           input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      output%val(idx_out, s) = input%val(idx_in, s)
                   end do
                end do
             end do
          case(5) ! replication
             select case(idim)
             case(1) ! Face perpendicular to dimension 1
                do k = dest(1,3), dest(2,3)
                   do j = dest(1,2), dest(2,2)
                      idx_in = orig(1,1) + &
                           (j - dest(1,2) + orig(1,2) - 1) * input_h + &
                           (k - dest(1,3) + orig(1,3) - 1) * input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      do i = dest(1,1), dest(2,1)
                         idx_out = i + (j - 1) * output_h + &
                              (k - 1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         output%val(idx_out, s) = input%val(idx_in, s)
                      end do
                   end do
                end do
             case(2) ! Face perpendicular to dimension 2
                do k = dest(1,3), dest(2,3)
                   do i = dest(1,1), dest(2,1)
                      idx_in = i - dest(1,1) + orig(1,1) + &
                           (orig(1,2) - 1) * input_h + &
                           (k - dest(1,3) + orig(1,3) - 1) * input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      do j = dest(1,2), dest(2,2)
                         idx_out = i + (j - 1) * output_h + &
                              (k - 1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         output%val(idx_out, s) = input%val(idx_in, s)
                      end do
                   end do
                end do
             case(3) ! Face perpendicular to dimension 3
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_in = i - dest(1,1) + orig(1,1) + &
                           (j - dest(1,2) + orig(1,2) - 1) * input_h + &
                           (orig(1,3) - 1) * input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      do k = dest(1,3), dest(2,3)
                         idx_out = i + (j - 1) * output_h + &
                              (k - 1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         output%val(idx_out, s) = input%val(idx_in, s)
                      end do
                   end do
                end do
             end select
          end select
       end do
    end do

  end subroutine fill_face_region_3d
!-------------------------------------------------------------------------------
  subroutine accumulate_corner_gradients_3d(input, output, indices, adj_ja)
    !! Accumulate gradients for 3D corner regions
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output
    integer, dimension(33), intent(in) :: indices
    integer, dimension(2,156), intent(in) :: adj_ja

    ! Local variables
    integer :: i, j, k, m, s, f
    integer :: step, idx_in, idx_out, idx_shift
    integer :: input_h, input_w, input_d
    integer :: output_h, output_w, output_d
    integer, dimension(2,3) :: orig, dest
    real(real32) :: grad_sum

    input_h = output%shape(1)
    input_w = output%shape(2)
    input_d = output%shape(3)
    output_h = input%shape(1)
    output_w = input%shape(2)
    output_d = input%shape(3)

    idx_shift = ( indices(5) + indices(6) ) * 6
    do f = 1, indices(7)
       orig(1:2,1) = adj_ja(1,(f-1)*6 + 1 + idx_shift:(f-1)*6 + 2 + idx_shift)
       orig(1:2,2) = adj_ja(1,(f-1)*6 + 3 + idx_shift:(f-1)*6 + 4 + idx_shift)
       orig(1:2,3) = adj_ja(1,(f-1)*6 + 5 + idx_shift:(f-1)*6 + 6 + idx_shift)
       dest(1:2,1) = adj_ja(2,(f-1)*6 + 1 + idx_shift:(f-1)*6 + 2 + idx_shift)
       dest(1:2,2) = adj_ja(2,(f-1)*6 + 3 + idx_shift:(f-1)*6 + 4 + idx_shift)
       dest(1:2,3) = adj_ja(2,(f-1)*6 + 5 + idx_shift:(f-1)*6 + 6 + idx_shift)

       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(4) )
          select case(indices(1))
          case(3, 4) ! circular or reflection
             step = merge(1, -1, indices(1) .eq. 3)
             do k = dest(1,3), dest(2,3)
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j-1) * output_h + &
                           (k-1) * output_h * output_w + &
                           (m - 1) * output_h * output_w * output_d
                      idx_in = orig(1,1) + step * (i - dest(1,1)) + &
                           (orig(1,2) + step * (j - dest(1,2)) - 1) * &
                           input_h + &
                           (orig(1,3) + step * (k - dest(1,3)) - 1) * &
                           input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      output%val(idx_in, s) = output%val(idx_in, s) + &
                           input%val(idx_out, s)
                   end do
                end do
             end do
          case(5) ! replication
             grad_sum = 0._real32
             do k = dest(1,3), dest(2,3)
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j-1) * output_h + &
                           (k-1) * output_h * output_w + &
                           (m - 1) * output_h * output_w * output_d
                      grad_sum = grad_sum + input%val(idx_out, s)
                   end do
                end do
             end do
             idx_in = orig(1,1) + (orig(1,2) - 1) * input_h + &
                  (orig(1,3) - 1) * input_h * input_w + &
                  (m - 1) * input_h * input_w * input_d
             output%val(idx_in, s) = output%val(idx_in, s) + grad_sum
          end select
       end do
    end do

  end subroutine accumulate_corner_gradients_3d
!-------------------------------------------------------------------------------
  subroutine accumulate_edge_gradients_3d(input, output, indices, adj_ja)
    !! Accumulate gradients for 3D edge regions
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output
    integer, dimension(33), intent(in) :: indices
    integer, dimension(2,156), intent(in) :: adj_ja

    ! Local variables
    integer :: i, j, k, m, s, f, idim
    integer :: step1, step2, step3, idx_in, idx_out, idx_shift
    integer :: input_h, input_w, input_d
    integer :: output_h, output_w, output_d
    integer, dimension(2,3) :: orig, dest
    real(real32) :: grad_sum

    input_h = output%shape(1)
    input_w = output%shape(2)
    input_d = output%shape(3)
    output_h = input%shape(1)
    output_w = input%shape(2)
    output_d = input%shape(3)

    idx_shift = indices(5) * 6
    do f = 1, indices(6)
       idim = indices(7 + indices(5) + f)
       orig(1:2,1) = adj_ja(1,(f-1)*6 + 1 + idx_shift:(f-1)*6 + 2 + idx_shift)
       orig(1:2,2) = adj_ja(1,(f-1)*6 + 3 + idx_shift:(f-1)*6 + 4 + idx_shift)
       orig(1:2,3) = adj_ja(1,(f-1)*6 + 5 + idx_shift:(f-1)*6 + 6 + idx_shift)
       dest(1:2,1) = adj_ja(2,(f-1)*6 + 1 + idx_shift:(f-1)*6 + 2 + idx_shift)
       dest(1:2,2) = adj_ja(2,(f-1)*6 + 3 + idx_shift:(f-1)*6 + 4 + idx_shift)
       dest(1:2,3) = adj_ja(2,(f-1)*6 + 5 + idx_shift:(f-1)*6 + 6 + idx_shift)

       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(4) )
          select case(indices(1))
          case(3, 4) ! circular or reflection
             step1 = merge(-1, 1, indices(1) .eq. 4 .and. idim .eq. 1)
             step2 = merge(-1, 1, indices(1) .eq. 4 .and. idim .eq. 2)
             step3 = merge(-1, 1, indices(1) .eq. 4 .and. idim .eq. 3)
             do k = dest(1,3), dest(2,3)
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j-1) * output_h + &
                           (k-1) * output_h * output_w + &
                           (m - 1) * output_h * output_w * output_d
                      idx_in = orig(1,1) + step1 * (i - dest(1,1)) + &
                           (orig(1,2) + step2 * (j - dest(1,2)) - 1) * &
                           input_h + &
                           (orig(1,3) + step3 * (k - dest(1,3)) - 1) * &
                           input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      output%val(idx_in, s) = output%val(idx_in, s) + &
                           input%val(idx_out, s)
                   end do
                end do
             end do
          case(5) ! replication
             select case(idim)
             case(1) ! Edge along dimension 1
                do i = dest(1,1), dest(2,1)
                   idx_in = i - dest(1,1) + 1 + &
                        (orig(1,2) - 1) * input_h + &
                        (orig(1,3) - 1) * input_h * input_w + &
                        (m - 1) * input_h * input_w * input_d
                   grad_sum = 0._real32
                   do k = dest(1,3), dest(2,3)
                      do j = dest(1,2), dest(2,2)
                         idx_out = i + (j - 1) * output_h + &
                              (k - 1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         grad_sum = grad_sum + input%val(idx_out, s)
                      end do
                   end do
                   output%val(idx_in, s) = output%val(idx_in, s) + grad_sum
                end do
             case(2) ! Edge along dimension 2
                do j = dest(1,2), dest(2,2)
                   idx_in = orig(1,1) + &
                        (j - dest(1,2)) * input_h + &
                        (orig(1,3) - 1) * input_h * input_w + &
                        (m - 1) * input_h * input_w * input_d
                   grad_sum = 0._real32
                   do k = dest(1,3), dest(2,3)
                      do i = dest(1,1), dest(2,1)
                         idx_out = i + (j - 1) * output_h + &
                              (k - 1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         grad_sum = grad_sum + input%val(idx_out, s)
                      end do
                   end do
                   output%val(idx_in, s) = output%val(idx_in, s) + grad_sum
                end do
             case(3) ! Edge along dimension 3
                do k = dest(1,3), dest(2,3)
                   idx_in = orig(1,1) + &
                        (orig(1,2) - 1) * input_h + &
                        (k - dest(1,3)) * input_h * input_w + &
                        (m - 1) * input_h * input_w * input_d
                   grad_sum = 0._real32
                   do j = dest(1,2), dest(2,2)
                      do i = dest(1,1), dest(2,1)
                         idx_out = i + (j - 1) * output_h + &
                              (k - 1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         grad_sum = grad_sum + input%val(idx_out, s)
                      end do
                   end do
                   output%val(idx_in, s) = output%val(idx_in, s) + grad_sum
                end do
             end select
          end select
       end do
    end do

  end subroutine accumulate_edge_gradients_3d
!-------------------------------------------------------------------------------
  subroutine accumulate_face_gradients_3d(input, output, indices, adj_ja)
    !! Accumulate gradients for 3D face regions
    implicit none

    ! Arguments
    type(array_type), intent(in) :: input
    type(array_type), intent(inout) :: output
    integer, dimension(33), intent(in) :: indices
    integer, dimension(2,156), intent(in) :: adj_ja

    ! Local variables
    integer :: i, j, k, m, s, f, idim
    integer :: step1, step2, step3, idx_in, idx_out
    integer :: input_h, input_w, input_d
    integer :: output_h, output_w, output_d
    integer, dimension(2,3) :: orig, dest
    real(real32) :: grad_sum

    input_h = output%shape(1)
    input_w = output%shape(2)
    input_d = output%shape(3)
    output_h = input%shape(1)
    output_w = input%shape(2)
    output_d = input%shape(3)

    do f = 1, indices(5)
       idim = indices(7 + f)
       orig(1:2,1) = adj_ja(1,(f-1)*6 + 1:(f-1)*6 + 2)
       orig(1:2,2) = adj_ja(1,(f-1)*6 + 3:(f-1)*6 + 4)
       orig(1:2,3) = adj_ja(1,(f-1)*6 + 5:(f-1)*6 + 6)
       dest(1:2,1) = adj_ja(2,(f-1)*6 + 1:(f-1)*6 + 2)
       dest(1:2,2) = adj_ja(2,(f-1)*6 + 3:(f-1)*6 + 4)
       dest(1:2,3) = adj_ja(2,(f-1)*6 + 5:(f-1)*6 + 6)


       do concurrent( s = 1:size(output%val, dim=2), m = 1:output%shape(4) )
          select case(indices(1))
          case(3, 4) ! circular or reflection
             step1 = merge(-1, 1, indices(1) .eq. 4 .and. idim .eq. 1)
             step2 = merge(-1, 1, indices(1) .eq. 4 .and. idim .eq. 2)
             step3 = merge(-1, 1, indices(1) .eq. 4 .and. idim .eq. 3)
             do k = dest(1,3), dest(2,3)
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_out = i + (j-1) * output_h + &
                           (k-1) * output_h * output_w + &
                           (m - 1) * output_h * output_w * output_d
                      idx_in = orig(1,1) + step1 * (i - dest(1,1)) + &
                           (orig(1,2) + step2 * (j - dest(1,2)) - 1) * &
                           input_h + &
                           (orig(1,3) + step3 * (k - dest(1,3)) - 1) * &
                           input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      output%val(idx_in, s) = output%val(idx_in, s) + &
                           input%val(idx_out, s)
                   end do
                end do
             end do
          case(5) ! replication
             select case(idim)
             case(1) ! Face perpendicular to dimension 1
                do k = dest(1,3), dest(2,3)
                   do j = dest(1,2), dest(2,2)
                      idx_in = orig(1,1) + &
                           ( j - dest(1,2) ) * input_h + &
                           ( k - dest(1,3) ) * input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      grad_sum = 0._real32
                      do i = dest(1,1), dest(2,1)
                         idx_out = i + (j-1) * output_h + &
                              (k-1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         grad_sum = grad_sum + input%val(idx_out, s)
                      end do
                      output%val(idx_in, s) = output%val(idx_in, s) + grad_sum
                   end do
                end do
             case(2) ! Face perpendicular to dimension 2
                do k = dest(1,3), dest(2,3)
                   do i = dest(1,1), dest(2,1)
                      idx_in = i - dest(1,1) + 1 + &
                           ( k - dest(1,3) ) * input_h * input_w + &
                           (m - 1) * input_h * input_w * input_d
                      grad_sum = 0._real32
                      do j = dest(1,2), dest(2,2)
                         idx_out = i + (j-1) * output_h + &
                              (k-1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         grad_sum = grad_sum + input%val(idx_out, s)
                      end do
                      output%val(idx_in, s) = output%val(idx_in, s) + grad_sum
                   end do
                end do
             case(3) ! Face perpendicular to dimension 3
                do j = dest(1,2), dest(2,2)
                   do i = dest(1,1), dest(2,1)
                      idx_in = i - dest(1,1) + 1 + &
                           ( j - dest(1,2) ) * input_h + &
                           (m - 1) * input_h * input_w * input_d
                      grad_sum = 0._real32
                      do k = dest(1,3), dest(2,3)
                         idx_out = i + (j-1) * output_h + &
                              (k-1) * output_h * output_w + &
                              (m - 1) * output_h * output_w * output_d
                         grad_sum = grad_sum + input%val(idx_out, s)
                      end do
                      output%val(idx_in, s) = output%val(idx_in, s) + grad_sum
                   end do
                end do
             end select
          end select
       end do
    end do

  end subroutine accumulate_face_gradients_3d
!###############################################################################


!###############################################################################
  module function pad1d(input, facets, pad_size, imethod) result(output)
    !! 1D padding operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    type(facets_type), intent(in) :: facets
    integer, intent(in) :: pad_size
    integer, intent(in) :: imethod
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, m, s
    integer :: idx_in, idx_out
    integer :: input_size, output_size
    integer, dimension(3) :: output_shape

    input_size = input%shape(1)
    output_size = input_size +  2 * pad_size

    output_shape = [ output_size, input%shape(2), size(input%val, dim=2) ]
    output => input%create_result(array_shape = output_shape)

    ! save the facet values to indices and adj_ja
    allocate(output%indices(2 + facets%num))
    output%indices(1) = imethod
    output%indices(2) = pad_size
    output%indices(3) = facets%num
    allocate(output%adj_ja(2, 2 * facets%num))
    do i = 1, facets%num
       output%adj_ja(1,(i-1)*2 + 1) = facets%orig_bound(1,1,i)
       output%adj_ja(2,(i-1)*2 + 1) = facets%dest_bound(1,1,i)
       output%adj_ja(1,(i-1)*2 + 2) = facets%orig_bound(2,1,i)
       output%adj_ja(2,(i-1)*2 + 2) = facets%dest_bound(2,1,i)
    end do

    ! Initialize with pad_value
    output%val = 0._real32

    ! Copy input into the correct location in output
    do concurrent( &
         s = 1:output_shape(3), &
         m = 1:output_shape(2), &
         i = 1:input_size)
       idx_in = i + (m-1) * input_size
       idx_out = i + pad_size + (m-1) * output_size
       output%val(idx_out, s) = input%val(idx_in, s)
    end do

    if(output%indices(1) .ge. 3 .and. output%indices(1) .le. 5)then
       call fill_edge_region_1d( input, output )
    end if


    output%get_partial_left => get_partial_pad1d
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'pad'
       output%left_operand => input
    end if

  end function pad1d
!-------------------------------------------------------------------------------
  function get_partial_pad1d(this, upstream_grad) result(output)
    !! Get the partial derivative for the pad1d operation
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, m, s, f
    integer :: idx_in, idx_out
    integer, dimension(3) :: input_shape


    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    call output%allocate(array_shape = input_shape)
    output%val = 0._real32
    output%indices = this%indices
    output%adj_ja = this%adj_ja

    do concurrent( &
         s = 1:input_shape(3), &
         m = 1:input_shape(2), &
         i = 1:input_shape(1))
       idx_in = i + (m-1) * input_shape(1)
       idx_out = i + this%indices(2) + (m-1) * input_shape(1)
       output%val(idx_in, s) = upstream_grad%val(idx_out, s)
    end do

    if(this%indices(1) .ge. 3 .and. this%indices(1) .le. 5)then
       call accumulate_edge_gradients_1d( upstream_grad, output, &
            this%indices, this%adj_ja &
       )
    end if

  end function get_partial_pad1d
!###############################################################################


!###############################################################################
  module function pad2d(input, facets, pad_size, imethod) result(output)
    !! 2D padding operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    type(facets_type), dimension(2), intent(in) :: facets
    integer, dimension(2), intent(in) :: pad_size
    integer, intent(in) :: imethod
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, m, s
    integer :: idx_in, idx_out, idx_shift
    integer :: input_size_h, input_size_w, num_channels
    integer :: output_size_h, output_size_w
    integer, dimension(4) :: output_shape

    input_size_h = input%shape(1)
    input_size_w = input%shape(2)
    num_channels = input%shape(3)
    output_size_h = input_size_h + 2 * pad_size(1)
    output_size_w = input_size_w + 2 * pad_size(2)

    output_shape = [ &
         output_size_h, output_size_w, num_channels, size(input%val, dim=2) &
    ]
    output => input%create_result(array_shape = output_shape)

    ! save the facet values to indices and adj_ja
    allocate(output%indices(3 + 2 + sum( facets(:)%num )))
    output%indices(1) = imethod
    output%indices(2) = pad_size(1)
    output%indices(3) = pad_size(2)
    output%indices(4) = facets(1)%num
    output%indices(5) = facets(2)%num
    output%indices(6:5 + facets(1)%num) = [(facets(1)%dim(i), i=1, facets(1)%num)]
    output%indices(6 + facets(1)%num:5 + facets(1)%num + facets(2)%num) = &
         [(facets(2)%dim(i), i=1, facets(2)%num)]
    allocate(output%adj_ja(2, 4 * ( facets(1)%num + facets(2)%num )))
    ! Edges (1D faces)
    do i = 1, facets(1)%num
       output%adj_ja(1,(i-1)*4 + 1) = facets(1)%orig_bound(1,1,i)
       output%adj_ja(2,(i-1)*4 + 1) = facets(1)%dest_bound(1,1,i)
       output%adj_ja(1,(i-1)*4 + 2) = facets(1)%orig_bound(2,1,i)
       output%adj_ja(2,(i-1)*4 + 2) = facets(1)%dest_bound(2,1,i)
       output%adj_ja(1,(i-1)*4 + 3) = facets(1)%orig_bound(1,2,i)
       output%adj_ja(2,(i-1)*4 + 3) = facets(1)%dest_bound(1,2,i)
       output%adj_ja(1,(i-1)*4 + 4) = facets(1)%orig_bound(2,2,i)
       output%adj_ja(2,(i-1)*4 + 4) = facets(1)%dest_bound(2,2,i)
    end do
    idx_shift = facets(1)%num * 4
    ! Corners (2D edges)
    do i = 1, facets(2)%num
       output%adj_ja(1,(i-1)*4 + 1 + idx_shift) = facets(2)%orig_bound(1,1,i)
       output%adj_ja(2,(i-1)*4 + 1 + idx_shift) = facets(2)%dest_bound(1,1,i)
       output%adj_ja(1,(i-1)*4 + 2 + idx_shift) = facets(2)%orig_bound(2,1,i)
       output%adj_ja(2,(i-1)*4 + 2 + idx_shift) = facets(2)%dest_bound(2,1,i)
       output%adj_ja(1,(i-1)*4 + 3 + idx_shift) = facets(2)%orig_bound(1,2,i)
       output%adj_ja(2,(i-1)*4 + 3 + idx_shift) = facets(2)%dest_bound(1,2,i)
       output%adj_ja(1,(i-1)*4 + 4 + idx_shift) = facets(2)%orig_bound(2,2,i)
       output%adj_ja(2,(i-1)*4 + 4 + idx_shift) = facets(2)%dest_bound(2,2,i)
    end do

    ! Initialize with zero
    output%val = 0._real32

    ! Copy input into the correct location in output
    do concurrent( &
         s = 1:output_shape(4), &
         m = 1:num_channels, &
         j = 1:input_size_w, &
         i = 1:input_size_h)
       idx_in = i + (j-1) * input_size_h + (m-1) * input_size_h * input_size_w
       idx_out = (i + pad_size(1)) + (j + pad_size(2) - 1) * output_size_h + &
            (m-1) * output_size_h * output_size_w
       output%val(idx_out, s) = input%val(idx_in, s)
    end do

    if(output%indices(1) .ge. 3 .and. output%indices(1) .le. 5)then
       call fill_corner_region_2d( input, output )
       call fill_edge_region_2d( input, output )
    end if

    output%get_partial_left => get_partial_pad2d
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'pad'
       output%left_operand => input
    end if

  end function pad2d
!-------------------------------------------------------------------------------
  function get_partial_pad2d(this, upstream_grad) result(output)
    !! Get the partial derivative for the pad2d operation
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, j, m, s
    integer :: idx_in, idx_out
    integer :: input_size_h, input_size_w, num_channels
    integer :: output_size_h, output_size_w
    integer, dimension(4) :: input_shape

    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    call output%allocate(array_shape = input_shape)
    output%val = 0._real32
    output%indices = this%indices
    output%adj_ja = this%adj_ja

    input_size_h = input_shape(1)
    input_size_w = input_shape(2)
    num_channels = input_shape(3)
    output_size_h = input_size_h + 2 * this%indices(2)
    output_size_w = input_size_w + 2 * this%indices(3)

    do concurrent( &
         s = 1:input_shape(4), &
         m = 1:num_channels, &
         j = 1:input_size_w, &
         i = 1:input_size_h)
       idx_in = i + (j-1) * input_size_h + (m-1) * input_size_h * input_size_w
       idx_out = (i + this%indices(2)) + &
            (j + this%indices(3) - 1) * output_size_h + &
            (m-1) * output_size_h * output_size_w
       output%val(idx_in, s) = upstream_grad%val(idx_out, s)
    end do

    if(this%indices(1) .ge. 3 .and. this%indices(1) .le. 5)then
       call accumulate_corner_gradients_2d( &
            upstream_grad, output, this%indices, this%adj_ja &
       )
       call accumulate_edge_gradients_2d( &
            upstream_grad, output, this%indices, this%adj_ja &
       )
    end if

  end function get_partial_pad2d
!###############################################################################


!###############################################################################
  module function pad3d(input, facets, pad_size, imethod) result(output)
    !! 3D padding operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    type(facets_type), dimension(3), intent(in) :: facets
    integer, dimension(3), intent(in) :: pad_size
    integer, intent(in) :: imethod
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, k, m, s
    integer :: idx_in, idx_out, idx_shift
    integer :: input_size_h, input_size_w, input_size_d, num_channels
    integer :: output_size_h, output_size_w, output_size_d
    integer, dimension(5) :: output_shape

    input_size_h = input%shape(1)
    input_size_w = input%shape(2)
    input_size_d = input%shape(3)
    num_channels = input%shape(4)
    output_size_h = input_size_h + 2 * pad_size(1)
    output_size_w = input_size_w + 2 * pad_size(2)
    output_size_d = input_size_d + 2 * pad_size(3)

    output_shape = [ output_size_h, output_size_w, output_size_d, num_channels, &
         size(input%val, dim=2) ]
    output => input%create_result(array_shape = output_shape)

    ! save the facet values to indices and adj_ja
    allocate(output%indices(4 + 3 + sum( facets(:)%num )))
    output%indices(1) = imethod
    output%indices(2) = pad_size(1)
    output%indices(3) = pad_size(2)
    output%indices(4) = pad_size(3)
    output%indices(5) = facets(1)%num
    output%indices(6) = facets(2)%num
    output%indices(7) = facets(3)%num
    output%indices(8:7 + facets(1)%num) = [(facets(1)%dim(i), i=1, facets(1)%num)]
    output%indices(8 + facets(1)%num:7 + facets(1)%num + facets(2)%num) = &
         [(facets(2)%dim(i), i=1, facets(2)%num)]
    output%indices(8 + facets(1)%num + facets(2)%num:7 + &
         facets(1)%num + facets(2)%num + facets(3)%num) = &
         [(facets(3)%dim(i), i=1, facets(3)%num)]
    allocate(output%adj_ja(2, 6 * (facets(1)%num + facets(2)%num + facets(3)%num)))
    ! Edges (1D edges)
    do i = 1, facets(1)%num
       output%adj_ja(1,(i-1)*6 + 1 : (i-1)*6 + 2) = facets(1)%orig_bound(1:2,1,i)
       output%adj_ja(1,(i-1)*6 + 3 : (i-1)*6 + 4) = facets(1)%orig_bound(1:2,2,i)
       output%adj_ja(1,(i-1)*6 + 5 : (i-1)*6 + 6) = facets(1)%orig_bound(1:2,3,i)
       output%adj_ja(2,(i-1)*6 + 1 : (i-1)*6 + 2) = facets(1)%dest_bound(1:2,1,i)
       output%adj_ja(2,(i-1)*6 + 3 : (i-1)*6 + 4) = facets(1)%dest_bound(1:2,2,i)
       output%adj_ja(2,(i-1)*6 + 5 : (i-1)*6 + 6) = facets(1)%dest_bound(1:2,3,i)
    end do
    idx_shift = facets(1)%num * 6
    ! Faces (2D faces)
    do i = 1, facets(2)%num
       output%adj_ja(1,(i-1)*6 + 1 + idx_shift : (i-1)*6 + 2 + idx_shift) = &
            facets(2)%orig_bound(1:2,1,i)
       output%adj_ja(1,(i-1)*6 + 3 + idx_shift : (i-1)*6 + 4 + idx_shift) = &
            facets(2)%orig_bound(1:2,2,i)
       output%adj_ja(1,(i-1)*6 + 5 + idx_shift : (i-1)*6 + 6 + idx_shift) = &
            facets(2)%orig_bound(1:2,3,i)
       output%adj_ja(2,(i-1)*6 + 1 + idx_shift : (i-1)*6 + 2 + idx_shift) = &
            facets(2)%dest_bound(1:2,1,i)
       output%adj_ja(2,(i-1)*6 + 3 + idx_shift : (i-1)*6 + 4 + idx_shift) = &
            facets(2)%dest_bound(1:2,2,i)
       output%adj_ja(2,(i-1)*6 + 5 + idx_shift : (i-1)*6 + 6 + idx_shift) = &
            facets(2)%dest_bound(1:2,3,i)
    end do
    idx_shift = idx_shift + facets(2)%num * 6
    ! Corners (3D corners)
    do i = 1, facets(3)%num
       output%adj_ja(1,(i-1)*6 + 1 + idx_shift : (i-1)*6 + 2 + idx_shift) = &
            facets(3)%orig_bound(1:2,1,i)
       output%adj_ja(1,(i-1)*6 + 3 + idx_shift : (i-1)*6 + 4 + idx_shift) = &
            facets(3)%orig_bound(1:2,2,i)
       output%adj_ja(1,(i-1)*6 + 5 + idx_shift : (i-1)*6 + 6 + idx_shift) = &
            facets(3)%orig_bound(1:2,3,i)
       output%adj_ja(2,(i-1)*6 + 1 + idx_shift : (i-1)*6 + 2 + idx_shift) = &
            facets(3)%dest_bound(1:2,1,i)
       output%adj_ja(2,(i-1)*6 + 3 + idx_shift : (i-1)*6 + 4 + idx_shift) = &
            facets(3)%dest_bound(1:2,2,i)
       output%adj_ja(2,(i-1)*6 + 5 + idx_shift : (i-1)*6 + 6 + idx_shift) = &
            facets(3)%dest_bound(1:2,3,i)
    end do

    ! Initialize with zero
    output%val = 0._real32

    ! Copy input into the correct location in output
    do concurrent( &
         s = 1:output_shape(5), &
         m = 1:num_channels, &
         k = 1:input_size_d, &
         j = 1:input_size_w, &
         i = 1:input_size_h)
       idx_in = i + (j-1) * input_size_h + (k-1) * input_size_h * input_size_w + &
            (m-1) * input_size_h * input_size_w * input_size_d
       idx_out = (i + pad_size(1)) + &
            (j + pad_size(2) - 1) * output_size_h + &
            (k + pad_size(3) - 1) * output_size_h * output_size_w + &
            (m-1) * output_size_h * output_size_w * output_size_d
       output%val(idx_out, s) = input%val(idx_in, s)
    end do

    if(output%indices(1) .ge. 3 .and. output%indices(1) .le. 5)then
       call fill_corner_region_3d( input, output )
       call fill_edge_region_3d( input, output )
       call fill_face_region_3d( input, output )
    end if

    output%get_partial_left => get_partial_pad3d
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'pad'
       output%left_operand => input
    end if

  end function pad3d
!-------------------------------------------------------------------------------
  function get_partial_pad3d(this, upstream_grad) result(output)
    !! Get the partial derivative for the pad3d operation
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, j, k, m, s
    integer :: idx_in, idx_out
    integer :: input_size_h, input_size_w, input_size_d, num_channels
    integer :: output_size_h, output_size_w, output_size_d
    integer, dimension(5) :: input_shape

    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    call output%allocate(array_shape = input_shape)
    output%val = 0._real32
    output%indices = this%indices
    output%adj_ja = this%adj_ja

    input_size_h = input_shape(1)
    input_size_w = input_shape(2)
    input_size_d = input_shape(3)
    num_channels = input_shape(4)
    output_size_h = input_size_h + 2 * this%indices(2)
    output_size_w = input_size_w + 2 * this%indices(3)
    output_size_d = input_size_d + 2 * this%indices(4)

    do concurrent( &
         s = 1:input_shape(5), &
         m = 1:num_channels, &
         k = 1:input_size_d, &
         j = 1:input_size_w, &
         i = 1:input_size_h)
       idx_in = i + (j-1) * input_size_h + (k-1) * input_size_h * input_size_w + &
            (m-1) * input_size_h * input_size_w * input_size_d
       idx_out = (i + this%indices(2)) + &
            (j + this%indices(3) - 1) * output_size_h + &
            (k + this%indices(4) - 1) * output_size_h * output_size_w + &
            (m-1) * output_size_h * output_size_w * output_size_d
       output%val(idx_in, s) = upstream_grad%val(idx_out, s)
    end do

    if(this%indices(1) .ge. 3 .and. this%indices(1) .le. 5)then
       call accumulate_corner_gradients_3d( &
            upstream_grad, output, this%indices, this%adj_ja &
       )
       call accumulate_edge_gradients_3d( &
            upstream_grad, output, this%indices, this%adj_ja &
       )
       call accumulate_face_gradients_3d( &
            upstream_grad, output, this%indices, this%adj_ja &
       )
    end if

  end function get_partial_pad3d
!###############################################################################

end submodule athena__diffstruc_extd_submodule_pad
