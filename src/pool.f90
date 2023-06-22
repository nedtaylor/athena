!!!###################################################################################
!!!###################################################################################
!!!###################################################################################
module PoolingLayer
  implicit none

  integer :: pool_size     ! Pooling window size (assumed square)
  integer :: pool_stride   ! Pooling stride

  private

  public :: initialise, forward, backward

contains

  subroutine initialise(size, stride)
    implicit none
    integer :: size, stride
    pool_size = size
    pool_stride = stride
  end subroutine initialise

  subroutine forward(input, output)
    implicit none
    real, dimension(:,:,:), intent(in) :: input
    real, dimension(:,:,:), intent(out) :: output

    integer :: i, j, k, l, m, n
    
    ! Compute the size of the input and output feature maps
    integer :: input_size, output_size
    input_size = size(input, 1)
    output_size = (input_size - pool_size) / pool_stride + 1

    ! Perform the pooling operation
    do m = 1, size(input, 3)
       do l = 1, size(input, 2), pool_stride
          do j = 1, output_size
             do i = 1, output_size
                output(i, j, m) = maxval(&
                     input((i-1)*pool_stride+1:(i-1)*pool_stride+pool_size, &
                     (j-1)*pool_stride+1:(j-1)*pool_stride+pool_size, m))
             end do
          end do
       end do
    end do
  end subroutine forward

  subroutine backward(input, output_gradients, input_gradients)
    implicit none
    real, dimension(:,:,:), intent(in) :: input
    real, dimension(:,:,:), intent(in) :: output_gradients
    real, dimension(:,:,:), intent(out) :: input_gradients

    integer :: i, j, k, l, m, n
    integer :: input_size, output_size
    integer, dimension(2) :: max_index

    ! Initialise input_gradients to zero
    input_gradients = 0.0

    ! Compute the size of the input and output feature maps
    input_size = size(input, 1)
    output_size = (input_size - pool_size) / pool_stride + 1

    ! Compute gradients for input feature map
    do m = 1, size(input, 3)
       do l = 1, size(input, 2), pool_stride
          do j = 1, output_size
             do i = 1, output_size
                ! Find the index of the maximum value in the corresponding pooling window
                max_index = maxloc(input((i-1)*pool_stride+1:(i-1)*pool_stride+pool_size, &
                     (j-1)*pool_stride+1:(j-1)*pool_stride+pool_size, m))

                ! Compute gradients for input feature map
                input_gradients((i-1)*pool_stride+max_index(1), &
                     (j-1)*pool_stride+max_index(2), m) = output_gradients(i, j, m)
                
                !input_gradients((i-1)*pool_stride+1:(i-1)*pool_stride+pool_size, &
                !     (j-1)*pool_stride+1:(j-1)*pool_stride+pool_size, m)) = output_gradients(i,j,m)
             end do
          end do
       end do
    end do
  end subroutine backward

end module PoolingLayer

!!!###################################################################################
!!!###################################################################################
!!!###################################################################################
