!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module PoolingLayer
  use constants, only: real12
  implicit none

  integer :: pool_size     ! pooling window size (assumed square)
  integer :: pool_stride   ! pooling stride


  private

  public :: initialise, forward, backward

contains

!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine initialise(size, stride)
    implicit none
    integer :: size, stride
    pool_size = size
    pool_stride = stride
  end subroutine initialise
!!!#############################################################################

!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine forward(input, output)
    implicit none
    real(real12), dimension(:,:,:), intent(in) :: input
    real(real12), dimension(:,:,:), intent(out) :: output

    integer :: i, j, l, m
    
    !! compute the size of the input and output feature maps
    integer :: input_size, output_size
    input_size = size(input, 1)
    output_size = (input_size - pool_size) / pool_stride + 1

    !! perform the pooling operation
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
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine backward(input, output_gradients, input_gradients)
    implicit none
    real(real12), dimension(:,:,:), intent(in) :: input
    real(real12), dimension(:,:,:), intent(in) :: output_gradients
    real(real12), dimension(:,:,:), intent(out) :: input_gradients

    integer :: i, j, m, istride, jstride
    integer :: input_size, output_size
    integer, dimension(2) :: max_index

    !! initialise input_gradients to zero
    input_gradients = 0._real12

    !! compute the size of the input and output feature maps
    input_size = size(input, 1)
    output_size = (input_size - pool_size) / pool_stride + 1
    
    !! compute gradients for input feature map
    do m = 1, size(input, 3)
       do j = 1, output_size
          jstride = (j-1)*pool_stride
          do i = 1, output_size
             istride = (i-1)*pool_stride
             !! find the index of the maximum value in the corresponding pooling window
             max_index = maxloc(input(istride+1:istride+pool_size, &
                  jstride+1:jstride+pool_size, m))

             !! compute gradients for input feature map
             input_gradients(istride+max_index(1), &
                  jstride+max_index(2), m) = output_gradients(i, j, m)

          end do
       end do
    end do
  end subroutine backward
!!!#############################################################################

end module PoolingLayer
!!!###################################################################################
