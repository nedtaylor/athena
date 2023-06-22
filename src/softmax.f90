!!!#############################################################################
!!!#############################################################################
module SoftmaxLayer
  use constants, only: real12
  implicit none

  integer :: sm_num_classes


  private

  public :: initialise, forward, backward



contains

!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine initialise(num_classes)
    implicit none
    integer, intent(in) :: num_classes
    sm_num_classes = num_classes
  end subroutine initialise
!!!#############################################################################
  
!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine forward(input, output)
    implicit none
    real(real12), dimension(:), intent(in) :: input
    real(real12), dimension(:), intent(out) :: output

    integer :: i
    real :: max_val

    ! Compute maximum value for numerical stability
    max_val = maxval(input)

    ! Compute softmax values
    do i = 1, sm_num_classes
       output(i) = exp(input(i) - max_val)
    end do

    ! Normalize softmax values
    output = output / sum(output)
  end subroutine forward
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine backward(output, expected, input_gradients)
    implicit none
    real(real12), dimension(:), intent(in) :: output
    integer :: expected
    real(real12), dimension(:), intent(out) :: input_gradients

    integer :: i

    ! Compute gradients for softmax layer
    input_gradients = output
    input_gradients(expected) = input_gradients(expected) - 1._real12

  end subroutine backward
!!!#############################################################################
  
end module SoftmaxLayer
!!!#############################################################################
