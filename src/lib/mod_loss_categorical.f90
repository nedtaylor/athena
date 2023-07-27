!!! https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e#:~:text=Categorical%20cross%2Dentropy%20is%20used,%5D%20for%203%2Dclass%20problem.
!!! https://stackoverflow.com/questions/8612466/how-to-alias-a-function-name-in-fortran
module loss_categorical
  use constants, only: real12
  implicit none

  abstract interface
     function compute_loss_function(predicted, expected) result(output)
       import real12
       real(real12), dimension(:), intent(in) :: predicted
       integer, intent(in) :: expected
       real(real12) :: output
     end function compute_loss_function
  end interface
  

  private

  public :: compute_loss_function
  public :: compute_loss_bce
  public :: compute_loss_cce
  public :: compute_loss_mse
  public :: compute_loss_nll


contains

!!!#############################################################################
!!! compute losses
!!! method: Binary cross entropy
!!!#############################################################################
  function compute_loss_bce(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted
    integer, intent(in) :: expected
    real(real12) :: output, epsilon

    epsilon = 1.E-10_real12
    output = -log(predicted(expected)+epsilon)

  end function compute_loss_bce
!!!#############################################################################


!!!#############################################################################
!!! compute 
!!! method: categorical cross entropy
!!!#############################################################################
  function compute_loss_cce(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted
    integer, intent(in) :: expected
    real(real12) :: output, epsilon

    epsilon = 1.E-10_real12
    output = -log(predicted(expected)+epsilon)

  end function compute_loss_cce
!!!#############################################################################


!!!!#############################################################################
!!!! compute loss derivative
!!!! method: categorical cross entropy
!!!! this is handled by the softmax backward subroutine
!!!!#############################################################################
!  subroutine compute_loss_derivative_cce(predicted, expected, gradient)
!    implicit none
!    integer, intent(in) :: expected
!    real(real12), dimension(:), intent(in) :: predicted
!    real(real12), dimension(:), intent(out) :: gradient
!    
!    gradient = predicted
!    gradient(expected) = predicted(expected) - 1._real12
!
!  end subroutine compute_loss_derivative_cce
!!!!#############################################################################


!!!#############################################################################
!!! compute losses
!!! method: mean squared error
!!!#############################################################################
  function compute_loss_mse(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted
    integer, intent(in) :: expected
    integer :: i
    real(real12) :: output, total

    !! Compute the cross-entropy loss
    total = 0._real12
    do i=1,size(predicted)
       if(i.eq.expected)then
          total = total + (predicted(i) - 1._real12)**2.E0
       else
          total = total + predicted(i)**2.E0
       end if
    end do
    output = total /(2*size(predicted))

  end function compute_loss_mse
!!!#############################################################################


!!!#############################################################################
!!! compute losses
!!! method: categorical cross entropy
!!!#############################################################################
  function compute_loss_nll(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted
    integer, intent(in) :: expected
    integer :: i
    real(real12) :: output, epsilon

    epsilon = 1.E-10_real12
    output = 0._real12
    do i=1,size(predicted)
       if(i.eq.expected)then
          output = output - log(predicted(i)+epsilon)
       else
          output = output - log(1-predicted(i)+epsilon)
       end if
    end do

  end function compute_loss_nll
!!!#############################################################################

end module loss_categorical
