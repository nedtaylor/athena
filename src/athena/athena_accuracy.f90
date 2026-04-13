module athena__accuracy
  !! Module containing functions to compute the accuracy of a model
  use coreutils, only: real32
  implicit none


  private

  public :: compute_accuracy_function
  public :: categorical_score
  public :: mae_score, mse_score, rmse_score
  public :: r2_score


  abstract interface
     !! Interface for the accuracy function
     pure function compute_accuracy_function(predicted, expected) result(output)
       !! Compute the accuracy of a model
       import real32
       real(real32), dimension(:,:), intent(in) :: predicted, expected
       !! Predicted and expected values
       real(real32), dimension(size(expected,2)) :: output
       !! Accuracy of the model
     end function compute_accuracy_function
  end interface

contains

!###############################################################################
  pure function categorical_score(predicted, expected) result(output)
    !! Compute the categorical accuracy of a model
    !!
    !! This function is only valid for categorical/classification datasets
    implicit none

    !! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(expected,2)) :: output
    !! Categorical accuracy

    ! Local variables
    integer :: s
    !! Loop index

    !! Compute the accuracy
    do concurrent(s=1:size(expected,2))
       if(maxloc(expected(:,s),dim=1).eq.maxloc(predicted(:,s),dim=1))then
          output(s) = 1._real32
       else
          output(s) = 0._real32
       end if
    end do

  end function categorical_score
!###############################################################################


!###############################################################################
  pure function mae_score(predicted, expected) result(output)
    !! Compute the mean absolute error of a model
    !!
    !! This function is only valid for continuous datasets
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(expected,2)) :: output
    !! Mean absolute error

    ! Compute the accuracy
    output = 1._real32 - sum(abs(expected - predicted),dim=1)/size(expected,1)

  end function mae_score
!###############################################################################


!###############################################################################
  pure function mse_score(predicted, expected) result(output)
    !! Compute the mean squared error of a model
    !!
    !! This function is only valid for continuous datasets
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(expected,2)) :: output
    !! Mean squared error

    ! Compute the accuracy
    output = 1._real32 - &
         sum((expected - predicted)**2._real32,dim=1)/size(expected,1)

  end function mse_score
!###############################################################################


!###############################################################################
  pure function rmse_score(predicted, expected) result(output)
    !! Compute the root mean squared error of a model
    !!
    !! This function is only valid for continuous datasets
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(expected,2)) :: output
    !! Root mean squared error

    ! Compute the accuracy
    output = 1._real32 - &
         sqrt(sum((expected - predicted)**2._real32,dim=1)/size(expected,1))

  end function rmse_score
!###############################################################################


!###############################################################################
  pure function r2_score(predicted, expected) result(output)
    !! Compute the R^2 score of a model
    !!
    !! This function is only valid for continuous datasets
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    real(real32), dimension(size(expected,2)) :: y_mean, rss, tss
    real(real32), dimension(size(expected,2)) :: output

    ! Local variables
    real(real32), parameter :: epsilon = 1.E-8_real32
    !! Small value to avoid division by zero
    integer :: s
    !! Loop index

    do s = 1, size(expected,2)
       ! compute mean of true/expected
       y_mean(s) = sum(expected(:,s),dim=1) / size(expected,dim=1)

       ! compute total sum of squares
       tss(s) = sum( ( expected(:,s) - y_mean(s) ) ** 2._real32, dim=1 )

       ! compute residual sum of squares
       rss(s) = sum( ( expected(:,s) - predicted(:,s) ) ** 2._real32, dim=1 )

       ! compute accuracy (R^2 score)
       if(abs(rss(s)).lt.epsilon)then
          output(s) = 1._real32
       elseif(abs(tss(s)).lt.epsilon.or.rss(s)/tss(s).gt.1._real32)then
          output(s) = 0._real32
       else
          output(s) = 1._real32 - rss(s)/tss(s)
       end if
    end do

  end function r2_score
!###############################################################################

end module athena__accuracy
