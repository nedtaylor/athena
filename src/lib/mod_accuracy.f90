!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains functions to compute the accuracy of a model
!!! module includes the following procedures:
!!! - categorical_score - computes the accuracy of a categorisation model
!!! - mae_score         - computes the mean absolute error of a continuous model
!!! - mse_score         - computes the mean squared error of a continuous model
!!! - r2_score          - computes the R^2 score of a continuous model
!!!#############################################################################
module accuracy
  use constants, only: real32
  implicit none


  private

  public :: compute_accuracy_function
  public :: categorical_score
  public :: mae_score, mse_score, rmse_score
  public :: r2_score


  abstract interface
     !! compute the accuracy function
     !! predicted = (R, in) predicted values
     !! expected  = (R, in) expected values
     !! output    = (R, in) accuracy function
     pure function compute_accuracy_function(predicted, expected) result(output)
       import real32
       real(real32), dimension(:,:), intent(in) :: predicted, expected
       real(real32), dimension(size(predicted,2)) :: output
     end function compute_accuracy_function
  end interface

contains

!!!#############################################################################
!!! compute accuracy
!!! this only works (and is only valid for?) categorisation problems
!!!#############################################################################
  pure function categorical_score(predicted, expected) result(output)
    implicit none
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    real(real32), dimension(size(expected,2)) :: output

    integer :: s

    !! Compute the accuracy
    do concurrent(s=1:size(expected,2))
       if (maxloc(expected(:,s),dim=1).eq.maxloc(predicted(:,s),dim=1)) then
          output(s) = 1._real32
       else
          output(s) = 0._real32
       end if
    end do

  end function categorical_score
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
!!! works for continuous datasets
  pure function mae_score(predicted, expected) result(output)
    implicit none
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    real(real32), dimension(size(expected,2)) :: output

    !! Compute the accuracy
    output = 1._real32 - sum(abs(expected - predicted),dim=1)/size(expected,1)

  end function mae_score
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
!!! works for continuous datasets
  pure function mse_score(predicted, expected) result(output)
    implicit none
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    real(real32), dimension(size(expected,2)) :: output

    !! Compute the accuracy
    output = 1._real32 - &
         sum((expected - predicted)**2._real32,dim=1)/size(expected,1)

  end function mse_score
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
!!! works for continuous datasets
  pure function rmse_score(predicted, expected) result(output)
    implicit none
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    real(real32), dimension(size(expected,2)) :: output

    !! Compute the accuracy
    output = 1._real32 - &
         sqrt(sum((expected - predicted)**2._real32,dim=1)/size(expected,1))

  end function rmse_score
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
!!! works for continuous datasets
  pure function r2_score(predicted, expected) result(output)
    implicit none
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    real(real32), dimension(size(expected,2)) :: y_mean, rss, tss
    real(real32), dimension(size(expected,2)) :: output

    real(real32), parameter :: epsilon = 1.E-8_real32

    integer :: s

    do s = 1, size(expected,2)
       !! compute mean of true/expected
       y_mean(s) = sum(expected(:,s),dim=1) / size(expected,dim=1)
 
       !! compute total sum of squares
       tss(s) = sum( ( expected(:,s) - y_mean(s) ) ** 2._real32, dim=1 )
 
       !! compute residual sum of squares
       rss(s) = sum( ( expected(:,s) - predicted(:,s) ) ** 2._real32, dim=1 )
 
       !! compute accuracy (R^2 score)
       if(abs(rss(s)).lt.epsilon)then
         output(s) = 1._real32
       elseif(abs(tss(s)).lt.epsilon.or.rss(s)/tss(s).gt.1._real32)then
         output(s) = 0._real32
       else
         output(s) = 1._real32 - rss(s)/tss(s)
       end if
    end do

  end function r2_score
!!!#############################################################################

end module accuracy