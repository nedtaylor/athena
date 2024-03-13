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
  use constants, only: real12
  implicit none


  private

  public :: categorical_score
  public :: mae_score, mse_score
  public :: r2_score


contains

!!!#############################################################################
!!! compute accuracy
!!! this only works (and is only valid for?) categorisation problems
!!!#############################################################################
  function categorical_score(output, expected) result(accuracy)
    implicit none
    real(real12), dimension(:,:), intent(in) :: output
    integer, dimension(:,:) :: expected
    real(real12), dimension(size(expected,2)) :: accuracy

    integer :: s

    !! Compute the accuracy
    do concurrent(s=1:size(expected,2))
       if (maxloc(expected(:,s),dim=1).eq.maxloc(output(:,s),dim=1)) then
          accuracy(s) = 1._real12
       else
          accuracy(s) = 0._real12
       end if
    end do

  end function categorical_score
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
!!! works for continuous datasets
  function mae_score(output, expected) result(accuracy)
    implicit none
    real(real12), dimension(:,:), intent(in) :: output, expected
    real(real12), dimension(size(expected,2)) :: accuracy

    !! Compute the accuracy
    accuracy = 1._real12 - sum(abs(expected - output),dim=1)/size(expected(:,1))

  end function mae_score
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
!!! works for continuous datasets
  function mse_score(output, expected) result(accuracy)
    implicit none
    real(real12), dimension(:,:), intent(in) :: output, expected
    real(real12), dimension(size(expected,2)) :: accuracy

    !! Compute the accuracy
    accuracy = 1._real12 - sum((expected - output)**2._real12,dim=1)/size(expected(:,1))

  end function mse_score
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
!!! works for continuous datasets (CURRENTLY UNAVAILABLE)
  function r2_score(output, expected) result(accuracy)
    implicit none
    real(real12), dimension(:,:), intent(in) :: output, expected
    real(real12), dimension(size(expected,2)) :: y_mean, rss, tss
    real(real12), dimension(size(expected,2)) :: accuracy

    real(real12), parameter :: epsilon = 1.E-8_real12

    integer :: s

    do s = 1, size(expected,2)
       !! compute mean of true/expected
       y_mean(s) = sum(expected(:,s),dim=1) / size(expected(:,s),dim=1)
 
       !! compute total sum of squares
       tss(s) = sum( ( expected(:,s) - y_mean(s) ) ** 2._real12, dim=1 )
 
       !! compute residual sum of squares
       rss(s) = sum( ( expected(:,s) - output(:,s) ) ** 2._real12, dim=1 )
 
       !! compute accuracy (R^2 score)
       if(abs(rss(s)).lt.epsilon)then
         accuracy(s) = 1._real12
       elseif(abs(tss(s)).lt.epsilon.or.rss(s)/tss(s).gt.1._real12)then
         accuracy(s) = 0._real12
       else
         accuracy(s) = 1._real12 - rss(s)/tss(s)
       end if
    end do

  end function r2_score
!!!#############################################################################

end module accuracy