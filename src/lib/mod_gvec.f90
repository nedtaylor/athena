module gvec
  implicit none

contains

  subroutine gauss(G, dist, nbins, centre, sigma, cutoff)
    integer, intent(in) :: nbins
    real, intent(in) :: cutoff, dist, sigma
    real, intent(in) :: centre(nbins)
    real, intent(inout) :: G(nbins)
    integer :: ibin
    real :: rtmp

    do ibin=1, nbins
       rtmp = ((dist-centre(ibin))/sigma)**2
       if(rtmp.gt.16.E0) cycle
       !call smooth_cutoff(rtmp, dist, cutoff) !! NOT SURE WHAT THIS IS MEANT TO DO
       G(ibin) = G(ibin) + rtmp
    end do
    return
  end subroutine gauss

  subroutine smooth_cutoff(val, dist, cutoff)
    real, intent(in) :: dist, cutoff
    real, intent(inout) :: val
    real :: pi = 4.0*atan(1.0)
    if(dist>cutoff)then
       val = 0
    else
       val = 0.5*(cos(pi*dist/cutoff) + 1.0)
    end if
    return
  end subroutine smooth_cutoff

end module gvec
