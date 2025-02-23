module athena__clipper
  !! Module containing functions to clip gradients
  !!
  !! This module implements clipping methods for layer gradients
  use athena__constants, only: real32
  implicit none


  private

  public :: clip_type


  type clip_type
     !! Type for clipping gradients
     logical :: l_min_max = .false.
     !! Boolean whether min/max values are set
     logical :: l_norm    = .false.
     !! Boolean whether a norm is set
     real(real32) :: min  =-huge(1._real32)
     !! Minimum value for clipping
     real(real32) :: max  = huge(1._real32)
     !! Maximum value for clipping
     real(real32) :: norm = huge(1._real32)
     !! Maximum L2-norm for clipping
   contains
     procedure, pass(this) :: read => read_clip
     !! Read clipping information
     procedure, pass(this) :: set => set_clip
     !! Set clipping information
     procedure, pass(this) :: apply => apply_clip
     !! Apply clipping to gradients
  end type clip_type

  interface clip_type
     !! Interface for the clip type
     module function clip_setup( &
          clip_min, clip_max, clip_norm) result(clip)
       !! Set up the clip dictionary
       real(real32), optional, intent(in) :: clip_min, clip_max, clip_norm
       !! Minimum, maximum, and norm values for clipping
       type(clip_type) :: clip
       !! Clip dictionary
     end function clip_setup
  end interface clip_type



contains

!###############################################################################
  module function clip_setup( &
       clip_min, clip_max, clip_norm) result(clip)
    !! Set up the clip dictionary
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: clip_min, clip_max, clip_norm
    !! Minimum, maximum, and norm values for clipping
    type(clip_type) :: clip
    !! Instance of the clip type


    !---------------------------------------------------------------------------
    ! Set up clipping limits
    !---------------------------------------------------------------------------
    if(present(clip_min))then
       clip%l_min_max = .true.
       clip%min = clip_min
    end if
    if(present(clip_max))then
       clip%l_min_max = .true.
       clip%max = clip_max
    end if
    if(present(clip_norm))then
       clip%l_norm = .true.
       clip%norm = clip_norm
    end if

   end function clip_setup
!###############################################################################


!###############################################################################
  subroutine read_clip(this, min_str, max_str, norm_str)
    !! Read clipping information
    implicit none

    ! Arguments
    class(clip_type), intent(inout) :: this
    !! Instance of the clip type
    character(*), intent(in) :: min_str, max_str, norm_str
    !! Strings for min, max, and norm values

    if(trim(min_str).ne."")then
       read(min_str,*) this%min
    else
       this%min = -huge(1._real32)
    end if
    if(trim(max_str).ne."")then
       read(max_str,*) this%max
    else
       this%max = huge(1._real32)
    end if

    if(trim(min_str).ne."".or.trim(max_str).ne."")then
       this%l_min_max = .true.
    end if
    if(trim(norm_str).ne."")then
       read(norm_str,*) this%norm
       this%l_norm = .true.
    end if

  end subroutine read_clip
!###############################################################################


!###############################################################################
  subroutine set_clip(this, clip_dict, clip_min, clip_max, clip_norm)
    !! Set clipping information
    implicit none

    ! Arguments
    class(clip_type), intent(inout) :: this
    !! Instance of the clip type
    type(clip_type), optional, intent(in) :: clip_dict
    !! Clip dictionary
    real(real32), optional, intent(in) :: clip_min, clip_max, clip_norm
    !! Minimum, maximum, and norm values for clipping


    !---------------------------------------------------------------------------
    ! Set up clipping limits
    !---------------------------------------------------------------------------
    if(present(clip_dict))then
       this%l_min_max = clip_dict%l_min_max
       this%l_norm = clip_dict%l_norm
       this%min = clip_dict%min
       this%max = clip_dict%max
       this%norm = clip_dict%norm
       if(present(clip_min).or.present(clip_max).or.present(clip_norm))then
          write(*,*) "Multiple clip options provided"
          write(*,*) "Ignoring all except clip_dict"
       end if
    else
       if(present(clip_min))then
          this%l_min_max = .true.
          this%min = clip_min
       end if
       if(present(clip_max))then
          this%l_min_max = .true.
          this%max = clip_max
       end if
       if(present(clip_norm))then
          this%l_norm = .true.
          this%norm = clip_norm
       end if
    end if

  end subroutine set_clip
!###############################################################################


!###############################################################################
  pure subroutine apply_clip(this, length, gradient, bias)
    !! Function to apply clipping to gradients
    implicit none

    ! Arguments
    class(clip_type), intent(in) :: this
    !! Instance of the clip type
    integer, intent(in) :: length
    !! Length of the gradient
    real(real32), dimension(length), intent(inout) :: gradient
    !! Gradient to be clipped
    real(real32), dimension(:), optional, intent(inout) :: bias
    !! Bias to be clipped

    ! Local variables
    real(real32) :: scale
    !! Scaling factor for the gradient
    real(real32), dimension(:), allocatable :: bias_
    !! Copy of the bias

    if(present(bias))then
       bias_ = bias
    else
       allocate(bias_(1), source=0._real32)
    end if

    ! Clip values to within limits of (min,max)
    if(this%l_min_max)then
       gradient = max(this%min,min(this%max,gradient))
       bias_   = max(this%min,min(this%max,bias_))
    end if

    ! Clip values to a maximum L2-norm
    if(this%l_norm)then
       scale = min(1._real32, &
            this%norm/sqrt(sum(gradient**2._real32) + &
            sum(bias_)**2._real32))
       if(scale.lt.1._real32)then
          gradient = gradient * scale
          bias_   = bias_ * scale
       end if
    end if

    if(present(bias)) bias = bias_

  end subroutine apply_clip
!###############################################################################

end module athena__clipper
