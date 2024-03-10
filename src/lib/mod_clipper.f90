!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!!#############################################################################
module clipper
  use constants, only: real12
  implicit none


!!!------------------------------------------------------------------------
!!! gradient clipping type
!!!------------------------------------------------------------------------
  type clip_type
     logical :: l_min_max = .false.
     logical :: l_norm    = .false.
     real(real12) :: min  =-huge(1._real12)
     real(real12) :: max  = huge(1._real12)
     real(real12) :: norm = huge(1._real12)
   contains
     procedure, pass(this) :: read => read_clip
     procedure, pass(this) :: set => set_clip
     procedure, pass(this) :: apply => apply_clip
  end type clip_type

  interface clip_type
     module function clip_setup( &
          clip_min, clip_max, clip_norm) result(clip)
        real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm
        type(clip_type) :: clip
     end function clip_setup
  end interface clip_type



  private

  public :: clip_type


contains

!!!#############################################################################
!!! set clip dictionary
!!!#############################################################################
  module function clip_setup( &
       clip_min, clip_max, clip_norm) result(clip)
    implicit none
    real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm
    type(clip_type) :: clip


    !!--------------------------------------------------------------------------
    !! set up clipping limits
    !!--------------------------------------------------------------------------
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
!!!#############################################################################

!!!#############################################################################
!!! get clipping information
!!!#############################################################################
  subroutine read_clip(this, min_str, max_str, norm_str)
    implicit none
    class(clip_type), intent(inout) :: this
    character(*), intent(in) :: min_str, max_str, norm_str

    if(trim(min_str).ne."")then
       read(min_str,*) this%min
    else
       this%min = -huge(1._real12)
    end if
    if(trim(max_str).ne."")then
       read(max_str,*) this%max
    else
       this%max = huge(1._real12)
    end if

    if(trim(min_str).ne."".or.trim(max_str).ne."")then
       this%l_min_max = .true.
    end if
    if(trim(norm_str).ne."")then
       read(norm_str,*) this%norm
       this%l_norm = .true.
    end if

  end subroutine read_clip
!!!#############################################################################


!!!#############################################################################
!!! set clip dictionary
!!!#############################################################################
  subroutine set_clip(this, clip_dict, clip_min, clip_max, clip_norm)
    implicit none
    class(clip_type), intent(inout) :: this
    type(clip_type), optional, intent(in) :: clip_dict
    real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm


    !!--------------------------------------------------------------------------
    !! set up clipping limits
    !!--------------------------------------------------------------------------
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
!!!#############################################################################


!!!#############################################################################
!!! gradient norm clipping
!!!#############################################################################
  pure subroutine apply_clip(this, length, gradient, bias)
    implicit none
    class(clip_type), intent(in) :: this
    integer, intent(in) :: length
    real(real12), dimension(length), intent(inout) :: gradient
    real(real12), dimension(:), optional, intent(inout) :: bias

    real(real12) :: scale
    real(real12), dimension(:), allocatable :: bias_

    if(present(bias))then
       bias_ = bias
    else
       allocate(bias_(1), source=0._real12)
    end if

    !! clip values to within limits of (min,max)
    if(this%l_min_max)then
       gradient = max(this%min,min(this%max,gradient))
       bias_   = max(this%min,min(this%max,bias_))
    end if

    !! clip values to a maximum L2-norm
    if(this%l_norm)then
       scale = min(1._real12, &
            this%norm/sqrt(sum(gradient**2._real12) + &
            sum(bias_)**2._real12))
       if(scale.lt.1._real12)then
          gradient = gradient * scale
          bias_   = bias_ * scale
       end if
    end if

    if(present(bias)) bias = bias_

  end subroutine apply_clip
!!!#############################################################################

end module clipper
