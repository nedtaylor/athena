module misc_ml
  implicit none


  private

  public :: get_padding_half


contains

!!!########################################################################
!!! return width of padding from kernel/filter size
!!!########################################################################
  function get_padding_half(width) result(output)
    implicit none
    integer, intent(in) :: width
    integer :: output
    
    output = ( width - (1 - mod(width,2)) - 1 ) / 2
        
  end function get_padding_half
!!!########################################################################

end module misc_ml
