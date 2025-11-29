module athena__activation_none
  !! Module containing implementation of no activation function (i.e. linear)
  !!
  !! This module implements the identity function f(x) = x
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type
  use athena__misc_types, only: base_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: none_actv_type


  type, extends(base_actv_type) :: none_actv_type
   contains
     procedure, pass(this) :: activate => none_activate
     procedure, pass(this) :: reset => none_reset
     procedure, pass(this) :: apply_attributes => none_apply_attributes
     procedure, pass(this) :: export_attributes => none_export_attributes
  end type none_actv_type

  interface none_actv_type
     procedure initialise
  end interface none_actv_type



contains

!###############################################################################
  pure function initialise() result(activation)
    !! Initialise a none (no-op) activation function
    implicit none

    ! Arguments
    type(none_actv_type) :: activation
    !! None activation type


    call activation%reset()

  end function initialise
!-------------------------------------------------------------------------------
  pure subroutine none_reset(this)
    !! Reset none activation function attributes and variables
    implicit none

    ! Arguments
    class(none_actv_type), intent(inout) :: this
    !! None activation type

    this%name = "none"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine none_reset
!###############################################################################


!###############################################################################
  subroutine none_apply_attributes(this, attributes)
    !! Load ONNX attributes into none activation function
    implicit none

    ! Arguments
    class(none_actv_type), intent(inout) :: this
    !! None activation type
    type(onnx_attribute_type), dimension(:), intent(in) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    integer :: i
    !! Loop variable

    ! Load provided attributes
    do i=1, size(attributes,dim=1)
       select case(trim(attributes(i)%name))
       case default
          call print_warning( &
               'None activation: unknown attribute '// &
               trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine none_apply_attributes
!###############################################################################


!###############################################################################
  pure function none_export_attributes(this) result(attributes)
    !! Export none activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(none_actv_type), intent(in) :: this
    !! None activation type
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes

    ! No attributes for none activation
    allocate(attributes(0))

  end function none_export_attributes
!###############################################################################


!###############################################################################
  function none_activate(this, val) result(output)
    !! Apply identity activation to 1D array
    !!
    !! Simply returns scaled input: f = scale * x
    implicit none

    ! Arguments
    class(none_actv_type), intent(in) :: this
    !! None activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Scaled output values

    output => val * 1._real32 ! multiplication by 1 to ensure new allocation
  end function none_activate
!###############################################################################

end module athena__activation_none
