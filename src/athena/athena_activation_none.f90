module athena__activation_none
  !! Module containing implementation of no activation function (identity)
  !!
  !! This module implements the identity function (no activation).
  !!
  !! Mathematical operation:
  !! \[ f(x) = x \]
  !!
  !! Derivative:
  !! \[ f'(x) = 1 \]
  !!
  !! Properties: Preserves input as-is, linear transformation
  !! Used for regression outputs or when no non-linearity is desired
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type
  use athena__misc_types, only: base_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: none_actv_type, create_from_onnx_none_activation


  type, extends(base_actv_type) :: none_actv_type
   contains
     procedure, pass(this) :: apply => apply_none
     procedure, pass(this) :: reset => reset_none
     procedure, pass(this) :: apply_attributes => apply_attributes_none
     procedure, pass(this) :: export_attributes => export_attributes_none
  end type none_actv_type

  interface none_actv_type
     procedure initialise
  end interface none_actv_type



contains

!###############################################################################
  function initialise(attributes) result(activation)
    !! Initialise a none (no-op) activation function
    implicit none

    ! Arguments
    type(none_actv_type) :: activation
    !! None activation type
    type(onnx_attribute_type), dimension(:), intent(in), optional :: attributes
    !! Optional array of ONNX attributes


    call activation%reset()
    if(present(attributes))then
       call activation%apply_attributes(attributes)
    end if

  end function initialise
!-------------------------------------------------------------------------------
  pure subroutine reset_none(this)
    !! Reset none activation function attributes and variables
    implicit none

    ! Arguments
    class(none_actv_type), intent(inout) :: this
    !! None activation type

    this%name = "none"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine reset_none
!-------------------------------------------------------------------------------
  function create_from_onnx_none_activation(attributes) result(activation)
    !! Create none activation function from ONNX attributes
    implicit none

    ! Arguments
    type(onnx_attribute_type), dimension(:), intent(in) :: attributes
    !! Array of ONNX attributes

    class(base_actv_type), allocatable :: activation
    !! Instance of activation type

    allocate(activation, source = none_actv_type(attributes = attributes))

  end function create_from_onnx_none_activation
!###############################################################################


!###############################################################################
  subroutine apply_attributes_none(this, attributes)
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
       case("name")
          if(trim(attributes(i)%val) .ne. trim(this%name))then
             call print_warning( &
                  'None activation: name attribute "' // &
                  trim(attributes(i)%val) // &
                  '"" does not match expected "' // trim(this%name)//'"' &
             )

          end if
       case default
          call print_warning( &
               'None activation: unknown attribute '// &
               trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine apply_attributes_none
!###############################################################################


!###############################################################################
  pure function export_attributes_none(this) result(attributes)
    !! Export none activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(none_actv_type), intent(in) :: this
    !! None activation type
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    character(50) :: buffer
    !! Temporary string buffer

    ! No attributes for none activation
    allocate(attributes(1))

    write(buffer, '(A)') this%name
    attributes(1) = onnx_attribute_type( &
         "name", "string", trim(adjustl(buffer)) )

  end function export_attributes_none
!###############################################################################


!###############################################################################
  function apply_none(this, val) result(output)
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
  end function apply_none
!###############################################################################

end module athena__activation_none
