module athena__activation_linear
  !! Module containing implementation of the linear activation function
  !!
  !! This module implements a scaled linear (affine) activation function.
  !!
  !! Mathematical operation:
  !! \[ f(x) = \alpha x \]
  !!
  !! where \(\alpha\) is a scaling factor (typically \(\alpha=1\))
  !!
  !! Derivative:
  !! \[ f'(x) = \alpha \]
  !!
  !! Properties: Linear transformation, no non-linearity introduced
  !! Used for regression outputs or when no activation is desired
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*)
  use athena__misc_types, only: base_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: linear_actv_type, create_from_onnx_linear_activation


  type, extends(base_actv_type) :: linear_actv_type
   contains
     procedure, pass(this) :: apply => apply_linear
     procedure, pass(this) :: reset => reset_linear
     procedure, pass(this) :: apply_attributes => apply_attributes_linear
     procedure, pass(this) :: export_attributes => export_attributes_linear
  end type linear_actv_type

  interface linear_actv_type
     procedure initialise
  end interface linear_actv_type



contains

!###############################################################################
  function initialise(scale, attributes) result(activation)
    !! Initialise a linear activation function
    implicit none

    ! Arguments
    real(real32), intent(in), optional :: scale
    !! Optional scale factor for activation output
    type(onnx_attribute_type), dimension(:), intent(in), optional :: attributes
    !! Optional array of ONNX attributes
    type(linear_actv_type) :: activation
    !! Linear activation type


    call activation%reset()

    if(present(scale)) activation%scale = scale
    if(abs(activation%scale-1._real32) .gt. 1.e-6_real32)then
       activation%apply_scaling = .true.
    end if

    if(present(attributes)) then
       call activation%apply_attributes(attributes)
    end if

  end function initialise
!-------------------------------------------------------------------------------
  pure subroutine reset_linear(this)
    !! Reset linear activation function attributes and variables
    implicit none

    ! Arguments
    class(linear_actv_type), intent(inout) :: this
    !! Linear activation type

    this%name = "linear"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine reset_linear
!-------------------------------------------------------------------------------
  function create_from_onnx_linear_activation(attributes) result(activation)
    !! Create linear activation function from ONNX attributes
    implicit none

    ! Arguments
    type(onnx_attribute_type), dimension(:), intent(in) :: attributes
    !! Array of ONNX attributes

    class(base_actv_type), allocatable :: activation
    !! Instance of activation type

    allocate(activation, source = linear_actv_type(attributes = attributes))

  end function create_from_onnx_linear_activation
!###############################################################################


!###############################################################################
  subroutine apply_attributes_linear(this, attributes)
    !! Load ONNX attributes into linear activation function
    implicit none

    ! Arguments
    class(linear_actv_type), intent(inout) :: this
    !! Linear activation type
    type(onnx_attribute_type), dimension(:), intent(in) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    integer :: i
    !! Loop variable

    ! Load provided attributes
    do i=1, size(attributes,dim=1)
       select case(trim(attributes(i)%name))
       case("scale")
          read(attributes(i)%val,*) this%scale
          if(abs(this%scale-1._real32) .gt. 1.e-6_real32)then
             this%apply_scaling = .true.
          else
             this%apply_scaling = .false.
          end if
       case("name")
          if(trim(attributes(i)%val) .ne. trim(this%name)) then
             call print_warning( &
                  'Linear activation: name attribute "' // &
                  trim(attributes(i)%val) // &
                  '"" does not match expected "' // trim(this%name)//'"' &
             )

          end if
       case default
          call print_warning( &
               'Linear activation: unknown attribute '// &
               trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine apply_attributes_linear
!###############################################################################


!###############################################################################
  pure function export_attributes_linear(this) result(attributes)
    !! Export linear activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(linear_actv_type), intent(in) :: this
    !! Linear activation type
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    character(50) :: buffer
    !! Temporary string buffer

    allocate(attributes(2))

    write(buffer, '(A)') this%name
    attributes(1) = onnx_attribute_type( &
         "name", "string", trim(adjustl(buffer)) )

    write(buffer, '(F10.6)') this%scale
    attributes(2) = onnx_attribute_type( &
         "scale", "float", trim(adjustl(buffer)) )

  end function export_attributes_linear
!###############################################################################


!###############################################################################
  function apply_linear(this, val) result(output)
    !! Apply linear activation to 1D array
    !!
    !! Computes: f = scale * x
    implicit none

    ! Arguments
    class(linear_actv_type), intent(in) :: this
    !! Linear activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Scaled output values

    if(this%apply_scaling)then
       output => val * this%scale
    else
       output => val * 1._real32 ! multiplication by 1 to ensure new allocation
    end if
  end function apply_linear
!###############################################################################

end module athena__activation_linear
