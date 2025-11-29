module athena__activation_relu
  !! Module containing implementation of the ReLU activation function
  !!
  !! This module implements the Rectified Linear Unit activation function
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*), max
  use athena__misc_types, only: base_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: relu_actv_type, create_from_onnx_relu_activation


  type, extends(base_actv_type) :: relu_actv_type
     !! Type for ReLU activation function with overloaded procedures
   contains
     procedure, pass(this) :: apply => apply_relu
     procedure, pass(this) :: reset => reset_relu
     procedure, pass(this) :: apply_attributes => apply_attributes_relu
     procedure, pass(this) :: export_attributes => export_attributes_relu
  end type relu_actv_type

  interface relu_actv_type
     procedure initialise
  end interface relu_actv_type



contains

!###############################################################################
  function initialise(scale, attributes) result(activation)
    !! Initialise a ReLU activation function
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output
    type(relu_actv_type) :: activation
    !! ReLU activation type
    type(onnx_attribute_type), optional, intent(in) :: attributes(:)
    !! Optional ONNX attributes


    call activation%reset()

    if(present(scale)) activation%scale = scale
    if(abs(activation%scale-1._real32) .gt. 1.e-6_real32)then
       activation%apply_scaling = .true.
    end if
    if(present(attributes))then
       call activation%apply_attributes(attributes)
    end if

  end function initialise
!-------------------------------------------------------------------------------
  pure subroutine reset_relu(this)
    !! Reset ReLU activation function attributes and variables
    implicit none

    ! Arguments
    class(relu_actv_type), intent(inout) :: this
    !! ReLU activation type

    this%name = "relu"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine reset_relu
!-------------------------------------------------------------------------------
  function create_from_onnx_relu_activation(attributes) result(activation)
    !! Create ReLU activation function from ONNX attributes
    implicit none

    ! Arguments
    type(onnx_attribute_type), dimension(:), intent(in) :: attributes
    !! Array of ONNX attributes

    class(base_actv_type), allocatable :: activation
    !! Instance of activation type

    allocate(activation, source = relu_actv_type(attributes = attributes))

  end function create_from_onnx_relu_activation
!###############################################################################


!###############################################################################
  subroutine apply_attributes_relu(this, attributes)
    !! Load ONNX attributes into ReLU activation function
    implicit none

    ! Arguments
    class(relu_actv_type), intent(inout) :: this
    !! ReLU activation type
    type(onnx_attribute_type), dimension(:), intent(in) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    integer :: i
    !! Loop variable
    type(onnx_attribute_type) :: attribute
    !! Temporary attribute holder
    character(20), allocatable, dimension(:) :: attribute_names

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
       case("threshold")
          read(attributes(i)%val,*) this%threshold
       case default
          call print_warning( &
               'ReLU activation: unknown attribute '//trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine apply_attributes_relu
!###############################################################################


!###############################################################################
  pure function export_attributes_relu(this) result(attributes)
    !! Export ReLU activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(relu_actv_type), intent(in) :: this
    !! ReLU activation type
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    character(50) :: buffer
    !! Temporary string buffer

    allocate(attributes(3))

    write(buffer, '(A)') this%name
    attributes(1) = onnx_attribute_type( &
         "name", "string", trim(adjustl(buffer)) )

    write(buffer, '(F10.6)') this%scale
    attributes(2) = onnx_attribute_type( &
         "scale", "float", trim(adjustl(buffer)) )

    write(buffer, '(F10.6)') this%threshold
    attributes(3) = onnx_attribute_type( &
         "threshold", "float", trim(adjustl(buffer)) )

  end function export_attributes_relu
!###############################################################################


!###############################################################################
  function apply_relu(this, val) result(output)
    !! Apply ReLU activation to 1D array
    !!
    !! Computes: f = max(0,x)
    implicit none

    ! Arguments
    class(relu_actv_type), intent(in) :: this
    !! ReLU activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values

    if(this%apply_scaling)then
       output => max(val, this%threshold) * this%scale
    else
       output => max(val, this%threshold)
    end if
  end function apply_relu
!###############################################################################

end module athena__activation_relu
