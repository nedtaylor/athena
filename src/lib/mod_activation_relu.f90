module athena__activation_relu
  !! Module containing implementation of the ReLU activation function
  !!
  !! This module implements the Rectified Linear Unit activation function
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*), max
  use athena__misc_types, only: activation_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: relu_actv_type


  type, extends(activation_type) :: relu_actv_type
     !! Type for ReLU activation function with overloaded procedures
   contains
     procedure, pass(this) :: activate => relu_activate
     procedure, pass(this) :: reset => relu_reset
     procedure, pass(this) :: apply_attributes => relu_apply_attributes
     procedure, pass(this) :: export_attributes => relu_export_attributes
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
  pure subroutine relu_reset(this)
    !! Reset ReLU activation function attributes and variables
    implicit none

    ! Arguments
    class(relu_actv_type), intent(inout) :: this
    !! ReLU activation type

    this%name = "relu"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine relu_reset
!###############################################################################


!###############################################################################
  subroutine relu_apply_attributes(this, attributes)
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

  end subroutine relu_apply_attributes
!###############################################################################


!###############################################################################
  pure function relu_export_attributes(this) result(attributes)
    !! Export ReLU activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(relu_actv_type), intent(in) :: this
    !! ReLU activation type
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    integer :: n_attributes
    !! Number of attributes
    character(50) :: buffer
    !! Temporary string buffer

    n_attributes = 2
    allocate(attributes(n_attributes))

    write(buffer, '(F10.6)') this%scale
    attributes(1) = onnx_attribute_type( &
         "scale", "float", trim(adjustl(buffer)) )

    write(buffer, '(F10.6)') this%threshold
    attributes(2) = onnx_attribute_type( &
         "threshold", "float", trim(adjustl(buffer)) )

  end function relu_export_attributes
!###############################################################################


!###############################################################################
  function relu_activate(this, val) result(output)
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
  end function relu_activate
!###############################################################################

end module athena__activation_relu
