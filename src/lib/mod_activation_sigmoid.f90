module athena__activation_sigmoid
  !! Module containing implementation of the sigmoid activation function
  !!
  !! This module implements the logistic sigmoid function for normalizing
  !! outputs between 0 and 1
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(+), operator(-), &
       operator(*), operator(/), exp, merge, operator(.gt.), sigmoid
  use athena__misc_types, only: base_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: sigmoid_actv_type, create_from_onnx_sigmoid_activation


  type, extends(base_actv_type) :: sigmoid_actv_type
     !! Type for sigmoid activation function with overloaded procedures
   contains
     procedure, pass(this) :: apply => apply_sigmoid
     procedure, pass(this) :: reset => reset_sigmoid
     procedure, pass(this) :: apply_attributes => apply_attributes_sigmoid
     procedure, pass(this) :: export_attributes => export_attributes_sigmoid
  end type sigmoid_actv_type

  interface sigmoid_actv_type
     procedure initialise
  end interface sigmoid_actv_type



contains

!###############################################################################
  function initialise(scale, attributes) result(activation)
    !! Initialise a sigmoid activation function
    implicit none

    ! Arguments
    real(real32), intent(in), optional :: scale
    !! Optional scale factor for activation output
    type(onnx_attribute_type), dimension(:), intent(in), optional :: attributes
    !! Optional array of ONNX attributes
    type(sigmoid_actv_type) :: activation
    !! Sigmoid activation type


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
  pure subroutine reset_sigmoid(this)
    !! Reset sigmoid activation function attributes and variables
    implicit none

    ! Arguments
    class(sigmoid_actv_type), intent(inout) :: this
    !! Sigmoid activation type

    this%name = "sigmoid"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine reset_sigmoid
!-------------------------------------------------------------------------------
  function create_from_onnx_sigmoid_activation(attributes) result(activation)
    !! Create sigmoid activation function from ONNX attributes
    implicit none

    ! Arguments
    type(onnx_attribute_type), dimension(:), intent(in) :: attributes
    !! Array of ONNX attributes

    class(base_actv_type), allocatable :: activation
    !! Instance of activation type

    allocate(activation, source = sigmoid_actv_type(attributes = attributes))

  end function create_from_onnx_sigmoid_activation
!###############################################################################


!###############################################################################
  subroutine apply_attributes_sigmoid(this, attributes)
    !! Load ONNX attributes into sigmoid activation function
    implicit none

    ! Arguments
    class(sigmoid_actv_type), intent(inout) :: this
    !! Sigmoid activation type
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
       case default
          call print_warning( &
               'Sigmoid activation: unknown attribute '//trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine apply_attributes_sigmoid
!###############################################################################


!###############################################################################
  pure function export_attributes_sigmoid(this) result(attributes)
    !! Export sigmoid activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(sigmoid_actv_type), intent(in) :: this
    !! Sigmoid activation type
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

  end function export_attributes_sigmoid
!###############################################################################


!###############################################################################
  function apply_sigmoid(this, val) result(output)
    !! Apply sigmoid activation to 1D array
    !!
    !! Computes: f = 1/(1+exp(-x))
    implicit none

    ! Arguments
    class(sigmoid_actv_type), intent(in) :: this
    !! Sigmoid activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values in range [0,1]

    if(this%apply_scaling)then
       output => sigmoid(val) * this%scale
    else
       output => sigmoid(val)
    end if
  end function apply_sigmoid
!###############################################################################

end module athena__activation_sigmoid
