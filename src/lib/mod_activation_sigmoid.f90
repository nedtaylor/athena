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

  public :: sigmoid_actv_type


  type, extends(base_actv_type) :: sigmoid_actv_type
     !! Type for sigmoid activation function with overloaded procedures
   contains
     procedure, pass(this) :: activate => sigmoid_activate
     procedure, pass(this) :: reset => sigmoid_reset
     procedure, pass(this) :: apply_attributes => sigmoid_apply_attributes
     procedure, pass(this) :: export_attributes => sigmoid_export_attributes
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
  pure subroutine sigmoid_reset(this)
    !! Reset sigmoid activation function attributes and variables
    implicit none

    ! Arguments
    class(sigmoid_actv_type), intent(inout) :: this
    !! Sigmoid activation type

    this%name = "sigmoid"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine sigmoid_reset
!###############################################################################


!###############################################################################
  subroutine sigmoid_apply_attributes(this, attributes)
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

  end subroutine sigmoid_apply_attributes
!###############################################################################


!###############################################################################
  pure function sigmoid_export_attributes(this) result(attributes)
    !! Export sigmoid activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(sigmoid_actv_type), intent(in) :: this
    !! Sigmoid activation type
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    integer :: n_attributes
    !! Number of attributes
    character(50) :: buffer
    !! Temporary string buffer

    n_attributes = 1
    allocate(attributes(n_attributes))

    write(buffer, '(F10.6)') this%scale
    attributes(1) = onnx_attribute_type( &
         "scale", "float", trim(adjustl(buffer)) )

  end function sigmoid_export_attributes
!###############################################################################


!###############################################################################
  function sigmoid_activate(this, val) result(output)
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
  end function sigmoid_activate
!###############################################################################

end module athena__activation_sigmoid
