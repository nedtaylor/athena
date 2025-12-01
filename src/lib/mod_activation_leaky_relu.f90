module athena__activation_leaky_relu
  !! Module containing implementation of the leaky ReLU activation function
  !!
  !! This module implements the Leaky Rectified Linear Unit function:
  !! f(x) = x if x > 0, 0.01x otherwise
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*), max
  use athena__misc_types, only: base_actv_type
  use athena__activation_relu, only: relu_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: leaky_relu_actv_type, create_from_onnx_leaky_relu_activation


  type, extends(relu_actv_type) :: leaky_relu_actv_type
     real(real32) :: alpha
   contains
     procedure, pass(this) :: apply => leaky_relu_activate
     procedure, pass(this) :: reset => leaky_relu_reset
     procedure, pass(this) :: apply_attributes => apply_attributes_leaky_relu
     procedure, pass(this) :: export_attributes => export_attributes_leaky_relu
  end type leaky_relu_actv_type

  interface leaky_relu_actv_type
     procedure initialise
  end interface leaky_relu_actv_type



contains

!###############################################################################
  function initialise(scale, attributes) result(activation)
    !! Initialise a leaky ReLU activation function
    implicit none

    ! Arguments
    real(real32), intent(in), optional :: scale
    !! Optional scale factor for activation output
    type(onnx_attribute_type), dimension(:), intent(in), optional :: attributes
    !! Optional array of ONNX attributes
    type(leaky_relu_actv_type) :: activation
    !! Leaky ReLU activation type


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
  pure subroutine leaky_relu_reset(this)
    !! Reset leaky ReLU activation function attributes and variables
    implicit none

    ! Arguments
    class(leaky_relu_actv_type), intent(inout) :: this
    !! Leaky ReLU activation type

    this%name = "leaky_relu"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%alpha = 0.01_real32
    this%apply_scaling = .false.

  end subroutine leaky_relu_reset
!-------------------------------------------------------------------------------
  function create_from_onnx_leaky_relu_activation(attributes) result(activation)
    !! Create leaky ReLU activation function from ONNX attributes
    implicit none

    ! Arguments
    type(onnx_attribute_type), dimension(:), intent(in) :: attributes
    !! Array of ONNX attributes

    class(base_actv_type), allocatable :: activation
    !! Instance of activation type

    allocate(activation, source = leaky_relu_actv_type(attributes = attributes))

  end function create_from_onnx_leaky_relu_activation
!###############################################################################


!###############################################################################
  subroutine apply_attributes_leaky_relu(this, attributes)
    !! Load ONNX attributes into leaky ReLU activation function
    implicit none

    ! Arguments
    class(leaky_relu_actv_type), intent(inout) :: this
    !! Leaky ReLU activation type
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
       case("alpha")
          read(attributes(i)%val,*) this%alpha
       case("name")
          if(trim(attributes(i)%val) .ne. trim(this%name)) then
             call print_warning( &
                  'Leaky ReLU activation: name attribute "' // &
                  trim(attributes(i)%val) // &
                  '"" does not match expected "' // trim(this%name)//'"' &
             )

          end if
       case default
          call print_warning( &
               'Leaky ReLU activation: unknown attribute '//trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine apply_attributes_leaky_relu
!###############################################################################


!###############################################################################
  pure function export_attributes_leaky_relu(this) result(attributes)
    !! Export leaky ReLU activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(leaky_relu_actv_type), intent(in) :: this
    !! Leaky ReLU activation type
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

    write(buffer, '(F10.6)') this%alpha
    attributes(3) = onnx_attribute_type( &
         "alpha", "float", trim(adjustl(buffer)) )

  end function export_attributes_leaky_relu
!###############################################################################


!###############################################################################
  function leaky_relu_activate(this, val) result(output)
    !! Apply leaky ReLU activation to 1D array
    !!
    !! Computes: f = max(0.01x, x)
    implicit none

    ! Arguments
    class(leaky_relu_actv_type), intent(in) :: this
    !! Leaky ReLU activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values

    ! allocate(output)
    if(this%apply_scaling)then
       output => max(val * this%alpha, val) * this%scale
    else
       output => max(val * this%alpha, val)
    end if
  end function leaky_relu_activate
!###############################################################################

end module athena__activation_leaky_relu
