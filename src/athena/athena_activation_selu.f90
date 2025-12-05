module athena__activation_selu
  !! Module containing implementation of the SELU activation function
  !!
  !! This module implements Scaled Exponential Linear Unit (SELU), which has
  !! self-normalizing properties for deep networks.
  !!
  !! Mathematical operation:
  !! \[ f(x) = \lambda \begin{cases} x & \text{if } x > 0 \\\\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases} \]
  !!
  !! where \(\lambda \approx 1.0507\) and \(\alpha \approx 1.6733\)
  !! preserve mean=0, variance=1
  !!
  !! Derivative:
  !! \[ f'(x) = \lambda \begin{cases} 1 & \text{if } x > 0 \\\\ \alpha e^x & \text{if } x \leq 0 \end{cases} \]
  !!
  !! Properties: Self-normalizing, enables very deep networks
  !! Requires: Lecun Normal initialisation, alpha dropout
  !! Reference: Klambauer et al. (2017), NeurIPS
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*), operator(-), operator(.gt.), &
       merge, exp
  use athena__misc_types, only: base_actv_type
  use athena__activation_relu, only: relu_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: selu_actv_type, create_from_onnx_selu_activation


  type, extends(relu_actv_type) :: selu_actv_type
     !! Type for SELU activation function with overloaded procedures
     real(real32) :: alpha = 1.6732632423543772848170429916717_real32
     !! Alpha parameter for SELU
     real(real32) :: lambda = 1.0507009873554804934193349852946_real32
     !! Lambda parameter for SELU
   contains
     procedure, pass(this) :: apply => apply_selu
     procedure, pass(this) :: reset => reset_selu
     procedure, pass(this) :: apply_attributes => apply_attributes_selu
     procedure, pass(this) :: export_attributes => export_attributes_selu
  end type selu_actv_type

  interface selu_actv_type
     procedure initialise
  end interface selu_actv_type



contains

!###############################################################################
  function initialise(scale, alpha, lambda, attributes) result(activation)
    !! Initialise a SELU activation function
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output
    real(real32), optional, intent(in) :: alpha
    !! Optional alpha parameter (default: 1.67326)
    real(real32), optional, intent(in) :: lambda
    !! Optional lambda parameter (default: 1.0507)
    type(selu_actv_type) :: activation
    !! SELU activation type
    type(onnx_attribute_type), optional, intent(in) :: attributes(:)
    !! Optional ONNX attributes


    call activation%reset()

    if(present(scale)) activation%scale = scale
    if(abs(activation%scale-1._real32) .gt. 1.e-6_real32)then
       activation%apply_scaling = .true.
    end if
    if(present(alpha)) activation%alpha = alpha
    if(present(lambda)) activation%lambda = lambda
    if(present(attributes))then
       call activation%apply_attributes(attributes)
    end if

  end function initialise
!-------------------------------------------------------------------------------
  pure subroutine reset_selu(this)
    !! Reset SELU activation function attributes and variables
    implicit none

    ! Arguments
    class(selu_actv_type), intent(inout) :: this
    !! SELU activation type

    this%name = "selu"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.
    this%alpha = 1.67326_real32
    this%lambda = 1.0507_real32

  end subroutine reset_selu
!-------------------------------------------------------------------------------
  function create_from_onnx_selu_activation(attributes) result(activation)
    !! Create SELU activation function from ONNX attributes
    implicit none

    ! Arguments
    type(onnx_attribute_type), dimension(:), intent(in) :: attributes
    !! Array of ONNX attributes

    class(base_actv_type), allocatable :: activation
    !! Instance of activation type

    allocate(activation, source = selu_actv_type(attributes = attributes))

  end function create_from_onnx_selu_activation
!###############################################################################


!###############################################################################
  subroutine apply_attributes_selu(this, attributes)
    !! Load ONNX attributes into SELU activation function
    implicit none

    ! Arguments
    class(selu_actv_type), intent(inout) :: this
    !! SELU activation type
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
       case("lambda")
          read(attributes(i)%val,*) this%lambda
       case("name")
          if(trim(attributes(i)%val) .ne. trim(this%name)) then
             call print_warning( &
                  'SELU activation: name attribute "' // &
                  trim(attributes(i)%val) // &
                  '"" does not match expected "' // trim(this%name)//'"' &
             )

          end if
       case default
          call print_warning( &
               'SELU activation: unknown attribute '// &
               trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine apply_attributes_selu
!###############################################################################


!###############################################################################
  pure function export_attributes_selu(this) result(attributes)
    !! Export SELU activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(selu_actv_type), intent(in) :: this
    !! SELU activation type
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    character(50) :: buffer
    !! Temporary string buffer

    allocate(attributes(4))

    write(buffer, '(A)') this%name
    attributes(1) = onnx_attribute_type( &
         "name", "string", trim(adjustl(buffer)) )

    write(buffer, '(F10.6)') this%scale
    attributes(2) = onnx_attribute_type( &
         "scale", "float", trim(adjustl(buffer)) )

    write(buffer, '(F10.6)') this%alpha
    attributes(3) = onnx_attribute_type( &
         "alpha", "float", trim(adjustl(buffer)) )

    write(buffer, '(F10.6)') this%lambda
    attributes(4) = onnx_attribute_type( &
         "lambda", "float", trim(adjustl(buffer)) )

  end function export_attributes_selu
!###############################################################################


!###############################################################################
  function apply_selu(this, val) result(output)
    !! Apply SELU activation to array
    !!
    !! Computes: f(x) = λ * x if x > 0
    !!           f(x) = λ * α * (exp(x) - 1) if x ≤ 0
    implicit none

    ! Arguments
    class(selu_actv_type), intent(in) :: this
    !! SELU activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values

    ! Local variables
    type(array_type), pointer :: positive_part, negative_part

    ! Compute SELU: λ * merge(x, α * (exp(x) - 1), x > 0)
    positive_part => val * this%lambda
    negative_part => (exp(val) - 1._real32) * this%alpha * this%lambda
    output => merge(positive_part, negative_part, val .gt. 0._real32)

    if(this%apply_scaling)then
       output => output * this%scale
    end if

  end function apply_selu
!###############################################################################

end module athena__activation_selu
