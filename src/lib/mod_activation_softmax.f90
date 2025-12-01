module athena__activation_softmax
  !! Module containing implementation of the softmax activation function
  !!
  !! This module implements softmax for converting logits into probability
  !! distributions. Commonly used for multi-class classification.
  !!
  !! Mathematical operation:
  !! \[ \text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} \]
  !!
  !! Properties:
  !!   - Outputs sum to 1: \(\sum_{i=1}^{n} \text{softmax}(\mathbf{x})_i = 1\)
  !!   - All outputs in range \((0, 1)\)
  !!   - Preserves ordering: \(x_i > x_j \Rightarrow f(x_i) > f(x_j)\)
  !!   - Translation invariant: \(\text{softmax}(\mathbf{x}+c) = \text{softmax}(\mathbf{x})\)
  !!
  !! Derivative (Jacobian):
  !! \[ \frac{\partial f_i}{\partial x_j} = f_i(\delta_{ij} - f_j) \]
  !! where \(\delta_{ij}\) is the Kronecker delta
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*)
  use athena__diffstruc_extd, only: softmax
  use athena__misc_types, only: base_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: softmax_actv_type, create_from_onnx_softmax_activation


  type, extends(base_actv_type) :: softmax_actv_type
     !! Type for softmax activation function with overloaded procedures
   contains
     procedure, pass(this) :: apply => apply_softmax
     procedure, pass(this) :: reset => reset_softmax
     procedure, pass(this) :: apply_attributes => apply_attributes_softmax
     procedure, pass(this) :: export_attributes => export_attributes_softmax
  end type softmax_actv_type

  interface softmax_actv_type
     procedure initialise
  end interface softmax_actv_type



contains

!###############################################################################
  function initialise(scale, attributes) result(activation)
    !! Initialise a softmax activation function
    implicit none

    ! Arguments
    real(real32), intent(in), optional :: scale
    !! Optional scale factor for activation output
    type(onnx_attribute_type), dimension(:), intent(in), optional :: attributes
    !! Optional array of ONNX attributes
    type(softmax_actv_type) :: activation
    !! Softmax activation type

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
  pure subroutine reset_softmax(this)
    !! Reset softmax activation function attributes and variables
    implicit none

    ! Arguments
    class(softmax_actv_type), intent(inout) :: this
    !! Softmax activation type

    this%name = "softmax"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine reset_softmax
!-------------------------------------------------------------------------------
  function create_from_onnx_softmax_activation(attributes) result(activation)
    !! Create softmax activation function from ONNX attributes
    implicit none

    ! Arguments
    type(onnx_attribute_type), dimension(:), intent(in) :: attributes
    !! Array of ONNX attributes

    class(base_actv_type), allocatable :: activation
    !! Instance of activation type

    allocate(activation, source = softmax_actv_type(attributes = attributes))

  end function create_from_onnx_softmax_activation
!###############################################################################


!###############################################################################
  subroutine apply_attributes_softmax(this, attributes)
    !! Load ONNX attributes into softmax activation function
    implicit none

    ! Arguments
    class(softmax_actv_type), intent(inout) :: this
    !! Softmax activation type
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
                  'Softmax activation: name attribute "' // &
                  trim(attributes(i)%val) // &
                  '"" does not match expected "' // trim(this%name)//'"' &
             )

          end if
       case default
          call print_warning( &
               'Softmax activation: unknown attribute '// &
               trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine apply_attributes_softmax
!###############################################################################


!###############################################################################
  pure function export_attributes_softmax(this) result(attributes)
    !! Export softmax activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(softmax_actv_type), intent(in) :: this
    !! Softmax activation type
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

  end function export_attributes_softmax
!###############################################################################


!###############################################################################
  function apply_softmax(this, val) result(output)
    !! Apply softmax activation to 1D array
    !!
    !! Computes: f = exp(x-max)/sum(exp(x-max))
    implicit none

    ! Arguments
    class(softmax_actv_type), intent(in) :: this
    !! Softmax activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Normalised probability distribution output

    !! compute softmax values
    if(this%apply_scaling)then
       output => softmax(val, dim=2) * this%scale
    else
       output => softmax(val, dim=2)
    end if
  end function apply_softmax
!###############################################################################

end module athena__activation_softmax
