module athena__activation_swish
  !! Module containing implementation of the swish activation function
  !!
  !! This module implements the swish activation function: f(x) = x * sigmoid(β*x)
  !! where β is a learnable parameter (default β=1)
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*)
  use athena__misc_types, only: base_actv_type
  use athena__misc_types, only: onnx_attribute_type
  use athena__diffstruc_extd, only: swish
  implicit none

  private

  public :: swish_actv_type

  type, extends(base_actv_type) :: swish_actv_type
     !! Type for swish activation function with overloaded procedures
     real(real32) :: beta = 1._real32
     !! Beta parameter for swish function
   contains
     procedure, pass(this) :: activate => swish_activate
     procedure, pass(this) :: reset => swish_reset
     procedure, pass(this) :: apply_attributes => swish_apply_attributes
     procedure, pass(this) :: export_attributes => swish_export_attributes
  end type swish_actv_type

  interface swish_actv_type
     !! Interface for setting up swish activation function
     procedure initialise
  end interface swish_actv_type

contains

!###############################################################################
  function initialise(scale, beta, attributes) result(activation)
    !! Initialise a swish activation function
    implicit none

    ! Arguments
    real(real32), intent(in), optional :: scale
    !! Optional scale factor for activation output
    real(real32), intent(in), optional :: beta
    !! Optional beta parameter for swish function
    type(onnx_attribute_type), dimension(:), intent(in), optional :: attributes
    !! Optional array of ONNX attributes
    type(swish_actv_type) :: activation
    !! Swish activation type


    call activation%reset()

    if(present(scale)) activation%scale = scale
    if(abs(activation%scale-1._real32) .gt. 1.e-6_real32)then
       activation%apply_scaling = .true.
    end if

    if(present(beta)) activation%beta = beta

    if(present(attributes)) then
       call activation%apply_attributes(attributes)
    end if

  end function initialise
!-------------------------------------------------------------------------------
  pure subroutine swish_reset(this)
    !! Reset swish activation function attributes and variables
    implicit none

    ! Arguments
    class(swish_actv_type), intent(inout) :: this
    !! Swish activation type

    this%name = "swish"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.
    this%beta = 1._real32

  end subroutine swish_reset
!###############################################################################


!###############################################################################
  subroutine swish_apply_attributes(this, attributes)
    !! Load ONNX attributes into swish activation function
    implicit none

    ! Arguments
    class(swish_actv_type), intent(inout) :: this
    !! Swish activation type
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
       case("beta")
          read(attributes(i)%val,*) this%beta
       case default
          call print_warning( &
               'Swish activation: unknown attribute '// &
               trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine swish_apply_attributes
!###############################################################################


!###############################################################################
  pure function swish_export_attributes(this) result(attributes)
    !! Export swish activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(swish_actv_type), intent(in) :: this
    !! Swish activation type
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

    write(buffer, '(F10.6)') this%beta
    attributes(2) = onnx_attribute_type( &
         "beta", "float", trim(adjustl(buffer)) )

  end function swish_export_attributes
!###############################################################################


!###############################################################################
  function swish_activate(this, val) result(output)
    !! Apply swish activation to 1D array
    !!
    !! Computes: f(x) = x * sigmoid(β*x) = x / (1 + exp(-β*x))
    implicit none

    ! Arguments
    class(swish_actv_type), intent(in) :: this
    !! Swish activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Swish activation output

    ! Compute sigmoid(β*x)
    ! Compute swish: x * sigmoid(β*x)
    if(this%apply_scaling)then
       output => swish(val, this%beta) * this%scale
    else
       output => swish(val, this%beta)
    end if
  end function swish_activate
!###############################################################################

end module athena__activation_swish
