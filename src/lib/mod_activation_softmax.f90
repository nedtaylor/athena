module athena__activation_softmax
  !! Module containing implementation of the softmax activation function
  !!
  !! This module implements the softmax activation function for normalising
  !! outputs into probability distributions
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*)
  use athena__diffstruc_extd, only: softmax
  use athena__misc_types, only: base_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: softmax_actv_type


  type, extends(base_actv_type) :: softmax_actv_type
     !! Type for softmax activation function with overloaded procedures
   contains
     procedure, pass(this) :: activate => softmax_activate
     procedure, pass(this) :: reset => softmax_reset
     procedure, pass(this) :: apply_attributes => softmax_apply_attributes
     procedure, pass(this) :: export_attributes => softmax_export_attributes
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
  pure subroutine softmax_reset(this)
    !! Reset softmax activation function attributes and variables
    implicit none

    ! Arguments
    class(softmax_actv_type), intent(inout) :: this
    !! Softmax activation type

    this%name = "softmax"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine softmax_reset
!###############################################################################


!###############################################################################
  subroutine softmax_apply_attributes(this, attributes)
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
       case default
          call print_warning( &
               'Softmax activation: unknown attribute '// &
               trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine softmax_apply_attributes
!###############################################################################


!###############################################################################
  pure function softmax_export_attributes(this) result(attributes)
    !! Export softmax activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(softmax_actv_type), intent(in) :: this
    !! Softmax activation type
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

  end function softmax_export_attributes
!###############################################################################


!###############################################################################
  function softmax_activate(this, val) result(output)
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
  end function softmax_activate
!###############################################################################

end module athena__activation_softmax
