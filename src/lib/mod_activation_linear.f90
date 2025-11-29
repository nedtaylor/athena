module athena__activation_linear
  !! Module containing implementation of the linear activation function
  !!
  !! This module implements a scaled linear function f(x) = scale * x
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*)
  use athena__misc_types, only: base_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: linear_actv_type


  type, extends(base_actv_type) :: linear_actv_type
   contains
     procedure, pass(this) :: activate => linear_activate
     procedure, pass(this) :: reset => linear_reset
     procedure, pass(this) :: apply_attributes => linear_apply_attributes
     procedure, pass(this) :: export_attributes => linear_export_attributes
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
  pure subroutine linear_reset(this)
    !! Reset linear activation function attributes and variables
    implicit none

    ! Arguments
    class(linear_actv_type), intent(inout) :: this
    !! Linear activation type

    this%name = "linear"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine linear_reset
!###############################################################################


!###############################################################################
  subroutine linear_apply_attributes(this, attributes)
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
       case default
          call print_warning( &
               'Linear activation: unknown attribute '// &
               trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine linear_apply_attributes
!###############################################################################


!###############################################################################
  pure function linear_export_attributes(this) result(attributes)
    !! Export linear activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(linear_actv_type), intent(in) :: this
    !! Linear activation type
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

  end function linear_export_attributes
!###############################################################################


!###############################################################################
  function linear_activate(this, val) result(output)
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
  end function linear_activate
!###############################################################################

end module athena__activation_linear
