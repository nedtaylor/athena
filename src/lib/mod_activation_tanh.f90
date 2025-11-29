module athena__activation_tanh
  !! Module containing implementation of the tanh activation function
  !!
  !! This module implements the hyperbolic tangent activation function
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*), tanh
  use athena__misc_types, only: base_actv_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: tanh_actv_type


  type, extends(base_actv_type) :: tanh_actv_type
     !! Type for tanh activation function with overloaded procedures
   contains
     procedure, pass(this) :: activate => tanh_activate
     procedure, pass(this) :: reset => tanh_reset
     procedure, pass(this) :: apply_attributes => tanh_apply_attributes
     procedure, pass(this) :: export_attributes => tanh_export_attributes
  end type tanh_actv_type

  interface tanh_actv_type
     procedure initialise
  end interface tanh_actv_type



contains

!###############################################################################
  function initialise(scale, attributes) result(activation)
    !! Initialise a tanh activation function
    implicit none

    ! Arguments
    real(real32), intent(in), optional :: scale
    !! Optional scale factor for activation output
    type(onnx_attribute_type), dimension(:), intent(in), optional :: attributes
    !! Optional array of ONNX attributes
    type(tanh_actv_type) :: activation
    !! tanh activation type


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
  pure subroutine tanh_reset(this)
    !! Reset tanh activation function attributes and variables
    implicit none

    ! Arguments
    class(tanh_actv_type), intent(inout) :: this
    !! Tanh activation type

    this%name = "tanh"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.

  end subroutine tanh_reset
!###############################################################################


!###############################################################################
  subroutine tanh_apply_attributes(this, attributes)
    !! Load ONNX attributes into tanh activation function
    implicit none

    ! Arguments
    class(tanh_actv_type), intent(inout) :: this
    !! Tanh activation type
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
               'Tanh activation: unknown attribute '//trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine tanh_apply_attributes
!###############################################################################


!###############################################################################
  pure function tanh_export_attributes(this) result(attributes)
    !! Export tanh activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(tanh_actv_type), intent(in) :: this
    !! Tanh activation type
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

  end function tanh_export_attributes
!###############################################################################


!###############################################################################
  function tanh_activate(this, val) result(output)
    !! Apply tanh activation to 1D array
    !!
    !! Applies the hyperbolic tangent function element-wise to input array:
    !! f = (exp(x) - exp(-x))/(exp(x) + exp(-x))
    implicit none

    ! Arguments
    class(tanh_actv_type), intent(in) :: this
    !! Tanh activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values

    if(this%apply_scaling)then
       output => tanh(val) * this%scale
    else
       output => tanh(val)
    end if
  end function tanh_activate
!###############################################################################

end module athena__activation_tanh
