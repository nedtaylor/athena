!###############################################################################
module athena__activation_piecewise
  !! Module containing implementation of the piecewise activation function
  !! https://doi.org/10.48550/arXiv.1809.09534
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, operator(*)
  use athena__diffstruc_extd, only: piecewise
  use athena__misc_types, only: activation_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: piecewise_actv_type


  type, extends(activation_type) :: piecewise_actv_type
     !! Type for piecewise activation function with overloaded procedures
     real(real32) :: gradient, limit
   contains
     procedure, pass(this) :: activate => piecewise_activate
     procedure, pass(this) :: reset => piecewise_reset
     procedure, pass(this) :: apply_attributes => piecewise_apply_attributes
     procedure, pass(this) :: export_attributes => piecewise_export_attributes
  end type piecewise_actv_type

  interface piecewise_actv_type
     procedure initialise
  end interface piecewise_actv_type



contains

!###############################################################################
  function initialise(scale, gradient, limit, attributes) result(activation)
    !! Initialise a piecewise activation function
    implicit none

    ! Arguments
    real(real32), intent(in), optional :: scale
    !! Optional scale factor for activation output
    real(real32), intent(in), optional :: gradient
    !! Optional gradient parameter for piecewise function
    real(real32), intent(in), optional :: limit
    !! Optional limit parameter for piecewise function
    !! -limit < x < limit
    type(onnx_attribute_type), dimension(:), intent(in), optional :: attributes
    !! Optional array of ONNX attributes
    type(piecewise_actv_type) :: activation
    !! Piecewise activation type


    call activation%reset()

    if(present(scale)) activation%scale = scale
    if(abs(activation%scale-1._real32) .gt. 1.e-6_real32)then
       activation%apply_scaling = .true.
    end if

    if(present(gradient)) activation%gradient = gradient
    if(present(limit)) activation%limit = limit

      if(present(attributes)) then
         call activation%apply_attributes(attributes)
      end if

  end function initialise
!-------------------------------------------------------------------------------
  pure subroutine piecewise_reset(this)
    !! Reset piecewise activation function attributes and variables
    implicit none

    ! Arguments
    class(piecewise_actv_type), intent(inout) :: this
    !! Piecewise activation type

    this%name = "piecewise"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.
    this%gradient = 0.1_real32
    this%limit = 1._real32

  end subroutine piecewise_reset
!###############################################################################


!###############################################################################
  subroutine piecewise_apply_attributes(this, attributes)
    !! Load ONNX attributes into piecewise activation function
    implicit none

    ! Arguments
    class(piecewise_actv_type), intent(inout) :: this
    !! Piecewise activation type
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
       case("gradient")
          read(attributes(i)%val,*) this%gradient
       case("limit")
          read(attributes(i)%val,*) this%limit
       case default
          call print_warning( &
               'Piecewise activation: unknown attribute '// &
               trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine piecewise_apply_attributes
!###############################################################################


!###############################################################################
  pure function piecewise_export_attributes(this) result(attributes)
    !! Export piecewise activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(piecewise_actv_type), intent(in) :: this
    !! Piecewise activation type
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    integer :: n_attributes
    !! Number of attributes
    character(50) :: buffer
    !! Temporary string buffer

    n_attributes = 3
    allocate(attributes(n_attributes))

    write(buffer, '(F10.6)') this%scale
    attributes(1) = onnx_attribute_type( &
         "scale", "float", trim(adjustl(buffer)) )

    write(buffer, '(F10.6)') this%gradient
    attributes(2) = onnx_attribute_type( &
         "gradient", "float", trim(adjustl(buffer)) )

    write(buffer, '(F10.6)') this%limit
    attributes(3) = onnx_attribute_type( &
         "limit", "float", trim(adjustl(buffer)) )

  end function piecewise_export_attributes
!###############################################################################


!###############################################################################
  function piecewise_activate(this, val) result(output)
    !! Apply piecewise activation to 1D array
    !!
    !! Computes piecewise function:
    !! f = 0 if x ≤ min
    !! f = scale if x ≥ max
    !! f = scale * x + intercept otherwise
    implicit none

    ! Arguments
    class(piecewise_actv_type), intent(in) :: this
    !! Piecewise activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values

    if(this%apply_scaling)then
       output => piecewise(val, this%gradient, this%limit) * this%scale
    else
       output => piecewise(val, this%gradient, this%limit)
    end if
  end function piecewise_activate
!###############################################################################

end module athena__activation_piecewise
