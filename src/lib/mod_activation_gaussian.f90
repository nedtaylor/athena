module athena__activation_gaussian
  !! Module containing implementation of the Gaussian activation function
  !!
  !! This module implements the Gaussian (bell curve) activation function
  use coreutils, only: real32, print_warning
  use diffstruc, only: array_type, gaussian, operator(*)
  use athena__misc_types, only: activation_type
  use athena__misc_types, only: onnx_attribute_type
  implicit none


  private

  public :: gaussian_actv_type


  type, extends(activation_type) :: gaussian_actv_type
     !! Type for Gaussian activation function with overloaded procedures
     real(real32) :: sigma
     !! Standard deviation parameter for Gaussian function
     real(real32) :: mu
     !! Mean parameter for Gaussian function
   contains
     procedure, pass(this) :: activate => gaussian_activate
     procedure, pass(this) :: reset => gaussian_reset
     procedure, pass(this) :: apply_attributes => gaussian_apply_attributes
     procedure, pass(this) :: export_attributes => gaussian_export_attributes
  end type gaussian_actv_type

  interface gaussian_actv_type
     procedure initialise
  end interface gaussian_actv_type



contains

!###############################################################################
  function initialise(scale, sigma, mu, attributes) result(activation)
    !! Initialise a Gaussian activation function
    implicit none

    ! Arguments
    real(real32), intent(in), optional :: scale
    !! Optional scale factor for activation output
    real(real32), intent(in), optional :: sigma
    !! Optional standard deviation parameter
    real(real32), intent(in), optional :: mu
    !! Optional mean parameter
    type(onnx_attribute_type), dimension(:), intent(in), optional :: attributes
    !! Optional array of ONNX attributes
    type(gaussian_actv_type) :: activation
    !! Gaussian activation type


    call activation%reset()

    if(present(scale)) activation%scale = scale
    if(abs(activation%scale-1._real32) .gt. 1.e-6_real32)then
       activation%apply_scaling = .true.
    end if

    if(present(sigma)) activation%sigma = sigma
    if(present(mu)) activation%mu = mu

    if(present(attributes)) then
       call activation%apply_attributes(attributes)
    end if

  end function initialise
!-------------------------------------------------------------------------------
  pure subroutine gaussian_reset(this)
    !! Reset Gaussian activation function attributes and variables
    implicit none

    ! Arguments
    class(gaussian_actv_type), intent(inout) :: this
    !! Gaussian activation type

    this%name = "gaussian"
    this%scale = 1._real32
    this%threshold = 0._real32
    this%apply_scaling = .false.
    this%sigma = 1.5_real32
    this%mu = 0._real32

  end subroutine gaussian_reset
!###############################################################################


!###############################################################################
  subroutine gaussian_apply_attributes(this, attributes)
    !! Load ONNX attributes into Gaussian activation function
    implicit none

    ! Arguments
    class(gaussian_actv_type), intent(inout) :: this
    !! Gaussian activation type
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
       case("sigma")
          read(attributes(i)%val,*) this%sigma
       case("mu")
          read(attributes(i)%val,*) this%mu
       case default
          call print_warning( &
               'Gaussian activation: unknown attribute '// &
               trim(attributes(i)%name) &
          )
       end select
    end do

  end subroutine gaussian_apply_attributes
!###############################################################################


!###############################################################################
  pure function gaussian_export_attributes(this) result(attributes)
    !! Export Gaussian activation function attributes as ONNX attributes
    implicit none

    ! Arguments
    class(gaussian_actv_type), intent(in) :: this
    !! Gaussian activation type
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

    write(buffer, '(F10.6)') this%sigma
    attributes(2) = onnx_attribute_type( &
         "sigma", "float", trim(adjustl(buffer)) )

    write(buffer, '(F10.6)') this%mu
    attributes(3) = onnx_attribute_type( &
         "mu", "float", trim(adjustl(buffer)) )

  end function gaussian_export_attributes
!###############################################################################


!###############################################################################
  function gaussian_activate(this, val) result(output)
    !! Apply Gaussian activation to array
    !!
    !! Applies the Gaussian function element-wise to input array:
    !! f = exp(-x^2/(2σ^2))/(σ√(2π))
    implicit none

    ! Arguments
    class(gaussian_actv_type), intent(in) :: this
    !! Gaussian activation type containing sigma parameter
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Gaussian activated output values

    if(this%apply_scaling)then
       output => gaussian(val, this%mu, this%sigma) * this%scale
    else
       output => gaussian(val, this%mu, this%sigma)
    end if

  end function gaussian_activate
!###############################################################################

end module athena__activation_gaussian
