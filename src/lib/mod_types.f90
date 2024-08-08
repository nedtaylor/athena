!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains custom derived types for the ATHENA library
!!! module contains the following derived types:
!!! - activation_type  - abstract type for activation functions
!!! - initialiser_type - abstract type for initialising weights and biases
!!!##################
!!! the activation_type contains the following deferred procedures:
!!! - activate_<N>d      - activation function for rank <N> input
!!! - differentiate_<N>d - derivative of activation function for rank <N> input
!!!##################
!!! the initialiser_type contains the following deferred procedures:
!!! - initialise - initialises weights and biases
!!!#############################################################################
module custom_types
  use constants, only: real32
  implicit none


  private

  public :: activation_type
  public :: initialiser_type
  public :: array_type
  public :: array1d_type, array2d_type, array3d_type, array4d_type, array5d_type


!!!-----------------------------------------------------------------------------
!!! activation (transfer) function base type
!!!-----------------------------------------------------------------------------
  type, abstract :: activation_type
     !! memory leak as allocatable character goes out of bounds
     !! change to defined length
     !character(:), allocatable :: name
     character(10) :: name
     real(real32) :: scale
     real(real32) :: threshold
   contains
     procedure (activation_function_1d), deferred, pass(this) :: activate_1d
     procedure (derivative_function_1d), deferred, pass(this) :: differentiate_1d
     procedure (activation_function_2d), deferred, pass(this) :: activate_2d
     procedure (derivative_function_2d), deferred, pass(this) :: differentiate_2d
     procedure (activation_function_3d), deferred, pass(this) :: activate_3d
     procedure (derivative_function_3d), deferred, pass(this) :: differentiate_3d
     procedure (activation_function_4d), deferred, pass(this) :: activate_4d
     procedure (derivative_function_4d), deferred, pass(this) :: differentiate_4d
     procedure (activation_function_5d), deferred, pass(this) :: activate_5d
     procedure (derivative_function_5d), deferred, pass(this) :: differentiate_5d
     generic :: activate => activate_1d, activate_2d, &
          activate_3d , activate_4d, activate_5d
     generic :: differentiate => differentiate_1d, differentiate_2d, &
          differentiate_3d, differentiate_4d, differentiate_5d
  end type activation_type
  

  !! interface for activation function
  !!----------------------------------------------------------------------------
  abstract interface
     pure function activation_function_1d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:), intent(in) :: val
       real(real32), dimension(size(val,1)) :: output
     end function activation_function_1d
     
     pure function activation_function_2d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2)) :: output
     end function activation_function_2d

     pure function activation_function_3d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function activation_function_3d

     pure function activation_function_4d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:,:), intent(in) :: val
       real(real32), dimension(&
            size(val,1),size(val,2),size(val,3),size(val,4)) :: output
     end function activation_function_4d

     pure function activation_function_5d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:,:,:), intent(in) :: val
       real(real32), dimension(&
            size(val,1),size(val,2),size(val,3), &
            size(val,4),size(val,5)) :: output
     end function activation_function_5d
  end interface


  !! interface for derivative function
  !!----------------------------------------------------------------------------
  abstract interface
     pure function derivative_function_1d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:), intent(in) :: val
       real(real32), dimension(size(val,1)) :: output
     end function derivative_function_1d

     pure function derivative_function_2d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2)) :: output
     end function derivative_function_2d

     pure function derivative_function_3d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function derivative_function_3d

     pure function derivative_function_4d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:,:), intent(in) :: val
       real(real32), dimension(&
            size(val,1),size(val,2),size(val,3),size(val,4)) :: output
     end function derivative_function_4d

     pure function derivative_function_5d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:,:,:), intent(in) :: val
       real(real32), dimension(&
            size(val,1),size(val,2),size(val,3), &
            size(val,4),size(val,5)) :: output
     end function derivative_function_5d
  end interface


!!!-----------------------------------------------------------------------------
!!! weights and biases initialiser base type
!!!-----------------------------------------------------------------------------
  type, abstract :: initialiser_type
     real(real32) :: scale = 1._real32, mean = 1._real32, std = 0.01_real32
   contains
     procedure (initialiser_subroutine), deferred, pass(this) :: initialise
  end type initialiser_type


  !! interface for initialiser function
  !!----------------------------------------------------------------------------
  abstract interface
     subroutine initialiser_subroutine(this, input, fan_in, fan_out)
       import initialiser_type, real32
       class(initialiser_type), intent(inout) :: this
       real(real32), dimension(..), intent(out) :: input
       integer, optional, intent(in) :: fan_in, fan_out
       real(real32) :: scale
     end subroutine initialiser_subroutine
  end interface


!!!-----------------------------------------------------------------------------
!!! base and extended array types
!!!-----------------------------------------------------------------------------
  type, abstract :: array_type
     integer :: rank
     integer, dimension(:), allocatable :: shape
     integer :: size
     logical :: allocated = .false.
   contains
     procedure (allocate_array), deferred, pass(this) :: allocate
     procedure (deallocate_array), deferred, pass(this) :: deallocate
     procedure (flatten_array), deferred, pass(this) :: flatten
     procedure (get_array), deferred, pass(this) :: get
     procedure (set_array), deferred, pass(this) :: set
  end type array_type


  !! interface for allocate, deallocate, and flattening array
  !!----------------------------------------------------------------------------
  abstract interface
     module subroutine allocate_array(this, array_shape, source)
       class(array_type), intent(inout) :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
     end subroutine allocate_array

     pure module subroutine deallocate_array(this)
       class(array_type), intent(inout) :: this
     end subroutine deallocate_array

    !  pure module function reshape_array(this, shape) result(output)
    !    class(array_type), intent(inout) :: this
    !    integer, dimension(:), intent(in) :: shape
    !    real(real32), dimension(size(shape)) :: output
    !  end function reshape_array

     pure module function flatten_array(this) result(output)
        class(array_type), intent(in) :: this
        real(real32), dimension(this%size) :: output
     end function flatten_array

     pure module subroutine get_array(this, output)
        class(array_type), intent(in) :: this
        real(real32), dimension(..), intent(out) :: output
      end subroutine get_array

      pure module subroutine set_array(this, input)
         class(array_type), intent(inout) :: this
         real(real32), dimension(..), intent(in) :: input
       end subroutine set_array 
  end interface

  !! extend the array type to 1d, 2d, 3d, 4d, and 5d arrays
  !!----------------------------------------------------------------------------
  type, extends(array_type) :: array1d_type
     real(real32), dimension(:), allocatable :: val
   contains
     procedure :: allocate => allocate_array1d
     procedure :: deallocate => deallocate_array1d
     procedure :: flatten => flatten_array1d
     procedure :: get => get_array1d
     procedure :: set => set_array1d
     end type array1d_type

  type, extends(array_type) :: array2d_type
     real(real32), dimension(:,:), allocatable :: val
   contains
     procedure :: allocate => allocate_array2d
     procedure :: deallocate => deallocate_array2d
     procedure :: flatten => flatten_array2d
     procedure :: get => get_array2d
     procedure :: set => set_array2d
  end type array2d_type

  type, extends(array_type) :: array3d_type
     real(real32), dimension(:,:,:), allocatable :: val
   contains
     procedure :: allocate => allocate_array3d
     procedure :: deallocate => deallocate_array3d
     procedure :: flatten => flatten_array3d
     procedure :: get => get_array3d
     procedure :: set => set_array3d
     end type array3d_type

  type, extends(array_type) :: array4d_type
     real(real32), dimension(:,:,:,:), allocatable :: val
   contains
     procedure :: allocate => allocate_array4d
     procedure :: deallocate => deallocate_array4d
     procedure :: flatten => flatten_array4d
     procedure :: get => get_array4d
     procedure :: set => set_array4d
     end type array4d_type

  type, extends(array_type) :: array5d_type
     real(real32), dimension(:,:,:,:,:), allocatable :: val
   contains
     procedure :: allocate => allocate_array5d
     procedure :: deallocate => deallocate_array5d
     procedure :: flatten => flatten_array5d
     procedure :: get => get_array5d
     procedure :: set => set_array5d
     end type array5d_type

  !! interface for allocating array
  !!----------------------------------------------------------------------------
  interface
    module subroutine allocate_array1d(this, array_shape, source)
      class(array1d_type), intent(inout) :: this
      integer, dimension(:), intent(in), optional :: array_shape
      class(*), dimension(..), intent(in), optional :: source
    end subroutine allocate_array1d

    module subroutine allocate_array2d(this, array_shape, source)
      class(array2d_type), intent(inout) :: this
      integer, dimension(:), intent(in), optional :: array_shape
      class(*), dimension(..), intent(in), optional :: source
    end subroutine allocate_array2d

    module subroutine allocate_array3d(this, array_shape, source)
      class(array3d_type), intent(inout) :: this
      integer, dimension(:), intent(in), optional :: array_shape
      class(*), dimension(..), intent(in), optional :: source
    end subroutine allocate_array3d

    module subroutine allocate_array4d(this, array_shape, source)
      class(array4d_type), intent(inout) :: this
      integer, dimension(:), intent(in), optional :: array_shape
      class(*), dimension(..), intent(in), optional :: source
    end subroutine allocate_array4d

    module subroutine allocate_array5d(this, array_shape, source)
      class(array5d_type), intent(inout) :: this
      integer, dimension(:), intent(in), optional :: array_shape
      class(*), dimension(..), intent(in), optional :: source
    end subroutine allocate_array5d
  end interface

  !! interface for deallocating array
  !!----------------------------------------------------------------------------
  interface
    pure module subroutine deallocate_array1d(this)
      class(array1d_type), intent(inout) :: this
    end subroutine deallocate_array1d

    pure module subroutine deallocate_array2d(this)
      class(array2d_type), intent(inout) :: this
    end subroutine deallocate_array2d

    pure module subroutine deallocate_array3d(this)
      class(array3d_type), intent(inout) :: this
    end subroutine deallocate_array3d

    pure module subroutine deallocate_array4d(this)
      class(array4d_type), intent(inout) :: this
    end subroutine deallocate_array4d

    pure module subroutine deallocate_array5d(this)
      class(array5d_type), intent(inout) :: this
    end subroutine deallocate_array5d
  end interface

  !! interface for flattening array
  !!----------------------------------------------------------------------------
  interface 
    pure module function flatten_array1d(this) result(output)
      class(array1d_type), intent(in) :: this
      real(real32), dimension(this%size) :: output
    end function flatten_array1d

    pure module function flatten_array2d(this) result(output)
      class(array2d_type), intent(in) :: this
      real(real32), dimension(this%size) :: output
    end function flatten_array2d

    pure module function flatten_array3d(this) result(output)
      class(array3d_type), intent(in) :: this
      real(real32), dimension(this%size) :: output
    end function flatten_array3d

    pure module function flatten_array4d(this) result(output)
      class(array4d_type), intent(in) :: this
      real(real32), dimension(this%size) :: output
    end function flatten_array4d

    pure module function flatten_array5d(this) result(output)
      class(array5d_type), intent(in) :: this
      real(real32), dimension(this%size) :: output
    end function flatten_array5d
  end interface

  !! interface for getting array
  !!----------------------------------------------------------------------------
  interface
    pure module subroutine get_array1d(this, output)
      class(array1d_type), intent(in) :: this
      real(real32), dimension(..), intent(out) :: output
    end subroutine get_array1d

    pure module subroutine get_array2d(this, output)
      class(array2d_type), intent(in) :: this
      real(real32), dimension(..), intent(out) :: output
    end subroutine get_array2d

    pure module subroutine get_array3d(this, output)
      class(array3d_type), intent(in) :: this
      real(real32), dimension(..), intent(out) :: output
    end subroutine get_array3d

    pure module subroutine get_array4d(this, output)
      class(array4d_type), intent(in) :: this
      real(real32), dimension(..), intent(out) :: output
    end subroutine get_array4d

    pure module subroutine get_array5d(this, output)
      class(array5d_type), intent(in) :: this
      real(real32), dimension(..), intent(out) :: output
    end subroutine get_array5d
  end interface

  !! interface for setting array
  !!----------------------------------------------------------------------------
  interface
    pure module subroutine set_array1d(this, input)
      class(array1d_type), intent(inout) :: this
      real(real32), dimension(..), intent(in) :: input
    end subroutine set_array1d

    pure module subroutine set_array2d(this, input)
      class(array2d_type), intent(inout) :: this
      real(real32), dimension(..), intent(in) :: input
    end subroutine set_array2d

    pure module subroutine set_array3d(this, input)
      class(array3d_type), intent(inout) :: this
      real(real32), dimension(..), intent(in) :: input
    end subroutine set_array3d

    pure module subroutine set_array4d(this, input)
      class(array4d_type), intent(inout) :: this
      real(real32), dimension(..), intent(in) :: input
    end subroutine set_array4d

    pure module subroutine set_array5d(this, input)
      class(array5d_type), intent(inout) :: this
      real(real32), dimension(..), intent(in) :: input
    end subroutine set_array5d
  end interface

  !! interface for initialising array
  !!----------------------------------------------------------------------------
  interface array1d_type
    pure module function init_array1d(array_shape) result(output)
      integer, dimension(:), intent(in), optional :: array_shape
      type(array1d_type) :: output
    end function init_array1d
  end interface array1d_type

  interface array2d_type
    pure module function init_array2d(array_shape) result(output)
      integer, dimension(:), intent(in), optional :: array_shape
      type(array2d_type) :: output
    end function init_array2d
  end interface array2d_type

  interface array3d_type
    pure module function init_array3d(array_shape) result(output)
      integer, dimension(:), intent(in), optional :: array_shape
      type(array3d_type) :: output
    end function init_array3d
  end interface array3d_type

  interface array4d_type
    pure module function init_array4d(array_shape) result(output)
      integer, dimension(:), intent(in), optional :: array_shape
      type(array4d_type) :: output
    end function init_array4d
  end interface array4d_type

  interface array5d_type
    pure module function init_array5d(array_shape) result(output)
      integer, dimension(:), intent(in), optional :: array_shape
      type(array5d_type) :: output
    end function init_array5d
  end interface array5d_type
  
end module custom_types
!!!#############################################################################
