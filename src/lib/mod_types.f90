!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module custom_types
  use constants, only: real12
  implicit none


!!!-----------------------------------------------------------------------------
!!! activation (transfer) function base type
!!!-----------------------------------------------------------------------------
  type, abstract :: activation_type
     !! memory leak as allocatable character goes out of bounds
     !! change to defined length
     !character(:), allocatable :: name
     character(10) :: name
     real(real12) :: scale
     real(real12) :: threshold
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
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:), intent(in) :: val
       real(real12), dimension(size(val,1)) :: output
     end function activation_function_1d
     
     pure function activation_function_2d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:), intent(in) :: val
       real(real12), dimension(size(val,1),size(val,2)) :: output
     end function activation_function_2d

     pure function activation_function_3d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:), intent(in) :: val
       real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function activation_function_3d

     pure function activation_function_4d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:,:), intent(in) :: val
       real(real12), dimension(&
            size(val,1),size(val,2),size(val,3),size(val,4)) :: output
     end function activation_function_4d

     pure function activation_function_5d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:,:,:), intent(in) :: val
       real(real12), dimension(&
            size(val,1),size(val,2),size(val,3), &
            size(val,4),size(val,5)) :: output
     end function activation_function_5d
  end interface


  !! interface for derivative function
  !!----------------------------------------------------------------------------
  abstract interface
     pure function derivative_function_1d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:), intent(in) :: val
       real(real12), dimension(size(val,1)) :: output
     end function derivative_function_1d

     pure function derivative_function_2d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:), intent(in) :: val
       real(real12), dimension(size(val,1),size(val,2)) :: output
     end function derivative_function_2d

     pure function derivative_function_3d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:), intent(in) :: val
       real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function derivative_function_3d

     pure function derivative_function_4d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:,:), intent(in) :: val
       real(real12), dimension(&
            size(val,1),size(val,2),size(val,3),size(val,4)) :: output
     end function derivative_function_4d

     pure function derivative_function_5d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:,:,:), intent(in) :: val
       real(real12), dimension(&
            size(val,1),size(val,2),size(val,3), &
            size(val,4),size(val,5)) :: output
     end function derivative_function_5d
  end interface


!!!-----------------------------------------------------------------------------
!!! weights and biases initialiser base type
!!!-----------------------------------------------------------------------------
  type, abstract :: initialiser_type
     real(real12) :: scale = 1._real12, mean = 1._real12, std = 0.01_real12
   contains
     procedure (initialiser_subroutine), deferred, pass(this) :: initialise
  end type initialiser_type


  !! interface for initialiser function
  !!----------------------------------------------------------------------------
  abstract interface
     subroutine initialiser_subroutine(this, input, fan_in, fan_out)
       import initialiser_type, real12
       class(initialiser_type), intent(inout) :: this
       real(real12), dimension(..), intent(out) :: input
       integer, optional, intent(in) :: fan_in, fan_out
       real(real12) :: scale
     end subroutine initialiser_subroutine
  end interface



  private

  public :: activation_type
  public :: initialiser_type


end module custom_types
!!!#############################################################################
