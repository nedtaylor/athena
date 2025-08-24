program example_hessian
  !! Example demonstrating second-order derivative computation using your autodiff system
  use athena__constants, only: real32
  use athena__misc_types
  implicit none

  ! Variables
  type(array_type) :: x, y, f
  type(array_type) :: hessian
  integer :: i, j
  integer :: itmp

  write(*,*) "=== Second-Order Derivatives Example ==="

  ! Create a 2D input vector x = [x1, x2]
  call x%allocate(array_shape=[2, 1])
  call y%allocate(array_shape=[2, 1])
  x%val(1, 1) = 1.0_real32
  x%val(2, 1) = 5.0_real32
  y%val(1, 1) = 3.0_real32
  y%val(2, 1) = 4.0_real32
  call x%set_requires_grad(.true.)

  write(*,*) "Computing function: f"
  write(*,*) "At point: x =", x%val(1,1)

!   f = x ** 4._real32 + x * y
!   ! implement indexing in the other direction
  f = pack(x, [1], 1) + pack(x, [2], 1) **2 * pack(x, [1], 1)

  write(*,*) "Function value f =", f%val(1,1)

  ! Compute first derivatives (gradient)
  call x%set_direction([0._real32, 1._real32])
  call f%backward()
  write(*,*) "First derivatives (gradient):"
  if(associated(x%grad)) then
     write(*,*) "  df/dx1 =", x%grad%val(1,1)
     write(*,*) "  df/dx2 =", x%grad%val(2,1)
  end if

  ! Compute second derivatives (Hessian)
  write(*,*) "Computing Hessian matrix..."
  call x%set_direction([1._real32, 0._real32])
  itmp = 0
  hessian = x%grad%forward(x, itmp)
  write(*,*) "  df/dx1dx1 =", hessian%val(1,1)
  write(*,*) "  df/dx2dx2 =", hessian%val(2,1)

  write(*,*) "=== Example Complete ==="

end program example_hessian
