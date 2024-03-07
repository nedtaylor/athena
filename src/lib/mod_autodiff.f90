module autodiff
  implicit none

  type :: ValueAndPartial
    real :: value
    real :: partial = 0.0
  end type ValueAndPartial

  type, abstract :: Expression
    contains
      procedure(evaluateAndDerive_gen), deferred, pass(this) :: evaluateAndDerive
  end type Expression

   
  abstract interface
     function evaluateAndDerive_gen(this, variable) result(output)
       import :: Expression, ValueAndPartial
       class(Expression), intent(inout) :: this
       class(Expression), intent(inout) :: variable
       type(ValueAndPartial) :: output
     end function evaluateAndDerive_gen
  end interface

  type, extends(Expression) :: Variable_expr
    real :: value
  contains
    procedure :: evaluateAndDerive => evaluateAndDerive_Variable
  end type Variable_expr

  type, extends(Expression) :: Plus_expr
    class(Expression), pointer :: a, b
  contains
    procedure :: evaluateAndDerive => evaluateAndDerive_Plus
  end type Plus_expr

  type, extends(Expression) :: Multiply_expr
    class(Expression), pointer :: a, b
  contains
    procedure :: evaluateAndDerive => evaluateAndDerive_Multiply
  end type Multiply_expr

  
contains

  ! Implementation of evaluateAndDerive for Variable
  type(ValueAndPartial) function evaluateAndDerive_Variable(this, variable)
    class(Variable_expr), intent(inout) :: this
    class(Expression), intent(inout) :: variable
    if(loc(this).eq.loc(variable))then
       evaluateAndDerive_Variable = ValueAndPartial(this%value, 1.0)
    else
       evaluateAndDerive_Variable = ValueAndPartial(this%value, 0.0)
    end if
    !write(*,*) "variable", this%value, evaluateAndDerive_Variable
  end function evaluateAndDerive_Variable

  ! Implementation of evaluateAndDerive for Plus
  type(ValueAndPartial) function evaluateAndDerive_Plus(this, variable)
    class(Plus_expr), intent(inout) :: this
    class(Expression), intent(inout) :: variable
    type(ValueAndPartial) :: vpA, vpB
    vpA = this%a%evaluateAndDerive(variable)
    vpB = this%b%evaluateAndDerive(variable)
    evaluateAndDerive_Plus = ValueAndPartial(vpA%value + vpB%value, vpA%partial + vpB%partial)
    !write(*,*) "plus", vpA, vpB, evaluateAndDerive_Plus
  end function evaluateAndDerive_Plus

  ! Implementation of evaluateAndDerive for Multiply
  type(ValueAndPartial) function evaluateAndDerive_Multiply(this, variable)
    class(Multiply_expr), intent(inout) :: this
    class(Expression), intent(inout) :: variable
    type(ValueAndPartial) :: vpA, vpB
    vpA = this%a%evaluateAndDerive(variable)
    vpB = this%b%evaluateAndDerive(variable)
    evaluateAndDerive_Multiply = ValueAndPartial(vpA%value * vpB%value, vpB%value * vpA%partial + vpA%value * vpB%partial)
    !write(*,*) "multiply", vpA, vpB, evaluateAndDerive_Multiply
  end function evaluateAndDerive_Multiply

end module autodiff

! program AutoDiffExample
!   use autodiff
!   implicit none

!   ! Example: Finding the partials of z = x * (x + y) + y * y at (x, y) = (2, 3)
!   ! z = m1 + m2
!   ! z = (x * p1) + (y * y)
!   ! z = (x * (x + y)) + (y * y)
!   type(Variable_expr), target :: x, y
!   type(Plus_expr), target :: p1, z
!   type(Multiply_expr), target :: m1, m2
!   type(ValueAndPartial) :: xPartial, yPartial

!   x%value = 2.0
!   y%value = 3.0

!   allocate(p1%a, source = x)
!   allocate(p1%b, source = y)
!   p1%a => x
!   p1%b => y
  
!   allocate(m1%a, source = x)
!   allocate(m1%b, source = p1)
!   m1%a => x
!   m1%b => p1

!   allocate(m2%a, source = y)
!   allocate(m2%b, source = y)
!   m2%a => y
!   m2%b => y

!   allocate(z%a, source = m1)
!   allocate(z%b, source = m2)
!   z%a => m1
!   z%b => m2

!   xPartial = z%evaluateAndDerive(x)
!   yPartial = z%evaluateAndDerive(y)

!   write(*,*) '∂z/∂x = ', xPartial%partial, ', ∂z/∂y = ', yPartial%partial
!   ! Output: ∂z/∂x = 7.0, ∂z/∂y = 8.0

! end program AutoDiffExample
