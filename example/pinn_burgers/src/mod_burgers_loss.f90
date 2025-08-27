module burgers_loss
  use constants_mnist, only: real32, pi
  use athena, only: base_loss_type, array_type
  implicit none

  private

  public :: burgers_loss_type


  type, extends(base_loss_type) :: burgers_loss_type
     real(real32) :: nu
     !type(graph_type), allocatable, dimension(:) :: graphs, gradient_graphs
   contains
     procedure :: compute => compute_loss_burgers
     procedure :: compute_pinn => compute_pinn_loss_burgers
     procedure :: compute_pinn_derivative => compute_pinn_derivative_burgers
  end type burgers_loss_type

  interface burgers_loss_type
     !! Interface for burgers loss function
     module function setup_loss_burgers() result(loss)
       !! Set up burgers loss function
       type(burgers_loss_type) :: loss
       !! Error loss function
     end function setup_loss_burgers
  end interface burgers_loss_type


contains

  module function setup_loss_burgers() result(loss)
    !! Set up burgers loss function
    type(burgers_loss_type) :: loss

    loss%nu = 0.01_real32 / pi
    loss%name = "burgers"
    loss%requires_autodiff = .true.
  end function setup_loss_burgers


  pure function compute_loss_burgers(this, predicted, expected) result(output)
    implicit none
    class(burgers_loss_type), intent(in) :: this
    !! Instance of the loss function type
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output

    output = 0._real32

  end function compute_loss_burgers


  module function compute_pinn_loss_burgers(this, predicted, expected, input) &
       result(output)
    implicit none
    class(burgers_loss_type), intent(in) :: this
    !! Instance of the loss function type
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    type(array_type), dimension(:), intent(in) :: input
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output

    integer :: s

    do s = 1, size(predicted,2)
       output(1,s) = &
            input(1)%grad%val(2,s) + &
            input(1)%val(1,s) * input(1)%grad%val(1,s)
    end do
  end function compute_pinn_loss_burgers


  module function compute_pinn_derivative_burgers(this, predicted, expected, input) &
       result(output)
    implicit none
    class(burgers_loss_type), intent(in) :: this
    !! Instance of the loss function type
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    type(array_type), dimension(:), intent(in) :: input
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Derivative of the loss function

    integer :: s

  end function compute_pinn_derivative_burgers

end module burgers_loss
