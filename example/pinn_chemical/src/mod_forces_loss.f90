module forces_loss
  use constants_mnist, only: real32
  use athena, only: base_loss_type, array_type
  implicit none

  private

  public :: forces_loss_type


  type, extends(base_loss_type) :: forces_loss_type
     real(real32) :: alpha, beta
     !type(graph_type), allocatable, dimension(:) :: graphs, gradient_graphs
   contains
     procedure :: compute => compute_loss_forces
     procedure :: compute_pinn => compute_pinn_loss_forces
     procedure :: compute_pinn_derivative => compute_pinn_derivative_forces
  end type forces_loss_type

  interface forces_loss_type
     !! Interface for forces loss function
     module function setup_loss_forces() result(loss)
       !! Set up forces loss function
       type(forces_loss_type) :: loss
       !! Error loss function
     end function setup_loss_forces
  end interface forces_loss_type


contains

  module function setup_loss_forces() result(loss)
    !! Set up forces loss function
    type(forces_loss_type) :: loss

    loss%alpha = 0.5_real32
    loss%beta = 5.E-2_real32
    loss%name = "for"
    loss%requires_autodiff = .true.
  end function setup_loss_forces


  pure function compute_loss_forces(this, predicted, expected) result(output)
    implicit none
    class(forces_loss_type), intent(in) :: this
    !! Instance of the loss function type
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output

    output = 0._real32

  end function compute_loss_forces


  module function compute_pinn_loss_forces(this, predicted, expected, input) result(output)
    implicit none
    class(forces_loss_type), intent(in) :: this
    !! Instance of the loss function type
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    type(array_type), dimension(:), intent(in) :: input
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output

    integer :: s

    do s = 1, size(input)
       output(:,s) = this%alpha * ( predicted(:,s) - expected(:,s) )**2 + &
            this%beta * &
            sum( input(s)%grad%val(1:3,:) - input(s)%val(4:6,:) ) ** 2 / &
            size(input(s)%val, dim = 2)
    end do
  end function compute_pinn_loss_forces


  module function compute_pinn_derivative_forces(this, predicted, expected, input) &
       result(output)
    implicit none
    class(forces_loss_type), intent(in) :: this
    !! Instance of the loss function type
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    type(array_type), dimension(:), intent(in) :: input
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Derivative of the loss function

    integer :: s

    do s = 1, size(input)
       output(:,s) = 2._real32 * this%alpha * ( predicted(:,s) - expected(:,s) ) + &
            2._real32 * this%beta * &
            sum( input(s)%grad%val(1:3,:) - input(s)%val(4:6,:) ) / &
            size(input(s)%val, dim = 2)
    end do
  end function compute_pinn_derivative_forces

end module forces_loss
