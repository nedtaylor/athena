module forces_loss
  use constants_mnist, only: real32
  use athena, only: base_loss_type
  implicit none

  private

  public :: forces_loss_type


  type, extends(base_loss_type) :: forces_loss_type
     type(graph_type), allocatable, dimension(:) :: graphs, gradient_graphs
   contains
     procedure :: compute => compute_loss_forces
     procedure :: compute_derivative => compute_derivative_forces
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

    allocate(loss%graphs(0))
    loss%name = "for"
  end function setup_loss_forces


  pure module function compute_derivative_forces(this, predicted, expected) &
       result(output)
    implicit none
    class(forces_loss_type), intent(in) :: this
    !! Instance of the loss function type
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Derivative of the loss function

    output = predicted - expected + &
         (this%graphs(this%batch_index)%vertex_features(1:3,:) - &
              this%gradient_graphs(this%batch_index)%vertex_features(1:3,:))
  end function compute_derivative_forces


  pure function compute_loss_forces(this, predicted, expected) result(output)
    implicit none
    class(forces_loss_type), intent(in) :: this
    !! Instance of the loss function type
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output

    output = ( predicted - expected )**2 + ( &
         this%graphs(this%batch_index)%vertex_features(1:3,:) - &
         this%gradient_graphs(this%batch_index)%vertex_features(1:3,:) )**2

  end function compute_loss_forces

end module forces_loss
