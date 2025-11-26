module forces_loss
  use constants_mnist, only: real32
  use athena, only: network_type, base_loss_type
  use diffstruc
  implicit none

  private

  public :: forces_loss_type


  type, extends(base_loss_type) :: forces_loss_type
     real(real32) :: alpha, beta
     type(network_type), pointer :: network
     type(array_type), dimension(:), allocatable :: expected_forces
   contains
     procedure :: compute => compute_forces
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
    loss%beta = 1.E-4_real32
    loss%name = "for"
    loss%requires_autodiff = .true.
  end function setup_loss_forces

  function compute_forces( this, predicted, expected ) result(output)
    implicit none
    class(forces_loss_type), intent(in), target :: this
    !! Instance of the loss function type
    type(array_type), dimension(:,:), intent(inout), target :: predicted
    type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
         expected
    !! Predicted and expected values
    type(array_type), pointer :: output

    integer :: s
    integer :: num_atoms
    type(array_type), pointer :: input, forces, forces_loss

    do s = 1, size(predicted, 2)
       input => this%network%model(this%network%root_vertices(1))%layer%output(1,s)
       call input%set_requires_grad(.true.)
       num_atoms = size(input%val, dim=2)
       forces => predicted(1,1)%grad_forward(input)
       if(s.eq.1)then
          forces_loss => sum( forces - this%expected_forces(s), dim=2 ) ** 2 / real(num_atoms, real32)
       else
          forces_loss => forces_loss + &
               sum( forces - this%expected_forces(s), dim=2 ) ** 2 / real(num_atoms, real32)
       end if
    end do
    output => this%alpha * ( predicted(1,1) - expected(1,1) ) ** 2._real32 + &
         this%beta * forces_loss

  end function compute_forces

end module forces_loss
