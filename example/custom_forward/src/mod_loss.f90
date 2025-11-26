module my_loss_module
  use athena
  use diffstruc
  implicit none

  type, extends(base_loss_type) :: my_loss_type
     class(network_type), pointer :: net => null()
   contains
     procedure, pass(this) :: compute => my_compute
  end type my_loss_type

contains

  function my_compute(this, predicted, expected) result(output)
    implicit none
    class(my_loss_type), intent(in), target :: this
    type(array_type), dimension(:,:), intent(inout), target :: predicted
    type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
         expected
    type(array_type), pointer :: output

    output => mean( ( predicted(1,1) - expected(1,1) )  ** 2._real32, dim=2 )

  end function my_compute

end module my_loss_module
