program test
  use my_loss_module
  use my_network_module
  implicit none

  integer :: i
  integer :: seed = 42
  integer :: num_epochs = 200
  type(my_loss_type) :: loss_method
  type(my_network_type), target :: network
  type(array_type) :: x(1,1), y(1,1)
  type(array_type), pointer :: loss, output(:,:)
  class(base_layer_type), pointer :: layer


  !-----------------------------------------------------------------------------
  ! initialise random seed
  !-----------------------------------------------------------------------------
  call random_setup(seed, restart=.false.)


  call network%add(full_layer_type(num_inputs=3,num_outputs=5, &
       activation="tanh"))
  call network%add(full_layer_type(num_outputs=2, activation="sigmoid"))
  network%model(1)%layer%id = 101
  network%model(2)%layer%id = 102
  call network%compile( &
       optimiser = sgd_optimiser_type(learning_rate=1._real32), verbose=1)
  call network%set_batch_size(1)

  layer => network%layer_from_id(101)
  select type(layer)
  type is(full_layer_type)
     write(*,'("Layer ID 101 has ",I0," outputs.")') layer%num_outputs
  end select


  !-----------------------------------------------------------------------------
  ! create train data
  !-----------------------------------------------------------------------------
  call x(1,1)%allocate(source=reshape([0.2, 0.4, 0.6], [3,1]))
  call y(1,1)%allocate(source=reshape([0.123456, 0.246802], [2,1]))

  do i = 1, num_epochs
     output => network%forward_eval(x)
     loss => loss_method%compute(output, y)
     call loss%grad_reverse(reset_graph=.true.)
     call network%update()
     if(mod(i,10) == 0) write(*,'("epoch ",I4,":",2(3X,F8.6))') i, output(1,1)%val
     if(all(abs(output(1,1)%val - y(1,1)%val) .lt. 1.E-5_real32)) exit
  end do
  if(i.le.num_epochs)then
     write(*,*) "Converged in ", i, " iterations."
  else
     write(*,*) "Did not converge within the maximum number of epochs."
  end if

end program test
