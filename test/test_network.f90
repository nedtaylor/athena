program test_network
  use ieee_arithmetic, only: ieee_is_finite
  use athena, only: &
       network_type, &
       metric_dict_type, &
       full_layer_type, &
       input_layer_type, &
       base_optimiser_type, &
       maxpool2d_layer_type, &
       conv2d_layer_type, &
       batchnorm2d_layer_type, &
       dropblock2d_layer_type, &
       conv3d_layer_type
  use athena__loss, only: &
       base_loss_type, &
       bce_loss_type, &
       cce_loss_type, &
       mae_loss_type, &
       mse_loss_type, &
       nll_loss_type, &
       huber_loss_type
  use athena__accuracy, only: &
       compute_accuracy_function, &
       categorical_score, &
       mae_score, &
       mse_score, &
       rmse_score, &
       r2_score
  use diffstruc, only: array_type
  implicit none

  type(metric_dict_type), dimension(2) :: metrics
  type(network_type) :: network, network3
  type(network_type), allocatable :: network2
  real, allocatable, dimension(:) :: gradients
  real, allocatable, dimension(:,:) :: x, y

  ! Additional variables for new tests
  type(input_layer_type) :: input_layer_test
  type(full_layer_type) :: full_layer_test
  real, allocatable, dimension(:) :: params
  integer :: num_params
  type(array_type) :: input_data(1,1), output_data(1,1)

  real, parameter :: learning_rate = 0.1

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

  integer :: unit
  integer :: i, n
  integer :: iter_before
  real :: rtmp1
  logical :: success = .true.
  procedure(compute_accuracy_function), pointer :: get_accuracy => null()


!-------------------------------------------------------------------------------
! Initialise random number generator with a seed
!-------------------------------------------------------------------------------
  call random_seed(size=seed_size)
  allocate(seed(seed_size), source=1)
  call random_seed(put=seed)


!-------------------------------------------------------------------------------
! Create network
!-------------------------------------------------------------------------------
  ! call network%add(input_layer_type(input_shape=[1]))
  call network%add(full_layer_type( &
       num_inputs=3, num_outputs=5, activation="tanh"))
  call network%add(full_layer_type( &
       num_outputs=2, activation="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=learning_rate), &
       loss_method="mse", accuracy_method="mse", metrics=["loss"], verbose=0 &
  )
  call network%set_batch_size(1)


!-------------------------------------------------------------------------------
! Train network
!-------------------------------------------------------------------------------
  !! create test data
  x = reshape([0.2, 0.4, 0.6], [3,1])
  y = reshape([0.123456, 0.246802], [2,1])
  call input_data(1,1)%allocate(array_shape=[3,1])
  call input_data(1,1)%set(x)
  call output_data(1,1)%allocate(array_shape=[2,1])
  call output_data(1,1)%set(y)

  !! train network
  write(*,*) "Training network"
  call network%train(input_data, output_data, num_epochs=600, batch_size=1, verbose=0)
  write(*,*) "Network trained"

  if(abs(network%metrics(1)%val).gt.1.E-3)then
     write(0,*) "Training loss higher than expected"
     write(0,*) "Loss: ", network%metrics(1)%val
     success = .false.
  end if
  if(abs(network%metrics(2)%val).lt.0.95)then
     write(0,*) "Training accuracy higher than expected"
     write(0,*) "Accuracy: ", network%accuracy_val
     success = .false.
  end if


!-------------------------------------------------------------------------------
! Train with non-divisible batch size (remainder batch)
!-------------------------------------------------------------------------------
  call network%reset()
  call network%add(full_layer_type( &
       num_inputs=3, num_outputs=5, activation="tanh"))
  call network%add(full_layer_type( &
       num_outputs=2, activation="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=learning_rate), &
       loss_method="mse", accuracy_method="mse", metrics=["loss"], verbose=0 &
  )

  call input_data(1,1)%deallocate()
  call output_data(1,1)%deallocate()
  x = reshape([ &
       0.1, 0.2, 0.3, &
       0.4, 0.5, 0.6, &
       0.7, 0.8, 0.9 &
  ], [3,3])
  y = reshape([ &
       0.11, 0.22, &
       0.33, 0.44, &
       0.55, 0.66 &
  ], [2,3])
  call input_data(1,1)%allocate(array_shape=[3,3])
  call input_data(1,1)%set(x)
  call output_data(1,1)%allocate(array_shape=[2,3])
  call output_data(1,1)%set(y)

  iter_before = network%optimiser%iter
  call network%train( &
       input_data, output_data, num_epochs=1, batch_size=2, verbose=0, &
       early_stopping=.false. &
  )
  if(network%optimiser%iter-iter_before.ne.2)then
     write(0,*) "Remainder batch was not processed as expected"
     write(0,*) "Expected optimiser iterations: 2"
     write(0,*) "Actual optimiser iterations: ", &
          network%optimiser%iter-iter_before
     success = .false.
  end if


!-------------------------------------------------------------------------------
! Test network
!-------------------------------------------------------------------------------
  !! create test data
  write(*,*)
  x = reshape([0.4, 0.6, 0.8], [3,1])
  y = reshape([0.370368, 0.493824], [2,1])
  call network%test(x, y)
  if(network%loss_val.gt.1.E-1)then
     write(0,*) "Test loss higher than expected"
     write(0,*) "Loss: ", network%loss_val
     success = .false.
  end if
  if(network%accuracy_val.lt.0.7)then
     write(0,*) "Test accuracy higher than expected"
     write(0,*) "Accuracy: ", network%accuracy_val
     success = .false.
  end if


!-------------------------------------------------------------------------------
! Train network with validation data
!-------------------------------------------------------------------------------
  write(*,*)
  write(*,*) "Training network with validation"

  call network%reset()
  call network%add(full_layer_type( &
       num_inputs=3, num_outputs=5, activation="tanh"))
  call network%add(full_layer_type( &
       num_outputs=2, activation="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=learning_rate), &
       loss_method="mse", accuracy_method="mse", metrics=["loss"], verbose=0 &
  )
  call network%set_batch_size(1)

  !! create training data
  call input_data(1,1)%deallocate()
  call output_data(1,1)%deallocate()
  x = reshape([0.2, 0.4, 0.6], [3,1])
  y = reshape([0.123456, 0.246802], [2,1])
  call input_data(1,1)%allocate(array_shape=[3,1])
  call input_data(1,1)%set(x)
  call output_data(1,1)%allocate(array_shape=[2,1])
  call output_data(1,1)%set(y)

  !! create validation data
  validation: block
    type(array_type) :: val_input_data(1,1), val_output_data(1,1)
    real, allocatable, dimension(:,:) :: val_x, val_y

    val_x = reshape([0.3, 0.5, 0.7], [3,1])
    val_y = reshape([0.185184, 0.370368], [2,1])
    call val_input_data(1,1)%allocate(array_shape=[3,1])
    call val_input_data(1,1)%set(val_x)
    call val_output_data(1,1)%allocate(array_shape=[2,1])
    call val_output_data(1,1)%set(val_y)

    !! train with validation
    call network%train( &
         input_data, output_data, num_epochs=600, batch_size=1, verbose=0, &
         val_input=val_input_data, val_output=val_output_data &
    )
    write(*,*) "Network trained with validation"

    !! check training loss is reasonable
    if(abs(network%metrics(1)%val).gt.1.E-3)then
       write(0,*) "Training loss (with validation) higher than expected"
       write(0,*) "Loss: ", network%metrics(1)%val
       success = .false.
    end if

    !! check training accuracy is reasonable
    if(abs(network%metrics(2)%val).lt.0.95)then
       write(0,*) "Training accuracy (with validation) lower than expected"
       write(0,*) "Accuracy: ", network%accuracy_val
       success = .false.
    end if

    !! verify that we can still test after training with validation
    call network%test(val_x, val_y)
    if(network%loss_val.gt.1.E-1)then
       write(0,*) "Test loss after validation training higher than expected"
       write(0,*) "Loss: ", network%loss_val
       success = .false.
    end if
  end block validation

!!! DOES NOT WORK DUE TO THE ORDER OF THE LAYERS NOT BEING UNDERSTOOD
!!! This results in there being multiple input layers being created
!!! Should we also enforce passing of an adjacency matrix?
!!! write(*,*) network%auto_graph%adjacency
!   !! check network allocation
!   allocate(network2, source=network_type(layers=network%model, batch_size=4))
!   call network2%compile( &
!        optimiser = base_optimiser_type(learning_rate=learning_rate), &
!        loss_method="mse", metrics=["loss"], verbose=1)
!   if(network2%batch_size.ne.4)then
!      write(0,*) "Batch size not set correctly"
!      success = .false.
!   end if

!   !! check gradients
!   call network2%set_gradients(0.1)
! !   write(*,*) "hehe", network2%get_gradients()
!   if(any(abs(network2%get_gradients()-0.1).gt.1.E-6))then
!      write(0,*) "Gradients not set correctly"
!      success = .false.
!   end if
!   allocate(gradients(network%get_num_params()))
!   gradients = 0.2
!   call network%set_gradients(gradients)
!   if(any(abs(network%get_gradients()-0.2).gt.1.E-6))then
!      write(0,*) "Gradients not set correctly"
!      success = .false.
!   end if


!-------------------------------------------------------------------------------
! Test network copy and reduce
!-------------------------------------------------------------------------------
  call network3%copy(network)
  if(abs(network3%metrics(1)%val-network%metrics(1)%val).gt.1.E-6)then
     write(0,*) "Network copy failed"
     success = .false.
  end if
  if(size(network3%model).ne.size(network%model))then
     write(0,*) "Network copy failed"
     success = .false.
  end if
  rtmp1 = network3%metrics(1)%val

  !! check network reduce
  call network3%reduce(network)
  if(abs(network3%metrics(1)%val-(network%metrics(1)%val+rtmp1)).gt.1.E-6)then
     write(0,*) "Network reduction failed"
     success = .false.
  end if


!-------------------------------------------------------------------------------
! Check all unique network layer adds
!-------------------------------------------------------------------------------
  call network3%reset()
  call network3%add(maxpool2d_layer_type())
  call network3%add(conv2d_layer_type())
  call network3%add(batchnorm2d_layer_type())
  call network3%add(dropblock2d_layer_type(block_size=3, rate=0.1))

  !! check automatic flatten layer adding
  call network3%reset()
  call network3%add(conv2d_layer_type(input_shape=[3,3,3]))
  call network3%add(full_layer_type(num_outputs=5))
  call network3%compile( &
       optimiser = base_optimiser_type(learning_rate=learning_rate), &
       loss_method="mse", metrics=["loss"], verbose=1)

  !! check automatic flatten layer adding
  call network3%reset()
  call network3%add(conv3d_layer_type(input_shape=[3,3,3,3]))
  call network3%add(full_layer_type(num_outputs=5))
  call network3%compile( &
       optimiser = base_optimiser_type(learning_rate=learning_rate), &
       loss_method="mse", metrics=["loss"], verbose=1)


!-------------------------------------------------------------------------------
! check loss procedure setup
!-------------------------------------------------------------------------------
  call network3%set_loss("binary_crossentropy")
  if(network3%loss%name.ne.'bce')then
     write(0,*) "BCE loss method not set correctly"
     success = .false.
  end if
  call network3%set_loss("categorical_crossentropy")
  if(network3%loss%name.ne.'cce')then
     write(0,*) "CCE loss method not set correctly"
     success = .false.
  end if
  call network3%set_loss("mean_absolute_error")
  if(network3%loss%name.ne.'mae')then
     write(0,*) "MAE loss method not set correctly"
     success = .false.
  end if
  call network3%set_loss("mean_squared_error")
  if(network3%loss%name.ne.'mse')then
     write(0,*) "MSE loss method not set correctly"
     success = .false.
  end if
  call network3%set_loss("negative_log_likelihood")
  if(network3%loss%name.ne.'nll')then
     write(0,*) "NLL loss method not set correctly"
     success = .false.
  end if
  call network3%set_loss("huber")
  if(network3%loss%name.ne.'hub')then
     write(0,*) "Huber loss method not set correctly"
     success = .false.
  end if


!-------------------------------------------------------------------------------
! check metric dict setting in network
!-------------------------------------------------------------------------------
  metrics(1)%history = [1.0, 2.0, 3.0]
  call network%set_metrics(metrics)
  if(size(network%metrics).ne.size(metrics))then
     write(0,*) "Metric dict failed to set the correct size."
     success = .false.
  end if
  if(any(abs(network%metrics(1)%history - metrics(1)%history).gt.1.E-6))then
     write(0,*) "Metric dict failed to set the correct history."
     success = .false.
  end if


!-------------------------------------------------------------------------------
! check accuracy procedure setup
!-------------------------------------------------------------------------------
  call network%reset()
  call network%set_accuracy("categorical")
  get_accuracy => categorical_score
  if(.not.associated(network%get_accuracy, get_accuracy))then
     write(0,*) "CAT accuracy method not set correctly"
     success = .false.
  end if
  call network%set_accuracy("mean_absolute_error")
  get_accuracy => mae_score
  if(.not.associated(network%get_accuracy, get_accuracy))then
     write(0,*) "MAE accuracy method not set correctly"
     success = .false.
  end if
  call network%set_accuracy("mean_squared_error")
  get_accuracy => mse_score
  if(.not.associated(network%get_accuracy, get_accuracy))then
     write(0,*) "MSE accuracy method not set correctly"
     success = .false.
  end if
  call network%set_accuracy("root_mean_squared_error")
  get_accuracy => rmse_score
  if(.not.associated(network%get_accuracy, get_accuracy))then
     write(0,*) "RMSE accuracy method not set correctly"
     success = .false.
  end if
  call network%set_accuracy("r2")
  get_accuracy => r2_score
  if(.not.associated(network%get_accuracy, get_accuracy))then
     write(0,*) "R2 accuracy method not set correctly"
     success = .false.
  end if


!-------------------------------------------------------------------------------
! check get_num_params and get_params
!-------------------------------------------------------------------------------
  call network%reset()
  call network%add(input_layer_type(input_shape=[3]))
  call network%add(full_layer_type(num_outputs=8, &
       activation="tanh"))
  call network%add(full_layer_type(num_outputs=4, &
       activation="sigmoid"))
  call network%compile(optimiser=base_optimiser_type(learning_rate=0.01), &
       loss_method="mse")
  call network%set_batch_size(1)

  ! Get network parameters count
  num_params = network%get_num_params()
  if(num_params.le.0)then
     write(0,*) "get_num_params() returned invalid count:", num_params
     success = .false.
  end if

  ! Expected parameter count for this network:
  ! Layer 1: (3 inputs + 1 bias) * 8 outputs = 32 params
  ! Layer 2: (8 inputs + 1 bias) * 4 outputs = 36 params
  ! Total expected: 68 params
  if(num_params.ne.68)then
     write(0,*) "get_num_params() returned unexpected count:", num_params, &
          " expected: 68"
     success = .false.
  end if

  ! Get parameter array
  params = network%get_params()
  if(.not.allocated(params) .or. size(params).ne.num_params)then
     write(0,*) "get_params() failed to return correct parameter array"
     success = .false.
  end if

  ! Verify parameters are finite values (not infinite or NaN)
  if(any(.not.ieee_is_finite(params)))then
     write(0,*) "get_params() returned non-finite values"
     success = .false.
  end if


!-------------------------------------------------------------------------------
! check file I/O
!-------------------------------------------------------------------------------
  call network%reset()
  network%name = "test network"
  call network%add(input_layer_type(input_shape=[3]))
  call network%add(full_layer_type(num_outputs=8, &
       activation="tanh"))
  call network%add(full_layer_type(num_outputs=4, &
       activation="sigmoid"))
  call network%compile(optimiser=base_optimiser_type(learning_rate=0.01), &
       loss_method="mse")
  call network%set_batch_size(1)

  write(*,*) "Saving network to 'test_network.dat'"
  call network%print("test_network.dat")

  write(*,*) "Loading network from 'test_network.dat'"
  call network3%reset()
  call network3%read("test_network.dat")
  call network3%compile(optimiser=base_optimiser_type(learning_rate=0.01))

  if(network3%name.ne.network%name)then
     write(0,*) "Network name mismatch after read"
     write(0,*) "Expected: ", network%name, " Found: ", network3%name
     success = .false.
  end if

  if(network3%get_num_params().ne.network%get_num_params())then
     write(0,*) "Network parameter count mismatch after read"
     success = .false.
  end if

  if(size(network3%model).ne.size(network%model))then
     write(0,*) "Network model size mismatch after read"
     success = .false.
  end if

  ! delete the file
  open(newunit=unit, file="test_network.dat")
  close(unit, status="delete")


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_network passed all tests'
  else
     write(0,*) 'test_network failed one or more tests'
     stop 1
  end if

end program test_network
