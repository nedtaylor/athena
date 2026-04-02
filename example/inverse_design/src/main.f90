program inverse_design_example
  !! Inverse design demonstration using athena
  !!
  !! This example demonstrates how to use the built-in `inverse_design`
  !! procedure to find an input that produces a desired output from a trained
  !! neural network.
  !!
  !! ## Overview
  !!
  !! Given a trained model \( y = f(x) \), inverse design solves:
  !! $$x^* = \arg\min_{x} \| f(x) - y_t \|^2$$
  !!
  !! where \( y_t \) is the target output.
  !!
  !! ## Workflow
  !!
  !! 1. Build and train a small network on \( y = 2x + 0.5 \)
  !! 2. Choose a target output
  !! 3. Use `inverse_design` to find the input that produces it
  !! 4. Verify by running the optimised input through the network
  !!
  !! ## Running
  !!
  !! ```
  !! fpm run --example inverse_design
  !! ```
  use athena
  use coreutils, only: real32
  implicit none

  type(network_type), target :: network
  real(real32), dimension(:,:), allocatable :: x_opt
  type(array_type) :: x_array(1), y_array(1,1)
  type(array_type), pointer :: loss

  integer, parameter :: num_train = 50000
  integer, parameter :: inverse_steps = 2000

  real(real32) :: x(1,1), y(1,1)
  real(real32) :: target_y(1,1), x_init(1,1)
  real(real32) :: predicted(1,1)

  integer :: i


  !-----------------------------------------------------------------------------
  ! set up random seed for reproducibility
  !-----------------------------------------------------------------------------
  call random_setup(42)


  !-----------------------------------------------------------------------------
  ! build network: 1 -> 16 -> 1
  !   learns the mapping y = 2*x + 0.5 over [0,1]
  !   output is normalised to [0,1] by dividing by 2.5
  !-----------------------------------------------------------------------------
  write(*,*) "============================================="
  write(*,*) " Inverse Design Example"
  write(*,*) "============================================="
  write(*,*)
  write(*,*) "Step 1: Build and train a network on y = 2*x + 0.5"
  write(*,*) "        (normalised to [0,1])"
  write(*,*)

  call network%add(full_layer_type( &
       num_inputs=1, num_outputs=16, activation="tanh"))
  call network%add(full_layer_type(num_outputs=1, activation="sigmoid"))
  call network%compile( &
       optimiser = sgd_optimiser_type(learning_rate=0.1_real32), &
       loss_method = "mse", &
       metrics = ["loss"], &
       verbose = 0 &
  )
  call network%set_batch_size(1)


  !-----------------------------------------------------------------------------
  ! train the network
  !-----------------------------------------------------------------------------
  call x_array(1)%allocate(array_shape=[1,1])
  call y_array(1,1)%allocate(array_shape=[1,1])

  write(*,'(A10, A15)') "Iteration", "Loss"
  do i = 1, num_train
     call random_number(x)
     y(1,1) = (2._real32 * x(1,1) + 0.5_real32) / 2.5_real32

     x_array(1)%val = x
     y_array(1,1)%val = y

     call network%set_batch_size(1)
     call network%forward(x)
     network%expected_array = y_array
     loss => network%loss_eval(1, 1)
     call loss%grad_reverse()
     call network%update()

     if (mod(i, 5000) == 0) then
        write(*,'(I10, F15.8)') i, sum(loss%val)
     end if
  end do
  write(*,*)

  ! verify the network is well trained
  predicted = network%predict(input=reshape([0.5_real32], [1,1]))
  write(*,'(A,F10.6)') "  Sanity check: predict(0.5) = ", predicted(1,1)
  write(*,'(A,F10.6)') "  Expected:     (2*0.5+0.5)/2.5 = ", 0.6_real32
  write(*,*)


  !-----------------------------------------------------------------------------
  ! Step 2: inverse design using built-in function
  !-----------------------------------------------------------------------------
  write(*,*) "Step 2: Inverse design using network%inverse_design()"
  write(*,*)

  ! Target: y = 0.6 (normalised), corresponding to x = 0.5
  ! Since y = (2*0.5 + 0.5) / 2.5 = 1.5 / 2.5 = 0.6
  target_y(1,1) = 0.6_real32
  x_init(1,1)   = 0.1_real32   ! start far from the true value

  write(*,'(A,F8.4)') "  Target output (normalised):  ", target_y(1,1)
  write(*,'(A,F8.4)') "  Initial input guess:         ", x_init(1,1)
  write(*,'(A,F8.4)') "  Expected optimal input:      ", 0.5_real32
  write(*,*)

  x_opt = network%inverse_design( &
       target = target_y, &
       x_init = x_init, &
       optimiser = sgd_optimiser_type(learning_rate=0.1_real32), &
       steps = inverse_steps &
  )

  ! verify by running optimised input through the network
  ! Use a real array (not reshape) to avoid gfortran assumed-rank bug
  predicted = network%predict(input=x_opt)

  write(*,*) "--- Built-in inverse_design results ---"
  write(*,'(A,F10.6)') "  Optimised input:    ", x_opt(1,1)
  write(*,'(A,F10.6)') "  Predicted output:   ", predicted(1,1)
  write(*,'(A,F10.6)') "  Target output:      ", target_y(1,1)
  write(*,'(A,ES10.3)') "  Output error:       ", &
       abs(predicted(1,1) - target_y(1,1))
  write(*,'(A,ES10.3)') "  Input error:        ", &
       abs(x_opt(1,1) - 0.5_real32)
  write(*,*)

  ! check if network is still intact
  predicted = network%predict(input=reshape([0.5_real32], [1,1]))
  write(*,'(A,F10.6)') "  Post-inverse predict(0.5) = ", predicted(1,1)


  !-----------------------------------------------------------------------------
  ! Step 3: custom inverse design loop (manual approach)
  !-----------------------------------------------------------------------------
  write(*,*) "Step 3: Custom inverse design loop (manual approach)"
  write(*,*)
  call custom_inverse_design()


contains

  subroutine custom_inverse_design()
    !! Demonstrates a manual inverse design loop without using the
    !! built-in inverse_design procedure.
    !! This gives full control over the optimisation process.
    implicit none

    type(array_type) :: cx(1,1), cy(1,1)
    type(array_type), pointer :: closs
    real(real32) :: cx_flat(1), cx_grad(1)
    integer :: step, root_id
    type(sgd_optimiser_type) :: opt

    ! set the optimiser
    opt = sgd_optimiser_type(learning_rate=0.1_real32)
    call opt%init(num_params=1)

    ! initialise x from the same starting point
    call cx(1,1)%allocate(source=reshape([0.1_real32], [1,1]))
    call cx(1,1)%set_requires_grad(.true.)

    ! target
    call cy(1,1)%allocate(source=reshape([0.6_real32], [1,1]))

    ! get the input layer id
    root_id = network%auto_graph%vertex(network%root_vertices(1))%id

    call network%set_batch_size(1)

    do step = 1, inverse_steps

       ! forward pass with current x
       call network%forward(cx)

       ! enable gradient tracking on input layer output
       call network%model(root_id)%layer%output(1,1)%set_requires_grad(.true.)

       ! compute loss via network's loss function
       network%expected_array = cy
       closs => network%loss_eval(1, 1)

       ! backward pass
       call closs%grad_reverse()

       ! extract gradient w.r.t. input
       if(associated( &
            network%model(root_id)%layer%output(1,1)%grad))then
          cx_grad = network%model(root_id)%layer%output(1,1)%grad%val(:,1)
       else
          cx_grad = 0._real32
       end if

       if (step .le. 5 .or. mod(step, 500) == 0) then
          write(*,'(A,I5,A,F10.6,A,ES12.4,A,ES12.4)') &
               "  step=",step, &
               " x=",cx(1,1)%val(1,1), &
               " loss=",sum(closs%val), &
               " grad=",cx_grad(1)
       end if

       ! gradient descent update on x
       cx_flat = cx(1,1)%val(:,1)
       call opt%minimise(param=cx_flat, gradient=cx_grad)
       cx(1,1)%val(:,1) = cx_flat

       ! clean up computation graph
       call closs%nullify_graph()
       deallocate(closs)
       nullify(closs)

       ! reset network gradients (do NOT call network%update)
       call network%reset_gradients()

    end do

    ! verify - use real array to avoid gfortran assumed-rank bug
    x(1,1) = cx(1,1)%val(1,1)
    predicted = network%predict(input=x)

    write(*,*) "--- Custom inverse design results ---"
    write(*,'(A,F10.6)') "  Optimised input:    ", cx(1,1)%val(1,1)
    write(*,'(A,F10.6)') "  Predicted output:   ", predicted(1,1)
    write(*,'(A,F10.6)') "  Target output:      ", 0.6_real32
    write(*,'(A,ES10.3)') "  Output error:       ", &
         abs(predicted(1,1) - 0.6_real32)
    write(*,'(A,ES10.3)') "  Input error:        ", &
         abs(cx(1,1)%val(1,1) - 0.5_real32)
    write(*,*)

  end subroutine custom_inverse_design

end program inverse_design_example
