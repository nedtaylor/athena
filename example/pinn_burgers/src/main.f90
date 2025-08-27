program pinn_burgers_example
  !! Program to demonstrate the use of PINN architecture
  !!
  !! Example is a port of the following python example:
  !! https://www.marktechpost.com/2025/03/28/a-step-by-step-guide-to-solve-1d-burgers-equation-with-physics-informed-neural-networks-pinns-a-pytorch-approach-using-automatic-differentiation-and-collocation-methods/
  use athena
  use constants_mnist, only: real32, pi
  use burgers_loss, only: burgers_loss_type

  implicit none

  integer :: seed = 42
  type(network_type), target :: network
  class(base_layer_type), allocatable :: layer
  type(metric_dict_type), dimension(2) :: metric_dict
  class(clip_type), allocatable :: clip

  logical :: restart = .false.

  ! data loading and preprocessing
  character(1024) :: file, train_file

  ! training loop variables
  integer :: num_tests = 10, num_epochs = 100, batch_size = 8

  integer :: i, num_params

  integer :: x_min, x_max, t_min, t_max
  integer :: N_f, N_0, N_b
  real(real32) :: nu
  real(real32), dimension(:), allocatable :: u0
  real(real32), dimension(:,:), allocatable :: X_f, X_0, X_b_left, X_b_right
  type(array_type), pointer :: u, u_i, u_xx

  type(array_type), pointer :: loss, loss_f, loss_0, loss_b, f_pred, input, u0_pred
  type(array_type), dimension(:,:), allocatable :: u_left_pred, u_right_pred


  !-----------------------------------------------------------------------------
  ! initialise random seed
  !-----------------------------------------------------------------------------
  call random_setup(seed, restart=.false.)


  !-----------------------------------------------------------------------------
  ! read training dataset
  !-----------------------------------------------------------------------------
  x_min = -1._real32
  x_max = 1._real32
  t_min = 0._real32
  t_max = 1._real32

  N_f = 100
  N_0 = 200
  N_b = 200

  nu = 0.01 / pi
  allocate(X_f(2,N_f))
  ! assign random
  call random_number(X_f)
  X_f(1,:) = x_min + (x_max - x_min) * X_f(1,:)
  X_f(2,:) = t_min + (t_max - t_min) * X_f(2,:)

  allocate(u0(N_0))
  allocate(X_0(2,N_0))
  ! assign random
  call random_number(X_0)
  ! fortran version of linspace from x_min to x_max with N_0 steps
  do i = 1, N_0
     X_0(1,i) = real(i-1) * (x_max - x_min) / real(N_0 - 1) + x_min
     X_0(2,i) = 0._real32
     u0(i) = -sin(pi * X_0(1,i))
  end do

  allocate(X_b_left(2,N_b))
  allocate(X_b_right(2,N_b))
  ! assign random
  do i = 1, N_b
     X_b_left(1,i) = x_min
     X_b_right(1,i) = x_max
     X_b_left(2,i) = real(i-1) * (t_max - t_min) / real(N_b - 1) + t_min
     X_b_right(2,i) = real(i-1) * (t_max - t_min) / real(N_b - 1) + t_min
  end do

  !-----------------------------------------------------------------------------
  ! initialise convolutional and pooling layers
  !-----------------------------------------------------------------------------
  if(restart)then
     write(*,*) "Reading network from file..."
     call network%read(file="network.txt")
     write(*,*) "Reading finished"
  else
     write(6,*) "Initialising PINN..."

     call network%add(full_layer_type( &
          num_inputs  = 2, &
          num_outputs = 50, &
          batch_size  = batch_size, &
          activation_function = 'tanh', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
     call network%add(full_layer_type( &
          num_outputs = 50, &
          batch_size  = batch_size, &
          activation_function = 'tanh', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
     call network%add(full_layer_type( &
          num_outputs = 50, &
          batch_size  = batch_size, &
          activation_function = 'tanh', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
     call network%add(full_layer_type( &
          num_outputs = 50, &
          batch_size  = batch_size, &
          activation_function = 'tanh', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
     call network%add(full_layer_type( &
          num_outputs = 1, &
          batch_size  = batch_size, &
          activation_function = 'tanh', &
          kernel_initialiser = 'he_normal', &
          bias_initialiser = 'ones' &
     ))
  end if


  !-----------------------------------------------------------------------------
  ! compile network
  !-----------------------------------------------------------------------------
  allocate(clip, source=clip_type(-1.E0_real32, 1.E0_real32))
!   allocate(clip, source=clip_type(clip_norm = 1.E-1_real32))
  metric_dict%active = .false.
  metric_dict(1)%key = "loss"
  metric_dict(2)%key = "accuracy"
  metric_dict%threshold = 1.E-1_real32
  call network%compile( &
       optimiser = adam_optimiser_type( &
            clip_dict = clip, &
            learning_rate = 1.E-3_real32 &
            ! lr_decay = exp_lr_decay_type(1.E-2_real32) &
            ! lr_decay = step_lr_decay_type(0.5_real32, 5) &
       ), &
       loss_method = burgers_loss_type(), &
       metrics = metric_dict, &
       batch_size = batch_size, verbose = 1, &
       accuracy_method = "mse" &
  )


  !-----------------------------------------------------------------------------
  ! print network and dataset summary
  !-----------------------------------------------------------------------------
  num_params = network%get_num_params()
  write(*,*) "NUMBER OF LAYERS",network%num_layers
  write(*,*) "Number of parameters", num_params
!   write(*,*) "Number of samples",size(output(1,1)%val,2)


  !-----------------------------------------------------------------------------
  ! training loop
  !-----------------------------------------------------------------------------
  call network%set_batch_size(batch_size)
  do i = 1, num_epochs
     ! write(*,*) "residual"
     ! set direction to only calculate u_xx
     call network%forward(X_f)
     input => network%model(network%root_vertices(1))%layer%output(1,1)
     u => network%model(network%leaf_vertices(1))%layer%output(1,1)
     ! write(*,*) "setting direction"
     call input%set_direction([1._real32, 1._real32])
     call u%grad_reverse(record_graph=.true., reset_graph=.true.)
     u_i => input%grad
     call input%set_direction([1._real32, 0._real32])
     allocate(u_xx)
     u_xx = u_i%grad_forward(input)

     f_pred => &
          pack(u_i, [2], dim = 1) + &
          u * pack(u_i, [1], dim = 1) - &
          nu * pack(u_xx, [1], dim = 1)
     loss_f => mean( f_pred ** 2, 2 )
     ! call f_pred%reset_graph()
     ! call loss_f%set_requires_grad(.false.)

     ! write(*,*) "boundary conditions"
     u_left_pred = network%predict_array(X_b_left)
     u_right_pred = network%predict_array(X_b_right)

     loss_b => mean( u_left_pred(1,1) ** 2, 2 ) + mean( u_right_pred(1,1) ** 2, 2 )

     ! write(*,*) "zero condition"
     call network%forward(X_0)
     u0_pred => network%model(network%leaf_vertices(1))%layer%output(1,1)
     loss_0 => mean( ( u0_pred - u0 ) ** 2, 2)

     ! write(*,*) "loss"
     loss =>  loss_f%val + loss_0 + loss_b
     ! write(*,*) "backward"
     call loss%grad_reverse(reset_graph=.true.)
     ! write(*,*) "updating"
     call network%update()
     write(*,*) "epoch: ", i, "loss: ", loss%val(1,1)

  end do
  !-----------------------------------------------------------------------------
  ! testing loop
  !-----------------------------------------------------------------------------
!   write(*,*) "Starting testing..."
!   call network%test( &
!        graphs_in, &
!        output &
!   )
!   write(*,*) "Testing finished"


  write(6,'("Overall accuracy=",F0.5)') network%accuracy_val
  write(6,'("Overall loss=",F0.5)')     network%loss_val

  if(.not.restart)then
     call network%print(file="network.txt")
  else
     call network%print(file="tmp.txt")
  end if

end program pinn_burgers_example
