program pinn_burgers_example
  !! Program to demonstrate the use of PINN architecture
  !!
  !! Example is a port of the following python example:
  !! https://www.marktechpost.com/2025/03/28/a-step-by-step-guide-to-solve-1d-burgers-equation-with-physics-informed-neural-networks-pinns-a-pytorch-approach-using-automatic-differentiation-and-collocation-methods/
  use athena
  use coreutils, only: real32
  use constants_mnist, only: pi

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
  integer :: num_tests = 10, num_epochs = 100, batch_size = 1
  character(32) :: activation_name
  class(*), allocatable :: kernel_initialiser, bias_initialiser

  integer :: i, j, itmp1, num_params
  real(real32), dimension(:), allocatable :: params

  integer :: x_min, x_max, t_min, t_max
  integer :: N_f, N_0, N_b, N_x, N_t
  real(real32) :: nu
  real(real32), dimension(:,:), allocatable :: u0
  real(real32), dimension(:,:), allocatable :: X_f, X_0, X_b_left, X_b_right, XT
  type(array_type), pointer :: u_pred(:,:), loss


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

  N_f = 1000
  N_0 = 200
  N_b = 200

  nu = 0.01_real32 / pi
  allocate(X_f(2,N_f))
  ! assign random
  call random_number(X_f)
  X_f(1,:) = x_min + (x_max - x_min) * X_f(1,:)
  X_f(2,:) = t_min + (t_max - t_min) * X_f(2,:)
  open(unit=10, file="X_f.txt", status="replace")
  do i = 1, N_f
     write(10,*) X_f(1,i), X_f(2,i)
  end do
  close(10)


  allocate(u0(1,N_0))
  allocate(X_0(2,N_0))
  ! assign random
  ! fortran version of linspace from x_min to x_max with N_0 steps
  do i = 1, N_0
     X_0(1,i) = real(i-1) * (x_max - x_min) / real(N_0 - 1) + x_min
     X_0(2,i) = 0._real32
     u0(1,i) = -sin(pi * X_0(1,i))
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
     activation_name = "tanh"
     kernel_initialiser = he_uniform_init_type(scale = 1._real32/sqrt(6._real32))
     bias_initialiser = he_uniform_init_type(scale = 1._real32/sqrt(6._real32))

     call network%add(full_layer_type( &
          num_inputs  = 2, &
          num_outputs = 50, &
          batch_size  = batch_size, &
          activation = activation_name, &
          kernel_initialiser = kernel_initialiser, &
          bias_initialiser = bias_initialiser &
     ))
     call network%add(full_layer_type( &
          num_outputs = 50, &
          batch_size  = batch_size, &
          activation = activation_name, &
          kernel_initialiser = kernel_initialiser, &
          bias_initialiser = bias_initialiser &
     ))
     call network%add(full_layer_type( &
          num_outputs = 50, &
          batch_size  = batch_size, &
          activation = activation_name, &
          kernel_initialiser = kernel_initialiser, &
          bias_initialiser = bias_initialiser &
     ))
     call network%add(full_layer_type( &
          num_outputs = 50, &
          batch_size  = batch_size, &
          activation = activation_name, &
          kernel_initialiser = kernel_initialiser, &
          bias_initialiser = bias_initialiser &
     ))
     call network%add(full_layer_type( &
          num_outputs = 1, &
          batch_size  = batch_size, &
          activation = "none", &
          kernel_initialiser = kernel_initialiser, &
          bias_initialiser = bias_initialiser &
     ))
  end if


  !-----------------------------------------------------------------------------
  ! compile network
  !-----------------------------------------------------------------------------
  ! allocate(clip, source=clip_type(-1.E-3_real32, 1.E-3_real32))
  ! allocate(clip, source=clip_type(clip_norm = 1.E-1_real32))
  metric_dict%active = .false.
  metric_dict(1)%key = "loss"
  metric_dict(2)%key = "accuracy"
  metric_dict%threshold = 1.E-1_real32
  call network%compile( &
       optimiser = adam_optimiser_type( &
            ! clip_dict = clip, &
            beta1 = 0.9_real32, &
            beta2 = 0.999_real32, &
            epsilon = 1.E-8_real32, &
            learning_rate = 1.E-3_real32 &
            ! lr_decay = exp_lr_decay_type(1.E-2_real32) &
            ! lr_decay = step_lr_decay_type(0.5_real32, 5) &
       ), &
       ! loss_method = burgers_loss_type(), &
       metrics = metric_dict, &
       ! batch_size = batch_size, &
       verbose = 1, &
       accuracy_method = "mse" &
  )
  params = network%get_params()
  open(unit=10, file="params.txt", status='replace')
  do i = 1, network%num_layers
     select type(layer => network%model(i)%layer)
     class is (learnable_layer_type)
        write(10,*) layer%params(1)%val(:,1)
        write(10,*) layer%params(2)%val(:,1)
     end select
  end do


  !-----------------------------------------------------------------------------
  ! print network and dataset summary
  !-----------------------------------------------------------------------------
  num_params = network%get_num_params()
  write(*,*) "NUMBER OF LAYERS",network%num_layers
  write(*,*) "Number of parameters", num_params


  !-----------------------------------------------------------------------------
  ! training loop
  !-----------------------------------------------------------------------------
  ! call network%set_batch_size(batch_size)
  do i = 1, num_epochs
     ! forward pass
     loss => loss_func(network)

     ! backward pass
     call loss%grad_reverse(reset_graph=.false.)

     ! update learnable parameters
     call network%update()

     write(*,'("epoch: ",I0,"/",I0," loss: ",F0.5)') i, num_epochs, loss%val(1,1)

     ! clean memory
     call loss%nullify_graph()
     deallocate(loss)
     nullify(loss)
  end do


  !-----------------------------------------------------------------------------
  ! testing loop
  !-----------------------------------------------------------------------------
  N_x = 256
  N_t = 100
  itmp1 = 0
  allocate(XT(2, N_x * N_t))
  do i = 1, N_t
     do j = 1, N_x
        itmp1 = itmp1 + 1
        XT(1,itmp1) = (real(j-1) * (x_max - x_min) / real(N_x - 1) + x_min)
        XT(2,itmp1) = (real(i-1) * (t_max - t_min) / real(N_t - 1) + t_min)
     end do
  end do
  write(*,*) "Starting testing..."
! check compiler, if gfortran, do the following
#ifdef __GNUC__
  allocate(u_pred(1,1))
  u_pred = network%predict_array(XT)
#else
  u_pred(1:1,1:1) => network%predict_array(XT)
#endif
  write(*,*) "Testing finished"
  open(unit=20, file="u_pred.txt", status='replace')
  itmp1 = 0
  do i = 1, N_t
     do j = 1, N_x
        itmp1 = itmp1 + 1
        write(20,'(3F12.5)') XT(1,itmp1), XT(2,itmp1), u_pred(1,1)%val(1,itmp1)
     end do
  end do
  close(20)

  if(.not.restart)then
     call network%print(file="network.txt")
  else
     call network%print(file="tmp.txt")
  end if

contains

  function loss_func(this) result(loss)
    class(network_type), intent(inout), target :: this
    !! Instance of the burgers network type

    type(array_type), pointer :: loss

    type(array_type), pointer :: u_i, u_xx, u_t, u_x

    type(array_type), pointer :: input, u0_pred, f_pred, u_left_pred, u_right_pred, u
    type(array_type), pointer :: loss_f, loss_0, loss_b

    call this%forward(X_f)
    this%model(this%root_vertices(1))%layer%output(1,1)%id = 1
    ! find what the new input loc is now
    u => this%model(this%leaf_vertices(1))%layer%output(1,1)%duplicate_graph()
    input => u%get_ptr_from_id(1)

    ! set direction (t,x) to compute u_t, u_x, u_xx
    call input%set_direction([0._real32, 1._real32])
    call input%set_requires_grad(.true.)
    u_t => u%grad_forward(input)
    call input%set_direction([1._real32, 0._real32])
    u_x => u%grad_forward(input)
    u_xx => u_x%grad_forward(input)

    f_pred => u_t + u * u_x - nu * u_xx
    loss_f => mean( f_pred ** 2._real32, 2 )

    ! boundary conditions
    call this%forward(X_b_left)
    u_left_pred => &
         this%model(this%leaf_vertices(1))%layer%output(1,1)%duplicate_graph()

    call this%forward(X_b_right)
    u_right_pred => &
         this%model(this%leaf_vertices(1))%layer%output(1,1)%duplicate_graph()
    loss_b => &
         mean( u_left_pred ** 2._real32, 2 ) + &
         mean( u_right_pred ** 2._real32, 2 )

    ! zero time condition
    call this%forward(X_0)
    u0_pred => this%model(this%leaf_vertices(1))%layer%output(1,1)
    loss_0 => mean( ( u0_pred - u0 ) ** 2._real32, 2)

    loss => loss_f + loss_0 + loss_b
    loss%is_temporary = .false.

  end function loss_func

end program pinn_burgers_example
