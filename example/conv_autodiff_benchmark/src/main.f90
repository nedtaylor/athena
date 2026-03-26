program conv_autodiff_benchmark
  use iso_fortran_env, only: int64
  use athena__activation_relu, only: relu_actv_type
  use athena__conv_autodiff_reference, only: &
       initialise_conv_autodiff_input, &
       initialise_conv_autodiff_parameters, &
       reference_conv2d_forward, &
       reference_conv2d_backward, &
       reference_conv2d_backward_original, &
       reference_conv2d_original
  use athena__diffstruc_extd, only: conv2d, add_bias
  use coreutils, only: real32
  use diffstruc, only: array_type, mean
  implicit none

  integer, parameter :: input_h = 32
  integer, parameter :: input_w = 32
  integer, parameter :: input_c = 3
  integer, parameter :: batch_size = 16
  integer, parameter :: filters1 = 16
  integer, parameter :: filters2 = 32
  integer, parameter :: kernel_size = 3
  integer, parameter :: num_iterations = 200
  integer, parameter :: conv1_h = input_h - kernel_size + 1
  integer, parameter :: conv1_w = input_w - kernel_size + 1
  integer, parameter :: conv2_h = conv1_h - kernel_size + 1
  integer, parameter :: conv2_w = conv1_w - kernel_size + 1

  real(real32), allocatable, dimension(:,:,:,:) :: input_data
  real(real32), allocatable, dimension(:,:,:,:) :: kernel1_ref
  real(real32), allocatable, dimension(:) :: bias1_ref
  real(real32), allocatable, dimension(:,:,:,:) :: kernel2_ref
  real(real32), allocatable, dimension(:) :: bias2_ref
  real(real32), allocatable, dimension(:,:,:,:) :: conv1_ref
  real(real32), allocatable, dimension(:,:,:,:) :: act1_ref
  real(real32), allocatable, dimension(:,:,:,:) :: conv2_ref
  real(real32), allocatable, dimension(:,:,:,:) :: grad_conv2_ref
  real(real32), allocatable, dimension(:,:,:,:) :: grad_act1_ref
  real(real32), allocatable, dimension(:,:,:,:) :: grad_conv1_ref
  real(real32), allocatable, dimension(:,:,:,:) :: grad_input_ref
  real(real32), allocatable, dimension(:,:,:,:) :: grad_kernel1_ref
  real(real32), allocatable, dimension(:) :: grad_bias1_ref
  real(real32), allocatable, dimension(:,:,:,:) :: grad_kernel2_ref
  real(real32), allocatable, dimension(:) :: grad_bias2_ref
  real(real32), allocatable, dimension(:,:) :: output_opt
  real(real32), allocatable, dimension(:,:) :: grad_input_opt
  real(real32), allocatable, dimension(:) :: grad_layer1_opt
  real(real32), allocatable, dimension(:) :: grad_layer2_opt
  type(array_type) :: input_opt
  type(array_type) :: input_base
  type(array_type) :: kernel1_opt
  type(array_type) :: kernel1_base
  type(array_type) :: bias1_opt
  type(array_type) :: bias1_base
  type(array_type) :: kernel2_opt
  type(array_type) :: kernel2_base
  type(array_type) :: bias2_opt
  type(array_type) :: bias2_base
  type(array_type), pointer :: z1
  type(array_type), pointer :: z1b
  type(array_type), pointer :: a1
  type(array_type), pointer :: z2
  type(array_type), pointer :: z2b
  type(array_type), pointer :: loss
  type(array_type), pointer :: z1_base
  type(array_type), pointer :: z1b_base
  type(array_type), pointer :: a1_base
  type(array_type), pointer :: z2_base
  type(array_type), pointer :: z2b_base
  type(array_type), pointer :: loss_base
  type(relu_actv_type) :: relu
  real(real32) :: forward_reference_time
  real(real32) :: backward_reference_time
  real(real32) :: total_reference_time
  real(real32) :: forward_optimised_time
  real(real32) :: backward_optimised_time
  real(real32) :: total_optimised_time
  real(real32) :: speedup
  real(real32) :: backward_speedup
  real(real32) :: max_deviation
  integer :: n
  integer(int64) :: count_rate
  integer(int64) :: tick_start
  integer(int64) :: tick_stop

  call system_clock(count_rate = count_rate)

  allocate(input_data(input_h, input_w, input_c, batch_size))
  allocate(kernel1_ref(kernel_size, kernel_size, input_c, filters1))
  allocate(bias1_ref(filters1))
  allocate(kernel2_ref(kernel_size, kernel_size, filters1, filters2))
  allocate(bias2_ref(filters2))
  allocate(conv1_ref(conv1_h, conv1_w, filters1, batch_size))
  allocate(act1_ref(conv1_h, conv1_w, filters1, batch_size))
  allocate(conv2_ref(conv2_h, conv2_w, filters2, batch_size))
  allocate(grad_conv2_ref(conv2_h, conv2_w, filters2, batch_size))
  allocate(grad_act1_ref(conv1_h, conv1_w, filters1, batch_size))
  allocate(grad_conv1_ref(conv1_h, conv1_w, filters1, batch_size))
  allocate(grad_input_ref(input_h, input_w, input_c, batch_size))
  allocate(grad_kernel1_ref(kernel_size, kernel_size, input_c, filters1))
  allocate(grad_bias1_ref(filters1))
  allocate(grad_kernel2_ref(kernel_size, kernel_size, filters1, filters2))
  allocate(grad_bias2_ref(filters2))

  call initialise_conv_autodiff_input(input_data)
  call initialise_conv_autodiff_parameters( &
       kernel1_ref, bias1_ref, kernel2_ref, bias2_ref)

  call input_opt%allocate( &
       array_shape = [input_h, input_w, input_c, batch_size], &
       source = 0._real32)
  call input_opt%set(input_data)
  call input_opt%set_requires_grad(.true.)
  call input_base%allocate( &
       array_shape = [input_h, input_w, input_c, batch_size], &
       source = 0._real32)
  call input_base%set(input_data)
  call input_base%set_requires_grad(.true.)
  call kernel1_opt%allocate( &
       array_shape = [kernel_size, kernel_size, input_c, filters1, 1], &
       source = 0._real32)
  call kernel1_opt%set(reshape( &
       kernel1_ref, [kernel_size, kernel_size, input_c, filters1, 1]))
  kernel1_opt%is_sample_dependent = .false.
  call kernel1_opt%set_requires_grad(.true.)
  call bias1_opt%allocate(array_shape = [filters1, 1], source = 0._real32)
  call bias1_opt%set(reshape(bias1_ref, [filters1, 1]))
  bias1_opt%is_sample_dependent = .false.
  call bias1_opt%set_requires_grad(.true.)
  call kernel1_base%allocate( &
       array_shape = [kernel_size, kernel_size, input_c, filters1, 1], &
       source = 0._real32)
  call kernel1_base%set(reshape( &
       kernel1_ref, [kernel_size, kernel_size, input_c, filters1, 1]))
  kernel1_base%is_sample_dependent = .false.
  call kernel1_base%set_requires_grad(.true.)
  call bias1_base%allocate(array_shape = [filters1, 1], source = 0._real32)
  call bias1_base%set(reshape(bias1_ref, [filters1, 1]))
  bias1_base%is_sample_dependent = .false.
  call bias1_base%set_requires_grad(.true.)
  call kernel2_opt%allocate( &
       array_shape = [kernel_size, kernel_size, filters1, filters2, 1], &
       source = 0._real32)
  call kernel2_opt%set(reshape( &
       kernel2_ref, [kernel_size, kernel_size, filters1, filters2, 1]))
  kernel2_opt%is_sample_dependent = .false.
  call kernel2_opt%set_requires_grad(.true.)
  call bias2_opt%allocate(array_shape = [filters2, 1], source = 0._real32)
  call bias2_opt%set(reshape(bias2_ref, [filters2, 1]))
  bias2_opt%is_sample_dependent = .false.
  call bias2_opt%set_requires_grad(.true.)
  call kernel2_base%allocate( &
       array_shape = [kernel_size, kernel_size, filters1, filters2, 1], &
       source = 0._real32)
  call kernel2_base%set(reshape( &
       kernel2_ref, [kernel_size, kernel_size, filters1, filters2, 1]))
  kernel2_base%is_sample_dependent = .false.
  call kernel2_base%set_requires_grad(.true.)
  call bias2_base%allocate(array_shape = [filters2, 1], source = 0._real32)
  call bias2_base%set(reshape(bias2_ref, [filters2, 1]))
  bias2_base%is_sample_dependent = .false.
  call bias2_base%set_requires_grad(.true.)
  call relu%reset()

  grad_conv2_ref = 1._real32 / real(size(grad_conv2_ref), real32)

  call reference_conv2d_forward(input_data, kernel1_ref, bias1_ref, conv1_ref)
  act1_ref = max(conv1_ref, 0._real32)
  call reference_conv2d_forward(act1_ref, kernel2_ref, bias2_ref, conv2_ref)
  call reference_conv2d_backward( &
       act1_ref, kernel2_ref, grad_conv2_ref, grad_act1_ref, &
       grad_kernel2_ref, grad_bias2_ref)
  grad_conv1_ref = 0._real32
  where(conv1_ref .gt. 0._real32)
     grad_conv1_ref = grad_act1_ref
  end where
  call reference_conv2d_backward( &
       input_data, kernel1_ref, grad_conv1_ref, grad_input_ref, &
       grad_kernel1_ref, grad_bias1_ref)

  call run_optimised_step( &
       input_opt, kernel1_opt, bias1_opt, kernel2_opt, bias2_opt, &
       input_data, output_opt, grad_input_opt, grad_layer1_opt, &
       grad_layer2_opt)

  max_deviation = 0._real32
  max_deviation = max(max_deviation, maxval(abs( &
       reshape(conv2_ref, shape(output_opt)) - output_opt)))
  max_deviation = max(max_deviation, maxval(abs( &
       reshape(grad_input_ref, shape(grad_input_opt)) - grad_input_opt)))
  max_deviation = max(max_deviation, maxval(abs( &
       [reshape(grad_kernel1_ref, [size(grad_kernel1_ref)]), grad_bias1_ref] - &
       grad_layer1_opt)))
  max_deviation = max(max_deviation, maxval(abs( &
       [reshape(grad_kernel2_ref, [size(grad_kernel2_ref)]), grad_bias2_ref] - &
       grad_layer2_opt)))

  forward_reference_time = 0._real32
  backward_reference_time = 0._real32
  do n = 1, num_iterations
     call system_clock(tick_start)
     call input_base%zero_grad()
     call kernel1_base%zero_grad()
     call bias1_base%zero_grad()
     call kernel2_base%zero_grad()
     call bias2_base%zero_grad()
     call input_base%set(input_data)
     z1_base => reference_conv2d_original(input_base, kernel1_base, [1, 1], [1, 1])
     z1b_base => add_bias( &
          z1_base, bias1_base, dim = 3, dim_act_on_shape = .true.)
     a1_base => relu%apply(z1b_base)
     z2_base => reference_conv2d_original(a1_base, kernel2_base, [1, 1], [1, 1])
     z2b_base => add_bias( &
          z2_base, bias2_base, dim = 3, dim_act_on_shape = .true.)
     loss_base => mean(z2b_base)
     call system_clock(tick_stop)
     forward_reference_time = forward_reference_time + &
          elapsed_seconds(tick_start, tick_stop, count_rate)

     call system_clock(tick_start)
     call loss_base%grad_reverse(reset_graph = .true.)
     call system_clock(tick_stop)
     backward_reference_time = backward_reference_time + &
          elapsed_seconds(tick_start, tick_stop, count_rate)
  end do
  total_reference_time = forward_reference_time + backward_reference_time

  forward_optimised_time = 0._real32
  backward_optimised_time = 0._real32
  do n = 1, num_iterations
     call input_opt%zero_grad()
     call kernel1_opt%zero_grad()
     call bias1_opt%zero_grad()
     call kernel2_opt%zero_grad()
     call bias2_opt%zero_grad()
     call input_opt%set(input_data)

     call system_clock(tick_start)
     z1 => conv2d(input_opt, kernel1_opt, [1, 1], [1, 1])
     z1b => add_bias(z1, bias1_opt, dim = 3, dim_act_on_shape = .true.)
     a1 => relu%apply(z1b)
     z2 => conv2d(a1, kernel2_opt, [1, 1], [1, 1])
     z2b => add_bias(z2, bias2_opt, dim = 3, dim_act_on_shape = .true.)
     loss => mean(z2b)
     call system_clock(tick_stop)
     forward_optimised_time = forward_optimised_time + &
          elapsed_seconds(tick_start, tick_stop, count_rate)

     call system_clock(tick_start)
     call loss%grad_reverse(reset_graph = .true.)
     call system_clock(tick_stop)
     backward_optimised_time = backward_optimised_time + &
          elapsed_seconds(tick_start, tick_stop, count_rate)
  end do
  total_optimised_time = forward_optimised_time + backward_optimised_time
  speedup = total_reference_time / total_optimised_time
  backward_speedup = backward_reference_time / backward_optimised_time

  write(*,'(A)') 'ATHENA Convolution Autodiff Benchmark'
  write(*,'(A)') '-------------------------------------'
  write(*,'(A,F10.6,A)') 'Forward pass time: ', forward_optimised_time, ' s'
  write(*,'(A,F10.6,A)') 'Backward pass time: ', backward_optimised_time, ' s'
  write(*,'(A,F10.6,A)') 'Total training time: ', total_optimised_time, ' s'
  write(*,'(A,F10.6,A)') 'Baseline forward time: ', forward_reference_time, ' s'
  write(*,'(A,F10.6,A)') 'Baseline backward time: ', backward_reference_time, ' s'
  write(*,'(A,F10.6,A)') 'Baseline total time: ', total_reference_time, ' s'
  write(*,'(A,F10.6,A)') 'Backward speedup vs baseline: ', &
       backward_speedup, 'x'
  write(*,'(A,F10.6,A)') 'Speedup vs baseline: ', speedup, 'x'
  write(*,'(A,ES12.4)') 'Max gradient deviation: ', max_deviation

contains

!-------------------------------------------------------------------------------
  subroutine run_optimised_step( &
       input_values, kernel1_values, bias1_values, kernel2_values, &
       bias2_values, batch_input, output, grad_input, grad_layer1, &
       grad_layer2)
    implicit none

    type(array_type), intent(inout) :: input_values
    type(array_type), intent(inout) :: kernel1_values
    type(array_type), intent(inout) :: bias1_values
    type(array_type), intent(inout) :: kernel2_values
    type(array_type), intent(inout) :: bias2_values
    real(real32), dimension(:,:,:,:), intent(in) :: batch_input
    real(real32), allocatable, dimension(:,:), intent(out) :: output
    real(real32), allocatable, dimension(:,:), intent(out) :: grad_input
    real(real32), allocatable, dimension(:), intent(out) :: grad_layer1
    real(real32), allocatable, dimension(:), intent(out) :: grad_layer2

    type(array_type), pointer :: layer_z1
    type(array_type), pointer :: layer_z1b
    type(array_type), pointer :: layer_a1
    type(array_type), pointer :: layer_z2
    type(array_type), pointer :: layer_z2b
    type(array_type), pointer :: loss_ptr
    type(relu_actv_type) :: relu_local

    call relu_local%reset()
    call input_values%zero_grad()
    call kernel1_values%zero_grad()
    call bias1_values%zero_grad()
    call kernel2_values%zero_grad()
    call bias2_values%zero_grad()
    call input_values%set(batch_input)
    layer_z1 => conv2d(input_values, kernel1_values, [1, 1], [1, 1])
    layer_z1b => add_bias( &
         layer_z1, bias1_values, dim = 3, dim_act_on_shape = .true.)
    layer_a1 => relu_local%apply(layer_z1b)
    layer_z2 => conv2d(layer_a1, kernel2_values, [1, 1], [1, 1])
    layer_z2b => add_bias( &
         layer_z2, bias2_values, dim = 3, dim_act_on_shape = .true.)
    loss_ptr => mean(layer_z2b)
    call loss_ptr%grad_reverse(reset_graph = .true.)

    allocate(output, source = layer_z2b%val)
    allocate(grad_input, source = input_values%grad%val)
    allocate(grad_layer1, source = [ &
         kernel1_values%grad%val(:,1), bias1_values%grad%val(:,1)])
    allocate(grad_layer2, source = [ &
         kernel2_values%grad%val(:,1), bias2_values%grad%val(:,1)])

  end subroutine run_optimised_step
  pure function elapsed_seconds( &
       start_tick, stop_tick, tick_rate) result(value)
    implicit none

    integer(int64), intent(in) :: start_tick
    integer(int64), intent(in) :: stop_tick
    integer(int64), intent(in) :: tick_rate
    real(real32) :: value

    value = real(stop_tick - start_tick, real32) / &
         real(tick_rate, real32)

  end function elapsed_seconds

end program conv_autodiff_benchmark
