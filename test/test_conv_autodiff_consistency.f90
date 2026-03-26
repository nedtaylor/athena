program test_conv_autodiff_consistency
  use athena, only: conv2d_layer_type
  use athena__conv_autodiff_reference, only: &
       initialise_conv_autodiff_input, &
       initialise_conv_autodiff_parameters, &
       reference_conv2d_forward, &
       reference_conv2d_backward
  use coreutils, only: real32
  use diffstruc, only: array_type, mean
  implicit none

  integer, parameter :: input_h = 32
  integer, parameter :: input_w = 32
  integer, parameter :: input_c = 3
  integer, parameter :: batch_size = 4
  integer, parameter :: filters1 = 16
  integer, parameter :: filters2 = 32
  integer, parameter :: kernel_size = 3
  integer, parameter :: conv1_h = input_h - kernel_size + 1
  integer, parameter :: conv1_w = input_w - kernel_size + 1
  integer, parameter :: conv2_h = conv1_h - kernel_size + 1
  integer, parameter :: conv2_w = conv1_w - kernel_size + 1
  real(real32), parameter :: tolerance = 1.e-10_real32

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
  real(real32), allocatable, dimension(:) :: params1
  real(real32), allocatable, dimension(:) :: params2
  type(array_type) :: input_array(1,1)
  type(conv2d_layer_type) :: layer1
  type(conv2d_layer_type) :: layer2
  real(real32) :: max_forward_deviation
  real(real32) :: max_input_grad_deviation
  real(real32) :: max_weight_grad_deviation
  real(real32) :: max_deviation
  logical :: success

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

  call reference_conv2d_forward(input_data, kernel1_ref, bias1_ref, conv1_ref)
  act1_ref = max(conv1_ref, 0._real32)
  call reference_conv2d_forward(act1_ref, kernel2_ref, bias2_ref, conv2_ref)
  grad_conv2_ref = 1._real32 / real(size(grad_conv2_ref), real32)
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

  layer1 = conv2d_layer_type( &
       input_shape = [input_h, input_w, input_c], &
       num_filters = filters1, &
       kernel_size = kernel_size, &
       activation = 'relu')
  layer2 = conv2d_layer_type( &
       num_filters = filters2, &
       kernel_size = kernel_size, &
       activation = 'linear')
  call layer2%init(layer1%output_shape)

  call pack_conv_params(kernel1_ref, bias1_ref, params1)
  call pack_conv_params(kernel2_ref, bias2_ref, params2)
  call layer1%set_params(params1)
  call layer2%set_params(params2)

  call input_array(1,1)%allocate( &
       array_shape = [input_h, input_w, input_c, batch_size], &
       source = 0._real32)
  call input_array(1,1)%set(input_data)
  call input_array(1,1)%set_requires_grad(.true.)

  call run_optimised_step( &
       layer1, layer2, input_array, input_data, output_opt, grad_input_opt, &
       grad_layer1_opt, grad_layer2_opt)

  max_forward_deviation = maxval(abs( &
       reshape(conv2_ref, shape(output_opt)) - output_opt))
  max_input_grad_deviation = maxval(abs( &
       reshape(grad_input_ref, shape(grad_input_opt)) - grad_input_opt))
  max_weight_grad_deviation = max( &
       maxval(abs( &
            [reshape(grad_kernel1_ref, [size(grad_kernel1_ref)]), grad_bias1_ref] - &
            grad_layer1_opt)), &
       maxval(abs( &
            [reshape(grad_kernel2_ref, [size(grad_kernel2_ref)]), grad_bias2_ref] - &
            grad_layer2_opt)))
  max_deviation = max(max_forward_deviation, max_input_grad_deviation)
  max_deviation = max(max_deviation, max_weight_grad_deviation)
  success = max_deviation .lt. tolerance

  write(*,'(A,ES12.4)') 'Max forward deviation: ', max_forward_deviation
  write(*,'(A,ES12.4)') 'Max input-gradient deviation: ', &
       max_input_grad_deviation
  write(*,'(A,ES12.4)') 'Max weight-gradient deviation: ', &
       max_weight_grad_deviation
  write(*,'(A,ES12.4)') 'Max deviation: ', max_deviation

  if(.not. success)then
     write(0,*) 'test_conv_autodiff_consistency failed tolerance check'
     stop 1
  end if

  write(*,*) 'test_conv_autodiff_consistency passed all tests'

contains

!-------------------------------------------------------------------------------
  subroutine pack_conv_params(kernel, bias, params)
    implicit none

    real(real32), dimension(:,:,:,:), intent(in) :: kernel
    real(real32), dimension(:), intent(in) :: bias
    real(real32), allocatable, dimension(:), intent(out) :: params

    allocate(params(size(kernel) + size(bias)))
    params = [reshape(kernel, [size(kernel)]), bias]

  end subroutine pack_conv_params
!-------------------------------------------------------------------------------
  subroutine run_optimised_step( &
       conv_layer1, conv_layer2, input_values, batch_input, output, &
       grad_input, &
       grad_layer1, grad_layer2)
    implicit none

    type(conv2d_layer_type), intent(inout) :: conv_layer1
    type(conv2d_layer_type), intent(inout) :: conv_layer2
    type(array_type), dimension(:,:), intent(inout) :: input_values
    real(real32), dimension(:,:,:,:), intent(in) :: batch_input
    real(real32), allocatable, dimension(:,:), intent(out) :: output
    real(real32), allocatable, dimension(:,:), intent(out) :: grad_input
    real(real32), allocatable, dimension(:), intent(out) :: grad_layer1
    real(real32), allocatable, dimension(:), intent(out) :: grad_layer2

    type(array_type), pointer :: loss

    call conv_layer1%set_gradients(0._real32)
    call conv_layer2%set_gradients(0._real32)
    call input_values(1,1)%zero_grad()
    call input_values(1,1)%set(batch_input)
    call conv_layer1%forward(input_values)
    call conv_layer2%forward(conv_layer1%output)
    loss => mean(conv_layer2%output(1,1))
    call loss%grad_reverse(reset_graph = .true.)

    allocate(output, source = conv_layer2%output(1,1)%val)
    allocate(grad_input, source = input_values(1,1)%grad%val)
    allocate(grad_layer1, source = conv_layer1%get_gradients())
    allocate(grad_layer2, source = conv_layer2%get_gradients())

  end subroutine run_optimised_step

end program test_conv_autodiff_consistency
