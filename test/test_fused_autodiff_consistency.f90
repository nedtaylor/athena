program test_fused_autodiff_consistency
  use athena__diffstruc_extd, only: &
       conv2d, conv2d_relu, full_relu, full_tanh, full_softmax, add_bias, &
       softmax
  use coreutils, only: real32
  use diffstruc, only: array_type, matmul, mean, tanh, max, operator(*), &
       operator(+)
  implicit none

  real(real32), parameter :: tolerance = 1.e-6_real32
  character(len=32) :: case_name
  character(len=512) :: executable_name
  integer :: exitstat
  logical :: success

  success = .true.
  if(command_argument_count() .eq. 0)then
     call get_command_argument(0, executable_name)
     call run_case_command(executable_name, 'full_relu', success)
     call run_case_command(executable_name, 'full_tanh', success)
     call run_case_command(executable_name, 'full_softmax', success)
  else
     call get_command_argument(1, case_name)
     select case(trim(case_name))
     case('full_relu')
        call check_full_relu(success)
     case('full_tanh')
        call check_full_tanh(success)
     case('full_softmax')
        call check_full_softmax(success)
     case default
        write(0,'(A,1X,A)') 'Unknown fused consistency case:', trim(case_name)
        success = .false.
     end select
  end if

  if(.not. success) stop 1
  write(*,*) 'test_fused_autodiff_consistency passed all tests'

contains

  subroutine run_case_command(executable_name, case_name, test_success)
    implicit none

    character(*), intent(in) :: executable_name
    character(*), intent(in) :: case_name
    logical, intent(inout) :: test_success
    integer :: command_status

    call execute_command_line( &
         trim(executable_name)//' '//trim(case_name), exitstat = command_status)
    if(command_status .ne. 0) test_success = .false.

  end subroutine run_case_command

  subroutine check_full_relu(test_success)
    implicit none

    logical, intent(inout) :: test_success

    call check_full_case('full_relu', test_success)

  end subroutine check_full_relu

  subroutine check_full_tanh(test_success)
    implicit none

    logical, intent(inout) :: test_success

    call check_full_case('full_tanh', test_success)

  end subroutine check_full_tanh

  subroutine check_full_softmax(test_success)
    implicit none

    logical, intent(inout) :: test_success

    call check_full_case('full_softmax', test_success)

  end subroutine check_full_softmax

  subroutine check_full_case(case_name, test_success)
    implicit none

    character(*), intent(in) :: case_name
    logical, intent(inout) :: test_success

    integer, parameter :: num_inputs = 12
    integer, parameter :: num_outputs = 7
    integer, parameter :: batch_size = 5

    real(real32), dimension(num_inputs, batch_size) :: input_data
    real(real32), dimension(num_outputs, num_inputs) :: weight_data
    real(real32), dimension(num_outputs) :: bias_data
    real(real32), dimension(num_outputs, batch_size) :: target_data

    type(array_type) :: input_split
    type(array_type) :: weight_split
    type(array_type) :: bias_split
    type(array_type) :: target_split
    type(array_type) :: input_fused
    type(array_type) :: weight_fused
    type(array_type) :: bias_fused
    type(array_type) :: target_fused
    type(array_type), pointer :: output_split
    type(array_type), pointer :: output_fused
    type(array_type), pointer :: z_split
    type(array_type), pointer :: zb_split
    type(array_type), pointer :: loss_split
    type(array_type), pointer :: loss_fused
    real(real32) :: max_output_error
    real(real32) :: max_input_error
    real(real32) :: max_weight_error
    real(real32) :: max_bias_error

    call initialise_full_case(input_data, weight_data, bias_data, target_data)
    call setup_full_arrays(&
         input_data, weight_data, bias_data, target_data, &
         input_split, weight_split, bias_split, target_split)
    call setup_full_arrays(&
         input_data, weight_data, bias_data, target_data, &
         input_fused, weight_fused, bias_fused, target_fused)

    z_split => matmul(weight_split, input_split)
    zb_split => z_split + bias_split
    select case(case_name)
    case('full_relu')
       output_split => max(zb_split, 0._real32)
       output_fused => full_relu(input_fused, weight_fused, bias_fused)
    case('full_tanh')
       output_split => tanh(zb_split)
       output_fused => full_tanh(input_fused, weight_fused, bias_fused)
    case('full_softmax')
       output_split => softmax(zb_split, dim = 2)
       output_fused => full_softmax(input_fused, weight_fused, bias_fused)
    end select

    loss_split => mean(output_split * target_split)
    loss_fused => mean(output_fused * target_fused)
    call loss_split%grad_reverse(reset_graph = .true.)
    call loss_fused%grad_reverse(reset_graph = .true.)

    max_output_error = maxval(abs(output_split%val - output_fused%val))
    max_input_error = maxval(abs(input_split%grad%val - input_fused%grad%val))
    max_weight_error = maxval( &
         abs(weight_split%grad%val - weight_fused%grad%val))
    max_bias_error = maxval(abs(bias_split%grad%val - bias_fused%grad%val))

    write(*,'(A,1X,A)') 'Checking', trim(case_name)
    write(*,'(A,ES12.4)') '  max output error: ', max_output_error
    write(*,'(A,ES12.4)') '  max input-grad error: ', max_input_error
    write(*,'(A,ES12.4)') '  max weight-grad error: ', max_weight_error
    write(*,'(A,ES12.4)') '  max bias-grad error: ', max_bias_error

    if(max(max_output_error, max_input_error) .gt. tolerance .or. &
         max(max_weight_error, max_bias_error) .gt. tolerance)then
       write(0,'(A,1X,A)') 'Fused consistency failed for', trim(case_name)
       test_success = .false.
    end if

  end subroutine check_full_case

  subroutine initialise_full_case( &
       input_data, weight_data, bias_data, target_data)
    implicit none

    real(real32), dimension(:,:), intent(out) :: input_data
    real(real32), dimension(:,:), intent(out) :: weight_data
    real(real32), dimension(:), intent(out) :: bias_data
    real(real32), dimension(:,:), intent(out) :: target_data

    integer :: i, j

    do j = 1, size(input_data, 2)
       do i = 1, size(input_data, 1)
          input_data(i,j) = 0.01_real32 * real(i + 2 * j, real32)
       end do
    end do
    do j = 1, size(weight_data, 2)
       do i = 1, size(weight_data, 1)
          weight_data(i,j) = 0.02_real32 * real(i - j, real32)
       end do
    end do
    do i = 1, size(bias_data)
       bias_data(i) = -0.05_real32 + 0.01_real32 * real(i, real32)
    end do
    do j = 1, size(target_data, 2)
       do i = 1, size(target_data, 1)
          target_data(i,j) = 0.03_real32 * real(i + j, real32)
       end do
    end do

  end subroutine initialise_full_case

  subroutine setup_full_arrays(&
       input_data, weight_data, bias_data, target_data, &
       input_array, weight_array, bias_array, target_array)
    implicit none

    real(real32), dimension(:,:), intent(in) :: input_data
    real(real32), dimension(:,:), intent(in) :: weight_data
    real(real32), dimension(:), intent(in) :: bias_data
    real(real32), dimension(:,:), intent(in) :: target_data
    type(array_type), intent(out) :: input_array
    type(array_type), intent(out) :: weight_array
    type(array_type), intent(out) :: bias_array
    type(array_type), intent(out) :: target_array

    call input_array%allocate( &
         array_shape = [size(input_data,1), size(input_data,2)], &
         source = 0._real32)
    call input_array%set(input_data)
    call input_array%set_requires_grad(.true.)

    call weight_array%allocate(&
         array_shape = [size(weight_data,1), size(weight_data,2), 1], &
         source = 0._real32)
    call weight_array%set(weight_data)
    weight_array%is_sample_dependent = .false.
    call weight_array%set_requires_grad(.true.)

    call bias_array%allocate( &
         array_shape = [size(bias_data), 1], source = 0._real32)
    call bias_array%set(reshape(bias_data, [size(bias_data), 1]))
    bias_array%is_sample_dependent = .false.
    call bias_array%set_requires_grad(.true.)

    call target_array%allocate( &
         array_shape = [size(target_data,1), size(target_data,2)], &
         source = 0._real32)
    call target_array%set(target_data)
    call target_array%set_requires_grad(.false.)

  end subroutine setup_full_arrays

  subroutine check_conv2d_relu(test_success)
    implicit none

    logical, intent(inout) :: test_success

    integer, parameter :: input_h = 9
    integer, parameter :: input_w = 8
    integer, parameter :: input_c = 3
    integer, parameter :: filters = 4
    integer, parameter :: kernel_h = 3
    integer, parameter :: kernel_w = 2
    integer, parameter :: batch_size = 3
    integer, parameter :: output_h = input_h - kernel_h + 1
    integer, parameter :: output_w = input_w - kernel_w + 1

    real(real32), dimension(input_h, input_w, input_c, batch_size) :: input_data
    real(real32), dimension(kernel_h, kernel_w, input_c, filters) :: kernel_data
    real(real32), dimension(filters) :: bias_data
    real(real32), dimension(output_h, output_w, filters, batch_size) :: &
         target_data

    type(array_type) :: input_split
    type(array_type) :: kernel_split
    type(array_type) :: bias_split
    type(array_type) :: target_split
    type(array_type) :: input_fused
    type(array_type) :: kernel_fused
    type(array_type) :: bias_fused
    type(array_type) :: target_fused
    type(array_type), pointer :: conv_split
    type(array_type), pointer :: bias_split_ptr
    type(array_type), pointer :: output_split
    type(array_type), pointer :: output_fused
    type(array_type), pointer :: loss_split
    type(array_type), pointer :: loss_fused
    real(real32) :: max_output_error
    real(real32) :: max_input_error
    real(real32) :: max_kernel_error
    real(real32) :: max_bias_error

    call initialise_conv_case(input_data, kernel_data, bias_data, target_data)
    call setup_conv_arrays(&
         input_data, kernel_data, bias_data, target_data, &
         input_split, kernel_split, bias_split, target_split)
    call setup_conv_arrays(&
         input_data, kernel_data, bias_data, target_data, &
         input_fused, kernel_fused, bias_fused, target_fused)

    conv_split => conv2d(input_split, kernel_split, [1,1], [1,1])
    bias_split_ptr => add_bias( &
         conv_split, bias_split, dim = 3, dim_act_on_shape = .true.)
    output_split => max(bias_split_ptr, 0._real32)
    output_fused => conv2d_relu( &
         input_fused, kernel_fused, bias_fused, [1,1], [1,1])

    loss_split => mean(output_split * target_split)
    loss_fused => mean(output_fused * target_fused)
    call loss_split%grad_reverse(reset_graph = .true.)
    call loss_fused%grad_reverse(reset_graph = .true.)

    max_output_error = maxval(abs(output_split%val - output_fused%val))
    max_input_error = maxval(abs(input_split%grad%val - input_fused%grad%val))
    max_kernel_error = maxval( &
         abs(kernel_split%grad%val - kernel_fused%grad%val))
    max_bias_error = maxval(abs(bias_split%grad%val - bias_fused%grad%val))

    write(*,'(A)') 'Checking conv2d_relu'
    write(*,'(A,ES12.4)') '  max output error: ', max_output_error
    write(*,'(A,ES12.4)') '  max input-grad error: ', max_input_error
    write(*,'(A,ES12.4)') '  max kernel-grad error: ', max_kernel_error
    write(*,'(A,ES12.4)') '  max bias-grad error: ', max_bias_error

    if(max(max_output_error, max_input_error) .gt. tolerance .or. &
         max(max_kernel_error, max_bias_error) .gt. tolerance)then
       write(0,*) 'Fused consistency failed for conv2d_relu'
       test_success = .false.
    end if

  end subroutine check_conv2d_relu

  subroutine initialise_conv_case( &
       input_data, kernel_data, bias_data, target_data)
    implicit none

    real(real32), dimension(:,:,:,:), intent(out) :: input_data
    real(real32), dimension(:,:,:,:), intent(out) :: kernel_data
    real(real32), dimension(:), intent(out) :: bias_data
    real(real32), dimension(:,:,:,:), intent(out) :: target_data

    integer :: i, j, c, k, s

    do s = 1, size(input_data, 4)
       do c = 1, size(input_data, 3)
          do j = 1, size(input_data, 2)
             do i = 1, size(input_data, 1)
                input_data(i,j,c,s) = 0.005_real32 * &
                     real(i + j + 2*c + s, real32)
             end do
          end do
       end do
    end do
    do k = 1, size(kernel_data, 4)
       do c = 1, size(kernel_data, 3)
          do j = 1, size(kernel_data, 2)
             do i = 1, size(kernel_data, 1)
                kernel_data(i,j,c,k) = 0.01_real32 * real(i - j + c + k, real32)
             end do
          end do
       end do
    end do
    do i = 1, size(bias_data)
       bias_data(i) = -0.03_real32 + 0.015_real32 * real(i, real32)
    end do
    do s = 1, size(target_data, 4)
       do k = 1, size(target_data, 3)
          do j = 1, size(target_data, 2)
             do i = 1, size(target_data, 1)
                target_data(i,j,k,s) = 0.004_real32 * &
                     real(i + 2*j + k + s, real32)
             end do
          end do
       end do
    end do

  end subroutine initialise_conv_case

  subroutine setup_conv_arrays(&
       input_data, kernel_data, bias_data, target_data, &
       input_array, kernel_array, bias_array, target_array)
    implicit none

    real(real32), dimension(:,:,:,:), intent(in) :: input_data
    real(real32), dimension(:,:,:,:), intent(in) :: kernel_data
    real(real32), dimension(:), intent(in) :: bias_data
    real(real32), dimension(:,:,:,:), intent(in) :: target_data
    type(array_type), intent(out) :: input_array
    type(array_type), intent(out) :: kernel_array
    type(array_type), intent(out) :: bias_array
    type(array_type), intent(out) :: target_array

    call input_array%allocate( &
         array_shape = shape(input_data), source = 0._real32)
    call input_array%set(input_data)
    call input_array%set_requires_grad(.true.)

    call kernel_array%allocate(&
         array_shape = [size(kernel_data,1), size(kernel_data,2), &
              size(kernel_data,3), size(kernel_data,4), 1], source = 0._real32)
    call kernel_array%set(kernel_data)
    kernel_array%is_sample_dependent = .false.
    call kernel_array%set_requires_grad(.true.)

    call bias_array%allocate( &
         array_shape = [size(bias_data), 1], source = 0._real32)
    call bias_array%set(reshape(bias_data, [size(bias_data), 1]))
    bias_array%is_sample_dependent = .false.
    call bias_array%set_requires_grad(.true.)

    call target_array%allocate( &
         array_shape = shape(target_data), source = 0._real32)
    call target_array%set(target_data)
    call target_array%set_requires_grad(.false.)

  end subroutine setup_conv_arrays

end program test_fused_autodiff_consistency
