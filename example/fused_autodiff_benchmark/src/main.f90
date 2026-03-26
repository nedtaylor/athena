program fused_autodiff_benchmark
  use athena__diffstruc_extd, only: full_relu, full_tanh, full_softmax, softmax
  use coreutils, only: real32
  use diffstruc, only: array_type, matmul, mean, tanh, max, operator(*), &
       operator(+)
  implicit none

  character(len=32) :: case_name
  character(len=512) :: executable_name

  if(command_argument_count() .eq. 0)then
     call get_command_argument(0, executable_name)
     call execute_command_line(trim(executable_name)//' full_relu')
     call execute_command_line(trim(executable_name)//' full_tanh')
     call execute_command_line(trim(executable_name)//' full_softmax')
  else
     call get_command_argument(1, case_name)
     select case(trim(case_name))
     case('full_relu')
        call benchmark_full_case('full_relu', 12, 7, 5)
     case('full_tanh')
        call benchmark_full_case('full_tanh', 12, 7, 5)
     case('full_softmax')
        call benchmark_full_case('full_softmax', 12, 7, 5)
     case default
        write(0,'(A,1X,A)') 'Unknown fused benchmark case:', trim(case_name)
        stop 1
     end select
  end if

contains

  subroutine benchmark_full_case( &
       benchmark_name, num_inputs, num_outputs, batch_size)
    implicit none

    character(*), intent(in) :: benchmark_name
    integer, intent(in) :: num_inputs
    integer, intent(in) :: num_outputs
    integer, intent(in) :: batch_size

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
    type(array_type), pointer :: z_split
    type(array_type), pointer :: zb_split
    type(array_type), pointer :: output_split
    type(array_type), pointer :: output_fused
    type(array_type), pointer :: loss_split
    type(array_type), pointer :: loss_fused
    integer :: clock_rate
    integer :: clock_start
    integer :: clock_mid
    integer :: clock_end
    integer :: iter
    integer, parameter :: num_repeats = 1000
    real(real32), allocatable, dimension(:,:) :: split_output_values
    real(real32), allocatable, dimension(:,:) :: split_input_grad
    real(real32), allocatable, dimension(:,:) :: split_weight_grad
    real(real32), allocatable, dimension(:,:) :: split_bias_grad
    real(real32) :: split_forward
    real(real32) :: split_backward
    real(real32) :: fused_forward
    real(real32) :: fused_backward
    real(real32) :: max_output_error
    real(real32) :: max_input_error
    real(real32) :: max_weight_error
    real(real32) :: max_bias_error

    call initialise_full_case(input_data, weight_data, bias_data, target_data)
    call system_clock(count_rate = clock_rate)
    split_forward = 0._real32
    split_backward = 0._real32
    do iter = 1, num_repeats
       call setup_full_array_values(&
            input_data, weight_data, bias_data, target_data, &
            input_split, weight_split, bias_split, target_split)
       call system_clock(clock_start)
       z_split => matmul(weight_split, input_split)
       zb_split => z_split + bias_split
       select case(benchmark_name)
       case('full_relu')
          output_split => max(zb_split, 0._real32)
       case('full_tanh')
          output_split => tanh(zb_split)
       case default
          output_split => softmax(zb_split, dim = 2)
       end select
       loss_split => mean(output_split * target_split)
       call system_clock(clock_mid)
       call loss_split%grad_reverse(reset_graph = .true.)
       call system_clock(clock_end)
       split_forward = split_forward + &
            real(clock_mid - clock_start, real32) / real(clock_rate, real32)
       split_backward = split_backward + &
            real(clock_end - clock_mid, real32) / real(clock_rate, real32)
    end do

    allocate(split_output_values, source = output_split%val)
    allocate(split_input_grad, source = input_split%grad%val)
    allocate(split_weight_grad, source = weight_split%grad%val)
    allocate(split_bias_grad, source = bias_split%grad%val)

    fused_forward = 0._real32
    fused_backward = 0._real32
    do iter = 1, num_repeats
       call setup_full_array_values(&
            input_data, weight_data, bias_data, target_data, &
            input_fused, weight_fused, bias_fused, target_fused)
       call system_clock(clock_start)
       select case(benchmark_name)
       case('full_relu')
          output_fused => full_relu(input_fused, weight_fused, bias_fused)
       case('full_tanh')
          output_fused => full_tanh(input_fused, weight_fused, bias_fused)
       case default
          output_fused => full_softmax(input_fused, weight_fused, bias_fused)
       end select
       loss_fused => mean(output_fused * target_fused)
       call system_clock(clock_mid)
       call loss_fused%grad_reverse(reset_graph = .true.)
       call system_clock(clock_end)
       fused_forward = fused_forward + &
            real(clock_mid - clock_start, real32) / real(clock_rate, real32)
       fused_backward = fused_backward + &
            real(clock_end - clock_mid, real32) / real(clock_rate, real32)
    end do

    max_output_error = maxval(abs(split_output_values - output_fused%val))
    max_input_error = maxval(abs(split_input_grad - input_fused%grad%val))
    max_weight_error = maxval(abs(split_weight_grad - weight_fused%grad%val))
    max_bias_error = maxval(abs(split_bias_grad - bias_fused%grad%val))

    write(*,'(A)') '----------------------------------------'
    write(*,'(A,1X,A)') 'Benchmark:', trim(benchmark_name)
    write(*,'(A,F10.6,A)') '  split forward:   ', split_forward, ' s'
    write(*,'(A,F10.6,A)') '  split backward:  ', split_backward, ' s'
    write(*,'(A,F10.6,A)') '  split total:     ', &
         split_forward + split_backward, ' s'
    write(*,'(A,F10.6,A)') '  fused forward:   ', fused_forward, ' s'
    write(*,'(A,F10.6,A)') '  fused backward:  ', fused_backward, ' s'
    write(*,'(A,F10.6,A)') '  fused total:     ', &
         fused_forward + fused_backward, ' s'
    write(*,'(A,F8.3,A)') '  total speedup:   ', &
         (split_forward + split_backward) / (fused_forward + fused_backward), 'x'
    write(*,'(A,F8.3,A)') '  backward speedup:', &
         split_backward / fused_backward, 'x'
    write(*,'(A,ES12.4)') '  max output error: ', max_output_error
    write(*,'(A,ES12.4)') '  max input-grad error: ', max_input_error
    write(*,'(A,ES12.4)') '  max weight-grad error: ', max_weight_error
    write(*,'(A,ES12.4)') '  max bias-grad error: ', max_bias_error

  end subroutine benchmark_full_case

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

  subroutine setup_full_array_values(&
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

    call input_array%allocate(&
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

    call bias_array%allocate(&
         array_shape = [size(bias_data), 1], &
         source = 0._real32)
    call bias_array%set(reshape(bias_data, [size(bias_data), 1]))
    bias_array%is_sample_dependent = .false.
    call bias_array%set_requires_grad(.true.)

    call target_array%allocate(&
         array_shape = [size(target_data,1), size(target_data,2)], &
         source = 0._real32)
    call target_array%set(target_data)
    call target_array%set_requires_grad(.false.)

  end subroutine setup_full_array_values

end program fused_autodiff_benchmark
