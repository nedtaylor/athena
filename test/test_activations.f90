program test_activations
   use athena, only: &
        full_layer_type, &
        conv2d_layer_type, &
        conv3d_layer_type, &
        base_layer_type
   implicit none
 
   class(base_layer_type), allocatable :: full_layer, conv2d_layer, conv3d_layer
   logical :: success = .true.
 
   integer :: i
   integer, parameter :: batch_size = 1
   integer, parameter :: num_inputs = 1
   integer, parameter :: num_outputs = 1
   integer, parameter :: width = 1
   integer, parameter :: stride = 1
   integer, parameter :: kernel_size = 1
   integer, parameter :: num_channels = 1
   integer, parameter :: num_filters = 1
   real :: input_data(num_inputs, batch_size) = 1.E0
   real :: gradient(num_outputs, batch_size) = 1.E0
   real :: input_data_conv2d(width,width,num_channels, batch_size) = 1.E0
   real :: input_data_conv3d(width,width, width,num_channels, batch_size) = 1.E0
   character(len=20) :: activation_names(9)
   integer :: k, j, l, s
   integer, dimension(2) :: stp_idx, start_idx, end_idx
   
   activation_names(1) = 'none'
   activation_names(2) = 'gaussian'
   activation_names(3) = 'leaky_relu'
   activation_names(4) = 'linear'
   activation_names(5) = 'piecewise'
   activation_names(6) = 'relu'
   activation_names(7) = 'sigmoid'
   activation_names(8) = 'softmax'
   activation_names(9) = 'tanh'
 

   do i = 1, size(activation_names)
      !! check for rank 2 data
      !!------------------------------------------------------------------------
      !! set up full layer
      full_layer = full_layer_type( &
           num_inputs = num_inputs, &
           num_outputs = num_outputs, &
           batch_size = batch_size, &
           activation_function = activation_names(i), &
           kernel_initialiser = 'ones', &
           bias_initialiser = 'zeros' )
   
      !! check layer name
      select type(full_layer)
      type is(full_layer_type)
         if(.not. full_layer%transfer%name .eq. trim(activation_names(i)))then
            success = .false.
            write(0,*) 'activation has wrong name for ', &
                 trim(activation_names(i))
         else
            call full_layer%forward(input_data)
            call compare_output( &
                 full_layer%output, input_data, activation_names(i), success)

            full_layer%output = 1.E0
            call compare_derivative( &
                 full_layer%transfer%differentiate(full_layer%output), &
                 full_layer%output, &
                 activation_names(i), success)
         end if
      class default
         success = .false.
         write(0,*) 'full layer is not of type full_layer_type'
      end select


      !! check for rank 3 data
      !!------------------------------------------------------------------------
      !! set up full layer
      conv2d_layer = conv2d_layer_type( &
           input_shape = [width, width, num_channels], &
           kernel_size = kernel_size, &
           batch_size = batch_size, &
           stride = stride, &
           padding = "none", &
           num_filters = num_filters, &
           activation_function = activation_names(i), &
           kernel_initialiser = 'ones', &
           bias_initialiser = 'zeros' )
   
      !! check layer name
      select type(conv2d_layer)
      type is(conv2d_layer_type)
         if(.not. conv2d_layer%transfer%name .eq. trim(activation_names(i)))then
            success = .false.
            write(0,*) 'activation has wrong name for ', &
                 trim(activation_names(i))
         else
            call conv2d_layer%forward(input_data_conv2d)
            call compare_output( &
                 conv2d_layer%output, &
                 input_data_conv2d, activation_names(i), success)
            
            conv2d_layer%output = 1.E0
            call compare_derivative( &
                  conv2d_layer%transfer%differentiate(conv2d_layer%output), &
                  conv2d_layer%output, &
                  activation_names(i), success)
         end if
      class default
         success = .false.
         write(0,*) 'full layer is not of type conv2d_layer_type'
      end select


      !! check for rank 4 data
      !!------------------------------------------------------------------------
      !! set up full layer
      conv3d_layer = conv3d_layer_type( &
           input_shape = [width, width, width, num_channels], &
           kernel_size = kernel_size, &
           batch_size = batch_size, &
           stride = stride, &
           padding = "none", &
           num_filters = num_filters, &
           activation_function = activation_names(i), &
           kernel_initialiser = 'ones', &
           bias_initialiser = 'zeros' )
   
      !! check layer name
      select type(conv3d_layer)
      type is(conv3d_layer_type)
         if(.not. conv3d_layer%transfer%name .eq. trim(activation_names(i)))then
            success = .false.
            write(0,*) 'activation has wrong name for ', &
                 trim(activation_names(i))
         else
            call conv3d_layer%forward(input_data_conv3d)
            call compare_output( &
                 conv3d_layer%output, &
                 input_data_conv3d, activation_names(i), success)
            
            conv3d_layer%output = 1.E0
            call compare_derivative( &
                  conv3d_layer%transfer%differentiate(conv3d_layer%output), &
                  conv3d_layer%output, &
                  activation_names(i), success)
         end if
      class default
         success = .false.
         write(0,*) 'full layer is not of type conv3d_layer_type'
      end select

   end do
 
   !! check for any fails
   write(*,*) "----------------------------------------"
   if(success)then
      write(*,*) 'test_activations passed all tests'
   else
      write(*,*) 'test_activations failed one or more tests'
      stop 1
   end if
 
contains

   subroutine compare_output(output, input, activation_name, success)
      real, intent(in) :: output(1:1)
      real, intent(in) :: input(1:1)
      character(len=*), intent(in) :: activation_name
      logical, intent(inout) :: success
      real, allocatable :: expected_output(:)
      integer :: i, j

      allocate(expected_output, source=output)
   
      select case(activation_name)
      case('none')
         expected_output = reshape(input, shape(output))
      case('gaussian')
         expected_output = 1.E0 / (sqrt(8.E0 * atan(1.E0)) * 1.5E0) * &
              exp( - 0.5E0 * (input/1.5E0) ** 2.E0 )
      case('leaky_relu')
         expected_output = input
      case('linear')
         expected_output = input
      case('piecewise')
         expected_output = input
      case('relu')
         expected_output = input
         do i = 1, size(input,1)
            if(input(i) .lt. 0.E0) expected_output(i) = 0.E0
         end do
      case('sigmoid')
         expected_output = 1.E0 / ( 1.E0 + exp(-input) )
      case('softmax')
         expected_output = input
      case('tanh')
         expected_output = tanh(input)
      end select
   
      if(all(abs(output - expected_output) .gt. 1.E-6))then
         success = .false.
         write(0,*) 'activation ', trim(activation_name), ' failed'
         write(0,*) 'input: ', input
         write(0,*) 'output: ', output
         write(0,*) 'expected_output: ', expected_output
      end if
   
   end subroutine compare_output

   subroutine compare_derivative(derivative, input, activation_name, success)
      real, intent(in) :: derivative(1:1)
      real, intent(in) :: input(1:1)
      character(len=*), intent(in) :: activation_name
      logical, intent(inout) :: success
      real, allocatable :: expected_output(:)
      integer :: i, j

      allocate(expected_output, source=derivative)
   
      select case(activation_name)
      case('none')
         expected_output = reshape(input, shape(derivative))
      case('gaussian')
         expected_output = 1.E0 / (sqrt(8.E0 * atan(1.E0)) * 1.5E0) * &
              exp( - 0.5E0 * (input/1.5E0) ** 2.E0 )
         expected_output = -input/1.5E0**2.E0 * expected_output
      case('leaky_relu')
         expected_output = input
      case('linear')
         expected_output = input
      case('piecewise')
         do i = 1, size(input,1)
            if(abs(input(i)) .ge. 1.E0) expected_output(i) = 0.E0
         end do
      case('relu')
         expected_output = input
         do i = 1, size(input,1)
            if(input(i) .lt. 0.E0) expected_output(i) = 0.E0
         end do
      case('sigmoid')
         expected_output = 1.E0 / ( 1.E0 + exp(-input) )
         expected_output = 1.E0 * expected_output * (1.E0 - expected_output)
      case('softmax')
         expected_output = input
         expected_output = expected_output * (1.E0 - expected_output)
      case('tanh')
         expected_output = tanh(input)
         expected_output = 1.E0 * &
              (1.E0 - (expected_output/1.E0) ** 2.E0)
      end select
   
      if(all(abs(derivative - expected_output) .gt. 1.E-6))then
         success = .false.
         write(0,*) 'activation ', trim(activation_name), ' failed'
         write(0,*) 'input: ', input
         write(0,*) 'derivative: ', derivative
         write(0,*) 'expected_output: ', expected_output
      end if
   
   end subroutine compare_derivative

end program test_activations