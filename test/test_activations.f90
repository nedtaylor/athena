program test_activations
   use athena, only: &
        full_layer_type, &
        conv2d_layer_type, &
        conv3d_layer_type, &
        base_layer_type
   use activation, only: activation_setup
   use activation_gaussian, only: gaussian_setup ! threshold, sigma
   use activation_piecewise, only: piecewise_setup ! intercept
   use activation_sigmoid, only: sigmoid_setup ! threshold
   use activation_softmax, only: softmax_setup ! threshold
   use activation_tanh, only: tanh_setup ! threshold
   use custom_types, only: activation_type
   implicit none
 
   class(base_layer_type), allocatable :: full_layer, conv2d_layer, conv3d_layer
   class(activation_type), allocatable :: activation_var
   logical :: success = .true.
 
   integer :: i
   real :: scale, value
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
   real, dimension(9) :: activate, differentiate
   real, dimension(1) :: value_1d, rtmp1_1d
   real, dimension(1,1,1) :: value_3d, rtmp1_3d
   real, dimension(:,:), allocatable :: output_2d
   real, dimension(:,:,:,:), allocatable :: output_4d
   real, dimension(:,:,:,:,:), allocatable :: output_5d


!!!-----------------------------------------------------------------------------
!!! Initialise activation names
!!!-----------------------------------------------------------------------------
   activation_names(1) = 'none'
   activation_names(2) = 'gaussian'
   activation_names(3) = 'leaky_relu'
   activation_names(4) = 'linear'
   activation_names(5) = 'piecewise'
   activation_names(6) = 'relu'
   activation_names(7) = 'sigmoid'
   activation_names(8) = 'softmax'
   activation_names(9) = 'tanh'
 
   !! initialise expected activation values
   value = 0.25E0
   value_1d = value
   value_3d = value
   scale = 2.E0
   activate(1) = scale * value
   activate(2) = scale / (sqrt(8*atan(1.E0))*1.5E0) * &
        exp(-0.5E0 * (value/1.5E0)**2.E0)
   activate(3) = scale * value
   activate(4) = scale * value
   activate(5) = scale * value + 1.E0
   activate(6) = scale * value
   activate(7) = scale / (1.E0 + exp(-value))
   activate(8) = exp(0.E0)
   activate(9) = scale * tanh(value)

   !! initialise expected differentiation values
   differentiate(1) = scale * value
   differentiate(2) = - value/1.5E0**2.E0 * activate(2)
   differentiate(3) = scale
   differentiate(4) = scale * value
   differentiate(5) = scale
   differentiate(6) = scale
   differentiate(7) = scale * activate(7) * (scale - activate(7))
   differentiate(8) = activate(8) * (1.E0 - activate(8))
   differentiate(9) = scale * (1.E0 - (activate(9)/scale)**2.E0)


!!!-----------------------------------------------------------------------------
!!! check gaussian setup
!!!-----------------------------------------------------------------------------
   activation_var = gaussian_setup(threshold = 2.E0, sigma = 2.E0)
   if(.not. activation_var%name .eq. 'gaussian')then
      success = .false.
      write(0,*) 'activation has wrong name for gaussian'
   else
      if (activation_var%threshold .ne. 2.E0) then
         success = .false.
         write(0,*) 'activation has wrong threshold for gaussian'
      end if
   end if


!!!-----------------------------------------------------------------------------
!!! check piecewise setup
!!!-----------------------------------------------------------------------------
   activation_var = piecewise_setup(intercept = 2.E0)
   if(.not. activation_var%name .eq. 'piecewise')then
      success = .false.
      write(0,*) 'activation has wrong name for piecewise'
   end if


!!!-----------------------------------------------------------------------------
!!! check sigmoid setup
!!!-----------------------------------------------------------------------------
   activation_var = sigmoid_setup(threshold = 2.E0)
   if(.not. activation_var%name .eq. 'sigmoid')then
      success = .false.
      write(0,*) 'activation has wrong name for sigmoid'
   else
      if (activation_var%threshold .ne. 2.E0) then
         success = .false.
         write(0,*) 'activation has wrong threshold for sigmoid'
      end if
   end if


!!!-----------------------------------------------------------------------------
!!! check softmax setup
!!!-----------------------------------------------------------------------------
   activation_var = softmax_setup(threshold = 2.E0)
   if(.not. activation_var%name .eq. 'softmax')then
      success = .false.
      write(0,*) 'activation has wrong name for softmax'
   else
      if (activation_var%threshold .ne. 2.E0) then
         success = .false.
         write(0,*) 'activation has wrong threshold for softmax'
      end if
   end if


!!!-----------------------------------------------------------------------------
!!! check tanh setup
!!!-----------------------------------------------------------------------------
   activation_var = tanh_setup(threshold = 2.E0)
   if(.not. activation_var%name .eq. 'tanh')then
      success = .false.
      write(0,*) 'activation has wrong name for tanh'
   else
      if (activation_var%threshold .ne. 2.E0) then
         success = .false.
         write(0,*) 'activation has wrong threshold for tanh'
      end if
   end if

!!!-----------------------------------------------------------------------------
!!! check activation setups more rigorously
!!! check for different scales, and ranks
!!!-----------------------------------------------------------------------------
   do i = 1, size(activation_names)
      deallocate(activation_var)
      allocate(activation_var, &
           source=activation_setup(activation_names(i), scale = scale))
      if(.not. activation_var%name .eq. trim(activation_names(i)))then
         success = .false.
         write(0,*) 'activation has wrong name for ', &
              trim(activation_names(i))
      else
         if (abs(activation_var%scale - scale).gt.1.E-6) then
            success = .false.
            write(0,*) 'activation has wrong scale for ', &
                 trim(activation_names(i))
         end if
         rtmp1_1d = activation_var%activate_1d(value_1d)
         if (any(abs(rtmp1_1d - activate(i)).gt.1.E-6)) then
            success = .false.
            write(0,*) 'activation has wrong activation for ', &
                 trim(activation_names(i))
            write(*,*) rtmp1_1d, activate(i)
         end if
         rtmp1_1d = activation_var%differentiate_1d(value_1d)
         if (any(abs(rtmp1_1d - differentiate(i)).gt.1.E-6)) then
            success = .false.
            write(0,*) 'activation has wrong differentiation for ', &
                 trim(activation_names(i))
            write(*,*) rtmp1_1d, differentiate(i)
         end if

         rtmp1_3d = activation_var%activate_3d(value_3d)
         if (any(abs(rtmp1_3d - activate(i)).gt.1.E-6)) then
            success = .false.
            write(0,*) 'activation has wrong activation for ', &
                 trim(activation_names(i))
         end if
         rtmp1_3d = activation_var%differentiate_3d(value_3d)
         if (any(abs(rtmp1_3d - differentiate(i)).gt.1.E-6)) then
            success = .false.
            write(0,*) 'activation has wrong differentiation for ', &
                 trim(activation_names(i))
         end if
      end if

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
            output_2d = full_layer%transfer%differentiate(full_layer%output)
            call compare_derivative( &
                 output_2d, &
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
            output_4d = conv2d_layer%transfer%differentiate(conv2d_layer%output)
            call compare_derivative( &
                  output_4d, &
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
            output_5d = conv3d_layer%transfer%differentiate(conv3d_layer%output)
            call compare_derivative( &
                  output_5d, &
                  conv3d_layer%output, &
                  activation_names(i), success)
         end if
      class default
         success = .false.
         write(0,*) 'full layer is not of type conv3d_layer_type'
      end select

   end do
 

!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
   write(*,*) "----------------------------------------"
   if(success)then
      write(*,*) 'test_activations passed all tests'
   else
      write(0,*) 'test_activations failed one or more tests'
      stop 1
   end if
 
contains

!!!-----------------------------------------------------------------------------
!!! compare output
!!!-----------------------------------------------------------------------------
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


!!!-----------------------------------------------------------------------------
!!! compare derivative
!!!-----------------------------------------------------------------------------
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