program test_activations
  use coreutils, only: real32, pi
  use athena, only: &
       full_layer_type, &
       conv2d_layer_type, &
       conv3d_layer_type, &
       base_layer_type
  use athena__activation, only: activation_setup
  use athena__activation_gaussian, only: gaussian_actv_type ! threshold, sigma
  use athena__activation_piecewise, only: piecewise_actv_type ! intercept
  use athena__activation_sigmoid, only: sigmoid_actv_type ! threshold
  use athena__activation_softmax, only: softmax_actv_type ! threshold
  use athena__activation_swish, only: swish_actv_type ! threshold
  use athena__activation_tanh, only: tanh_actv_type ! threshold
  use athena__misc_types, only: base_actv_type, onnx_attribute_type
  use diffstruc, only: array_type
  implicit none

  class(base_layer_type), allocatable :: full_layer, conv2d_layer, conv3d_layer
  class(base_actv_type), allocatable :: activation
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
  character(len=20) :: activation_names(10)
  integer :: k, j, l, s
  integer, dimension(2) :: stp_idx, start_idx, end_idx
  real, dimension(10) :: activate, differentiate
  type(array_type) :: input(1,1)
  type(array_type) :: val_array
  type(array_type), pointer :: result_array
  type(onnx_attribute_type) :: scale_attr1, scale_attr2


!-------------------------------------------------------------------------------
! Initialise activation names
!-------------------------------------------------------------------------------
  activation_names(1) = 'none'
  activation_names(2) = 'gaussian'
  activation_names(3) = 'leaky_relu'
  activation_names(4) = 'linear'
  activation_names(5) = 'piecewise'
  activation_names(6) = 'relu'
  activation_names(7) = 'sigmoid'
  activation_names(8) = 'softmax'
  activation_names(9) = 'swish'
  activation_names(10) = 'tanh'

  !! initialise expected activation values
  value = 0.25E0
  call val_array%allocate([1], source=value)
  call val_array%set_requires_grad(.true.)
  val_array%is_temporary = .false.
  val_array%fix_pointer = .true.
  scale = 2.E0
  activate(1) = value
  activate(2) = scale / (1.5E0 * sqrt(2.E0 * pi)) * exp(-0.5E0 * (value/1.5E0)**2.E0)
  activate(3) = scale * value
  activate(4) = scale * value
  activate(5) = scale * value
  activate(6) = scale * value
  activate(7) = scale / (1.E0 + exp(-value))
  activate(8) = scale * exp(0.E0)
  activate(9) = scale * value / (1.E0 + exp(-value))
  activate(10) = scale * tanh(value)

  !! initialise expected differentiation values
  differentiate(1) = 1.E0
  differentiate(2) = - (value / 1.5E0**2) * activate(2)
  differentiate(3) = scale
  differentiate(4) = scale
  differentiate(5) = scale
  differentiate(6) = scale
  differentiate(7) =  activate(7) * (1.E0 - activate(7) / scale)
  differentiate(8) = activate(8) * (1.E0 - activate(8)/scale)
  differentiate(9) = &
       ( activate(9)  /value + activate(9) * (1.E0 - activate(9) / scale / value) )
  differentiate(10) = scale * (1.E0 - (activate(10)/scale)**2.E0)


!-------------------------------------------------------------------------------
! check gaussian setup
!-------------------------------------------------------------------------------
  activation = gaussian_actv_type(mu = 2.E0, sigma = 2.E0)
  if(.not. activation%name .eq. 'gaussian')then
     success = .false.
     write(0,*) 'activation has wrong name for gaussian'
  else
     select type(activation)
     type is(gaussian_actv_type)
        if(abs(activation%mu - 2.E0).gt.1.E-6)then
           success = .false.
           write(0,*) 'activation has wrong mu for gaussian'
        end if
     class default
        success = .false.
        write(0,*) 'activation is not of type gaussian_actv_type'
     end select
  end if


!-------------------------------------------------------------------------------
! check piecewise setup
!-------------------------------------------------------------------------------
  activation = piecewise_actv_type(gradient = 2.E0)
  if(.not. activation%name .eq. 'piecewise')then
     success = .false.
     write(0,*) 'activation has wrong name for piecewise'
  end if


!-------------------------------------------------------------------------------
! check sigmoid setup
!-------------------------------------------------------------------------------
  activation = sigmoid_actv_type(scale = 2.E0)
  if(.not. activation%name .eq. 'sigmoid')then
     success = .false.
     write(0,*) 'activation has wrong name for sigmoid'
  else
     if(abs(activation%scale - 2.E0).gt.1.E-6)then
        success = .false.
        write(0,*) 'activation has wrong scale for sigmoid'
     end if
  end if


!-------------------------------------------------------------------------------
! check softmax setup
!-------------------------------------------------------------------------------
  activation = softmax_actv_type(scale = 2.E0)
  if(.not. activation%name .eq. 'softmax')then
     success = .false.
     write(0,*) 'activation has wrong name for softmax'
  else
     if(abs(activation%scale - 2.E0).gt.1.E-6)then
        success = .false.
        write(0,*) 'activation has wrong scale for softmax'
     end if
  end if


!-------------------------------------------------------------------------------
! check swish setup
!-------------------------------------------------------------------------------
  activation = swish_actv_type(beta = 2.E0)
  if(.not. activation%name .eq. 'swish')then
     success = .false.
     write(0,*) 'activation has wrong name for swish'
  else
     select type(activation)
     type is(swish_actv_type)
        if(abs(activation%beta - 2.E0).gt.1.E-6)then
           success = .false.
           write(0,*) 'activation has wrong beta for swish'
        end if
     class default
        success = .false.
        write(0,*) 'activation is not of type swish_actv_type'
     end select
  end if


!-------------------------------------------------------------------------------
! check tanh setup
!-------------------------------------------------------------------------------
  activation = tanh_actv_type(scale = 2.E0)
  if(.not. activation%name .eq. 'tanh')then
     success = .false.
     write(0,*) 'activation has wrong name for tanh'
  else
     if(abs(activation%scale - 2.E0).gt.1.E-6)then
        success = .false.
        write(0,*) 'activation has wrong scale for tanh'
     end if
  end if

!!!-----------------------------------------------------------------------------
!!! check activation setups more rigorously
!!! check for different scales, and ranks
!!!-----------------------------------------------------------------------------
  scale_attr1 = onnx_attribute_type( &
       name = 'scale', &
       type = 'float', &
       val = '1.0' &
  )
  scale_attr2 = onnx_attribute_type( &
       name = 'scale', &
       type = 'float', &
       val = '2.0' &
  )
  do i = 1, size(activation_names)
     deallocate(activation)
     allocate(activation, source=activation_setup(activation_names(i)))
     if(trim(activation_names(i)).ne.'none')then
        call activation%apply_attributes([scale_attr2])
     end if
     if(.not. activation%name .eq. trim(activation_names(i)))then
        success = .false.
        write(0,*) 'activation has wrong name for ', &
             trim(activation_names(i))
     else
        if(trim(activation%name).ne.trim(activation_names(i)))then
           success = .false.
           write(0,*) 'activation name mismatch for ', &
                trim(activation_names(i))
        end if
        result_array => activation%apply(val_array)
        if(abs(result_array%val(1,1) - activate(i)).gt.1.E-6)then
           success = .false.
           write(0,*) 'activation has wrong activation for ', &
                trim(activation_names(i))
           write(*,*) result_array%val(1,1), activate(i)
        end if
        call result_array%grad_reverse(reset_graph=.true.)
        if(abs(val_array%grad%val(1,1) - differentiate(i)).gt.1.E-6)then
           success = .false.
           write(0,*) 'activation has wrong differentiation for ', &
                trim(activation_names(i))
           write(*,*) val_array%grad%val(1,1), differentiate(i)
        end if
        call result_array%nullify_graph()
        nullify(result_array)
     end if

     if(trim(activation_names(i)).ne.'none')then
        call activation%apply_attributes([scale_attr1])
     end if
     !! check for rank 2 data
     !!-------------------------------------------------------------------------
     !! set up full layer
     full_layer = full_layer_type( &
          num_inputs = num_inputs, &
          num_outputs = num_outputs, &
          activation = activation, &
          kernel_initialiser = 'ones', &
          bias_initialiser = 'zeros' )

     !! check layer name
     select type(full_layer)
     type is(full_layer_type)
        if(.not. full_layer%activation%name .eq. trim(activation_names(i)))then
           success = .false.
           write(0,*) 'activation has wrong name for ', &
                trim(activation_names(i))
        else
           call input(1,1)%allocate(array_shape=[num_inputs, batch_size], &
                source=1._real32)
           call input(1,1)%set_requires_grad(.true.)
           input(1,1)%is_temporary = .false.
           call full_layer%forward(input)
           call compare_output( &
                full_layer%output(1,1)%val, &
                input(1,1)%val, activation_names(i), "full", success)

           call full_layer%output(1,1)%grad_reverse(reset_graph=.true.)
           call compare_derivative( &
                input(1,1)%grad%val, &
                input(1,1)%val, &
                activation_names(i), "full", success)
           call input(1,1)%deallocate()
        end if
     class default
        success = .false.
        write(0,*) 'full layer is not of type full_layer_type'
     end select
     call full_layer%output(1,1)%nullify_graph()


     !! check for rank 3 data
     !!-------------------------------------------------------------------------
     !! set up full layer
     conv2d_layer = conv2d_layer_type( &
          input_shape = [width, width, num_channels], &
          kernel_size = kernel_size, &
          stride = stride, &
          padding = "none", &
          num_filters = num_filters, &
          activation = activation, &
          kernel_initialiser = 'ones', &
          bias_initialiser = 'zeros' )

     !! check layer name
     select type(conv2d_layer)
     type is(conv2d_layer_type)
        if(.not. conv2d_layer%activation%name .eq. trim(activation_names(i)))then
           success = .false.
           write(0,*) 'activation has wrong name for ', &
                trim(activation_names(i))
        else
           call input(1,1)%allocate( &
                array_shape = [width, width, num_channels, batch_size], &
                source = 1._real32 &
           )
           call input(1,1)%set_requires_grad(.true.)
           input(1,1)%is_temporary = .false.
           call conv2d_layer%forward(input)
           call compare_output( &
                conv2d_layer%output(1,1)%val, &
                input(1,1)%val, activation_names(i), "conv2d", success)

           call conv2d_layer%output(1,1)%grad_reverse(reset_graph=.true.)
           call compare_derivative( &
                input(1,1)%grad%val, &
                input(1,1)%val, &
                activation_names(i), "conv2d", success)
           call input(1,1)%deallocate()
        end if
     class default
        success = .false.
        write(0,*) 'conv layer is not of type conv2d_layer_type'
     end select
     call conv2d_layer%output(1,1)%nullify_graph()


     !! check for rank 4 data
     !!-------------------------------------------------------------------------
     !! set up full layer
     conv3d_layer = conv3d_layer_type( &
          input_shape = [width, width, width, num_channels], &
          kernel_size = kernel_size, &
          stride = stride, &
          padding = "none", &
          num_filters = num_filters, &
          activation = activation, &
          kernel_initialiser = 'ones', &
          bias_initialiser = 'zeros' )

     !! check layer name
     select type(conv3d_layer)
     type is(conv3d_layer_type)
        if(.not. conv3d_layer%activation%name .eq. trim(activation_names(i)))then
           success = .false.
           write(0,*) 'activation has wrong name for ', &
                trim(activation_names(i))
        else
           call input(1,1)%allocate( &
                array_shape = [width, width, width, num_channels, batch_size], &
                source = 1._real32 &
           )
           call input(1,1)%set_requires_grad(.true.)
           input(1,1)%is_temporary = .false.
           call conv3d_layer%forward(input)
           call compare_output( &
                conv3d_layer%output(1,1)%val, &
                input(1,1)%val, activation_names(i), "conv3d", success)
           call conv3d_layer%output(1,1)%grad_reverse(reset_graph=.true.)
           call compare_derivative( &
                input(1,1)%grad%val, &
                input(1,1)%val, &
                activation_names(i), "conv3d", success)
           call input(1,1)%deallocate()
        end if
     class default
        success = .false.
        write(0,*) 'conv3d layer is not of type conv3d_layer_type'
     end select
     call conv3d_layer%output(1,1)%nullify_graph()

  end do


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_activations passed all tests'
  else
     write(0,*) 'test_activations failed one or more tests'
     stop 1
  end if

contains

!-------------------------------------------------------------------------------
! compare output
!-------------------------------------------------------------------------------
  subroutine compare_output(output, input, activation_name, layer_name, success)
    real, intent(in) :: output(1:1)
    real, intent(in) :: input(1:1)
    character(len=*), intent(in) :: activation_name, layer_name
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
       write(0,*) 'activation ', trim(activation_name), ' failed for layer ', &
            trim(layer_name)
       write(0,*) 'input: ', input
       write(0,*) 'output: ', output
       write(0,*) 'expected_output: ', expected_output
    end if

  end subroutine compare_output


!-------------------------------------------------------------------------------
! compare derivative
!-------------------------------------------------------------------------------
  subroutine compare_derivative( &
       derivative, input, activation_name, layer_name, success &
  )
    real, intent(in) :: derivative(1:1)
    real, intent(in) :: input(1:1)
    character(len=*), intent(in) :: activation_name, layer_name
    logical, intent(inout) :: success
    real, allocatable :: expected_output(:)
    integer :: i, j

    allocate(expected_output, source=derivative)

    select case(activation_name)
    case('none')
       expected_output = reshape(input, shape(derivative))
    case('gaussian')
       expected_output = 1.E0 / (sqrt(2.E0 * pi) * 1.5E0) * &
            exp( - 0.5E0 * (input/1.5E0) ** 2.E0 )
       expected_output = -input / ( 1.5E0**2 ) * expected_output
    case('leaky_relu')
       expected_output = input
    case('linear')
       expected_output = input
    case('piecewise')
       do i = 1, size(input,1)
          if(abs(input(i)) .ge. 1.E0) expected_output(i) = 1.E0
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
       write(0,*) 'derivative of ', trim(activation_name), ' failed for layer ', &
            trim(layer_name)
       write(0,*) 'input: ', input
       write(0,*) 'derivative: ', derivative
       write(0,*) 'expected_derivative: ', expected_output
    end if

  end subroutine compare_derivative

end program test_activations
