program test_batchnorm3d_layer
  use coreutils, only: real32
  use athena, only: &
       batchnorm3d_layer_type, &
       base_layer_type, &
       learnable_layer_type
  use diffstruc, only: array_type
  use athena__batchnorm3d_layer, only: read_batchnorm3d_layer
  implicit none

  class(base_layer_type), allocatable, target :: bn_layer, bn_layer1, bn_layer2
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: num_channels = 3, width = 8, batch_size = 1
  integer :: unit
  real, parameter :: gamma  = 0.5, beta = 0.3
  type(array_type) :: input(1,1)
  real, allocatable, dimension(:,:,:,:,:) :: input_5d, output_5d, gradient_5d
  real, allocatable, dimension(:) :: output_1d, params1, params2
  real, allocatable, dimension(:,:) :: output_2d
  real, parameter :: tol = 0.5E-3
  logical :: success = .true.
  class(array_type), pointer :: output => null()
  class(array_type), pointer :: gradient => null()

  integer :: i, j, output_width, num_params
  integer :: seed_size
  real :: mean, std
  integer, allocatable, dimension(:) :: seed
  real, parameter :: max_value = 3.0

  ! interface assignment(=)
  !    module procedure assign_base
  ! end interface

!-------------------------------------------------------------------------------
! Initialise random number generator with a seed
!-------------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=0)
  call random_seed(put = seed)


!-------------------------------------------------------------------------------
! set up layer
!-------------------------------------------------------------------------------
  bn_layer = batchnorm3d_layer_type( &
       input_shape = [width, width, width, num_channels], &
       batch_size = batch_size, &
       momentum = 0.0, &
       epsilon = 1e-5, &
       gamma_init_mean = (gamma), &
       gamma_init_std = 0.0, &
       beta_init_mean = beta, &
       beta_init_std = 0.0, &
       kernel_initialiser = 'gaussian', &
       bias_initialiser = 'gaussian', &
       moving_mean_initialiser = 'zeros', &
       moving_variance_initialiser = 'zeros' &
  )
  call bn_layer%set_ptrs()

  !! check layer name
  if(.not. bn_layer%name .eq. 'batchnorm3d')then
     success = .false.
     write(0,*) 'batchnorm3d layer has wrong name'
  end if

  !! check layer type
  select type(bn_layer)
  type is(batchnorm3d_layer_type)
     !! check input shape
     if(any(bn_layer%input_shape .ne. [width,width,width,num_channels]))then
        success = .false.
        write(0,*) 'batchnorm3d layer has wrong input_shape'
     end if

     !! check output shape
     if(any(bn_layer%output_shape .ne. [width,width,width,num_channels]))then
        success = .false.
        write(0,*) 'batchnorm3d layer has wrong output shape'
     end if

     !! check batch size
     if(bn_layer%batch_size .ne. batch_size)then
        success = .false.
        write(0,*) 'batchnorm3d layer has wrong batch size'
     end if
  class default
     success = .false.
     write(0,*) 'batchnorm3d layer has wrong type'
  end select


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output for single-valued input
!!! use existing layer
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing forward pass with single-valued input..."
  !! initialise sample input
  call input(1,1)%allocate(&
       array_shape=[width,width,width,num_channels,batch_size], &
       source = max_value)
  call input(1,1)%set_requires_grad(.true.)

  !! run forward pass
  write(*,*) "Running forward..."
  call bn_layer%forward(input)
  write(*,*) "Extracting output..."
  call bn_layer%extract_output(output_5d)

  !! check outputs all get normalised to zero
  write(*,*) "Checking output values..."
  if (any(output_5d-beta.gt. tol)) then
     success = .false.
     write(0,*) 'batchnorm3d layer forward pass failed: &
          &output should all equal beta'
  end if

  deallocate(output_5d)
  call input(1,1)%deallocate()


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output for randomised input
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  call input(1,1)%allocate(&
       array_shape=[width, width, width, num_channels, batch_size])
  call input(1,1)%set_requires_grad(.true.)
  call random_number(input(1,1)%val)

  !! run forward pass
  call bn_layer%forward(input)
  call bn_layer%extract_output(output_5d)

  !! check outputs all get normalised to zero
  do i = 1, num_channels
     mean = sum(output_5d(:,:,:,i,:))/(width**3*batch_size)
     std = sqrt(sum((output_5d(:,:,:,i,:) - mean)**2)/(width**3*batch_size))
     if (abs(mean - beta) .gt. tol) then
        success = .false.
        write(0,*) 'batchnorm3d layer forward pass failed: &
             &mean should equal beta'
     end if
     if (abs(std - gamma) .gt. tol) then
        success = .false.
        write(0,*) 'batchnorm3d layer forward pass failed: &
             &std should equal gamma'
     end if
  end do

  deallocate(output_5d)


!!!-----------------------------------------------------------------------------
!!! test backward pass and check expected output for randomised input
!!! use existing layer
!!!-----------------------------------------------------------------------------
  output => bn_layer%output(1,1)
  allocate(output%grad)
  call output%grad%allocate(array_shape=[output%shape, size(output%val, 2)])
  output%grad%val = output%val
  call output%grad%extract(gradient_5d)
  call output%grad_reverse()
  gradient => input(1,1)%grad
  call gradient%extract(input_5d)

  !! check gradient has expected value
  select type(current => bn_layer)
  type is(batchnorm3d_layer_type)
     do i = 1, num_channels
        mean = sum(input_5d(:,:,:,i,:))/(width**3*batch_size)
        std = sqrt( &
             sum((input_5d(:,:,:,i,:) - mean)**2)/(width**3*batch_size) &
        )
        if (abs(mean) .gt. tol) then
           success = .false.
           write(0,*) 'batchnorm3d layer backward pass failed: &
                &mean gradient should be zero'
        end if
        if (abs(std) .gt. tol) then
           success = .false.
           write(0,*) 'batchnorm3d layer backward pass failed: &
                &std gradient should equal gamma'
        end if
     end do
  end select

  call input(1,1)%deallocate()
  call input(1,1)%reset_graph()
  deallocate(input_5d, gradient_5d)


!-------------------------------------------------------------------------------
! check handling of layer parameters and gradients
!-------------------------------------------------------------------------------
  select type(bn_layer)
  class is(learnable_layer_type)
     !! check parameters
     num_params = bn_layer%get_num_params()
     if (num_params .ne. 2 * num_channels)then
        write(0,*) 'batchnorm3d layer has wrong number of parameters'
        success = .false.
     end if
     allocate(params1(num_params), source = 12.E0)
     call bn_layer%set_params(params1)
     params2 = bn_layer%get_params()
     if(any(abs(params1 - params2).gt.1.E-6))then
        write(0,*) 'batchnorm3d layer has wrong parameters'
        success = .false.
     end if

     !! check gradients
     deallocate(params1, params2)
     allocate(params1(num_params), source = 15.E0)
     call bn_layer%set_gradients(params1)
     params2 = bn_layer%get_gradients()
     if(any(abs(params1 - params2).gt.1.E-6))then
        write(0,*) 'batchnorm3d layer has wrong gradients'
        success = .false.
     end if
     call bn_layer%set_gradients(20.E0)
     params2 = bn_layer%get_gradients()
     if(any(abs(params2 - 20.E0).gt.1.E-6))then
        write(0,*) 'batchnorm3d layer has wrong gradients'
        success = .false.
     end if
  class default
     write(0,*) 'batchnorm3d layer has wrong type'
     success = .false.
  end select


!-------------------------------------------------------------------------------
! check layer operations
!-------------------------------------------------------------------------------
  bn_layer1 = batchnorm3d_layer_type(input_shape=[2,2,2,1], batch_size=1)
  bn_layer2 = batchnorm3d_layer_type(input_shape=[2,2,2,1], batch_size=1)
  select type(bn_layer1)
  type is(batchnorm3d_layer_type)
     call bn_layer1%set_gradients(1.E0)
     select type(bn_layer2)
     type is(batchnorm3d_layer_type)
        call bn_layer2%set_gradients(2.E0)
        bn_layer = bn_layer1 + bn_layer2
        select type(bn_layer)
        type is(batchnorm3d_layer_type)
           !! check layer addition
           call compare_batchnorm3d_layers(&
                bn_layer, bn_layer1, bn_layer2, success)
        class default
           success = .false.
           write(0,*) 'batchnorm3d layer has wrong type'
        end select
     class default
        success = .false.
        write(0,*) 'batchnorm3d layer has wrong type'
     end select
  class default
     success = .false.
     write(0,*) 'batchnorm3d layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! check output request using rank 1 and rank 2 arrays is consistent
!-------------------------------------------------------------------------------
  allocate(output_1d(width**3*num_channels*batch_size))
  allocate(output_2d(width**3*num_channels, batch_size))
  call bn_layer%extract_output(output_1d)
  call bn_layer%extract_output(output_2d)
  if(any(abs(output_1d - reshape(output_2d, [size(output_2d)])) .gt. 1.E-6))then
     success = .false.
     write(0,*) 'output_1d and output_2d are not consistent'
  end if
  deallocate(output_1d, output_2d)


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_batchnorm3d_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("BATCHNORM3D")')
  call bn_layer%print_to_unit(unit)
  write(unit,'("END BATCHNORM3D")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_batchnorm3d_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_batchnorm3d_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (batchnorm3d_layer_type)
     if (.not. read_layer%name .eq. 'batchnorm3d') then
        success = .false.
        write(0,*) 'read batchnorm3d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not batchnorm3d_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_batchnorm3d_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_batchnorm3d_layer passed all tests'
  else
     write(0,*) 'test_batchnorm3d_layer failed one or more tests'
     stop 1
  end if

contains

!-------------------------------------------------------------------------------
! compare three layers
!-------------------------------------------------------------------------------
  subroutine compare_batchnorm3d_layers(layer1, layer2, layer3, success_flag)
    type(batchnorm3d_layer_type), intent(in) :: layer1, layer2, layer3
    logical, intent(inout) :: success_flag
    real(real32), allocatable, dimension(:) :: gradients1, gradients2, gradients3

    gradients1 = layer1%get_gradients()
    gradients2 = layer2%get_gradients()
    gradients3 = layer3%get_gradients()
    if(any(abs(gradients1-gradients2-gradients3).gt.tol))then
       success_flag = .false.
       write(0,*) 'batchnorm3d layer has wrong gradients'
    end if

  end subroutine compare_batchnorm3d_layers

end program test_batchnorm3d_layer
