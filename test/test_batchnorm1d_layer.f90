program test_batchnorm1d_layer
  use athena, only: &
       batchnorm1d_layer_type, &
       base_layer_type, &
       learnable_layer_type
  use coreutils, only: real32
  use diffstruc, only: array_type
  use athena__batchnorm1d_layer, only: read_batchnorm1d_layer
  implicit none

  class(base_layer_type), allocatable, target :: bn_layer, bn_layer1, bn_layer2
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: num_channels = 3, width = 8, batch_size = 5
  integer :: unit
  real, parameter :: gamma  = 0.5, beta = 0.3
  type(array_type) :: input(1,1)
  real, allocatable, dimension(:,:) :: input_data, output_data, gradient_data
  real, allocatable, dimension(:,:,:) :: input_3d, output_3d, gradient_3d
  real, allocatable, dimension(:) :: output_1d, params1, params2
  real, parameter :: tol = 0.5E-3
  logical :: success = .true.
  class(array_type), pointer :: &
       output => null(), &
       gradient => null(), &
       params_grad => null()

  integer :: i, j, output_width, num_params
  integer :: seed_size
  real :: mean, std
  integer, allocatable, dimension(:) :: seed
  real, parameter :: max_value = 3.0


!-------------------------------------------------------------------------------
! Initialise random number generator with a seed
!-------------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=0)
  call random_seed(put = seed)


!-------------------------------------------------------------------------------
! test num_channels and num_inputs
!-------------------------------------------------------------------------------
  bn_layer = batchnorm1d_layer_type( &
       input_shape=[width], &
       batch_size=batch_size &
  )
  call bn_layer%set_ptrs()
  select type(bn_layer)
  type is(batchnorm1d_layer_type)
     if(bn_layer%num_channels .ne. width)then
        success = .false.
        write(0,*) 'batchnorm1d layer has wrong num_channels'
     end if
  end select
  if(.not.allocated(bn_layer%input_shape))then
     success = .false.
     write(0,*) 'batchnorm1d input_shape is not allocated'
  elseif(size(bn_layer%input_shape) .ne. 2)then
     success = .false.
     write(0,*) 'batchnorm1d layer has wrong input_shape size'
  elseif(bn_layer%input_shape(2) .ne. width)then
     success = .false.
     write(0,*) 'batchnorm1d layer has wrong input_shape'
  end if
  deallocate(bn_layer)
  bn_layer = batchnorm1d_layer_type( &
       num_inputs=width, &
       num_channels=num_channels, &
       batch_size=batch_size &
  )
  select type(bn_layer)
  type is(batchnorm1d_layer_type)
     if(bn_layer%num_channels .ne. num_channels)then
        success = .false.
        write(0,*) 'batchnorm1d layer has wrong num_channels'
     end if
  end select
  if(.not.allocated(bn_layer%input_shape))then
     success = .false.
     write(0,*) 'batchnorm1d input_shape is not allocated'
  elseif(size(bn_layer%input_shape) .ne. 2)then
     success = .false.
     write(0,*) 'batchnorm1d layer has wrong input_shape size'
  elseif(bn_layer%input_shape(1) .ne. width)then
     success = .false.
     write(0,*) 'batchnorm1d layer has wrong input_shape'
  end if
  deallocate(bn_layer)
  bn_layer = batchnorm1d_layer_type( &
       num_inputs=width, &
       num_channels=num_channels, &
       batch_size=batch_size &
  )
  select type(bn_layer)
  type is(batchnorm1d_layer_type)
     if(bn_layer%num_channels .ne. num_channels)then
        success = .false.
        write(0,*) 'batchnorm1d layer has wrong num_channels'
     end if
  end select
  if(.not.allocated(bn_layer%input_shape))then
     success = .false.
     write(0,*) 'batchnorm1d input_shape is not allocated'
  elseif(size(bn_layer%input_shape) .ne. 2)then
     success = .false.
     write(0,*) 'batchnorm1d layer has wrong input_shape size'
  elseif(bn_layer%input_shape(1) .ne. width)then
     success = .false.
     write(0,*) 'batchnorm1d layer has wrong input_shape'
  end if
  deallocate(bn_layer)


!-------------------------------------------------------------------------------
! set up layer
!-------------------------------------------------------------------------------
  bn_layer = batchnorm1d_layer_type( &
       input_shape = [width], &
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
  if(.not. bn_layer%name .eq. 'batchnorm1d')then
     success = .false.
     write(0,*) 'batchnorm1d layer has wrong name'
  end if

  !! check layer type
  select type(bn_layer)
  type is(batchnorm1d_layer_type)
     !! check input shape
     if(any(bn_layer%input_shape .ne. [1, width]))then
        success = .false.
        write(0,*) 'batchnorm1d layer has wrong input_shape'
     end if

     !! check output shape
     if(bn_layer%output_shape(2) .ne. width)then
        success = .false.
        write(0,*) 'batchnorm1d layer has wrong output shape'
        write(0,*) "output shape", bn_layer%output_shape
        write(0,*) "width", width
     end if

     !! check batch size
     if(bn_layer%batch_size .ne. batch_size)then
        success = .false.
        write(0,*) 'batchnorm1d layer has wrong batch size'
     end if
  class default
     success = .false.
     write(0,*) 'batchnorm1d layer has wrong type'
  end select


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output for single-valued input
!!! use existing layer
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing forward pass with single-valued input..."
  !! initialise sample input
  if(input(1,1)%allocated) call input(1,1)%deallocate()
  call input(1,1)%allocate(array_shape=[1, width, batch_size], source = max_value)
  call input(1,1)%set_requires_grad(.true.)

  !! run forward pass
  write(*,*) "Running forward_derived..."
  call bn_layer%forward_derived(input)
  write(*,*) "Extracting output..."
  call bn_layer%extract_output(output_3d)

  !! check outputs all get normalised to zero
  write(*,*) "Checking output values..."
  if (any(output_3d(:,1,:)-beta.gt. tol)) then
     success = .false.
     write(0,*) 'batchnorm1d layer forward pass failed: &
          &output should all equal beta'
  end if

  deallocate(output_3d)
  call input(1,1)%deallocate()


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output for randomised input
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  call input(1,1)%allocate(array_shape=[1, width, batch_size])
  call input(1,1)%set_requires_grad(.true.)
  call random_number(input(1,1)%val)

  !! run forward pass
  call bn_layer%forward_derived(input)
  call bn_layer%extract_output(output_3d)

  !! check outputs all get normalised to zero
  do i = 1, width
     mean = sum(output_3d(1,i,:))/real(batch_size)
     std = sqrt(sum((output_3d(1,i,:) - mean)**2)/real(batch_size))
     if (abs(mean - beta) .gt. 1.E-3) then
        success = .false.
        write(0,*) 'batchnorm1d layer forward pass failed: &
             &mean should equal beta'
        write(*,*) "mean", mean, "beta", beta
     end if
     !! check std is close to gamma
     !! does not have to be exact due to random numbers and batch size
     if (std .gt. gamma) then
        success = .false.
        write(0,*) 'batchnorm1d layer forward pass failed: &
             &std should equal gamma'
     end if
  end do

  deallocate(output_3d)


!!!-----------------------------------------------------------------------------
!!! test backward pass and check expected output for randomised input
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! run backward pass
  output => bn_layer%output(1,1)
  allocate(output%grad)
  call output%grad%allocate(array_shape=[output%shape, size(output%val, 2)])
  output%grad%val = output%val
  call output%grad%extract(gradient_3d)
  !call output%extract(gradient_3d)
  call output%grad_reverse()
  gradient => input(1,1)%grad
  call gradient%extract(input_3d)

  !! check gradient has expected value
  select type(current => bn_layer)
  class is(batchnorm1d_layer_type)
     params_grad => current%params_array(1)%grad
     do i = 1, width
        mean = sum(input_3d(1,i,:))/real(batch_size)
        std = sqrt(sum((input_3d(1,i,:) - mean)**2)/real(batch_size))
        if (abs(mean) .gt. tol) then
           success = .false.
           write(0,*) 'batchnorm1d layer backward pass failed: &
                &mean gradient should be zero'
        end if
        !! does not have to be exact due to random numbers and batch size
        if (abs(std) .gt. 1.E-2) then
           success = .false.
           write(0,*) 'batchnorm1d layer backward pass failed: &
                &std gradient should equal gamma'
        end if
        if (abs(params_grad%val(current%num_channels+i,1) - &
             sum(gradient_3d(1,i,:))) .gt. tol) then
           success = .false.
           write(0,*) 'batchnorm1d layer backward pass failed: &
                &std gradient should equal sum of gradients'
           write(0,*) "Expected:", gradient_3d(1,i,:)
           write(0,*) "Found:", params_grad%val(current%num_channels+i,1)
        end if
     end do
  end select

  call input(1,1)%reset_graph()
  call input(1,1)%deallocate()
  deallocate(input_3d, gradient_3d)


!-------------------------------------------------------------------------------
! check handling of layer parameters and gradients
!-------------------------------------------------------------------------------
  select type(bn_layer)
  class is(learnable_layer_type)
     !! check parameters
     num_params = bn_layer%get_num_params()
     if (num_params .ne. 2 * width)then
        write(0,*) 'batchnorm1d layer has wrong number of parameters'
        success = .false.
     end if
     allocate(params1(num_params), source = 12.E0)
     call bn_layer%set_params(params1)
     params2 = bn_layer%get_params()
     if(any(abs(params1 - params2).gt.1.E-6))then
        write(0,*) 'batchnorm1d layer has wrong parameters'
        success = .false.
     end if

     !! check gradients
     deallocate(params1, params2)
     allocate(params1(num_params), source = 15.E0)
     call bn_layer%set_gradients(params1)
     params2 = bn_layer%get_gradients()
     if(any(abs(params1 - params2).gt.1.E-6))then
        write(0,*) 'batchnorm1d layer has wrong gradients'
        write(0,*) "Expected:", params1
        write(0,*) "Found:", params2
        success = .false.
     end if
     call bn_layer%set_gradients(20.E0)
     params2 = bn_layer%get_gradients()
     if(any(abs(params2 - 20.E0).gt.1.E-6))then
        write(0,*) 'batchnorm1d layer has wrong gradients'
        write(0,*) "Expected:", 20._real32
        write(0,*) "Found:", params2
        success = .false.
     end if
  class default
     write(0,*) 'batchnorm1d layer has wrong type'
     success = .false.
  end select


!-------------------------------------------------------------------------------
! check layer operations
!-------------------------------------------------------------------------------
  bn_layer1 = batchnorm1d_layer_type( &
       input_shape=[2], &
       batch_size=1 &
  )
  bn_layer2 = batchnorm1d_layer_type( &
       input_shape=[2], &
       batch_size=1 &
  )
  select type(bn_layer1)
  type is(batchnorm1d_layer_type)
     call bn_layer1%set_gradients(1.E0)
     select type(bn_layer2)
     type is(batchnorm1d_layer_type)
        call bn_layer2%set_gradients(2.E0)
        bn_layer = bn_layer1 + bn_layer2
        select type(bn_layer)
        type is(batchnorm1d_layer_type)
           !! check layer addition
           call compare_batchnorm1d_layers(&
                bn_layer, bn_layer1, bn_layer2, success)

           ! !! commented out for now due to pointer sharing issue when assigning
           ! !! ... i.e. all %grad pointers end up pointing to same location
           ! !! ... which means one can wipe the other
           ! !! check layer reduction
           ! bn_layer = bn_layer1
           ! call bn_layer%reduce(bn_layer2)
           ! call compare_batchnorm1d_layers(&
           !      bn_layer, bn_layer1, bn_layer2, success)
        class default
           success = .false.
           write(0,*) 'batchnorm1d layer has wrong type'
        end select
     class default
        success = .false.
        write(0,*) 'batchnorm1d layer has wrong type'
     end select
  class default
     success = .false.
     write(0,*) 'batchnorm1d layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! check output request using rank 1 and rank 2 arrays is consistent
!-------------------------------------------------------------------------------
  allocate(output_1d(width*batch_size))
  allocate(output_data(width, batch_size))
  call bn_layer%extract_output(output_1d)
  call bn_layer%extract_output(output_data)
  if(any(abs(output_1d - reshape(output_data, [size(output_data)])) .gt. 1.E-6))then
     success = .false.
     write(0,*) 'output_1d and output_2d are not consistent'
  end if
  deallocate(output_1d, output_data)


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_batchnorm1d_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("BATCHNORM1D")')
  call bn_layer%print_to_unit(unit)
  write(unit,'("END BATCHNORM1D")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_batchnorm1d_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_batchnorm1d_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (batchnorm1d_layer_type)
     if (.not. read_layer%name .eq. 'batchnorm1d') then
        success = .false.
        write(0,*) 'read batchnorm1d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not batchnorm1d_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_batchnorm1d_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_batchnorm1d_layer passed all tests'
  else
     write(0,*) 'test_batchnorm1d_layer failed one or more tests'
     stop 1
  end if

contains

!-------------------------------------------------------------------------------
! compare three layers
!-------------------------------------------------------------------------------
  subroutine compare_batchnorm1d_layers(layer1, layer2, layer3, success)
    type(batchnorm1d_layer_type), intent(in) :: layer1, layer2, layer3
    logical, intent(inout) :: success

    real, allocatable, dimension(:) :: gradients1, gradients2, gradients3

    gradients1 = layer1%get_gradients()
    gradients2 = layer2%get_gradients()
    gradients3 = layer3%get_gradients()
    if(any(abs(gradients1-gradients2-gradients3).gt.tol))then
       success = .false.
       write(0,*) 'batchnorm1d layer has wrong gradients'
       write(0,*) "Expected:", gradients2 + gradients3
       write(0,*) "Found:", gradients1
    end if

  end subroutine compare_batchnorm1d_layers

end program test_batchnorm1d_layer
