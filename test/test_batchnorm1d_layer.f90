program test_batchnorm1d_layer
  use athena, only: &
     batchnorm1d_layer_type, &
     base_layer_type, &
     learnable_layer_type
  implicit none

  class(base_layer_type), allocatable :: bn_layer, bn_layer1, bn_layer2
  integer, parameter :: width = 8, batch_size = 5
  real, parameter :: gamma  = 0.5, beta = 0.3
  real, allocatable, dimension(:,:) :: input_data, output, gradient
  real, allocatable, dimension(:) :: output_1d, params1, params2
  real, parameter :: tol = 0.5E-3
  logical :: success = .true.

  integer :: i, j, output_width, num_params
  integer :: seed_size = 1
  real :: mean, std
  integer, allocatable, dimension(:) :: seed
  real, parameter :: max_value = 3.0


!!!-----------------------------------------------------------------------------
!!! Initialize random number generator with a seed
!!!-----------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=0)
  call random_seed(put = seed)


!!!-----------------------------------------------------------------------------
!!! test num_channels and num_inputs
!!!-----------------------------------------------------------------------------
  bn_layer = batchnorm1d_layer_type(input_shape=[width], batch_size=batch_size)
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
  elseif(size(bn_layer%input_shape) .ne. 1)then
    success = .false.
    write(0,*) 'batchnorm1d layer has wrong input_shape size'
  elseif(bn_layer%input_shape(1) .ne. width)then
    success = .false.
    write(0,*) 'batchnorm1d layer has wrong input_shape'
  end if
  deallocate(bn_layer)
  bn_layer = batchnorm1d_layer_type(num_inputs=width, batch_size=batch_size)
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
  elseif(size(bn_layer%input_shape) .ne. 1)then
    success = .false.
    write(0,*) 'batchnorm1d layer has wrong input_shape size'
  elseif(bn_layer%input_shape(1) .ne. width)then
    success = .false.
    write(0,*) 'batchnorm1d layer has wrong input_shape'
  end if
  deallocate(bn_layer)
  bn_layer = batchnorm1d_layer_type(num_channels=width, batch_size=batch_size)
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
  elseif(size(bn_layer%input_shape) .ne. 1)then
    success = .false.
    write(0,*) 'batchnorm1d layer has wrong input_shape size'
  elseif(bn_layer%input_shape(1) .ne. width)then
    success = .false.
    write(0,*) 'batchnorm1d layer has wrong input_shape'
  end if
  deallocate(bn_layer)


!!!-----------------------------------------------------------------------------
!!! set up layer
!!!-----------------------------------------------------------------------------
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

  !! check layer name
  if(.not. bn_layer%name .eq. 'batchnorm1d')then
    success = .false.
    write(0,*) 'batchnorm1d layer has wrong name'
  end if

  !! check layer type
  select type(bn_layer)
  type is(batchnorm1d_layer_type)
    !! check input shape
    if(any(bn_layer%input_shape .ne. [width]))then
      success = .false.
      write(0,*) 'batchnorm1d layer has wrong input_shape'
    end if

    !! check output shape
    if(any(bn_layer%output_shape .ne. [width]))then
      success = .false.
      write(0,*) 'batchnorm1d layer has wrong output_shape'
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
  !! initialise sample input
  allocate(input_data(width, batch_size), source = 0.0)
  input_data = max_value

  !! run forward pass
  call bn_layer%forward(input_data)
  call bn_layer%get_output(output)

  !! check outputs all get normalised to zero
  if (any(output-beta.gt. tol)) then
    success = .false.
    write(0,*) 'batchnorm1d layer forward pass failed: &
         &output should all equal beta'
  end if


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output for randomised input
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  call random_number(input_data)

  !! run forward pass
  call bn_layer%forward(input_data)
  call bn_layer%get_output(output)

  !! check outputs all get normalised to zero
  do i = 1, width
     mean = sum(output(i,:))/real(batch_size)
     std = sqrt(sum((output(i,:) - mean)**2)/real(batch_size))
     if (abs(mean - beta) .gt. 1.E-3) then
       success = .false.
       write(0,*) 'batchnorm1d layer forward pass failed: &
           &mean should equal beta'
     end if
     !! check std is close to gamma
     !! does not have to be exact due to random numbers and batch size
     if (std .gt. gamma) then
       success = .false.
       write(0,*) 'batchnorm1d layer forward pass failed: &
           &std should equal gamma'
     end if
  end do


!!!-----------------------------------------------------------------------------
!!! test backward pass and check expected output for randomised input
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! run backward pass
  allocate(gradient, source = output)
  call bn_layer%backward(input_data, gradient)

  !! check gradient has expected value
  select type(current => bn_layer)
  type is(batchnorm1d_layer_type)
     do i = 1, width
        mean = sum(current%di(:,i,:))/real(batch_size)
        std = sqrt(sum((current%di(:,i,:) - mean)**2)/real(batch_size))
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
        if (abs(current%db(i) - sum(gradient(i,:))) .gt. tol) then
          success = .false.
          write(0,*) 'batchnorm1d layer backward pass failed: &
                &std gradient should equal sum of gradients'
        end if
     end do
  end select


!!!-----------------------------------------------------------------------------
!!! check handling of layer parameters and gradients
!!!-----------------------------------------------------------------------------
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
       success = .false.
     end if
     call bn_layer%set_gradients(20.E0)
     params2 = bn_layer%get_gradients()
     if(any(abs(params2 - 20.E0).gt.1.E-6))then
       write(0,*) 'batchnorm1d layer has wrong gradients'
       success = .false.
       write(*,*) params2
     end if
  class default
     write(0,*) 'batchnorm1d layer has wrong type'
     success = .false.
  end select


!!!-----------------------------------------------------------------------------
!!! check layer operations
!!!-----------------------------------------------------------------------------
  bn_layer1 = batchnorm1d_layer_type(input_shape=[2], batch_size=1)
  bn_layer2 = batchnorm1d_layer_type(input_shape=[2], batch_size=1)
  select type(bn_layer1)
  type is(batchnorm1d_layer_type)
     bn_layer1%dg = 1.E0
     bn_layer1%db = 1.E0
     select type(bn_layer2)
     type is(batchnorm1d_layer_type)
        bn_layer2%dg = 2.E0
        bn_layer2%db = 2.E0
        bn_layer = bn_layer1 + bn_layer2
        select type(bn_layer)
        type is(batchnorm1d_layer_type)
           !! check layer addition
           call compare_batchnorm1d_layers(&
                bn_layer, bn_layer1, bn_layer2, success)

           !! check layer reduction
           bn_layer = bn_layer1
           call bn_layer%reduce(bn_layer2)
           call compare_batchnorm1d_layers(&
                bn_layer, bn_layer1, bn_layer2, success)

           !! check layer merge
           bn_layer = bn_layer1
           call bn_layer%merge(bn_layer2)
           call compare_batchnorm1d_layers(&
                bn_layer, bn_layer1, bn_layer2, success)
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


!!!-----------------------------------------------------------------------------
!!! check output request using rank 1 and rank 2 arrays is consistent
!!!-----------------------------------------------------------------------------
  call bn_layer%get_output(output_1d)
  call bn_layer%get_output(output)
  if(any(abs(output_1d - reshape(output, [size(output)])) .gt. 1.E-6))then
     success = .false.
     write(0,*) 'output_1d and output_2d are not consistent'
  end if


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_batchnorm1d_layer passed all tests'
  else
     write(0,*) 'test_batchnorm1d_layer failed one or more tests'
     stop 1
  end if

contains

!!!-----------------------------------------------------------------------------
!!! compare three layers
!!!-----------------------------------------------------------------------------
  subroutine compare_batchnorm1d_layers(layer1, layer2, layer3, success)
     type(batchnorm1d_layer_type), intent(in) :: layer1, layer2, layer3
     logical, intent(inout) :: success

     if(any(abs(layer1%dg-layer2%dg-layer3%dg).gt.tol))then
         success = .false.
         write(0,*) 'batchnorm1d layer has wrong gradients'
     end if
     if(any(abs(layer1%db-layer2%db-layer3%db).gt.tol))then
         success = .false.
         write(0,*) 'batchnorm1d layer has wrong gradients'
     end if

  end subroutine compare_batchnorm1d_layers

end program test_batchnorm1d_layer