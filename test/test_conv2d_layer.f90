program test_conv2d_layer
  use athena, only: &
       conv2d_layer_type, &
       input_layer_type, &
       base_layer_type, &
       learnable_layer_type
  use athena__conv2d_layer, only: read_conv2d_layer
  use athena__diffstruc_extd, only: conv2d, add_bias
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(+), operator(-), sum, squared
  implicit none

  class(base_layer_type), allocatable, target :: conv_layer
  class(base_layer_type), allocatable :: conv_layer1, conv_layer2
  class(base_layer_type), allocatable :: input_layer
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: num_filters = 32, kernel_size = 3
  integer :: unit
  type(array_type) :: input(1,1)
  type(array_type), pointer :: output
  real, parameter :: tol = 1.E-7
  logical :: success = .true.

  real, allocatable, dimension(:) :: params
  real, allocatable, dimension(:,:) :: outputs
  character(len=16) :: mode


!-------------------------------------------------------------------------------
! optional modes: `test_conv2d_layer bench | grad`
!-------------------------------------------------------------------------------
  mode = ''
  if(command_argument_count() >= 1) call get_command_argument(1, mode)
  select case(trim(mode))
  case('bench')
     call run_bench()
     stop
  case('grad')
     call run_gradient_check()
     stop
  end select


!-------------------------------------------------------------------------------
! set up layer
!-------------------------------------------------------------------------------
  conv_layer = conv2d_layer_type( &
       num_filters = num_filters, &
       kernel_size = kernel_size &
  )

  !! check layer name
  if(.not. conv_layer%name .eq. 'conv2d')then
     success = .false.
     write(0,*) 'conv2d layer has wrong name'
  end if

  !! check layer type
  select type(conv_layer)
  type is(conv2d_layer_type)
     !! check default layer transfer/activation function
     if(conv_layer%activation%name .ne. 'none')then
        success = .false.
        write(0,*) 'conv2d layer has wrong activation: '//conv_layer%activation%name
     end if

     !! check number of filters
     if(conv_layer%num_filters .ne. num_filters)then
        success = .false.
        write(0,*) 'conv2d layer has wrong num_filters'
     end if

     !! check kernel size
     if(any(conv_layer%knl .ne. kernel_size))then
        success = .false.
        write(0,*) 'conv2d layer has wrong kernel_size'
     end if

     !! check input shape allocated
     if(allocated(conv_layer%input_shape))then
        success = .false.
        write(0,*) 'conv2d layer shape should not be allocated yet'
     end if
  class default
     success = .false.
     write(0,*) 'conv2d layer has wrong type'
  end select


!!!-----------------------------------------------------------------------------
!!! check layer input and output shape based on input layer
!!! conv2d layer: 32 x 32 pixel image, 3 channels
!!!-----------------------------------------------------------------------------
  input_layer = input_layer_type([32,32,3])
  call conv_layer%init(input_layer%input_shape)
  select type(conv_layer)
  type is(conv2d_layer_type)
     if(any(conv_layer%input_shape .ne. [32,32,3]))then
        success = .false.
        write(0,*) 'conv2d layer has wrong input_shape'
     end if
     if(any(conv_layer%output_shape .ne. [30,30,num_filters]))then
        success = .false.
        write(0,*) 'conv2d layer has wrong output shape'
     end if
  end select


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  !! conv2d layer: 3x3 pixel image, 1 channel
  call input(1,1)%allocate(array_shape=[3,3,1,1], source = 0._real32)
  call input(1,1)%set_requires_grad(.true.)
  conv_layer = conv2d_layer_type( &
       num_filters = 1, &
       kernel_size = 3, &
       activation = 'sigmoid' &
  )
  call conv_layer%init(input(1,1)%shape)

  !! run forward pass
  call conv_layer%forward(input)
  output => conv_layer%output(1,1)

  !! check outputs have expected value
  if(any(abs(output%val(:,1) - 0.5) .gt. tol))then
     success = .false.
     write(*,*) 'conv2d layer with zero input and sigmoid activation must &
          &return outputs all equal to 0.5'
     write(*,*) output%val(:,1)
  end if


!-------------------------------------------------------------------------------
! check handling of layer parameters, gradients, and outputs
!-------------------------------------------------------------------------------
  select type(conv_layer)
  class is(learnable_layer_type)
     !! check layer parameter handling
     params = conv_layer%get_params()
     if(size(params) .eq. 0)then
        success = .false.
        write(0,*) 'conv2d layer has wrong number of parameters'
     end if
     params = 1.E0
     call conv_layer%set_params(params)
     params = conv_layer%get_params()
     if(any(abs(params - 1.E0).gt.tol))then
        success = .false.
        write(0,*) 'conv2d layer has wrong parameters'
     end if

     !! check layer gradient handling
     params = conv_layer%get_gradients()
     if(size(params) .eq. 0)then
        success = .false.
        write(0,*) 'conv2d layer has wrong number of gradients'
     end if
     params = 1.E0
     call conv_layer%set_gradients(params)
     params = conv_layer%get_gradients()
     if(any(abs(params - 1.E0).gt.tol))then
        success = .false.
        write(0,*) 'conv2d layer has wrong gradients'
     end if
     call conv_layer%set_gradients(10.E0)
     params = conv_layer%get_gradients()
     if(any(abs(params - 10.E0).gt.tol))then
        success = .false.
        write(0,*) 'conv2d layer has wrong gradients'
     end if

     !! check layer output handling
     output => conv_layer%output(1,1)
     if(size(output%val,dim=1) .ne. product(conv_layer%output_shape))then
        success = .false.
        write(0,*) 'conv2d layer has wrong number of outputs'
     end if
     if(any(shape(output%val) .ne. [product(conv_layer%output_shape), 1]))then
        success = .false.
        write(0,*) 'conv2d layer has wrong number of outputs'
     end if
  end select


!-------------------------------------------------------------------------------
! check expected initialisation of kernel size and stride
!-------------------------------------------------------------------------------
  conv_layer = conv2d_layer_type( &
       kernel_size = [2, 2], &
       stride = [2, 2] &
  )
  select type(conv_layer)
  type is (conv2d_layer_type)
     if(any(conv_layer%knl .ne. 2))then
        success = .false.
        write(0,*) 'conv2d layer has wrong pool size'
     end if
     if(any(conv_layer%stp .ne. 2))then
        success = .false.
        write(0,*) 'conv2d layer has wrong stride size'
     end if
  end select


!-------------------------------------------------------------------------------
! check expected initialisation of kernel size and stride
!-------------------------------------------------------------------------------
  conv_layer = conv2d_layer_type( &
       kernel_size = [4], &
       stride = [4] &
  )
  select type(conv_layer)
  type is (conv2d_layer_type)
     if(any(conv_layer%knl .ne. 4))then
        success = .false.
        write(0,*) 'conv2d layer has wrong pool size'
     end if
     if(any(conv_layer%stp .ne. 4))then
        success = .false.
        write(0,*) 'conv2d layer has wrong stride size'
     end if
  end select


!-------------------------------------------------------------------------------
! check layer operations
!-------------------------------------------------------------------------------
  allocate(conv_layer1, source = conv2d_layer_type( &
       num_filters = 1, &
       kernel_size = 3, &
       activation = 'sigmoid' &
  ))
  call conv_layer1%init(input_layer%input_shape)
  allocate(conv_layer2, source = conv2d_layer_type( &
       num_filters = 1, &
       kernel_size = 3, &
       activation = 'sigmoid' &
  ))
  call conv_layer2%init(input_layer%input_shape)
  select type(conv_layer1)
  type is(conv2d_layer_type)
     select type(conv_layer2)
     type is(conv2d_layer_type)
        if(allocated(conv_layer)) deallocate(conv_layer)
        conv_layer = conv_layer1 + conv_layer2
        select type(conv_layer)
        type is(conv2d_layer_type)
           !! check layer addition
           call compare_conv2d_layers(&
                conv_layer, conv_layer1, success, conv_layer2)

           !! check layer reduction
           conv_layer = conv_layer1
           call conv_layer%reduce(conv_layer2)
           call compare_conv2d_layers(&
                conv_layer, conv_layer1, success, conv_layer2)
        class default
           success = .false.
           write(0,*) 'conv2d layer has wrong type'
        end select
     class default
        success = .false.
        write(0,*) 'conv2d layer has wrong type'
     end select
  class default
     success = .false.
     write(0,*) 'conv2d layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_conv2d_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("CONV2D")')
  call conv_layer%print_to_unit(unit)
  write(unit,'("END CONV2D")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_conv2d_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_conv2d_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (conv2d_layer_type)
     if(.not. read_layer%name .eq. 'conv2d')then
        success = .false.
        write(0,*) 'read conv2d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not conv2d_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_conv2d_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! Check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_conv2d_layer passed all tests'
  else
     write(0,*) 'test_conv2d_layer failed one or more tests'
     stop 1
  end if


contains

!-------------------------------------------------------------------------------
! finite-difference check of the conv2d value partials
!-------------------------------------------------------------------------------
  subroutine run_gradient_check()
    integer, parameter :: N_CASES = 4
    integer, parameter :: STRIDES(N_CASES) = [1, 2, 1, 2]
    integer, parameter :: DILATIONS(N_CASES) = [1, 1, 2, 2]
    type(conv2d_layer_type) :: layer
    type(array_type) :: xin(1,1)
    type(array_type), pointer :: loss
    real(real32), allocatable :: x0(:,:), w0(:), b0(:), ana_x(:,:), ana_w(:), ana_b(:)
    real(real32) :: eps, num
    real(8) :: num_loss
    integer :: c, i, s, n_w, n_b, fails

    eps = 1.E-2_real32
    fails = 0
    do c = 1, N_CASES
       layer = conv2d_layer_type( &
            input_shape = [5,5,2], num_filters = 3, kernel_size = 3, &
            stride = STRIDES(c), dilation = DILATIONS(c), &
            padding = "none", activation = "none" &
       )
       call xin(1,1)%allocate(array_shape=[5,5,2,2])
       xin(1,1)%is_temporary = .false.
       call xin(1,1)%set_requires_grad(.true.)
       call random_number(xin(1,1)%val)
       call random_number(layer%params(1)%val)
       layer%params(1)%val = (layer%params(1)%val - 0.5_real32) * 0.2_real32
       call random_number(layer%params(2)%val)
       layer%params(2)%val = (layer%params(2)%val - 0.5_real32) * 0.2_real32

       ! analytic gradients
       call layer%forward(xin)
       loss => sum(squared(layer%output(1,1)))
       call loss%grad_reverse()
       n_w = size(layer%params(1)%val)
       n_b = size(layer%params(2)%val)
       allocate(x0(5*5*2,2), ana_x(5*5*2,2), w0(n_w), ana_w(n_w), &
            b0(n_b), ana_b(n_b))
       x0 = xin(1,1)%val
       w0 = layer%params(1)%val(:,1)
       b0 = layer%params(2)%val(:,1)
       ana_x = xin(1,1)%grad%val
       ana_w = layer%params(1)%grad%val(:,1)
       ana_b = layer%params(2)%grad%val(:,1)
       call loss%release_expression()

       ! numeric: kernel
       do i = 1, n_w
          layer%params(1)%val(i,1) = w0(i) + eps
          num_loss = numeric_loss(layer, xin)
          layer%params(1)%val(i,1) = w0(i) - eps
          num_loss = num_loss - numeric_loss(layer, xin)
          layer%params(1)%val(i,1) = w0(i)
          num = real(num_loss / real(2._8 * eps, 8), real32)
          call check_value(ana_w(i), num, eps, 'kernel', c, fails)
       end do
       ! numeric: bias
       do i = 1, n_b
          layer%params(2)%val(i,1) = b0(i) + eps
          num_loss = numeric_loss(layer, xin)
          layer%params(2)%val(i,1) = b0(i) - eps
          num_loss = num_loss - numeric_loss(layer, xin)
          layer%params(2)%val(i,1) = b0(i)
          num = real(num_loss / real(2._8 * eps, 8), real32)
          call check_value(ana_b(i), num, eps, 'bias', c, fails)
       end do
       ! numeric: input
       do s = 1, 2
          do i = 1, 5*5*2
             xin(1,1)%val(i,s) = x0(i,s) + eps
             num_loss = numeric_loss(layer, xin)
             xin(1,1)%val(i,s) = x0(i,s) - eps
             num_loss = num_loss - numeric_loss(layer, xin)
             xin(1,1)%val(i,s) = x0(i,s)
             num = real(num_loss / real(2._8 * eps, 8), real32)
             call check_value(ana_x(i,s), num, eps, 'input', c, fails)
          end do
       end do

       deallocate(x0, ana_x, w0, ana_w, b0, ana_b)
       call xin(1,1)%deallocate()
    end do

    if(fails == 0)then
       write(*,*) 'conv2d finite-difference gradients passed'
    else
       write(0,*) 'conv2d finite-difference gradients failed:', fails
       stop 1
    end if
  end subroutine run_gradient_check

!-------------------------------------------------------------------------------
  real(8) function numeric_loss(layer, xin) result(l)
    type(conv2d_layer_type), intent(inout) :: layer
    type(array_type), intent(in) :: xin(1,1)
    call layer%forward(xin)
    l = sum(real(layer%output(1,1)%val, 8)**2)
  end function numeric_loss

!-------------------------------------------------------------------------------
  subroutine check_value(analytic, numeric, eps, name, case_id, fails)
    real(real32), intent(in) :: analytic, numeric, eps
    character(len=*), intent(in) :: name
    integer, intent(in) :: case_id
    integer, intent(inout) :: fails
    real(real32) :: rel
    rel = abs(analytic - numeric) / max(1.E-4_real32, abs(numeric))
    if(rel > 5.E-2_real32)then
       fails = fails + 1
       if(fails <= 5) &
            write(0,'("grad mismatch case ",I0," ",A,&
                 &" analytic=",E12.4," numeric=",E12.4)') &
            case_id, name, analytic, numeric
    end if
  end subroutine check_value

!-------------------------------------------------------------------------------
! forward/backward timing for a representative conv2d layer
!-------------------------------------------------------------------------------
  subroutine run_bench()
    type(conv2d_layer_type) :: layer
    type(array_type) :: xin(1,1), kernel, bias
    type(array_type), pointer :: loss, raw, ab
    real(8) :: t0, t1, t_fwd, t_rev, t_raw
    integer :: iter, c
    integer, parameter :: N_FWD = 200, N_REV = 100
    character(len=6), parameter :: PADS(4) = [character(len=6) :: &
         "none", "same", "none", "same"]
    character(len=6), parameter :: ACTS(4) = [character(len=6) :: &
         "none", "none", "relu", "relu"]

    call xin(1,1)%allocate(array_shape=[28,28,1,32])
    xin(1,1)%is_temporary = .false.
    call xin(1,1)%set_requires_grad(.true.)
    call random_number(xin(1,1)%val)
    call kernel%allocate(array_shape=[3,3,1,32,1])
    call random_number(kernel%val)
    kernel%is_temporary = .false.
    call bias%allocate(array_shape=[32,1])
    call random_number(bias%val)
    bias%is_temporary = .false.
    call bias%set_requires_grad(.true.)

    ! raw convolution op
    raw => conv2d(xin(1,1), kernel, [1,1], [1,1])
    call raw%release_expression()
    call cpu_time(t0)
    do iter = 1, N_FWD
       raw => conv2d(xin(1,1), kernel, [1,1], [1,1])
       call raw%release_expression()
    end do
    call cpu_time(t1)
    t_raw = (t1 - t0) / real(N_FWD, 8) * 1000._8

    raw => conv2d(xin(1,1), kernel, [1,1], [1,1])
    loss => sum(squared(raw))
    call loss%grad_reverse()
    call loss%release_expression()
    call cpu_time(t0)
    do iter = 1, N_REV
       raw => conv2d(xin(1,1), kernel, [1,1], [1,1])
       loss => sum(squared(raw))
       call loss%grad_reverse()
       call loss%release_expression()
    end do
    call cpu_time(t1)
    t_rev = (t1 - t0) / real(N_REV, 8) * 1000._8 - t_raw

    ! conv + bias
    raw => conv2d(xin(1,1), kernel, [1,1], [1,1])
    ab => add_bias(raw, bias, 3, .true.)
    loss => sum(squared(ab))
    call loss%grad_reverse()
    call loss%release_expression()
    call cpu_time(t0)
    do iter = 1, N_FWD
       raw => conv2d(xin(1,1), kernel, [1,1], [1,1])
       ab => add_bias(raw, bias, 3, .true.)
       call ab%release_expression()
    end do
    call cpu_time(t1)
    t_fwd = (t1 - t0) / real(N_FWD, 8) * 1000._8
    write(*,'(A)') ""
    write(*,'(A)') "=== conv2d benchmark (28x28, 32 filters, b=32) ==="
    write(*,'(" raw op(ms)=",F8.3,"  raw rev(ms)=",F8.3, &
         &"  conv+bias fwd(ms)=",F8.3)') t_raw, t_rev, t_fwd
    write(*,'(" op GFLOP/s=",F8.1)') &
         2._8 * 28._8 * 28._8 * 9._8 * 32._8 * 32._8 / (t_raw * 1.E6_8)
    write(*,'(A)') " layer    pad     act     fwd(ms)   rev(ms)"

    do c = 1, 4
       layer = conv2d_layer_type( &
            input_shape = [28,28,1], num_filters = 32, kernel_size = 3, &
            stride = 1, padding = trim(PADS(c)), activation = trim(ACTS(c)) &
       )
       call layer%forward(xin)
       loss => sum(squared(layer%output(1,1)))
       call loss%grad_reverse()
       call loss%release_expression()

       call cpu_time(t0)
       do iter = 1, N_FWD
          call layer%forward(xin)
       end do
       call cpu_time(t1)
       t_fwd = (t1 - t0) / real(N_FWD, 8) * 1000._8

       call layer%forward(xin)
       call cpu_time(t0)
       do iter = 1, N_REV
          call layer%forward(xin)
          loss => sum(squared(layer%output(1,1)))
          call loss%grad_reverse()
          call loss%release_expression()
       end do
       call cpu_time(t1)
       t_rev = (t1 - t0) / real(N_REV, 8) * 1000._8 - t_fwd

       write(*,'(I6,2x,A6,2x,A6,2(F9.3))') c, trim(PADS(c)), trim(ACTS(c)), &
            t_fwd, t_rev
    end do

    call xin(1,1)%deallocate()
    call kernel%deallocate()
  end subroutine run_bench

!-------------------------------------------------------------------------------
! compare two or three layers
!-------------------------------------------------------------------------------
  subroutine compare_conv2d_layers(layer1, layer2, success, layer3)
    type(conv2d_layer_type), intent(in) :: layer1, layer2
    logical, intent(inout) :: success
    type(conv2d_layer_type), optional, intent(in) :: layer3

    if(all(layer1%knl .ne. layer2%knl))then
       success = .false.
       write(0,*) 'conv2d layer has wrong kernel_size'
    end if
    if(layer1%num_filters .ne. layer2%num_filters)then
       success = .false.
       write(0,*) 'conv2d layer has wrong num_filters'
    end if
    if(layer1%activation%name .ne. 'sigmoid')then
       success = .false.
       write(0,*) 'conv2d layer has wrong activation: '//&
            layer1%activation%name
    end if
    if(present(layer3))then
       if( &
            associated(layer1%params(1)%grad).and. &
            associated(layer2%params(1)%grad).and. &
            associated(layer3%params(1)%grad) &
       )then
          if(any(abs( &
               layer1%params(1)%grad%val - &
               layer2%params(1)%grad%val - &
               layer3%params(1)%grad%val &
          ).gt.1.E-6))then
             success = .false.
             write(0,*) 'conv1d layer has wrong gradients'
          end if
       end if

       if( &
            associated(layer1%params(2)%grad).and. &
            associated(layer2%params(2)%grad).and. &
            associated(layer3%params(2)%grad) &
       )then
          if(any(abs( &
               layer1%params(2)%grad%val - &
               layer2%params(2)%grad%val - &
               layer3%params(2)%grad%val &
          ).gt.1.E-6))then
             success = .false.
             write(0,*) 'conv1d layer has wrong bias gradients'
          end if
       end if
    end if

  end subroutine compare_conv2d_layers

end program test_conv2d_layer
