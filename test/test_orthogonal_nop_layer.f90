program test_orthogonal_nop_block
  !! Tests for the Orthogonal Neural Operator layer
  use athena, only: &
       orthogonal_nop_block_type, &
       base_layer_type
  use athena__orthogonal_nop_block, only: read_orthogonal_nop_block
  use diffstruc, only: array_type, squared
  use diffstruc, only: diff_sum => sum
  use coreutils, only: real32
  implicit none

  class(base_layer_type), allocatable :: layer1, layer2, layer3
  class(base_layer_type), allocatable :: read_layer
  integer :: unit
  logical :: success = .true.


!-------------------------------------------------------------------------------
! Deterministic seed so randomised parameters and inputs are reproducible
!-------------------------------------------------------------------------------
  block
    integer :: seed_size
    integer, allocatable :: seed(:)
    call random_seed(size=seed_size)
    allocate(seed(seed_size))
    seed = 20261127
    call random_seed(put=seed)
    deallocate(seed)
  end block


!-------------------------------------------------------------------------------
! Test 1: basic layer construction with explicit num_inputs
!-------------------------------------------------------------------------------
  write(*,*) "Test 1: layer construction..."
  layer1 = orthogonal_nop_block_type( &
       num_inputs=8, num_outputs=6, num_basis=3)

  if(layer1%name .ne. 'orthogonal_nop')then
     success = .false.
     write(0,*) 'orthogonal_nop layer has wrong name: '//trim(layer1%name)
  end if

  if(any(layer1%input_shape .ne. [8]))then
     success = .false.
     write(0,*) 'orthogonal_nop layer has wrong input_shape'
  end if

  if(any(layer1%output_shape .ne. [6]))then
     success = .false.
     write(0,*) 'orthogonal_nop layer has wrong output_shape'
  end if

  select type(layer1)
  type is(orthogonal_nop_block_type)
     if(layer1%num_inputs .ne. 8)then
        success = .false.
        write(0,*) 'orthogonal_nop layer has wrong num_inputs'
     end if
     if(layer1%num_outputs .ne. 6)then
        success = .false.
        write(0,*) 'orthogonal_nop layer has wrong num_outputs'
     end if
     if(layer1%num_basis .ne. 3)then
        success = .false.
        write(0,*) 'orthogonal_nop layer has wrong num_basis'
     end if
     if(layer1%activation%name .ne. 'none')then
        success = .false.
        write(0,*) 'orthogonal_nop layer has wrong default activation: '// &
             layer1%activation%name
     end if
     ! num_params = k^2 + n_in*k + n_out*n_in + n_out (bias)
     !            = 9 + 24 + 48 + 6 = 87
     if(layer1%num_params .ne. 87)then
        success = .false.
        write(0,'("orthogonal_nop layer has wrong num_params: ",I0)') &
             layer1%num_params
     end if
     ! Check basis is allocated
     block
       type(array_type) :: phi
       phi = layer1%get_bases()
       if(.not.phi%allocated)then
          success = .false.
          write(0,*) 'orthogonal_nop phi not allocated'
       end if
     end block
  class default
     success = .false.
     write(0,*) 'layer1 has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 2: orthogonality check
!-------------------------------------------------------------------------------
  write(*,*) "Test 2: orthogonality check..."

  select type(layer1)
  type is(orthogonal_nop_block_type)
     block
       real(real32) :: orth_metric
       orth_metric = layer1%get_orthogonality_metric()
       write(*,'("  Orthogonality metric max(|Q^T Q - I|) = ",ES12.5)') &
            orth_metric
       if(orth_metric .gt. 1.0e-5_real32)then
          success = .false.
          write(0,'("Basis not orthogonal! metric = ",ES12.5)') orth_metric
       end if
     end block
  end select


!-------------------------------------------------------------------------------
! Test 3: deferred initialisation
!-------------------------------------------------------------------------------
  write(*,*) "Test 3: deferred initialisation..."
  layer2 = orthogonal_nop_block_type(num_outputs=4, num_basis=2)
  call layer2%init(layer1%output_shape)

  if(any(layer2%input_shape .ne. [6]))then
     success = .false.
     write(0,*) 'deferred orthogonal_nop layer has wrong input_shape'
  end if
  if(any(layer2%output_shape .ne. [4]))then
     success = .false.
     write(0,*) 'deferred orthogonal_nop layer has wrong output_shape'
  end if


!-------------------------------------------------------------------------------
! Test 4: no-bias layer
!-------------------------------------------------------------------------------
  write(*,*) "Test 4: no-bias layer..."
  layer3 = orthogonal_nop_block_type( &
       num_inputs=4, num_outputs=4, num_basis=2, use_bias=.false.)

  select type(layer3)
  type is(orthogonal_nop_block_type)
     if(layer3%use_bias)then
        success = .false.
        write(0,*) 'no-bias orthogonal_nop layer has use_bias = T'
     end if
     ! num_params = 4 + 8 + 16 = 28 (no bias)
     if(layer3%num_params .ne. 28)then
        success = .false.
        write(0,'("no-bias layer has wrong num_params: ",I0)') &
             layer3%num_params
     end if
  class default
     success = .false.
     write(0,*) 'layer3 has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 5: activation function
!-------------------------------------------------------------------------------
  write(*,*) "Test 5: activation function..."
  layer1 = orthogonal_nop_block_type( &
       num_inputs=4, num_outputs=3, num_basis=2, activation="relu")

  select type(layer1)
  type is(orthogonal_nop_block_type)
     if(layer1%activation%name .ne. 'relu')then
        success = .false.
        write(0,*) 'orthogonal_nop layer has wrong activation: '// &
             layer1%activation%name
     end if
  class default
     success = .false.
     write(0,*) 'layer1 has wrong type after re-assignment'
  end select


!-------------------------------------------------------------------------------
! Test 6: forward pass produces correct output shape
!-------------------------------------------------------------------------------
  write(*,*) "Test 6: forward pass..."
  layer1 = orthogonal_nop_block_type( &
       num_inputs=6, num_outputs=4, num_basis=3, &
       use_bias=.true., activation="none")

  block
    type(array_type), dimension(1,1) :: inp
    call inp(1,1)%allocate([6, 1], source=1.0_real32)
    call layer1%forward(inp)
    if(any(layer1%output(1,1)%shape .ne. [4]))then
       success = .false.
       write(0,*) 'forward pass produced wrong output shape'
    end if
    call inp(1,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Test 7: forward pass with activation
!-------------------------------------------------------------------------------
  write(*,*) "Test 7: forward pass with activation..."
  layer1 = orthogonal_nop_block_type( &
       num_inputs=6, num_outputs=4, num_basis=3, &
       use_bias=.true., activation="relu")

  block
    type(array_type), dimension(1,1) :: inp
    call inp(1,1)%allocate([6, 2], source=0.5_real32)
    call layer1%forward(inp)
    if(any(layer1%output(1,1)%shape .ne. [4]))then
       success = .false.
       write(0,*) 'forward pass with activation produced wrong output shape'
    end if
    call inp(1,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Test 8: gradient propagation (via forward + backward)
!-------------------------------------------------------------------------------
  write(*,*) "Test 8: gradient propagation..."
  layer1 = orthogonal_nop_block_type( &
       num_inputs=4, num_outputs=3, num_basis=2, &
       use_bias=.true., activation="none")

  block
    type(array_type), dimension(1,1) :: inp
    real(real32) :: loss_val

    call inp(1,1)%allocate([4, 1], source=1.0_real32)
    call inp(1,1)%set_requires_grad(.true.)

    call layer1%forward(inp)

    ! Verify output is finite and correct shape
    loss_val = sum(layer1%output(1,1)%val(:,1))
    write(*,'("  Forward output sum: ",ES12.5)') loss_val

    if(loss_val .ne. loss_val)then
       success = .false.
       write(0,*) 'Forward pass produced NaN'
    end if

    ! Trigger backward pass via the autodiff graph
    call layer1%output(1,1)%grad_reverse()
    if(.not.associated(inp(1,1)%grad))then
       success = .false.
       write(0,*) 'input gradient was not allocated by the reverse pass'
    else
       write(*,*) "  Gradient propagation completed without error."
    end if
    call inp(1,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Test 9: file I/O
!-------------------------------------------------------------------------------
  write(*,*) "Test 9: file I/O..."

  layer1 = orthogonal_nop_block_type( &
       num_inputs=5, num_outputs=3, num_basis=2, &
       activation="relu")

  open(newunit=unit, file='test_orthogonal_nop_block.txt', &
       status='replace', action='write')
  write(unit,'(A)') 'ORTHOGONAL_NOP'
  call layer1%print_to_unit(unit)
  write(unit,'(A)') 'END ORTHOGONAL_NOP'
  close(unit)

  open(newunit=unit, file='test_orthogonal_nop_block.txt', &
       status='old', action='read')
  block
    character(256) :: buffer
    read(unit,'(A)') buffer   ! Read 'ORTHOGONAL_NOP' header
  end block
  read_layer = read_orthogonal_nop_block(unit)
  close(unit, status='delete')

  select type(rl => read_layer)
  type is(orthogonal_nop_block_type)
     if(rl%num_inputs .ne. 5)then
        success = .false.
        write(0,*) 'read layer has wrong num_inputs'
     end if
     if(rl%num_outputs .ne. 3)then
        success = .false.
        write(0,*) 'read layer has wrong num_outputs'
     end if
     if(rl%num_basis .ne. 2)then
        success = .false.
        write(0,*) 'read layer has wrong num_basis'
     end if
     if(rl%activation%name .ne. 'relu')then
        success = .false.
        write(0,*) 'read layer has wrong activation: '//rl%activation%name
     end if
     ! Check parameter values match
     select type(layer1)
     type is(orthogonal_nop_block_type)
        if(maxval(abs(rl%params(1)%val(:,1) - layer1%params(1)%val(:,1))) &
             .gt. 1.0e-6_real32)then
           success = .false.
           write(0,*) 'read layer R params do not match'
        end if
        if(maxval(abs(rl%params(3)%val(:,1) - layer1%params(3)%val(:,1))) &
             .gt. 1.0e-6_real32)then
           success = .false.
           write(0,*) 'read layer W params do not match'
        end if
     end select
  class default
     success = .false.
     write(0,*) 'read_layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 10: forward pass against a plain-array reference
!-------------------------------------------------------------------------------
  write(*,*) "Test 10: forward reference..."
  block
    type(orthogonal_nop_block_type) :: layer
    type(array_type) :: inp(1,1), phi
    integer, parameter :: n_in = 6, n_out = 4, k = 3, batch = 2
    real(real32) :: phi_val(n_in,k), R(k,k), W(n_out,n_in), b(n_out)
    real(real32) :: u(n_in,batch), ref(n_out,batch)

    layer = orthogonal_nop_block_type( &
         num_inputs=n_in, num_outputs=n_out, num_basis=k, &
         use_bias=.true., activation="none")

    call inp(1,1)%allocate([n_in,batch])
    call random_number(inp(1,1)%val)
    call layer%forward(inp)

    phi = layer%get_bases()
    phi_val = reshape(phi%val(:,1), [n_in,k])
    R = reshape(layer%params(1)%val(:,1), [k,k])
    W = reshape(layer%params(3)%val(:,1), [n_out,n_in])
    b = layer%params(4)%val(:,1)
    u = inp(1,1)%val

    ref = matmul(matmul(matmul(matmul(W, phi_val), R), transpose(phi_val)), u) &
         + matmul(W, u)
    ref = ref + spread(b, 2, batch)

    if(maxval(abs(layer%output(1,1)%val - ref)) .gt. 1.0e-4_real32)then
       success = .false.
       write(0,*) 'orthogonal_nop forward output does not match reference'
    end if

    call inp(1,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Test 11: batch columns match separate single-sample forwards
!-------------------------------------------------------------------------------
  write(*,*) "Test 11: batch consistency..."
  block
    type(orthogonal_nop_block_type) :: layer
    type(array_type) :: inp(1,1), col(1,1)
    integer, parameter :: n_in = 6, n_out = 4, k = 3, batch = 2
    real(real32), allocatable :: batch_out(:,:)
    integer :: s

    layer = orthogonal_nop_block_type( &
         num_inputs=n_in, num_outputs=n_out, num_basis=k, &
         use_bias=.true., activation="none")

    call inp(1,1)%allocate([n_in,batch])
    inp(1,1)%is_temporary = .false.
    call random_number(inp(1,1)%val)
    call layer%forward(inp)
    batch_out = layer%output(1,1)%val

    call col(1,1)%allocate([n_in,1])
    col(1,1)%is_temporary = .false.
    do s = 1, batch
       col(1,1)%val(:,1) = inp(1,1)%val(:,s)
       call layer%forward(col)
       if(maxval(abs(layer%output(1,1)%val(:,1) - &
            batch_out(:,s))) .gt. 1.0e-6_real32)then
          success = .false.
          write(0,'("batch consistency failed for sample ",I0)') s
       end if
    end do

    call col(1,1)%deallocate()
    call inp(1,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Test 12: finite-difference gradient check (k=3, batch=2)
!-------------------------------------------------------------------------------
  write(*,*) "Test 12: finite-difference gradients..."
  call run_gradient_check(success)


!-------------------------------------------------------------------------------
! Test 13: multi-batch parameter-gradient reduction
!-------------------------------------------------------------------------------
  write(*,*) "Test 13: multi-batch parameter gradients..."
  call run_param_reduction_check(success)


!-------------------------------------------------------------------------------
! Test 14: repeated init reinitialises without crashing
!-------------------------------------------------------------------------------
  write(*,*) "Test 14: repeated init..."
  block
    type(orthogonal_nop_block_type) :: layer
    type(array_type) :: inp(1,1)
    integer, parameter :: n_in = 6, n_out = 4, k = 3
    integer :: expected

    layer = orthogonal_nop_block_type(num_outputs=n_out, num_basis=k)
    call layer%init([n_in])
    call layer%init([n_in])

    if(any(layer%input_shape .ne. [n_in]))then
       success = .false.
       write(0,*) 'repeated init has wrong input_shape'
    end if
    if(any(layer%output_shape .ne. [n_out]))then
       success = .false.
       write(0,*) 'repeated init has wrong output_shape'
    end if
    expected = k*k + n_in*k + n_out*n_in + n_out
    if(layer%num_params .ne. expected)then
       success = .false.
       write(0,*) 'repeated init has wrong num_params'
    end if
    if(.not.allocated(layer%params))then
       success = .false.
       write(0,*) 'repeated init did not allocate params'
    elseif(size(layer%params) .ne. 4)then
       success = .false.
       write(0,*) 'repeated init allocated wrong number of params'
    end if

    call inp(1,1)%allocate([n_in,1], source=1.0_real32)
    call layer%forward(inp)
    if(any(layer%output(1,1)%shape .ne. [n_out]))then
       success = .false.
       write(0,*) 'repeated init forward produced wrong shape'
    end if
    call inp(1,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Test 15: large multi-record I/O round-trip
!-------------------------------------------------------------------------------
  write(*,*) "Test 15: large I/O round-trip..."
  block
    type(orthogonal_nop_block_type) :: layer
    class(base_layer_type), allocatable :: rl
    type(array_type) :: inp(1,1)
    integer, parameter :: n_in = 40, n_out = 16, k = 8
    logical :: ok

    layer = orthogonal_nop_block_type( &
         num_inputs=n_in, num_outputs=n_out, num_basis=k, &
         use_bias=.true., activation="relu")

    open(newunit=unit, file='test_orthogonal_nop_block_large.txt', &
         status='replace', action='write')
    write(unit,'(A)') 'ORTHOGONAL_NOP'
    call layer%print_to_unit(unit)
    write(unit,'(A)') 'END ORTHOGONAL_NOP'
    close(unit)

    open(newunit=unit, file='test_orthogonal_nop_block_large.txt', &
         status='old', action='read')
    read(unit,*)
    rl = read_orthogonal_nop_block(unit)
    close(unit, status='delete')

    ok = .true.
    select type(rl)
    type is(orthogonal_nop_block_type)
       if(rl%num_inputs .ne. n_in .or. rl%num_outputs .ne. n_out .or. &
            rl%num_basis .ne. k)then
          ok = .false.
       end if
       if(maxval(abs(rl%params(1)%val(:,1) - &
            layer%params(1)%val(:,1))) .gt. 1.0e-6_real32) ok = .false.
       if(maxval(abs(rl%params(2)%val(:,1) - &
            layer%params(2)%val(:,1))) .gt. 1.0e-6_real32) ok = .false.
       if(maxval(abs(rl%params(3)%val(:,1) - &
            layer%params(3)%val(:,1))) .gt. 1.0e-6_real32) ok = .false.
       if(maxval(abs(rl%params(4)%val(:,1) - &
            layer%params(4)%val(:,1))) .gt. 1.0e-6_real32) ok = .false.

       call inp(1,1)%allocate([n_in,1])
       inp(1,1)%is_temporary = .false.
       call random_number(inp(1,1)%val)
       call layer%forward(inp)
       call rl%forward(inp)
       if(maxval(abs(rl%output(1,1)%val - &
            layer%output(1,1)%val)) .gt. 1.0e-5_real32) ok = .false.
       call inp(1,1)%deallocate()
    class default
       ok = .false.
    end select
    if(.not.ok)then
       success = .false.
       write(0,*) 'large orthogonal_nop I/O round-trip failed'
    end if
  end block


!-------------------------------------------------------------------------------
! Test 16: missing WEIGHTS card falls back to default initialisers
!-------------------------------------------------------------------------------
  write(*,*) "Test 16: missing WEIGHTS card..."
  block
    class(base_layer_type), allocatable :: rl
    type(array_type) :: inp(1,1)
    logical :: ok

    open(newunit=unit, file='test_orthogonal_nop_block_noweights.txt', &
         status='replace', action='write')
    write(unit,'(A)') 'ORTHOGONAL_NOP'
    write(unit,'(3X,"NUM_INPUTS = ",I0)') 5
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') 4
    write(unit,'(3X,"NUM_BASIS = ",I0)') 2
    write(unit,'(3X,"USE_BIAS = ",L1)') .true.
    write(unit,'(A)') 'END ORTHOGONAL_NOP'
    close(unit)

    open(newunit=unit, file='test_orthogonal_nop_block_noweights.txt', &
         status='old', action='read')
    read(unit,*)
    rl = read_orthogonal_nop_block(unit)
    close(unit, status='delete')

    ok = .true.
    select type(rl)
    type is(orthogonal_nop_block_type)
       if(rl%num_inputs .ne. 5 .or. rl%num_outputs .ne. 4 .or. &
            rl%num_basis .ne. 2) ok = .false.
       if(rl%num_params .ne. 38) ok = .false.
       if(.not.allocated(rl%params))then
          ok = .false.
       elseif(size(rl%params) .ne. 4)then
          ok = .false.
       end if
       call inp(1,1)%allocate([5,1], source=1.0_real32)
       call rl%forward(inp)
       if(any(rl%output(1,1)%shape .ne. [4])) ok = .false.
       call inp(1,1)%deallocate()
    class default
       ok = .false.
    end select
    if(.not.ok)then
       success = .false.
       write(0,*) 'missing WEIGHTS card read failed'
    end if
  end block


!-------------------------------------------------------------------------------
! Final result
!-------------------------------------------------------------------------------
  write(*,*)
  if(success)then
     write(*,*) "All orthogonal_nop_block tests passed."
  else
     write(*,*) "Some orthogonal_nop_block tests FAILED."
     stop 1
  end if

contains

!-------------------------------------------------------------------------------
! finite-difference check of the ONO value partials
!-------------------------------------------------------------------------------
  subroutine run_gradient_check(success)
    implicit none
    logical, intent(inout) :: success
    type(orthogonal_nop_block_type) :: layer
    type(array_type) :: xin(1,1)
    type(array_type), pointer :: loss
    integer, parameter :: n_in = 6, n_out = 4, k = 3, batch = 2
    real(real32), allocatable :: x0(:,:), ana_x(:,:), p0(:,:), pa(:,:)
    integer :: np(4), ip, i, s, fails
    real(real32) :: eps, num
    real(8) :: num_loss

    eps = 1.E-3_real32
    fails = 0

    layer = orthogonal_nop_block_type( &
         num_inputs=n_in, num_outputs=n_out, num_basis=k, &
         use_bias=.true., activation="none")

    call xin(1,1)%allocate(array_shape=[n_in,batch])
    xin(1,1)%is_temporary = .false.
    call xin(1,1)%set_requires_grad(.true.)
    call random_number(xin(1,1)%val)
    call random_number(layer%params(1)%val)
    layer%params(1)%val = (layer%params(1)%val - 0.5_real32) * 0.2_real32
    call random_number(layer%params(2)%val)
    layer%params(2)%val = (layer%params(2)%val - 0.5_real32) * 0.2_real32
    call random_number(layer%params(3)%val)
    layer%params(3)%val = (layer%params(3)%val - 0.5_real32) * 0.2_real32
    call random_number(layer%params(4)%val)
    layer%params(4)%val = (layer%params(4)%val - 0.5_real32) * 0.1_real32

    ! analytic gradients
    call layer%forward(xin)
    loss => diff_sum(squared(layer%output(1,1)))
    call loss%grad_reverse()

    allocate(x0(n_in,batch), ana_x(n_in,batch))
    x0 = xin(1,1)%val
    if(.not.associated(xin(1,1)%grad))then
       success = .false.
       write(0,*) 'orthogonal_nop input gradient was not accumulated'
       deallocate(x0, ana_x)
       if(associated(loss)) call loss%release_expression()
       call xin(1,1)%deallocate()
       return
    end if
    ana_x = xin(1,1)%grad%val

    np = [ size(layer%params(1)%val), size(layer%params(2)%val), &
           size(layer%params(3)%val), size(layer%params(4)%val) ]
    allocate(p0(maxval(np),4), pa(maxval(np),4))
    p0 = 0.0_real32
    pa = 0.0_real32
    do ip = 1, 4
       if(.not.associated(layer%params(ip)%grad))then
          success = .false.
          write(0,'("orthogonal_nop missing parameter gradient for ", &
               &"params(",I0,")")') ip
          deallocate(p0, pa, x0, ana_x)
          if(associated(loss)) call loss%release_expression()
          call xin(1,1)%deallocate()
          return
       end if
       p0(1:np(ip),ip) = layer%params(ip)%val(:,1)
       pa(1:np(ip),ip) = layer%params(ip)%grad%val(:,1)
    end do
    call loss%release_expression()

    ! numeric: parameters
    do ip = 1, 4
       do i = 1, np(ip)
          layer%params(ip)%val(i,1) = p0(i,ip) + eps
          num_loss = numeric_loss(layer, xin)
          layer%params(ip)%val(i,1) = p0(i,ip) - eps
          num_loss = num_loss - numeric_loss(layer, xin)
          layer%params(ip)%val(i,1) = p0(i,ip)
          num = real(num_loss / real(2._8*eps, 8), real32)
          call check_value(pa(i,ip), num, 'parameter', ip, fails)
       end do
    end do

    ! numeric: input
    do s = 1, batch
       do i = 1, n_in
          xin(1,1)%val(i,s) = x0(i,s) + eps
          num_loss = numeric_loss(layer, xin)
          xin(1,1)%val(i,s) = x0(i,s) - eps
          num_loss = num_loss - numeric_loss(layer, xin)
          xin(1,1)%val(i,s) = x0(i,s)
          num = real(num_loss / real(2._8*eps, 8), real32)
          call check_value(ana_x(i,s), num, 'input', s, fails)
       end do
    end do

    if(fails .ne. 0)then
       success = .false.
       write(0,'("orthogonal_nop finite-difference failures: ",I0)') fails
    end if

    deallocate(x0, ana_x, p0, pa)
    call xin(1,1)%deallocate()

  end subroutine run_gradient_check

!-------------------------------------------------------------------------------
! per-sample parameter gradients must sum to the batch gradient
!-------------------------------------------------------------------------------
  subroutine run_param_reduction_check(success)
    implicit none
    logical, intent(inout) :: success
    type(orthogonal_nop_block_type) :: layer
    type(array_type) :: xin(1,1), xcol(1,1)
    type(array_type), pointer :: loss
    integer, parameter :: n_in = 6, n_out = 4, k = 3, batch = 2
    real(real32), allocatable :: batch_grad(:,:), acc(:,:)
    real(real32) :: u(n_in,batch), tol
    integer :: np(4), ip, s

    tol = 1.0E-4_real32

    layer = orthogonal_nop_block_type( &
         num_inputs=n_in, num_outputs=n_out, num_basis=k, &
         use_bias=.true., activation="none")

    call random_number(layer%params(1)%val)
    layer%params(1)%val = (layer%params(1)%val - 0.5_real32) * 0.2_real32
    call random_number(layer%params(2)%val)
    layer%params(2)%val = (layer%params(2)%val - 0.5_real32) * 0.2_real32
    call random_number(layer%params(3)%val)
    layer%params(3)%val = (layer%params(3)%val - 0.5_real32) * 0.2_real32
    call random_number(layer%params(4)%val)
    layer%params(4)%val = (layer%params(4)%val - 0.5_real32) * 0.1_real32

    np = [ size(layer%params(1)%val), size(layer%params(2)%val), &
           size(layer%params(3)%val), size(layer%params(4)%val) ]

    call xin(1,1)%allocate(array_shape=[n_in,batch])
    xin(1,1)%is_temporary = .false.
    call xin(1,1)%set_requires_grad(.true.)
    call random_number(xin(1,1)%val)
    u = xin(1,1)%val

    ! batch-2 gradients
    call layer%forward(xin)
    loss => diff_sum(squared(layer%output(1,1)))
    call loss%grad_reverse()
    allocate(batch_grad(maxval(np),4), acc(maxval(np),4))
    batch_grad = 0.0_real32
    acc = 0.0_real32
    do ip = 1, 4
       if(.not.associated(layer%params(ip)%grad))then
          success = .false.
          write(0,'("orthogonal_nop missing batch gradient for params(",I0,")")') ip
          if(associated(loss)) call loss%release_expression()
          deallocate(batch_grad, acc)
          call xin(1,1)%deallocate()
          return
       end if
       batch_grad(1:np(ip),ip) = layer%params(ip)%grad%val(:,1)
       call layer%params(ip)%zero_grad()
    end do

    ! sum of single-sample gradients
    call xcol(1,1)%allocate(array_shape=[n_in,1])
    xcol(1,1)%is_temporary = .false.
    call xcol(1,1)%set_requires_grad(.true.)
    do s = 1, batch
       do ip = 1, 4
          call layer%params(ip)%zero_grad()
       end do
       xcol(1,1)%val(:,1) = u(:,s)
       call layer%forward(xcol)
       loss => diff_sum(squared(layer%output(1,1)))
       call loss%grad_reverse()
       do ip = 1, 4
          if(.not.associated(layer%params(ip)%grad))then
             success = .false.
             write(0,'("orthogonal_nop missing sample gradient for ", &
                  &"params(",I0,")")') ip
             if(associated(loss)) call loss%release_expression()
             deallocate(batch_grad, acc)
             call xcol(1,1)%deallocate()
             call xin(1,1)%deallocate()
             return
          end if
          acc(1:np(ip),ip) = acc(1:np(ip),ip) + &
               layer%params(ip)%grad%val(:,1)
       end do
    end do

    do ip = 1, 4
       if(maxval(abs(acc(1:np(ip),ip) - batch_grad(1:np(ip),ip))) .gt. tol)then
          success = .false.
          write(0,'("orthogonal_nop batch parameter reduction mismatch for ", &
               &"params(",I0,")")') ip
       end if
    end do

    if(associated(loss)) call loss%release_expression()

    deallocate(batch_grad, acc)
    call xcol(1,1)%deallocate()
    call xin(1,1)%deallocate()

  end subroutine run_param_reduction_check

!-------------------------------------------------------------------------------
  real(8) function numeric_loss(layer, xin) result(l)
    implicit none
    type(orthogonal_nop_block_type), intent(inout) :: layer
    type(array_type), intent(in) :: xin(1,1)
    call layer%forward(xin)
    l = sum(real(layer%output(1,1)%val, 8)**2)
  end function numeric_loss

!-------------------------------------------------------------------------------
  subroutine check_value(analytic, numeric, name, case_id, fails)
    implicit none
    real(real32), intent(in) :: analytic, numeric
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

end program test_orthogonal_nop_block
