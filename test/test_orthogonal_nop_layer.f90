program test_orthogonal_nop_block
  !! Tests for the Orthogonal Neural Operator layer
  use athena, only: &
       orthogonal_nop_block_type, &
       base_layer_type
  use athena__orthogonal_nop_block, only: read_orthogonal_nop_block
  use diffstruc, only: array_type
  use coreutils, only: real32
  implicit none

  class(base_layer_type), allocatable :: layer1, layer2, layer3
  class(base_layer_type), allocatable :: read_layer
  integer :: unit
  logical :: success = .true.


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
    inp(1,1)%requires_grad = .true.

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

    write(*,*) "  Gradient propagation completed without error."
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
! Final result
!-------------------------------------------------------------------------------
  write(*,*)
  if(success)then
     write(*,*) "All orthogonal_nop_block tests passed."
  else
     write(*,*) "Some orthogonal_nop_block tests FAILED."
     stop 1
  end if

end program test_orthogonal_nop_block
