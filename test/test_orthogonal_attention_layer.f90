program test_orthogonal_attention_layer
  !! Tests for the Orthogonal Attention layer
  use athena, only: &
       orthogonal_attention_layer_type, &
       base_layer_type
  use athena__orthogonal_attention_layer, only: &
       read_orthogonal_attention_layer
  use diffstruc, only: array_type
  use coreutils, only: real32
  implicit none

  class(base_layer_type), allocatable :: layer1, layer2, layer3
  class(base_layer_type), allocatable :: read_layer
  integer :: unit
  logical :: success = .true.


!-------------------------------------------------------------------------------
! Test 1: basic layer construction
!-------------------------------------------------------------------------------
  write(*,*) "Test 1: layer construction..."
  layer1 = orthogonal_attention_layer_type( &
       num_inputs=8, num_outputs=6, num_basis=3, key_dim=4)

  if(layer1%name .ne. 'orthogonal_attention')then
     success = .false.
     write(0,*) 'orthogonal_attention layer has wrong name: '// &
          trim(layer1%name)
  end if

  if(any(layer1%input_shape .ne. [8]))then
     success = .false.
     write(0,*) 'orthogonal_attention layer has wrong input_shape'
  end if

  if(any(layer1%output_shape .ne. [6]))then
     success = .false.
     write(0,*) 'orthogonal_attention layer has wrong output_shape'
  end if

  select type(layer1)
  type is(orthogonal_attention_layer_type)
     if(layer1%num_inputs .ne. 8)then
        success = .false.
        write(0,*) 'layer has wrong num_inputs'
     end if
     if(layer1%num_outputs .ne. 6)then
        success = .false.
        write(0,*) 'layer has wrong num_outputs'
     end if
     if(layer1%num_basis .ne. 3)then
        success = .false.
        write(0,*) 'layer has wrong num_basis'
     end if
     if(layer1%key_dim .ne. 4)then
        success = .false.
        write(0,*) 'layer has wrong key_dim'
     end if
     ! num_params = 4*8 + 4*8 + 6*8 + 8*3 + 6*8 + 6
     !            = 32 + 32 + 48 + 24 + 48 + 6 = 190
     if(layer1%num_params .ne. 190)then
        success = .false.
        write(0,'("layer has wrong num_params: ",I0)') &
             layer1%num_params
     end if
     ! Check phi basis is allocated
     block
       type(array_type) :: phi
       phi = layer1%get_bases()
       if(.not.phi%allocated)then
          success = .false.
          write(0,*) 'phi not allocated'
       end if
     end block
  class default
     success = .false.
     write(0,*) 'layer1 has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 2: orthogonality of basis
!-------------------------------------------------------------------------------
  write(*,*) "Test 2: orthogonality of basis..."

  select type(layer1)
  type is(orthogonal_attention_layer_type)
     block
       integer :: n, k, i, j
       real(real32), allocatable :: Q(:,:), QtQ(:,:)
       real(real32) :: metric, val
       type(array_type) :: phi

       n = layer1%num_inputs
       k = layer1%num_basis
       allocate(Q(n, k), QtQ(k, k))
       phi = layer1%get_bases()
       Q = reshape(phi%val(:,1), [n, k])
       QtQ = matmul(transpose(Q), Q)
       metric = 0.0_real32
       do j = 1, k
          do i = 1, k
             if(i .eq. j)then
                val = abs(QtQ(i,j) - 1.0_real32)
             else
                val = abs(QtQ(i,j))
             end if
             if(val .gt. metric) metric = val
          end do
       end do
       write(*,'("  Orthogonality metric max(|Q^T Q - I|) = ",ES12.5)') &
            metric
       if(metric .gt. 1.0e-5_real32)then
          success = .false.
          write(0,'("Basis not orthogonal! metric = ",ES12.5)') metric
       end if
       deallocate(Q, QtQ)
     end block
  end select


!-------------------------------------------------------------------------------
! Test 3: deferred initialisation
!-------------------------------------------------------------------------------
  write(*,*) "Test 3: deferred initialisation..."
  layer2 = orthogonal_attention_layer_type( &
       num_outputs=4, num_basis=2)
  call layer2%init(layer1%output_shape)

  if(any(layer2%input_shape .ne. [6]))then
     success = .false.
     write(0,*) 'deferred layer has wrong input_shape'
  end if
  if(any(layer2%output_shape .ne. [4]))then
     success = .false.
     write(0,*) 'deferred layer has wrong output_shape'
  end if


!-------------------------------------------------------------------------------
! Test 4: no-bias layer
!-------------------------------------------------------------------------------
  write(*,*) "Test 4: no-bias layer..."
  layer3 = orthogonal_attention_layer_type( &
       num_inputs=4, num_outputs=4, num_basis=2, use_bias=.false.)

  select type(layer3)
  type is(orthogonal_attention_layer_type)
     if(layer3%use_bias)then
        success = .false.
        write(0,*) 'no-bias layer has use_bias = T'
     end if
  class default
     success = .false.
     write(0,*) 'layer3 has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 5: forward pass shape correctness
!-------------------------------------------------------------------------------
  write(*,*) "Test 5: forward pass..."
  layer1 = orthogonal_attention_layer_type( &
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
! Test 6: forward pass with activation
!-------------------------------------------------------------------------------
  write(*,*) "Test 6: forward pass with activation..."
  layer1 = orthogonal_attention_layer_type( &
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
! Test 7: file I/O
!-------------------------------------------------------------------------------
  write(*,*) "Test 7: file I/O..."

  layer1 = orthogonal_attention_layer_type( &
       num_inputs=5, num_outputs=3, num_basis=2, key_dim=4, &
       activation="relu")

  open(newunit=unit, file='test_orthogonal_attention_layer.txt', &
       status='replace', action='write')
  write(unit,'(A)') 'ORTHOGONAL_ATTENTION'
  call layer1%print_to_unit(unit)
  write(unit,'(A)') 'END ORTHOGONAL_ATTENTION'
  close(unit)

  open(newunit=unit, file='test_orthogonal_attention_layer.txt', &
       status='old', action='read')
  block
    character(256) :: buffer
    read(unit,'(A)') buffer   ! Read header
  end block
  read_layer = read_orthogonal_attention_layer(unit)
  close(unit, status='delete')

  select type(rl => read_layer)
  type is(orthogonal_attention_layer_type)
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
     if(rl%key_dim .ne. 4)then
        success = .false.
        write(0,*) 'read layer has wrong key_dim'
     end if
     if(rl%activation%name .ne. 'relu')then
        success = .false.
        write(0,*) 'read layer has wrong activation: '//rl%activation%name
     end if
  class default
     success = .false.
     write(0,*) 'read_layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! Final result
!-------------------------------------------------------------------------------
  write(*,*)
  if(success)then
     write(*,*) "All orthogonal_attention_layer tests passed."
  else
     write(*,*) "Some orthogonal_attention_layer tests FAILED."
     stop 1
  end if

end program test_orthogonal_attention_layer
