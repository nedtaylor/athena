program test_nop_laplace_layer
  !! Tests for the Laplace Neural Operator layer
  use athena, only: &
       nop_laplace_layer_type, &
       base_layer_type
  use athena__nop_laplace_layer, only: read_nop_laplace_layer
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
  layer1 = nop_laplace_layer_type(num_inputs=4, num_outputs=8, num_modes=3)

  if(layer1%name .ne. 'nop_laplace')then
     success = .false.
     write(0,*) 'nop_laplace layer has wrong name: '//trim(layer1%name)
  end if

  if(any(layer1%input_shape .ne. [4]))then
     success = .false.
     write(0,*) 'nop_laplace layer has wrong input_shape'
  end if

  if(any(layer1%output_shape .ne. [8]))then
     success = .false.
     write(0,*) 'nop_laplace layer has wrong output_shape'
  end if

  select type(layer1)
  type is(nop_laplace_layer_type)
     if(layer1%num_inputs .ne. 4)then
        success = .false.
        write(0,*) 'nop_laplace layer has wrong num_inputs'
     end if
     if(layer1%num_outputs .ne. 8)then
        success = .false.
        write(0,*) 'nop_laplace layer has wrong num_outputs'
     end if
     if(layer1%num_modes .ne. 3)then
        success = .false.
        write(0,*) 'nop_laplace layer has wrong num_modes'
     end if
     if(layer1%activation%name .ne. 'none')then
        success = .false.
        write(0,*) 'nop_laplace layer has wrong default activation: '// &
             layer1%activation%name
     end if
     ! num_params = M^2 + n_out * n_in + n_out (bias)
     !            = 9 + 32 + 8 = 49
     if(layer1%num_params .ne. 49)then
        success = .false.
        write(0,'("nop_laplace layer has wrong num_params: ",I0)') &
             layer1%num_params
     end if
     ! Check basis matrices are allocated
     if(.not.layer1%encoder_basis%allocated)then
        success = .false.
        write(0,*) 'nop_laplace encoder_basis not allocated'
     end if
     if(.not.layer1%decoder_basis%allocated)then
        success = .false.
        write(0,*) 'nop_laplace decoder_basis not allocated'
     end if
  class default
     success = .false.
     write(0,*) 'layer1 has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 2: deferred initialisation (num_inputs provided later)
!-------------------------------------------------------------------------------
  write(*,*) "Test 2: deferred initialisation..."
  layer2 = nop_laplace_layer_type(num_outputs=6, num_modes=2)
  call layer2%init(layer1%output_shape)

  if(any(layer2%input_shape .ne. [8]))then
     success = .false.
     write(0,*) 'deferred nop_laplace layer has wrong input_shape'
  end if
  if(any(layer2%output_shape .ne. [6]))then
     success = .false.
     write(0,*) 'deferred nop_laplace layer has wrong output_shape'
  end if


!-------------------------------------------------------------------------------
! Test 3: layer without bias
!-------------------------------------------------------------------------------
  write(*,*) "Test 3: no-bias layer..."
  layer3 = nop_laplace_layer_type( &
       num_inputs=4, num_outputs=4, num_modes=2, use_bias=.false.)

  select type(layer3)
  type is(nop_laplace_layer_type)
     if(layer3%use_bias)then
        success = .false.
        write(0,*) 'no-bias nop_laplace layer has use_bias = T'
     end if
     ! num_params = 4 + 16 = 20 (no bias)
     if(layer3%num_params .ne. 20)then
        success = .false.
        write(0,'("no-bias layer has wrong num_params: ",I0)') layer3%num_params
     end if
  class default
     success = .false.
     write(0,*) 'layer3 has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 4: activation function
!-------------------------------------------------------------------------------
  write(*,*) "Test 4: activation function..."
  layer1 = nop_laplace_layer_type( &
       num_inputs=3, num_outputs=5, num_modes=2, activation="relu")

  select type(layer1)
  type is(nop_laplace_layer_type)
     if(layer1%activation%name .ne. 'relu')then
        success = .false.
        write(0,*) 'nop_laplace layer has wrong activation: '// &
             layer1%activation%name
     end if
  class default
     success = .false.
     write(0,*) 'layer1 has wrong type after re-assignment'
  end select


!-------------------------------------------------------------------------------
! Test 5: forward pass produces correct output shape
!-------------------------------------------------------------------------------
  write(*,*) "Test 5: forward pass..."
  layer1 = nop_laplace_layer_type( &
       num_inputs=4, num_outputs=3, num_modes=2, &
       use_bias=.true., activation="none")

  block
    type(array_type), dimension(1,1) :: inp
    call inp(1,1)%allocate([4, 1], source=1.0_real32)
    call layer1%forward(inp)
    ! Output should have shape [3]
    if(any(layer1%output(1,1)%shape .ne. [3]))then
       success = .false.
       write(0,*) 'forward pass produced wrong output shape'
    end if
    call inp(1,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Test 6: layer addition and reduction (inherited from learnable_layer_type)
!-------------------------------------------------------------------------------
  write(*,*) "Test 6: layer addition and reduction..."
  layer2 = nop_laplace_layer_type( &
       num_inputs=4, num_outputs=3, num_modes=2, activation="sigmoid")
  layer3 = nop_laplace_layer_type( &
       num_inputs=4, num_outputs=3, num_modes=2, activation="sigmoid")

  select type(layer2)
  type is(nop_laplace_layer_type)
     select type(layer3)
     type is(nop_laplace_layer_type)
        layer1 = layer2 + layer3
        select type(layer1)
        type is(nop_laplace_layer_type)
           call compare_layers(layer1, layer2, success, layer3)
           layer1 = layer2
           call layer1%reduce(layer3)
           call compare_layers(layer1, layer2, success, layer3)
        class default
           success = .false.
           write(0,*) 'layer addition returned wrong type'
        end select
     class default
        success = .false.
        write(0,*) 'layer3 has wrong type'
     end select
  class default
     success = .false.
     write(0,*) 'layer2 has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 7: file I/O
!-------------------------------------------------------------------------------
  write(*,*) "Test 7: file I/O..."

  layer1 = nop_laplace_layer_type( &
       num_inputs=3, num_outputs=4, num_modes=2, &
       use_bias=.true., activation="relu")

  open(newunit=unit, file='test_nop_laplace_layer.tmp', &
       status='replace', action='write')
  write(unit,'("NOP_LAPLACE")')
  call layer1%print_to_unit(unit)
  write(unit,'("END NOP_LAPLACE")')
  close(unit)

  open(newunit=unit, file='test_nop_laplace_layer.tmp', &
       status='old', action='read')
  read(unit,*)   ! skip NOP_LAPLACE header line
  read_layer = read_nop_laplace_layer(unit)
  close(unit)

  select type(read_layer)
  type is(nop_laplace_layer_type)
     if(read_layer%name .ne. 'nop_laplace')then
        success = .false.
        write(0,*) 'read layer has wrong name'
     end if
     if(read_layer%num_inputs .ne. 3)then
        success = .false.
        write(0,*) 'read layer has wrong num_inputs'
     end if
     if(read_layer%num_outputs .ne. 4)then
        success = .false.
        write(0,*) 'read layer has wrong num_outputs'
     end if
     if(read_layer%num_modes .ne. 2)then
        success = .false.
        write(0,*) 'read layer has wrong num_modes'
     end if
     if(read_layer%activation%name .ne. 'relu')then
        success = .false.
        write(0,*) 'read layer has wrong activation: '// &
             read_layer%activation%name
     end if
     select type(layer1)
     type is(nop_laplace_layer_type)
        if(any(abs(read_layer%params(1)%val(:,1) - &
             layer1%params(1)%val(:,1)) .gt. 1.0e-6_real32))then
           success = .false.
           write(0,*) 'read layer R weights differ from original'
        end if
        if(any(abs(read_layer%params(2)%val(:,1) - &
             layer1%params(2)%val(:,1)) .gt. 1.0e-6_real32))then
           success = .false.
           write(0,*) 'read layer W weights differ from original'
        end if
        if(any(abs(read_layer%params(3)%val(:,1) - &
             layer1%params(3)%val(:,1)) .gt. 1.0e-6_real32))then
           success = .false.
           write(0,*) 'read layer bias differs from original'
        end if
     end select
  class default
     success = .false.
     write(0,*) 'read layer is not nop_laplace_layer_type'
  end select

  open(newunit=unit, file='test_nop_laplace_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! Result
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_nop_laplace_layer passed all tests'
  else
     write(0,*) 'test_nop_laplace_layer failed one or more tests'
     stop 1
  end if


contains

!-------------------------------------------------------------------------------
  subroutine compare_layers(layer1, layer2, success, layer3)
    type(nop_laplace_layer_type), intent(in) :: layer1, layer2
    logical, intent(inout) :: success
    type(nop_laplace_layer_type), optional, intent(in) :: layer3

    if(layer1%num_inputs .ne. layer2%num_inputs)then
       success = .false.
       write(0,*) 'nop_laplace layer comparison: wrong num_inputs'
    end if
    if(layer1%activation%name .ne. 'sigmoid')then
       success = .false.
       write(0,*) 'nop_laplace layer comparison: wrong activation: '// &
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
          ).gt.1.0e-6_real32))then
             success = .false.
             write(0,*) 'nop_laplace layer: R gradient sum mismatch'
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
          ).gt.1.0e-6_real32))then
             success = .false.
             write(0,*) 'nop_laplace layer: W gradient sum mismatch'
          end if
       end if
    end if
  end subroutine compare_layers

end program test_nop_laplace_layer
