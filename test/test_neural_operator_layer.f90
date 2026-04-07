program test_neural_operator_layer
  !! Tests for the neural operator layer
  use athena, only: &
       neural_operator_layer_type, &
       base_layer_type
  use athena__neural_operator_layer, only: read_neural_operator_layer
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
  layer1 = neural_operator_layer_type(num_inputs=4, num_outputs=8)

  if(layer1%name .ne. 'neural_operator')then
     success = .false.
     write(0,*) 'neural_operator layer has wrong name: '//trim(layer1%name)
  end if

  if(any(layer1%input_shape .ne. [4]))then
     success = .false.
     write(0,*) 'neural_operator layer has wrong input_shape'
  end if

  if(any(layer1%output_shape .ne. [8]))then
     success = .false.
     write(0,*) 'neural_operator layer has wrong output_shape'
  end if

  select type(layer1)
  type is(neural_operator_layer_type)
     if(layer1%num_inputs .ne. 4)then
        success = .false.
        write(0,*) 'neural_operator layer has wrong num_inputs'
     end if
     if(layer1%num_outputs .ne. 8)then
        success = .false.
        write(0,*) 'neural_operator layer has wrong num_outputs'
     end if
     if(layer1%activation%name .ne. 'none')then
        success = .false.
        write(0,*) 'neural_operator layer has wrong default activation: '// &
             layer1%activation%name
     end if
     ! num_params = n_out * n_in + n_out (kernel coupling) + n_out (bias)
     !            = 8*4 + 8 + 8 = 48
     if(layer1%num_params .ne. 48)then
        success = .false.
        write(0,'("neural_operator layer has wrong num_params: ",I0)') &
             layer1%num_params
     end if
  class default
     success = .false.
     write(0,*) 'layer1 has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 2: deferred initialisation (num_inputs provided later)
!-------------------------------------------------------------------------------
  write(*,*) "Test 2: deferred initialisation..."
  layer2 = neural_operator_layer_type(num_outputs=6)
  call layer2%init(layer1%output_shape)

  if(any(layer2%input_shape .ne. [8]))then
     success = .false.
     write(0,*) 'deferred neural_operator layer has wrong input_shape'
  end if
  if(any(layer2%output_shape .ne. [6]))then
     success = .false.
     write(0,*) 'deferred neural_operator layer has wrong output_shape'
  end if


!-------------------------------------------------------------------------------
! Test 3: layer without bias
!-------------------------------------------------------------------------------
  write(*,*) "Test 3: no-bias layer..."
  layer3 = neural_operator_layer_type( &
       num_inputs=4, num_outputs=4, use_bias=.false.)

  select type(layer3)
  type is(neural_operator_layer_type)
     if(layer3%use_bias)then
        success = .false.
        write(0,*) 'no-bias neural_operator layer has use_bias = T'
     end if
     ! num_params = 4*4 + 4 = 20 (no bias)
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
  layer1 = neural_operator_layer_type( &
       num_inputs=3, num_outputs=5, activation="relu")

  select type(layer1)
  type is(neural_operator_layer_type)
     if(layer1%activation%name .ne. 'relu')then
        success = .false.
        write(0,*) 'neural_operator layer has wrong activation: '// &
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
  layer1 = neural_operator_layer_type( &
       num_inputs=4, num_outputs=3, use_bias=.true., activation="none")

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
  layer2 = neural_operator_layer_type( &
       num_inputs=4, num_outputs=3, activation="sigmoid")
  layer3 = neural_operator_layer_type( &
       num_inputs=4, num_outputs=3, activation="sigmoid")

  select type(layer2)
  type is(neural_operator_layer_type)
     select type(layer3)
     type is(neural_operator_layer_type)
        layer1 = layer2 + layer3
        select type(layer1)
        type is(neural_operator_layer_type)
           call compare_no_layers(layer1, layer2, success, layer3)

           ! Test reduce
           layer1 = layer2
           call layer1%reduce(layer3)
           call compare_no_layers(layer1, layer2, success, layer3)
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

  layer1 = neural_operator_layer_type( &
       num_inputs=3, num_outputs=4, use_bias=.true., activation="relu")

  open(newunit=unit, file='test_neural_operator_layer.tmp', &
       status='replace', action='write')
  write(unit,'("NEURAL_OPERATOR")')
  call layer1%print_to_unit(unit)
  write(unit,'("END NEURAL_OPERATOR")')
  close(unit)

  open(newunit=unit, file='test_neural_operator_layer.tmp', &
       status='old', action='read')
  read(unit,*)   ! skip NEURAL_OPERATOR header line
  read_layer = read_neural_operator_layer(unit)
  close(unit)

  select type(read_layer)
  type is(neural_operator_layer_type)
     if(read_layer%name .ne. 'neural_operator')then
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
     if(read_layer%activation%name .ne. 'relu')then
        success = .false.
        write(0,*) 'read layer has wrong activation: '//read_layer%activation%name
     end if
     ! Check params(1) weights match original
     select type(layer1)
     type is(neural_operator_layer_type)
        if(any(abs(read_layer%params(1)%val(:,1) - &
             layer1%params(1)%val(:,1)) .gt. 1.0e-6_real32))then
           success = .false.
           write(0,*) 'read layer W weights differ from original'
        end if
        if(any(abs(read_layer%params(2)%val(:,1) - &
             layer1%params(2)%val(:,1)) .gt. 1.0e-6_real32))then
           success = .false.
           write(0,*) 'read layer W_k weights differ from original'
        end if
        if(any(abs(read_layer%params(3)%val(:,1) - &
             layer1%params(3)%val(:,1)) .gt. 1.0e-6_real32))then
           success = .false.
           write(0,*) 'read layer bias differs from original'
        end if
     end select
  class default
     success = .false.
     write(0,*) 'read layer is not neural_operator_layer_type'
  end select

  open(newunit=unit, file='test_neural_operator_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! Result
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_neural_operator_layer passed all tests'
  else
     write(0,*) 'test_neural_operator_layer failed one or more tests'
     stop 1
  end if


contains

!-------------------------------------------------------------------------------
! Compare two or three neural operator layers (params sums)
!-------------------------------------------------------------------------------
  subroutine compare_no_layers(layer1, layer2, success, layer3)
    type(neural_operator_layer_type), intent(in) :: layer1, layer2
    logical, intent(inout) :: success
    type(neural_operator_layer_type), optional, intent(in) :: layer3

    if(layer1%num_inputs .ne. layer2%num_inputs)then
       success = .false.
       write(0,*) 'neural_operator layer comparison: wrong num_inputs'
    end if
    if(layer1%activation%name .ne. 'sigmoid')then
       success = .false.
       write(0,*) 'neural_operator layer comparison: wrong activation: '// &
            layer1%activation%name
    end if
    if(present(layer3))then
       ! Check that params sum correctly for W
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
             write(0,*) 'neural_operator layer: W gradient sum mismatch'
          end if
       end if
       ! Check that params sum correctly for W_k
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
             write(0,*) 'neural_operator layer: W_k gradient sum mismatch'
          end if
       end if
    end if
  end subroutine compare_no_layers

end program test_neural_operator_layer
