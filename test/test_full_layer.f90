program test_full_layer
  use athena, only: &
       full_layer_type, &
       base_layer_type
  use athena__full_layer, only: read_full_layer
  implicit none

  class(base_layer_type), allocatable :: full_layer1, full_layer2, full_layer3
  class(base_layer_type), allocatable :: read_layer
  integer :: unit
  logical :: success = .true.


!-------------------------------------------------------------------------------
! set up layer
!-------------------------------------------------------------------------------
  full_layer1 = full_layer_type(num_inputs=1, num_outputs=10)

  !! check layer name
  if(.not. full_layer1%name .eq. 'full')then
     success = .false.
     write(0,*) 'full layer has wrong name'
  end if

  if(any(full_layer1%input_shape .ne. [1]))then
     success = .false.
     write(0,*) 'full layer has wrong input_shape'
  end if

  if(any(full_layer1%output_shape .ne. [10]))then
     success = .false.
     write(0,*) 'full layer has wrong output shape'
  end if

  !! check layer type
  select type(full_layer1)
  type is(full_layer_type)
     !! check default layer transfer/activation function
     if(full_layer1%activation%name .ne. 'none')then
        success = .false.
        write(0,*) 'full layer has wrong activation: '//full_layer1%activation%name
     end if
  class default
     success = .false.
     write(0,*) 'full layer has wrong type'
  end select

  full_layer2 = full_layer_type(num_outputs=20)
  call full_layer2%init(full_layer1%output_shape)

  if(any(full_layer2%input_shape .ne. [10]))then
     success = .false.
     write(0,*) 'full layer has wrong input_shape'
  end if

  if(any(full_layer2%output_shape .ne. [20]))then
     success = .false.
     write(0,*) 'full layer has wrong input_shape'
  end if


!-------------------------------------------------------------------------------
! check layer operations
!-------------------------------------------------------------------------------
  full_layer2 = full_layer_type(num_inputs=1, num_outputs=10, &
       activation="sigmoid")
  full_layer3 = full_layer_type(num_inputs=1, num_outputs=10, &
       activation="sigmoid")
  select type(full_layer2)
  type is(full_layer_type)
     select type(full_layer3)
     type is(full_layer_type)
        full_layer1 = full_layer2 + full_layer3
        select type(full_layer1)
        type is(full_layer_type)
           !! check layer addition
           call compare_full_layers(&
                full_layer1, full_layer2, success, full_layer3)

           !! check layer reduction
           full_layer1 = full_layer2
           call full_layer1%reduce(full_layer3)
           call compare_full_layers(&
                full_layer1, full_layer2, success, full_layer3)
        class default
           success = .false.
           write(0,*) 'full layer has wrong type'
        end select
     class default
        success = .false.
        write(0,*) 'full layer has wrong type'
     end select
  class default
     success = .false.
     write(0,*) 'full layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_full_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("FULL")')
  call full_layer1%print_to_unit(unit)
  write(unit,'("END FULL")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_full_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_full_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (full_layer_type)
     if (.not. read_layer%name .eq. 'full') then
        success = .false.
        write(0,*) 'read full layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not full_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_full_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_full_layer passed all tests'
  else
     write(0,*) 'test_full_layer failed one or more tests'
     stop 1
  end if


contains

!-------------------------------------------------------------------------------
! compare two or three layers
!-------------------------------------------------------------------------------
  subroutine compare_full_layers(layer1, layer2, success, layer3)
    type(full_layer_type), intent(in) :: layer1, layer2
    logical, intent(inout) :: success
    type(full_layer_type), optional, intent(in) :: layer3

    if(layer1%num_inputs .ne. layer2%num_inputs)then
       success = .false.
       write(0,*) 'full layer has wrong num_inputs'
    end if
    if(layer1%activation%name .ne. 'sigmoid')then
       success = .false.
       write(0,*) 'full layer has wrong activation: '//&
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
             write(0,*) 'full layer has wrong gradients'
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
             write(0,*) 'full layer has wrong bias gradients'
          end if
       end if
    end if

  end subroutine compare_full_layers

end program test_full_layer
