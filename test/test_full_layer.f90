program test_full_layer
  use athena, only: &
       full_layer_type, &
       base_layer_type
  use athena__full_layer, only: read_full_layer
  use diffstruc, only: array_type, sum, squared
  implicit none

  class(base_layer_type), allocatable :: full_layer1, full_layer2, full_layer3
  class(base_layer_type), allocatable :: read_layer
  integer :: unit
  logical :: success = .true.
  character(len=16) :: mode


!-------------------------------------------------------------------------------
! optional forward/backward timing mode: `test_full_layer bench`
!-------------------------------------------------------------------------------
  mode = ''
  if(command_argument_count() >= 1) call get_command_argument(1, mode)
  if(trim(mode) == 'bench')then
     call run_bench()
     stop
  end if


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
     if(.not. read_layer%name .eq. 'full')then
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
! forward/backward timing for fully connected shapes
!-------------------------------------------------------------------------------
  subroutine run_bench
    type(full_layer_type) :: layer
    type(array_type) :: input(1,1)
    type(array_type), pointer :: loss
    real(8) :: t0, t1, t_fwd, t_rev
    integer :: in_dim, out_dim, batch, n_fwd, n_rev, c, iter

    integer, parameter :: N_CASES = 3
    integer, parameter :: IN_DIMS(N_CASES)  = [784, 100, 1024]
    integer, parameter :: OUT_DIMS(N_CASES) = [100,  10, 1024]
    integer, parameter :: BATCHES(N_CASES)  = [ 32,  32,  128]
    integer, parameter :: ITER_FWD(N_CASES) = [4000, 8000, 400]
    integer, parameter :: ITER_REV(N_CASES) = [2000, 4000, 200]
    character(len=8), parameter :: ACTS(N_CASES) = &
         [character(len=8) :: 'relu', 'softmax', 'relu']

    write(*,'(A)') ""
    write(*,'(A)') "=== fully connected forward/backward benchmark ==="
    write(*,'(A32,A12,A12,A12,A12)') &
         "case", "fwd(ms)", "rev(ms)", "fwd GFLOP/s", "rev GFLOP/s"
    write(*,'(A)') repeat('-', 80)

    do c = 1, N_CASES
       in_dim  = IN_DIMS(c)
       out_dim = OUT_DIMS(c)
       batch   = BATCHES(c)
       n_fwd   = ITER_FWD(c)
       n_rev   = ITER_REV(c)
       layer = full_layer_type(num_inputs=in_dim, num_outputs=out_dim, &
            activation=trim(ACTS(c)))
       call input(1,1)%allocate(array_shape=[in_dim, batch])
       input(1,1)%is_temporary = .false.
       call random_number(input(1,1)%val)

       ! warm up
       call layer%forward(input)
       loss => sum(squared(layer%output(1,1)))
       call loss%grad_reverse()
       call loss%release_expression()

       ! forward: layer%forward discards the previous expression tree
       call cpu_time(t0)
       do iter = 1, n_fwd
          call layer%forward(input)
       end do
       call cpu_time(t1)
       t_fwd = (t1 - t0) / real(n_fwd, 8) * 1000._8

       ! backward: forward + loss + reverse + cleanup
       call layer%forward(input)
       call cpu_time(t0)
       do iter = 1, n_rev
          call layer%forward(input)
          loss => sum(squared(layer%output(1,1)))
          call loss%grad_reverse()
          call loss%release_expression()
       end do
       call cpu_time(t1)
       t_rev = (t1 - t0) / real(n_rev, 8) * 1000._8 - t_fwd

       write(*,'(I6,"x",I6," b=",I4,2x,A8,2(F8.3),2(2x,F10.1))') &
            in_dim, out_dim, batch, trim(ACTS(c)), t_fwd, t_rev, &
            2._8 * in_dim * out_dim * batch / (t_fwd * 1.E6_8), &
            2._8 * 3._8 * in_dim * out_dim * batch / (t_rev * 1.E6_8)

       call input(1,1)%deallocate()
    end do
    write(*,'(A)') repeat('-', 80)
  end subroutine run_bench

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
