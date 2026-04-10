program test_concat_layer
  use coreutils, only: real32
  use athena__concat_layer, only: concat_layer_type, read_concat_layer
  use athena__base_layer, only: base_layer_type
  use athena__diffstruc_extd, only: array_ptr_type
  use diffstruc, only: array_type
  implicit none

  logical :: success
  type(concat_layer_type) :: layer
  type(array_type), target :: input_a(1,1), input_b(1,1)
  type(array_ptr_type) :: input_list(2)
  integer :: input_shapes(1,2)
  integer, allocatable :: calculated_shape(:)
  class(base_layer_type), allocatable :: read_layer
  integer :: unit

  success = .true.

  write(*,'("Testing concat layer forward/backward")')

  layer = concat_layer_type(input_layer_ids=[1, 2], input_rank=1, verbose=0)
  call layer%init(input_shape=[4])

  input_shapes = reshape([2, 2], shape(input_shapes))
  calculated_shape = layer%calc_input_shape(input_shapes)
  if(.not. allocated(calculated_shape) .or. &
       any(calculated_shape .ne. [4]))then
     success = .false.
     write(0,*) 'concat layer calculated the wrong input shape'
  end if

  call input_a(1,1)%allocate(array_shape=[2, 1])
  input_a(1,1)%val(:,1) = [1._real32, 2._real32]
  call input_a(1,1)%set_requires_grad(.true.)
  input_a(1,1)%is_temporary = .false.

  call input_b(1,1)%allocate(array_shape=[2, 1])
  input_b(1,1)%val(:,1) = [3._real32, 4._real32]
  call input_b(1,1)%set_requires_grad(.true.)
  input_b(1,1)%is_temporary = .false.

  input_list(1)%array => input_a
  input_list(2)%array => input_b

  call layer%combine(input_list)

  if(any(layer%output(1,1)%val(:,1) .ne. &
       [1._real32, 2._real32, 3._real32, 4._real32]))then
     success = .false.
     write(0,*) 'concat layer returned the wrong merged values'
  end if

  call layer%output(1,1)%grad_reverse(reset_graph=.true.)

  if(.not. associated(input_a(1,1)%grad) .or. &
       any(abs(input_a(1,1)%grad%val(:,1) - 1._real32) > 1.e-6_real32))then
     success = .false.
     write(0,*) 'concat layer returned the wrong gradient for the first input'
  end if

  if(.not. associated(input_b(1,1)%grad) .or. &
       any(abs(input_b(1,1)%grad%val(:,1) - 1._real32) > 1.e-6_real32))then
     success = .false.
     write(0,*) 'concat layer returned the wrong gradient for the second input'
  end if

  call layer%output(1,1)%nullify_graph()
  call input_a(1,1)%deallocate()
  call input_b(1,1)%deallocate()

  write(*,'("Testing concat layer file I/O")')

  layer = concat_layer_type(input_layer_ids=[3, 5], input_rank=1, verbose=0)
  call layer%init(input_shape=[4])

  open(newunit=unit, file='test_concat_layer.tmp', status='replace', action='write')
  write(unit,'(A)') 'CONCATENATE'
  call layer%print_to_unit(unit)
  write(unit,'(A)') 'END CONCATENATE'
  close(unit)

  open(newunit=unit, file='test_concat_layer.tmp', status='old', action='read')
  read(unit,*)
  read_layer = read_concat_layer(unit)
  close(unit, status='delete')

  select type (read_layer)
  type is (concat_layer_type)
     if(read_layer%input_rank .ne. 1)then
        success = .false.
        write(0,*) 'concat layer file I/O lost input rank'
     end if
     if(any(read_layer%input_shape .ne. [4]))then
        success = .false.
        write(0,*) 'concat layer file I/O lost input shape'
     end if
     if(any(read_layer%input_layer_ids .ne. [3, 5]))then
        success = .false.
        write(0,*) 'concat layer file I/O lost input layer ids'
     end if
  class default
     success = .false.
     write(0,*) 'concat layer file I/O returned the wrong layer type'
  end select

  if(success)then
     write(*,*) 'test_concat_layer passed all tests'
  else
     write(0,*) 'test_concat_layer failed one or more tests'
     stop 1
  end if

end program test_concat_layer
