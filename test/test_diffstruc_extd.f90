program test_diffstruc_extd
  use coreutils, only: real32
  use athena__diffstruc_extd, only: add_bias, conv1d, conv2d, conv3d
  use diffstruc, only: array_type
  implicit none

  logical :: success
  type(array_type), target :: input, bias, kernel
  type(array_type), pointer :: output

  success = .true.

  write(*,'("Testing add_bias forward/backward")')

  call input%allocate([2, 3, 1])
  input%val(:,1) = [1._real32, 2._real32, 3._real32, 4._real32, &
       5._real32, 6._real32]
  call input%set_requires_grad(.true.)
  input%is_temporary = .false.

  call bias%allocate([3, 1])
  bias%val(:,1) = [10._real32, 20._real32, 30._real32]
  call bias%set_requires_grad(.true.)
  bias%is_temporary = .false.

  output => add_bias(input, bias, dim=2, dim_act_on_shape=.true.)
  if(any(abs(output%val(:,1) - [11._real32, 12._real32, 23._real32, &
       24._real32, 35._real32, 36._real32]) > 1.e-6_real32))then
     success = .false.
     write(0,*) 'add_bias returned the wrong forward values'
  end if

  call output%grad_reverse(reset_graph=.true.)

  if(.not. associated(input%grad) .or. &
       any(abs(input%grad%val(:,1) - 1._real32) > 1.e-6_real32))then
     success = .false.
     write(0,*) 'add_bias returned the wrong input gradient'
  end if

  if(.not. associated(bias%grad) .or. &
       any(abs(bias%grad%val(:,1) - 2._real32) > 1.e-6_real32))then
     success = .false.
     write(0,*) 'add_bias returned the wrong bias gradient'
  end if

  call output%nullify_graph()
  nullify(output)
  call input%deallocate()
  call bias%deallocate()

  write(*,'("Testing conv1d forward/backward")')

  call input%allocate([4, 1, 1])
  input%val(:,1) = [1._real32, 2._real32, 3._real32, 4._real32]
  call input%set_requires_grad(.true.)
  input%is_temporary = .false.

  call kernel%allocate([2, 1, 1, 1])
  kernel%val(:,1) = [1._real32, 2._real32]
  call kernel%set_requires_grad(.true.)
  kernel%is_temporary = .false.

  output => conv1d(input, kernel, stride=1, dilation=1)
  if(any(output%shape .ne. [3, 1]))then
     success = .false.
     write(0,*) 'conv1d returned the wrong output shape'
  end if
  if(any(abs(output%val(:,1) - [5._real32, 8._real32, 11._real32]) > &
       1.e-6_real32))then
     success = .false.
     write(0,*) 'conv1d returned the wrong forward values'
  end if

  call output%grad_reverse(reset_graph=.true.)

  if(.not. associated(input%grad) .or. &
       any(abs(input%grad%val(:,1) - [1._real32, 3._real32, 3._real32, &
       2._real32]) > 1.e-6_real32))then
     success = .false.
     write(0,*) 'conv1d returned the wrong input gradient'
  end if

  if(.not. associated(kernel%grad) .or. &
       any(abs(kernel%grad%val(:,1) - [6._real32, 9._real32]) > &
       1.e-6_real32))then
     success = .false.
     write(0,*) 'conv1d returned the wrong kernel gradient'
  end if

  call output%nullify_graph()
  nullify(output)
  call input%deallocate()
  call kernel%deallocate()

  write(*,'("Testing conv2d forward/backward")')

  call input%allocate([2, 2, 1, 1])
  input%val(:,1) = [1._real32, 2._real32, 3._real32, 4._real32]
  call input%set_requires_grad(.true.)
  input%is_temporary = .false.

  call kernel%allocate([2, 2, 1, 1, 1])
  kernel%val(:,1) = [1._real32, 2._real32, 3._real32, 4._real32]
  call kernel%set_requires_grad(.true.)
  kernel%is_temporary = .false.

  output => conv2d(input, kernel, stride=[1, 1], dilation=[1, 1])
  if(any(output%shape .ne. [1, 1, 1]))then
     success = .false.
     write(0,*) 'conv2d returned the wrong output shape'
  end if
  if(abs(output%val(1,1) - 30._real32) > 1.e-6_real32)then
     success = .false.
     write(0,*) 'conv2d returned the wrong forward value'
  end if

  call output%grad_reverse(reset_graph=.true.)

  if(.not. associated(input%grad) .or. &
       any(abs(input%grad%val(:,1) - [4._real32, 3._real32, 2._real32, &
       1._real32]) > 1.e-6_real32))then
     success = .false.
     write(0,*) 'conv2d returned the wrong input gradient'
  end if

  if(.not. associated(kernel%grad) .or. &
       any(abs(kernel%grad%val(:,1) - [1._real32, 2._real32, 3._real32, &
       4._real32]) > 1.e-6_real32))then
     success = .false.
     write(0,*) 'conv2d returned the wrong kernel gradient'
  end if

  call output%nullify_graph()
  nullify(output)
  call input%deallocate()
  call kernel%deallocate()

  write(*,'("Testing conv3d forward/backward")')

  call input%allocate([2, 2, 2, 1, 1])
  input%val(:,1) = [1._real32, 2._real32, 3._real32, 4._real32, &
       5._real32, 6._real32, 7._real32, 8._real32]
  call input%set_requires_grad(.true.)
  input%is_temporary = .false.

  call kernel%allocate([2, 2, 2, 1, 1, 1])
  kernel%val(:,1) = [1._real32, 2._real32, 3._real32, 4._real32, &
       5._real32, 6._real32, 7._real32, 8._real32]
  call kernel%set_requires_grad(.true.)
  kernel%is_temporary = .false.

  output => conv3d(input, kernel, stride=[1, 1, 1], dilation=[1, 1, 1])
  if(any(output%shape .ne. [1, 1, 1, 1]))then
     success = .false.
     write(0,*) 'conv3d returned the wrong output shape'
  end if
  if(abs(output%val(1,1) - 204._real32) > 1.e-6_real32)then
     success = .false.
     write(0,*) 'conv3d returned the wrong forward value'
  end if

  call output%grad_reverse(reset_graph=.true.)

  if(.not. associated(input%grad) .or. &
       any(abs(input%grad%val(:,1) - [1._real32, 2._real32, 3._real32, &
       4._real32, 5._real32, 6._real32, 7._real32, 8._real32]) > &
       1.e-6_real32))then
     success = .false.
     write(0,*) 'conv3d returned the wrong input gradient'
  end if

  if(.not. associated(kernel%grad) .or. &
       any(abs(kernel%grad%val(:,1) - [1._real32, 2._real32, 3._real32, &
       4._real32, 5._real32, 6._real32, 7._real32, 8._real32]) > &
       1.e-6_real32))then
     success = .false.
     write(0,*) 'conv3d returned the wrong kernel gradient'
  end if

  call output%nullify_graph()
  nullify(output)

  if(success)then
     write(*,*) 'test_diffstruc passed all tests'
  else
     write(0,*) 'test_diffstruc failed one or more tests'
     stop 1
  end if

end program test_diffstruc_extd
