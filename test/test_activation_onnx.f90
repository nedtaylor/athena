program test_activation_onnx
  use coreutils, only: real32
  use athena__activation_none, only: create_from_onnx_none_activation
  use athena__activation_selu, only: create_from_onnx_selu_activation, &
       selu_actv_type
  use athena__misc_types, only: base_actv_type, onnx_attribute_type
  use diffstruc, only: array_type
  implicit none

  logical :: success
  class(base_actv_type), allocatable :: activation
  type(onnx_attribute_type), allocatable :: attributes(:)
  type(array_type), target :: input
  type(array_type), pointer :: output, loss
  real(real32) :: expected_negative
  real(real32) :: expected_negative_grad

  success = .true.

  write(*,'("Testing none activation ONNX import/export")')

  activation = create_from_onnx_none_activation([ &
       onnx_attribute_type('name', 'string', 'none') &
  ])

  call input%allocate([3, 1])
  input%val(:,1) = [-1._real32, 0._real32, 2._real32]
  call input%set_requires_grad(.true.)
  input%is_temporary = .false.

  output => activation%apply(input)
  if(any(abs(output%val(:,1) - input%val(:,1)) .gt. 1.e-6_real32))then
     success = .false.
     write(0,*) 'none activation returned the wrong forward values'
  end if

  loss => output - 0._real32
  call loss%grad_reverse(reset_graph=.true.)
  if(.not. associated(input%grad) .or. &
       any(abs(input%grad%val(:,1) - 1._real32) .gt. 1.e-6_real32))then
     success = .false.
     write(0,*) 'none activation returned the wrong backward values'
  end if

  attributes = activation%export_attributes()
  if(size(attributes) .ne. 1 .or. &
       trim(attributes(1)%name) .ne. 'name' .or. &
       trim(attributes(1)%val) .ne. 'none')then
     success = .false.
     write(0,*) 'none activation exported the wrong ONNX attributes'
  end if

  call output%nullify_graph()
  nullify(output)
  call input%deallocate()
  if(allocated(activation)) deallocate(activation)
  if(allocated(attributes)) deallocate(attributes)

  write(*,'("Testing SELU activation ONNX import/export")')

  activation = create_from_onnx_selu_activation([ &
       onnx_attribute_type('name', 'string', 'selu'), &
       onnx_attribute_type('scale', 'float', '1.5'), &
       onnx_attribute_type('alpha', 'float', '1.7'), &
       onnx_attribute_type('lambda', 'float', '1.05') &
  ])

  call input%allocate([2, 1])
  input%val(:,1) = [-1._real32, 2._real32]
  call input%set_requires_grad(.true.)
  input%is_temporary = .false.

  output => activation%apply(input)
  expected_negative = (exp(-1._real32) - 1._real32) * &
       1.7_real32 * 1.05_real32 * 1.5_real32
  if(abs(output%val(1,1) - expected_negative) .gt. 1.e-5_real32 .or. &
       abs(output%val(2,1) - 3.15_real32) .gt. 1.e-6_real32)then
     success = .false.
     write(0,*) 'SELU activation returned the wrong forward values'
  end if

  call output%grad_reverse(reset_graph=.true.)
  expected_negative_grad = exp(-1._real32) * &
       1.7_real32 * 1.05_real32 * 1.5_real32
  if(.not. associated(input%grad) .or. &
       abs(input%grad%val(1,1) - expected_negative_grad) .gt. 1.e-5_real32 &
       .or. abs(input%grad%val(2,1) - 1.575_real32) .gt. 1.e-6_real32)then
     success = .false.
     write(0,*) 'SELU activation returned the wrong backward values'
  end if

  attributes = activation%export_attributes()
  if(size(attributes) .ne. 4)then
     success = .false.
     write(0,*) 'SELU activation exported the wrong number of ONNX attributes'
  end if

  select type (activation)
  type is (selu_actv_type)
     if(abs(activation%scale - 1.5_real32) .gt. 1.e-6_real32 .or. &
          abs(activation%alpha - 1.7_real32) .gt. 1.e-6_real32 .or. &
          abs(activation%lambda - 1.05_real32) .gt. 1.e-6_real32)then
        success = .false.
        write(0,*) 'SELU activation did not retain imported attributes'
     end if
  class default
     success = .false.
     write(0,*) 'SELU ONNX import returned the wrong activation type'
  end select

  call output%nullify_graph()
  nullify(output)

  if(success)then
     write(*,*) 'test_activation_onnx passed all tests'
  else
     write(0,*) 'test_activation_onnx failed one or more tests'
     stop 1
  end if

end program test_activation_onnx
