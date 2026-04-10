program test_reshape_layer
  use athena
  use coreutils, only: test_error_handling
  use diffstruc, only: array_type
  use athena__misc_types, only: onnx_node_type, onnx_initialiser_type, &
       onnx_tensor_type
  use athena__flatten_layer, only: create_from_onnx_flatten_layer, &
       read_flatten_layer
  use athena__reshape_layer, only: create_from_onnx_reshape_layer, &
       read_reshape_layer
  implicit none

  logical :: success
  real(real32), parameter :: tol = 1.e-5_real32
  type(reshape_layer_type) :: reshape_layer
  type(flatten_layer_type) :: flatten_layer
  type(array_type) :: input(1,1)
  real(real32), allocatable :: flat_output(:)
  real(real32), allocatable :: reshape_output(:,:)
  real(real32) :: flat_expected(12)
  real(real32) :: reshape_expected(3,4)
  class(base_layer_type), allocatable :: read_layer
  class(base_layer_type), allocatable :: onnx_layer
  type(onnx_node_type) :: node
  type(onnx_initialiser_type), allocatable :: initialisers(:)
  type(onnx_tensor_type), allocatable :: value_info(:)
  integer, dimension(2) :: input_shape_2d, output_shape_2d
  integer, dimension(1) :: input_shape_1d, output_shape_1d
  integer, dimension(3) :: input_shape_3d
  integer :: index
  integer :: unit

  success = .true.

  write(*,'("Testing reshape flatten 2D to 1D")')
  input_shape_2d = [28, 28]
  output_shape_1d = [784]
  reshape_layer = reshape_layer_type(output_shape=output_shape_1d, &
       input_shape=input_shape_2d, verbose=0)
  if(size(reshape_layer%output_shape) .ne. 1)then
     success = .false.
     write(0,*) 'Expected output rank 1 for flatten 2D to 1D'
  end if
  if(reshape_layer%output_shape(1) .ne. 784)then
     success = .false.
     write(0,*) 'Expected output shape [784] for flatten 2D to 1D'
  end if

  write(*,'("Testing reshape unflatten 1D to 2D")')
  input_shape_1d = [784]
  output_shape_2d = [28, 28]
  reshape_layer = reshape_layer_type(output_shape=output_shape_2d, &
       input_shape=input_shape_1d, verbose=0)
  if(size(reshape_layer%output_shape) .ne. 2)then
     success = .false.
     write(0,*) 'Expected output rank 2 for unflatten 1D to 2D'
  end if
  if(any(reshape_layer%output_shape .ne. [28, 28]))then
     success = .false.
     write(0,*) 'Expected output shape [28, 28] for unflatten 1D to 2D'
  end if

  write(*,'("Testing reshape 3D to 2D")')
  input_shape_3d = [64, 32, 32]
  output_shape_2d = [64, 1024]
  reshape_layer = reshape_layer_type(output_shape=output_shape_2d, &
       input_shape=input_shape_3d, verbose=0)
  if(size(reshape_layer%output_shape) .ne. 2)then
     success = .false.
     write(0,*) 'Expected output rank 2 for reshape 3D to 2D'
  end if
  if(any(reshape_layer%output_shape .ne. [64, 1024]))then
     success = .false.
     write(0,*) 'Expected output shape [64, 1024] for reshape 3D to 2D'
  end if

  write(*,'("Testing reshape incompatible shapes")')
  test_error_handling = .true.
  reshape_layer = reshape_layer_type(output_shape=[22], &
       input_shape=[28, 28], verbose=0)
  test_error_handling = .false.

  write(*,'("Testing flatten forward pass")')
  flatten_layer = flatten_layer_type(input_shape=[2, 3, 2], verbose=0)
  call input(1,1)%allocate(array_shape=[2, 3, 2, 1], source=0._real32)
  flat_expected = [(real(index, real32), index=1, size(flat_expected))]
  input(1,1)%val(:,1) = flat_expected
  call flatten_layer%forward(input)
  allocate(flat_output, source=flatten_layer%output(1,1)%val(:,1))
  if(any(flatten_layer%output(1,1)%shape .ne. [size(flat_expected)]))then
     success = .false.
     write(0,*) 'flatten layer forward pass stored wrong output shape'
  elseif(size(flat_output) .ne. size(flat_expected))then
     success = .false.
     write(0,*) 'flatten layer forward pass returned wrong output size'
  elseif(any(abs(flat_output - flat_expected) .gt. tol))then
     success = .false.
     write(0,*) 'flatten layer forward pass changed tensor ordering'
  end if
  deallocate(flat_output)
  call input(1,1)%deallocate()

  write(*,'("Testing reshape forward pass")')
  reshape_layer = reshape_layer_type(output_shape=[3, 4], &
       input_shape=[2, 2, 3], verbose=0)
  call input(1,1)%allocate(array_shape=[2, 2, 3, 1], source=0._real32)
  input(1,1)%val(:,1) = [(real(index, real32), index=1, 12)]
  reshape_expected = reshape(input(1,1)%val(:,1), shape(reshape_expected))
  allocate(reshape_layer%output(1,1))
  call reshape_layer%forward(input)
  allocate(reshape_output, source=reshape( &
       reshape_layer%output(1,1)%val(:,1), [3, 4]))
  if(any(reshape_layer%output(1,1)%shape .ne. [3, 4]))then
     success = .false.
     write(0,*) 'reshape layer forward pass stored wrong output shape'
  elseif(any(shape(reshape_output) .ne. shape(reshape_expected)))then
     success = .false.
     write(0,*) 'reshape layer forward pass returned wrong output shape'
  elseif(any(abs(reshape_output - reshape_expected) .gt. tol))then
     success = .false.
     write(0,*) 'reshape layer forward pass changed tensor values'
  end if
  deallocate(reshape_output)
  call input(1,1)%deallocate()

  write(*,'("Testing reshape file I/O")')
  reshape_layer = reshape_layer_type(output_shape=[3, 4], &
       input_shape=[2, 2, 3], verbose=0)
  open(newunit=unit, file='test_reshape_layer.tmp', status='replace', &
       action='write')
  write(unit,'(A)') 'RESHAPE'
  call reshape_layer%print_to_unit(unit)
  write(unit,'(A)') 'END RESHAPE'
  close(unit)
  open(newunit=unit, file='test_reshape_layer.tmp', status='old', &
       action='read')
  read(unit,*)
  read_layer = read_reshape_layer(unit)
  close(unit, status='delete')
  select type(read_layer)
  type is(reshape_layer_type)
     if(any(read_layer%input_shape .ne. [2, 2, 3]))then
        success = .false.
        write(0,*) 'reshape layer file I/O lost input shape'
     end if
     if(any(read_layer%output_shape .ne. [3, 4]))then
        success = .false.
        write(0,*) 'reshape layer file I/O lost output shape'
     end if
  class default
     success = .false.
     write(0,*) 'reshape layer file I/O returned wrong layer type'
  end select

  write(*,'("Testing reshape ONNX builder")')
  allocate(initialisers(0))
  allocate(value_info(1))
  value_info(1)%name = 'reshape_out'
  value_info(1)%elem_type = 1
  allocate(value_info(1)%dims(3))
  value_info(1)%dims = [1, 3, 4]
  onnx_layer = create_from_onnx_reshape_layer(node, initialisers, value_info)
  select type(onnx_layer)
  type is(reshape_layer_type)
     if(any(onnx_layer%output_shape .ne. [3, 4]))then
        success = .false.
        write(0,*) 'reshape ONNX builder restored wrong output shape'
     end if
  class default
     success = .false.
     write(0,*) 'reshape ONNX builder returned wrong layer type'
  end select

  write(*,'("Testing flatten file I/O")')
  flatten_layer = flatten_layer_type(input_shape=[2, 2, 3], verbose=0)
  open(newunit=unit, file='test_flatten_layer.tmp', status='replace', &
       action='write')
  write(unit,'(A)') 'FLATTEN'
  call flatten_layer%print_to_unit(unit)
  write(unit,'(A)') 'END FLATTEN'
  close(unit)
  open(newunit=unit, file='test_flatten_layer.tmp', status='old', &
       action='read')
  read(unit,*)
  if(allocated(read_layer)) deallocate(read_layer)
  read_layer = read_flatten_layer(unit)
  close(unit, status='delete')
  select type(read_layer)
  type is(flatten_layer_type)
     if(any(read_layer%input_shape .ne. [2, 2, 3]))then
        success = .false.
        write(0,*) 'flatten layer file I/O lost input shape'
     end if
     if(any(read_layer%output_shape .ne. [12]))then
        success = .false.
        write(0,*) 'flatten layer file I/O lost output shape'
     end if
  class default
     success = .false.
     write(0,*) 'flatten layer file I/O returned wrong layer type'
  end select

  open(newunit=unit, file='test_flatten_rank_only.tmp', status='replace', &
       action='write')
  write(unit,'(A)') 'FLATTEN'
  write(unit,'(A)') '   INPUT_RANK = 3'
  write(unit,'(A)') 'END FLATTEN'
  close(unit)
  open(newunit=unit, file='test_flatten_rank_only.tmp', status='old', &
       action='read')
  read(unit,*)
  if(allocated(read_layer)) deallocate(read_layer)
  read_layer = read_flatten_layer(unit)
  close(unit, status='delete')
  select type(read_layer)
  type is(flatten_layer_type)
     if(read_layer%input_rank .ne. 3)then
        success = .false.
        write(0,*) 'flatten layer rank-only read lost input rank'
     end if
  class default
     success = .false.
     write(0,*) 'flatten layer rank-only read returned wrong layer type'
  end select

  write(*,'("Testing flatten ONNX builder")')
  deallocate(initialisers)
  deallocate(value_info)
  allocate(initialisers(0))
  allocate(value_info(0))
  if(allocated(onnx_layer)) deallocate(onnx_layer)
  onnx_layer = create_from_onnx_flatten_layer(node, initialisers, value_info)
  select type(onnx_layer)
  type is(flatten_layer_type)
     if(trim(onnx_layer%name) .ne. 'flatten' .or. &
          trim(onnx_layer%type) .ne. 'flat')then
        success = .false.
        write(0,*) 'flatten ONNX builder returned inconsistent metadata'
     end if
     if(onnx_layer%output_rank .ne. 1)then
        success = .false.
        write(0,*) 'flatten ONNX builder returned wrong output rank'
     end if
  class default
     success = .false.
     write(0,*) 'flatten ONNX builder returned wrong layer type'
  end select

  if(success)then
     write(*,*) 'test_reshape_layer passed all tests'
  else
     write(0,*) 'test_reshape_layer failed one or more tests'
     stop 1
  end if

end program test_reshape_layer
