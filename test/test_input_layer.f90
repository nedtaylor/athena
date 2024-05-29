program test_input_layer
  use athena, only: &
       input1d_layer_type, &
       input2d_layer_type, &
       input3d_layer_type, &
       input4d_layer_type, &
       base_layer_type
  implicit none

  class(base_layer_type), allocatable :: input_layer

  integer :: batch_size = 1
  real, dimension(2) :: input_1d = 1.E0
  real, allocatable, dimension(:) :: output_1d
  real, dimension(2,1) :: input_2d = 2.E0
  real, allocatable, dimension(:,:) :: output_2d
  real, dimension(2,1,1) :: input_3d = 3.E0
  real, allocatable, dimension(:,:,:) :: output_3d
  real, dimension(2,1,1,1) :: input_4d = 4.E0
  real, allocatable, dimension(:,:,:,:) :: output_4d
  real, dimension(2,1,1,1,1) :: input_5d = 5.E0
  real, allocatable, dimension(:,:,:,:) :: output_5d
  real, dimension(2,1,1,1,1,1) :: input_6d = 6.E0
  real, allocatable, dimension(:,:,:,:) :: output_6d

  logical :: success = .true.


!!!-----------------------------------------------------------------------------
!!! set up input1d layer
!!!-----------------------------------------------------------------------------
  input_layer = input1d_layer_type( &
       input_shape=shape(input_1d), &
       batch_size=batch_size)
  if(input_layer%batch_size.ne.batch_size)then
     write(0,*) 'input1d_layer batch_size failed'
     success = .false.
  end if
  select type(input_layer)
  type is (input1d_layer_type)
     call input_layer%set(input_1d)
     call input_layer%forward(input_1d)
     call input_layer%get_output(output_1d)
     if(any(abs(output_1d-1.E0).gt.1.E-6))then
        write(0,*) 'input1d_layer forward 1d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_2d)
     call input_layer%forward(input_2d)
     call input_layer%get_output(output_2d)
     if(any(abs(output_2d-2.E0).gt.1.E-6))then
        write(0,*) 'input1d_layer forward 2d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_3d)
     call input_layer%forward(input_3d)
     if(any(abs(input_layer%output-3.E0).gt.1.E-6))then
        write(0,*) 'input1d_layer forward 3d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_4d)
     call input_layer%forward(input_4d)
     if(any(abs(input_layer%output-4.E0).gt.1.E-6))then
        write(0,*) 'input1d_layer forward 4d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_5d)
     call input_layer%forward(input_5d)
     if(any(abs(input_layer%output-5.E0).gt.1.E-6))then
        write(0,*) 'input1d_layer forward 5d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_6d)
     call input_layer%forward(input_6d)
     if(any(abs(input_layer%output-6.E0).gt.1.E-6))then
        write(0,*) 'input1d_layer forward 6d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%backward([1.E0], [1.E0])
  end select


!!!-----------------------------------------------------------------------------
!!! set up input2d layer
!!!-----------------------------------------------------------------------------
  deallocate(input_layer)
  input_layer = input2d_layer_type( &
       input_shape=shape(input_3d), &
       batch_size=batch_size)
  if(input_layer%batch_size.ne.batch_size)then
     write(0,*) 'input2d_layer batch_size failed'
     success = .false.
  end if
  select type(input_layer)
  type is (input2d_layer_type)
     call input_layer%set(input_1d)
     call input_layer%forward(input_1d)
     call input_layer%get_output(output_1d)
     if(any(abs(output_1d-1.E0).gt.1.E-6))then
        write(0,*) 'input2d_layer forward 1d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_2d)
     call input_layer%forward(input_2d)
     call input_layer%get_output(output_2d)
     if(any(abs(output_2d-2.E0).gt.1.E-6))then
        write(0,*) 'input2d_layer forward 2d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_3d)
     call input_layer%forward(input_3d)
     if(any(abs(input_layer%output-3.E0).gt.1.E-6))then
        write(0,*) 'input2d_layer forward 3d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_4d)
     call input_layer%forward(input_4d)
     if(any(abs(input_layer%output-4.E0).gt.1.E-6))then
        write(0,*) 'input2d_layer forward 4d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_5d)
     call input_layer%forward(input_5d)
     if(any(abs(input_layer%output-5.E0).gt.1.E-6))then
        write(0,*) 'input2d_layer forward 5d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_6d)
     call input_layer%forward(input_6d)
     if(any(abs(input_layer%output-6.E0).gt.1.E-6))then
        write(0,*) 'input2d_layer forward 6d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%backward([1.E0], [1.E0])
  end select


!!!-----------------------------------------------------------------------------
!!! set up input3d layer
!!!-----------------------------------------------------------------------------
  deallocate(input_layer)
  input_layer = input3d_layer_type( &
       input_shape=shape(input_3d), &
       batch_size=batch_size)
  if(input_layer%batch_size.ne.batch_size)then
     write(0,*) 'input3d_layer batch_size failed'
     success = .false.
  end if
  select type(input_layer)
  type is (input3d_layer_type)
     call input_layer%set(input_1d)
     call input_layer%forward(input_1d)
     call input_layer%get_output(output_1d)
     if(any(abs(output_1d-1.E0).gt.1.E-6))then
        write(0,*) 'input3d_layer forward 1d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_2d)
     call input_layer%forward(input_2d)
     call input_layer%get_output(output_2d)
     if(any(abs(output_2d-2.E0).gt.1.E-6))then
        write(0,*) 'input3d_layer forward 2d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_3d)
     call input_layer%forward(input_3d)
     if(any(abs(input_layer%output-3.E0).gt.1.E-6))then
        write(0,*) 'input3d_layer forward 3d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_4d)
     call input_layer%forward(input_4d)
     if(any(abs(input_layer%output-4.E0).gt.1.E-6))then
        write(0,*) 'input3d_layer forward 4d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_5d)
     call input_layer%forward(input_5d)
     if(any(abs(input_layer%output-5.E0).gt.1.E-6))then
        write(0,*) 'input3d_layer forward 5d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_6d)
     call input_layer%forward(input_6d)
     if(any(abs(input_layer%output-6.E0).gt.1.E-6))then
        write(0,*) 'input3d_layer forward 6d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%backward([1.E0], [1.E0])
  end select


!!!-----------------------------------------------------------------------------
!!! set up input4d layer
!!!-----------------------------------------------------------------------------
  deallocate(input_layer)
  input_layer = input4d_layer_type( &
       input_shape=shape(input_4d), &
       batch_size=batch_size)
  if(input_layer%batch_size.ne.batch_size)then
     write(0,*) 'input4d_layer batch_size failed'
     success = .false.
  end if
  select type(input_layer)
  type is (input4d_layer_type)
     call input_layer%set(input_1d)
     call input_layer%forward(input_1d)
     call input_layer%get_output(output_1d)
     if(any(abs(output_1d-1.E0).gt.1.E-6))then
        write(0,*) 'input4d_layer forward 1d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_2d)
     call input_layer%forward(input_2d)
     call input_layer%get_output(output_2d)
     if(any(abs(output_2d-2.E0).gt.1.E-6))then
        write(0,*) 'input4d_layer forward 2d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_3d)
     call input_layer%forward(input_3d)
     if(any(abs(input_layer%output-3.E0).gt.1.E-6))then
        write(0,*) 'input4d_layer forward 3d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_4d)
     call input_layer%forward(input_4d)
     if(any(abs(input_layer%output-4.E0).gt.1.E-6))then
        write(0,*) 'input4d_layer forward 4d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_5d)
     call input_layer%forward(input_5d)
     if(any(abs(input_layer%output-5.E0).gt.1.E-6))then
        write(0,*) 'input4d_layer forward 5d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%set(input_6d)
     call input_layer%forward(input_6d)
     if(any(abs(input_layer%output-6.E0).gt.1.E-6))then
        write(0,*) 'input4d_layer forward 6d failed'
        write(*,*) input_layer%output
        success = .false.
     end if
     call input_layer%backward([1.E0], [1.E0])
  end select


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_input_layer passed all tests'
  else
     write(0,*) 'test_input_layer failed one or more tests'
     stop 1
  end if

end program test_input_layer