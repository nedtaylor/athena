program test_input_layer
  use athena, only: &
       input_layer_type, &
       base_layer_type
  use diffstruc, only: array_type
  implicit none

  class(input_layer_type), allocatable :: input_layer

  real, dimension(2) :: input_1d = 1.E0
  real, allocatable, dimension(:) :: output_1d
  real, dimension(2,1) :: input_2d = 2.E0
  real, allocatable, dimension(:,:) :: output_2d
  real, dimension(2,1,1) :: input_3d = 3.E0
  real, allocatable, dimension(:,:,:) :: output_3d
  real, dimension(2,1,1,1) :: input_4d = 4.E0
  real, allocatable, dimension(:,:,:,:) :: output_4d
  real, dimension(2,1,1,1,1) :: input_5d = 5.E0
  real, allocatable, dimension(:,:,:,:,:) :: output_5d
  type(array_type) :: input_array(1,1), output_array

  logical :: success = .true.


!-------------------------------------------------------------------------------
! set up input layer
!-------------------------------------------------------------------------------
!   input_layer = input_layer_type( &
!        input_shape=shape(input_1d) &
!   )
!   call input_layer%set(input_1d)
!   call input_layer%forward(input_1d)
!   call input_layer%get_output(output_1d)
!   if(any(abs(output_1d-1.E0).gt.1.E-6))then
!      write(0,*) 'input_layer forward 1d failed'
!      write(*,*) output_1d
!      success = .false.
!   end if
!   deallocate(input_layer)
  input_layer = input_layer_type( &
       input_shape=shape(input_2d(:,1)) &
  )
  call input_array(1,1)%allocate(array_shape=shape(input_2d))
  call input_array(1,1)%set(input_2d)
  call input_layer%forward(input_array)
  output_array = input_layer%output(1,1)
  if(any(abs(output_array%val-2.E0).gt.1.E-6))then
     write(0,*) 'input_layer forward 2d failed'
     write(*,*) output_array%val
     success = .false.
  end if
  call input_array(1,1)%deallocate()
  deallocate(input_layer)
  input_layer = input_layer_type( &
       input_shape=shape(input_3d(:,:,1)) &
  )
  call input_array(1,1)%allocate(array_shape=shape(input_3d))
  call input_array(1,1)%set(input_3d)
  call input_layer%forward(input_array)
  output_array = input_layer%output(1,1)
  if(any(abs(output_array%val-3.E0).gt.1.E-6))then
     write(0,*) 'input_layer forward 3d failed'
     write(*,*) output_array%val
     success = .false.
  end if
  deallocate(input_layer)
  call input_array(1,1)%deallocate()
  input_layer = input_layer_type( &
       input_shape=shape(input_4d(:,:,:,1)) &
  )
  call input_array(1,1)%allocate(array_shape=shape(input_4d))
  call input_array(1,1)%set(input_4d)
  call input_layer%forward(input_array)
  output_array = input_layer%output(1,1)
  if(any(abs(output_array%val-4.E0).gt.1.E-6))then
     write(0,*) 'input_layer forward 4d failed'
     write(*,*) output_array%val
     success = .false.
  end if
  deallocate(input_layer)
  call input_array(1,1)%deallocate()
  input_layer = input_layer_type( &
       input_shape=shape(input_5d(:,:,:,:,1)) &
  )
  call input_array(1,1)%allocate(array_shape=shape(input_5d))
  call input_array(1,1)%set(input_5d)
  call input_layer%forward(input_array)
  output_array = input_layer%output(1,1)
  if(any(abs(output_array%val-5.E0).gt.1.E-6))then
     write(0,*) 'input_layer forward 5d failed'
     write(*,*) output_array%val
     success = .false.
  end if
  !   call input_layer%set(input_6d)
  !   call input_layer%forward(input_6d)
  !   if(any(abs(input_layer%output-6.E0).gt.1.E-6))then
  !      write(0,*) 'input_layer forward 4d failed'
  !      write(*,*) input_layer%output
  !      success = .false.
  !   end if
  call output_array%grad_reverse()


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_input_layer passed all tests'
  else
     write(0,*) 'test_input_layer failed one or more tests'
     stop 1
  end if

end program test_input_layer
