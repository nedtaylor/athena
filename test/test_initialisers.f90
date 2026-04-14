program test_initialisers
  use coreutils, only: real32
  use athena, only: &
       full_layer_type, &
       conv2d_layer_type, &
       conv3d_layer_type, &
       base_layer_type
  use athena__misc_types, only: base_init_type
  use athena__initialiser, only: initialiser_setup, get_default_initialiser
  use athena__initialiser_data, only: data_init_type
  implicit none

  class(base_init_type), allocatable :: initialiser_var
  class(base_layer_type), allocatable :: full_layer, conv2d_layer, conv3d_layer
  logical :: success = .true.

  integer :: i
  integer :: width = 5, num_channels = 3, batch_size = 2
  real :: input_0d, input_1d(1), input_2d(1,1), &
       input_3d(1,1,1), input_6d(1,1,1,1,1,1)
  character(len=20) :: initialiser_names(11)


!-------------------------------------------------------------------------------
! Initialise initialiser names
!-------------------------------------------------------------------------------
  initialiser_names(1)  = 'zeros'
  initialiser_names(2)  = 'ones'
  initialiser_names(3)  = 'ident'
  initialiser_names(4)  = 'gaussian'
  initialiser_names(5)  = 'normal'
  initialiser_names(6)  = 'glorot_normal'
  initialiser_names(7)  = 'glorot_uniform'
  initialiser_names(8)  = 'he_normal'
  initialiser_names(9)  = 'he_uniform'
  initialiser_names(10)  = 'lecun_normal'
  initialiser_names(11) = 'lecun_uniform'


!-------------------------------------------------------------------------------
! check default initialiser names
!-------------------------------------------------------------------------------
  if(get_default_initialiser("selu").ne."lecun_normal")then
     success = .false.
     write(0,*) 'get_default_initialiser failed for selu'
     write(*,*)
  end if
  if(get_default_initialiser("relu").ne."he_uniform")then
     success = .false.
     write(0,*) 'get_default_initialiser failed for relu'
     write(*,*)
  end if
  if(get_default_initialiser("batch").ne."gaussian")then
     success = .false.
     write(0,*) 'get_default_initialiser failed for batch'
     write(*,*)
  end if
  if(get_default_initialiser("none").ne."glorot_uniform")then
     success = .false.
     write(0,*) 'get_default_initialiser failed for other'
     write(*,*)
  end if


!-------------------------------------------------------------------------------
! check initialisers work as expected for each rank
!-------------------------------------------------------------------------------
  do i = 1, size(initialiser_names)
     if(allocated(initialiser_var)) deallocate(initialiser_var)
     allocate(initialiser_var, source=initialiser_setup(initialiser_names(i)))
     if(.not.trim(initialiser_names(i)).eq."ident") &
          call initialiser_var%initialise(input_0d, fan_in = 1, fan_out = 1)
     call initialiser_var%initialise(input_1d, fan_in = 1, fan_out = 1)
     call initialiser_var%initialise(input_3d, fan_in = 1, fan_out = 1)
     call initialiser_var%initialise(input_6d, fan_in = 1, fan_out = 1)


     !! check for rank 2 data
     !!------------------------------------------------------------------------
     !! set up full layer
     full_layer = full_layer_type( &
          num_inputs=1, &
          num_outputs=10, &
          kernel_initialiser = initialiser_names(i), verbose = 1 )

     !! check layer name
     select type(full_layer)
     type is(full_layer_type)
        if(.not. trim(adjustl(full_layer%kernel_init%name)) .eq. &
             trim(initialiser_names(i)))then
           success = .false.
           write(0,*) 'kernel initialiser has wrong name for ', &
                trim(initialiser_names(i))
           write(0,*) 'returned name: ', full_layer%kernel_init%name
           write(*,*)
        end if
     class default
        success = .false.
        write(0,*) 'full layer is not of type full_layer_type'
     end select

     !! check for rank 3 data
     !!------------------------------------------------------------------------
     !! set up full layer
     conv2d_layer = conv2d_layer_type( &
          input_shape = [width, width, num_channels], &
          kernel_initialiser = initialiser_names(i), verbose = 1 )

     !! check layer name
     select type(conv2d_layer)
     type is(conv2d_layer_type)
        if(.not. trim(adjustl(conv2d_layer%kernel_init%name)) .eq. &
             trim(initialiser_names(i)))then
           success = .false.
           write(0,*) 'kernel initialiser has wrong name for ', &
                trim(initialiser_names(i))
           write(0,*) 'returned name: ', conv2d_layer%kernel_init%name
           write(*,*)
        end if
     class default
        success = .false.
        write(0,*) 'conv layer is not of type conv2d_layer_type'
     end select

     !! check for rank 4 data
     !!------------------------------------------------------------------------
     conv3d_layer = conv3d_layer_type( &
          input_shape = [width, width, width, num_channels], &
          kernel_initialiser = initialiser_names(i), verbose = 1 )

     !! check layer name
     select type(conv3d_layer)
     type is(conv3d_layer_type)
        if(.not. trim(adjustl(conv3d_layer%kernel_init%name)) .eq. &
             trim(initialiser_names(i)))then
           success = .false.
           write(0,*) 'kernel initialiser has wrong name for ', &
                trim(initialiser_names(i))
           write(0,*) 'returned name: ', conv3d_layer%kernel_init%name
           write(*,*)
        end if
     class default
        success = .false.
        write(0,*) 'conv layer is not of type conv2d_layer_type'
     end select
  end do


!-------------------------------------------------------------------------------
! check data initialiser preserves rank-2, rank-4, and rank-5 tensors
!-------------------------------------------------------------------------------
  data_initialiser_checks: block
    type(data_init_type) :: data_initialiser
    real(real32) :: data_2d(2,2), data_4d(2,2,1,2), data_5d(2,1,1,1,2)
    real(real32) :: init_2d(2,2), init_4d(2,2,1,2), init_5d(2,1,1,1,2)

    data_2d = reshape( &
         [1._real32, 2._real32, 3._real32, 4._real32], &
         shape(data_2d) &
    )
    data_initialiser = data_init_type(data_2d)
    call data_initialiser%initialise(init_2d)
    if(any(abs(init_2d - data_2d).gt.1.e-6_real32))then
       success = .false.
       write(0,*) 'data initialiser failed for rank-2 input'
    end if

    data_4d = reshape([(real(i, real32), i=1, size(data_4d))], shape(data_4d))
    data_initialiser = data_init_type(data_4d)
    call data_initialiser%initialise(init_4d)
    if(any(abs(init_4d - data_4d).gt.1.e-6_real32))then
       success = .false.
       write(0,*) 'data initialiser failed for rank-4 input'
    end if

    data_5d = reshape([(real(i, real32), i=1, size(data_5d))], shape(data_5d))
    data_initialiser = data_init_type(data_5d)
    call data_initialiser%initialise(init_5d)
    if(any(abs(init_5d - data_5d).gt.1.e-6_real32))then
       success = .false.
       write(0,*) 'data initialiser failed for rank-5 input'
    end if
  end block data_initialiser_checks


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_initialisers passed all tests'
  else
     write(0,*) 'test_initialisers failed one or more tests'
     stop 1
  end if

end program test_initialisers
