program test_full_layer
  use athena, only: &
       full_layer_type, &
       base_layer_type
  implicit none

  class(base_layer_type), allocatable :: full_layer1, full_layer2
  logical :: success = .true.


  !! set up full layer
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
     write(0,*) 'full layer has wrong output_shape'
  end if

  !! check layer type
  select type(full_layer1)
  type is(full_layer_type)
     !! check default layer transfer/activation function
     if(full_layer1%transfer%name .ne. 'none')then
        success = .false.
        write(0,*) 'full layer has wrong transfer: '//full_layer1%transfer%name
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

  !! check for any fails
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_full_layer passed all tests'
  else
     write(*,*) 'test_full_layer failed one or more tests'
     stop 1
  end if

end program test_full_layer