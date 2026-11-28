program test_dropblock2d_layer
  use coreutils, only: real32
  use athena, only: &
       dropblock2d_layer_type, &
       base_layer_type
  use diffstruc, only: array_type
  use athena__dropblock2d_layer, only: read_dropblock2d_layer
  implicit none

  class(base_layer_type), allocatable, target :: db_layer
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: num_channels = 3, width = 6
  integer :: unit
  type(array_type) :: input(1,1)
  real, allocatable, dimension(:,:,:,:) :: input_4d, output_4d, gradient_4d
  real, allocatable, dimension(:) :: output_1d
  real, allocatable, dimension(:,:,:,:) :: output_2d
  real, parameter :: tol = 1.E-7
  logical :: success = .true.
  class(array_type), pointer :: output => null()
  class(array_type), pointer :: gradient => null()
  logical, allocatable, dimension(:,:) :: mask_first, mask_second
  type(dropblock2d_layer_type), target :: extra_layer
  real(real32) :: scale = 0._real32, gamma_expected = 0._real32

  integer :: c

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed


!-------------------------------------------------------------------------------
! Initialise random number generator with a seed
!-------------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=0)
  call random_seed(put = seed)


!-------------------------------------------------------------------------------
! set up layer
!-------------------------------------------------------------------------------
  db_layer = dropblock2d_layer_type( &
       rate = 0.0, &
       block_size = 5, &
       input_shape = [width, width, num_channels] &
  )

  !! check layer name
  if(.not. db_layer%name .eq. 'dropblock2d')then
     success = .false.
     write(0,*) 'dropblock2d layer has wrong name'
  end if

  !! check layer type
  select type(db_layer)
  type is(dropblock2d_layer_type)
     !! check input shape
     if(any(db_layer%input_shape .ne. [width,width,num_channels]))then
        success = .false.
        write(0,*) 'dropblock2d layer has wrong input_shape'
     end if

     !! check output shape
     if(any(db_layer%output_shape .ne. [width,width,num_channels]))then
        success = .false.
        write(0,*) 'dropblock2d layer has wrong output shape'
     end if

     if(any(.not.db_layer%mask))then
        success = .false.
        write(0,*) 'dropblock2d layer has wrong mask, should all be true for &
             &rate = 0.0'
     end if
  class default
     success = .false.
     write(0,*) 'dropblock2d layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! test forward pass and check expected output
! use varied per-channel inputs so channel mapping errors are caught
!-------------------------------------------------------------------------------
  allocate(input_4d(width, width, num_channels, 1))
  do c = 1, num_channels
     input_4d(:,:,c,1) = real(c)
  end do

  db_layer = dropblock2d_layer_type( &
       rate = 0.5, &
       block_size = 5, &
       input_shape = [width, width, num_channels] &
  )

  call input(1,1)%allocate(array_shape=[width, width, num_channels, 1])
  call input(1,1)%set_requires_grad(.true.)
  call input(1,1)%set(input_4d)

  !! run forward pass
  call db_layer%forward(input)
  call db_layer%extract_output(output_4d)

  !! check outputs have expected value for every channel
  select type(db_layer)
  type is(dropblock2d_layer_type)
     if(count(db_layer%mask) .gt. 0)then
        scale = real(product(shape(db_layer%mask)),real32) / &
             real(count(db_layer%mask),real32)
     else
        scale = 0._real32
     end if
     do c = 1, num_channels
        if(any( &
             abs( &
                  merge(input_4d(:,:,c,1),0.0,db_layer%mask) * scale - &
                  output_4d(:,:,c,1) &
             ) .gt. tol) &
        )then
           success = .false.
           write(0,*) &
                'dropblock2d layer forward pass failed: mask incorrectly applied'
        end if
     end do
  end select

  !! check gamma matches the documented formula
  select type(db_layer)
  type is(dropblock2d_layer_type)
     gamma_expected = ( db_layer%rate / &
          real(db_layer%block_size,real32)**2 ) * &
          real(db_layer%input_shape(1),real32) / &
          real(db_layer%input_shape(1) - db_layer%block_size + 1,real32) * &
          real(db_layer%input_shape(2),real32) / &
          real(db_layer%input_shape(2) - db_layer%block_size + 1,real32)
     if(abs(db_layer%gamma - gamma_expected) .gt. tol)then
        success = .false.
        write(0,*) 'dropblock2d layer gamma does not match expected formula'
     end if
     if(abs(db_layer%gamma - 0.18_real32) .gt. 1.0E-6_real32)then
        success = .false.
        write(0,*) 'dropblock2d layer gamma does not match expected constant'
     end if
  end select


!-------------------------------------------------------------------------------
! test backward pass and check expected output
!-------------------------------------------------------------------------------
  !! run backward pass with unit upstream gradient
  output => db_layer%output(1,1)
  call output%set_requires_grad(.true.)
  output%grad%val = 1._real32
  call output%grad_reverse()
  gradient => input(1,1)%grad
  call gradient%extract(gradient_4d)

  !! check gradient has expected value for every channel
  select type(db_layer)
  type is(dropblock2d_layer_type)
     do c = 1, num_channels
        if(any( &
             abs( &
                  merge(1.0_real32,0.0_real32,db_layer%mask) * scale - &
                  gradient_4d(:,:,c,1) &
             ) .gt. tol ) &
        )then
           success = .false.
           write(0,*) 'dropblock2d layer backward pass failed: mask &
                &incorrectly applied'
        end if
     end do
  end select


!-------------------------------------------------------------------------------
! test inference is the identity (no scaling)
!-------------------------------------------------------------------------------
  deallocate(output_4d)
  db_layer%inference = .true.
  call db_layer%forward(input)
  call db_layer%extract_output(output_4d)
  if(any(abs(input_4d - output_4d) .gt. tol))then
     success = .false.
     write(0,*) 'dropblock2d layer inference pass is not the identity'
  end if
  db_layer%inference = .false.

  deallocate(output_4d, gradient_4d, input_4d)
  call input(1,1)%deallocate()
  call input(1,1)%reset_graph()


!-------------------------------------------------------------------------------
! check output request using rank 1 and rank 2 arrays is consistent
!-------------------------------------------------------------------------------
  call db_layer%extract_output(output_1d)
  call db_layer%extract_output(output_2d)
  if(any(abs(output_1d - reshape(output_2d, [size(output_2d)])) .gt. 1.E-6))then
     success = .false.
     write(0,*) 'output_1d and output_2d are not consistent'
  end if
  deallocate(output_1d, output_2d)


!-------------------------------------------------------------------------------
! check rate=0 training forward is the identity (inference is false)
!-------------------------------------------------------------------------------
  extra_layer = dropblock2d_layer_type( &
       rate = 0.0, &
       block_size = 5, &
       input_shape = [width, width, num_channels] &
  )
  allocate(input_4d(width, width, num_channels, 1))
  do c = 1, num_channels
     input_4d(:,:,c,1) = real(c)
  end do
  call input(1,1)%allocate(array_shape=[width, width, num_channels, 1])
  call input(1,1)%set(input_4d)
  call extra_layer%forward(input)
  call extra_layer%extract_output(output_4d)
  do c = 1, num_channels
     if(any(abs(output_4d(:,:,c,1) - input_4d(:,:,c,1)) .gt. tol))then
        success = .false.
        write(0,*) 'dropblock2d layer rate=0 training pass is not the identity'
     end if
  end do
  deallocate(output_4d, input_4d)
  call input(1,1)%deallocate()
  call input(1,1)%reset_graph()


!-------------------------------------------------------------------------------
! check a fresh mask is sampled on every training forward
!-------------------------------------------------------------------------------
  extra_layer = dropblock2d_layer_type( &
       rate = 0.5, &
       block_size = 1, &
       input_shape = [width, width, 1] &
  )
  call input(1,1)%allocate(array_shape=[width, width, 1, 1], source=1._real32)
  call extra_layer%forward(input)
  mask_first = extra_layer%mask
  call extra_layer%forward(input)
  mask_second = extra_layer%mask
  if(.not. any(mask_first .neqv. mask_second))then
     success = .false.
     write(0,*) 'dropblock2d layer did not resample the mask on forward'
  end if
  deallocate(mask_first, mask_second)
  call input(1,1)%deallocate()
  call input(1,1)%reset_graph()


!-------------------------------------------------------------------------------
! check even and odd block extents and repeat initialisation
!-------------------------------------------------------------------------------
  extra_layer = dropblock2d_layer_type( &
       rate = 0.5, &
       block_size = 4, &
       input_shape = [width, width, num_channels] &
  )
  if(extra_layer%half + extra_layer%upper + 1 .ne. 4)then
     success = .false.
     write(0,*) 'dropblock2d layer block extent is wrong for block_size = 4'
  end if
  call extra_layer%init(input_shape=[width, width, num_channels])
  call extra_layer%init(input_shape=[width, width, num_channels])
  if(.not. allocated(extra_layer%output_shape))then
     success = .false.
     write(0,*) 'dropblock2d layer output_shape not allocated after init'
  else if(any(extra_layer%output_shape .ne. [width,width,num_channels]))then
     success = .false.
     write(0,*) 'dropblock2d layer output_shape wrong after repeated init'
  end if
  if(any(extra_layer%input_shape .ne. [width,width,num_channels]))then
     success = .false.
     write(0,*) 'dropblock2d layer input_shape wrong after repeated init'
  end if
  if(.not. allocated(extra_layer%mask))then
     success = .false.
     write(0,*) 'dropblock2d layer mask not allocated after init'
  else if(any(shape(extra_layer%mask) .ne. [width,width]))then
     success = .false.
     write(0,*) 'dropblock2d layer mask shape wrong after repeated init'
  else if(any(.not. extra_layer%mask))then
     success = .false.
     write(0,*) 'dropblock2d layer mask wrong after repeated init'
  end if

  extra_layer = dropblock2d_layer_type( &
       rate = 0.5, &
       block_size = 5, &
       input_shape = [width, width, num_channels] &
  )
  if(extra_layer%half + extra_layer%upper + 1 .ne. 5)then
     success = .false.
     write(0,*) 'dropblock2d layer block extent is wrong for block_size = 5'
  end if
  call extra_layer%init(input_shape=[width, width, num_channels])
  call extra_layer%init(input_shape=[width, width, num_channels])
  if(.not. allocated(extra_layer%output_shape))then
     success = .false.
     write(0,*) 'dropblock2d layer output_shape not allocated after init'
  else if(any(extra_layer%output_shape .ne. [width,width,num_channels]))then
     success = .false.
     write(0,*) 'dropblock2d layer output_shape wrong after repeated init'
  end if
  if(any(extra_layer%input_shape .ne. [width,width,num_channels]))then
     success = .false.
     write(0,*) 'dropblock2d layer input_shape wrong after repeated init'
  end if
  if(.not. allocated(extra_layer%mask))then
     success = .false.
     write(0,*) 'dropblock2d layer mask not allocated after init'
  else if(any(shape(extra_layer%mask) .ne. [width,width]))then
     success = .false.
     write(0,*) 'dropblock2d layer mask shape wrong after repeated init'
  end if


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_dropblock2d_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("DROPBLOCK2D")')
  call db_layer%print_to_unit(unit)
  write(unit,'("END DROPBLOCK2D")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_dropblock2d_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_dropblock2d_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (dropblock2d_layer_type)
     if(.not. read_layer%name .eq. 'dropblock2d')then
        success = .false.
        write(0,*) 'read dropblock2d layer has wrong name'
     end if
     if(abs(read_layer%rate - 0.5_real32) .gt. tol)then
        success = .false.
        write(0,*) 'read dropblock2d layer has wrong rate'
     end if
     if(read_layer%block_size .ne. 5)then
        success = .false.
        write(0,*) 'read dropblock2d layer has wrong block_size'
     end if
     if(any(read_layer%input_shape .ne. [width,width,num_channels]))then
        success = .false.
        write(0,*) 'read dropblock2d layer has wrong input_shape'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not dropblock2d_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_dropblock2d_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_dropblock2d_layer passed all tests'
  else
     write(0,*) 'test_dropblock2d_layer failed one or more tests'
     stop 1
  end if

end program test_dropblock2d_layer
