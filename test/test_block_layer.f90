program test_block_layer
  !! Unit tests for the block layer module
  use coreutils, only: real32
  use athena, only: &
       block_layer_type, &
       full_layer_type, &
       actv_layer_type, &
       dropblock2d_layer_type, &
       batchnorm0d_layer_type, &
       container_layer_type, &
       base_layer_type, &
       read_block_layer
  use diffstruc, only: array_type, sum, squared
  implicit none

  class(base_layer_type), allocatable :: read_layer
  logical :: success = .true.
  real(real32), parameter :: tol = 1.E-6_real32

  call test_constructor_shapes(success)
  call test_actv_first(success)
  call test_forward_chain(success)
  call test_batchnorm_last(success)
  call test_input_gradients(success)
  call test_repeated_forward_actv(success)
  call test_inference(success)
  call test_io_roundtrip(success)

  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_block_layer passed all tests'
  else
     write(0,*) 'test_block_layer failed one or more tests'
     stop 1
  end if

contains

!-------------------------------------------------------------------------------
! Constructor shape, shape preservation, and parameter round-trip
!-------------------------------------------------------------------------------
  subroutine test_constructor_shapes(success)
    logical, intent(inout) :: success
    type(block_layer_type) :: blk
    type(array_type) :: xin(1,1)
    real(real32), allocatable :: params(:), zero_params(:), before(:), after(:)

    blk = block_layer_type( &
         layers = [ &
              make_container(full_layer_type( &
                   num_outputs=3, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )), &
              make_container(full_layer_type( &
                   num_outputs=2, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )) &
         ], &
         input_shape = [4] &
    )

    if(blk%num_layers.ne.2)then
       success = .false.
       write(0,*) 'block layer has wrong number of layers'
    end if
    if(.not.allocated(blk%input_shape))then
       success = .false.
       write(0,*) 'block layer input_shape is not allocated'
    elseif(any(blk%input_shape.ne.[4]))then
       success = .false.
       write(0,*) 'block layer input_shape was overwritten by a layer shape'
    end if
    if(any(blk%output_shape.ne.[2]))then
       success = .false.
       write(0,*) 'block layer has wrong output_shape'
    end if
    if(blk%num_params.ne.23)then
       success = .false.
       write(0,*) 'block layer has wrong number of parameters'
    end if

    params = blk%get_params()

    if(size(params).ne.blk%num_params)then
       success = .false.
       write(0,*) 'block layer get_params returned the wrong size'
    end if

    call xin(1,1)%allocate(array_shape=[4,2])
    xin(1,1)%val(:,1) = [0.1_real32, -0.2_real32, 0.3_real32, 0.4_real32]
    xin(1,1)%val(:,2) = [0.5_real32, 0.6_real32, -0.7_real32, 0.8_real32]
    xin(1,1)%is_temporary = .false.

    call blk%forward(xin)

    before = blk%output(1,1)%val(:,1)

    zero_params = params * 0._real32
    call blk%set_params(zero_params)

    call blk%forward(xin)
    if(any(abs(blk%output(1,1)%val).gt.tol))then
       success = .false.
       write(0,*) 'block layer with zero parameters is not zero'
    end if

    call blk%set_params(params)
    call blk%forward(xin)
    after = blk%output(1,1)%val(:,1)
    if(any(abs(before - after).gt.tol))then
       success = .false.
       write(0,*) 'block layer parameter restore changed the output'
    end if
    call xin(1,1)%deallocate()

  end subroutine test_constructor_shapes
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! Block bounded by activation layers without input shapes
!-------------------------------------------------------------------------------
  subroutine test_actv_first(success)
    logical, intent(inout) :: success
    type(block_layer_type) :: blk
    type(array_type) :: xin(1,1)

    blk = block_layer_type( &
         layers = [ &
              make_container(actv_layer_type(activation='relu')), &
              make_container(full_layer_type( &
                   num_outputs=3, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )), &
              make_container(actv_layer_type(activation='relu')) &
         ], &
         input_shape = [4] &
    )

    if(.not.allocated(blk%input_shape))then
       success = .false.
       write(0,*) 'actv-first block has no input_shape'
    elseif(any(blk%input_shape.ne.[4]))then
       success = .false.
       write(0,*) 'actv-first block has wrong input_shape'
    end if
    if(blk%input_rank.ne.1)then
       success = .false.
       write(0,*) 'actv-first block has wrong input_rank'
    end if
    if(.not.allocated(blk%output_shape))then
       success = .false.
       write(0,*) 'actv-first block has no output_shape'
    elseif(any(blk%output_shape.ne.[3]))then
       success = .false.
       write(0,*) 'actv-first block has wrong output_shape'
    end if
    if(blk%output_rank.ne.1)then
       success = .false.
       write(0,*) 'actv-first block has wrong output_rank'
    end if

    call xin(1,1)%allocate(array_shape=[4,2])
    xin(1,1)%val(:,1) = [-0.1_real32, 0.2_real32, -0.3_real32, 0.4_real32]
    xin(1,1)%val(:,2) = [0.5_real32, -0.6_real32, 0.7_real32, -0.8_real32]
    xin(1,1)%is_temporary = .false.

    call blk%forward(xin)
    if(any(shape(blk%output(1,1)%val).ne.[3,2]))then
       success = .false.
       write(0,*) 'actv-first block forward output has the wrong shape'
    end if
    call xin(1,1)%deallocate()

  end subroutine test_actv_first
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! Forward chain, with and without an activation layer
!-------------------------------------------------------------------------------
  subroutine test_forward_chain(success)
    logical, intent(inout) :: success
    type(block_layer_type) :: blk
    type(array_type) :: xin(1,1)
    real(real32), allocatable :: block_out(:,:), layer_out(:,:)

    blk = block_layer_type( &
         layers = [ &
              make_container(full_layer_type( &
                   num_outputs=3, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )), &
              make_container(full_layer_type( &
                   num_outputs=2, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )) &
         ], &
         input_shape = [4] &
    )
    call xin(1,1)%allocate(array_shape=[4,2])
    xin(1,1)%val(:,1) = [0.1_real32, -0.2_real32, 0.3_real32, 0.4_real32]
    xin(1,1)%val(:,2) = [0.5_real32, 0.6_real32, -0.7_real32, 0.8_real32]
    xin(1,1)%is_temporary = .false.

    call blk%forward(xin)
    block_out = blk%output(1,1)%val
    call blk%layers(1)%layer%forward(xin)
    call blk%layers(2)%layer%forward(blk%layers(1)%layer%output)
    layer_out = blk%layers(2)%layer%output(1,1)%val
    if(any(abs(block_out - layer_out).gt.tol))then
       success = .false.
       write(0,*) 'block forward output does not match the manual chain'
    end if
    call xin(1,1)%deallocate()

    blk = block_layer_type( &
         layers = [ &
              make_container(full_layer_type( &
                   num_outputs=3, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )), &
              make_container(actv_layer_type(activation='relu')), &
              make_container(full_layer_type( &
                   num_outputs=2, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )) &
         ], &
         input_shape = [4] &
    )
    call xin(1,1)%allocate(array_shape=[4,2])
    xin(1,1)%val(:,1) = [0.1_real32, -0.2_real32, 0.3_real32, 0.4_real32]
    xin(1,1)%val(:,2) = [0.5_real32, 0.6_real32, -0.7_real32, 0.8_real32]
    xin(1,1)%is_temporary = .false.

    call blk%forward(xin)
    block_out = blk%output(1,1)%val
    call blk%layers(1)%layer%forward(xin)
    call blk%layers(2)%layer%forward(blk%layers(1)%layer%output)
    call blk%layers(3)%layer%forward(blk%layers(2)%layer%output)
    layer_out = blk%layers(3)%layer%output(1,1)%val
    if(any(abs(block_out - layer_out).gt.tol))then
       success = .false.
       write(0,*) 'block forward with activation does not match the chain'
    end if
    call xin(1,1)%deallocate()

  end subroutine test_forward_chain
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! Forward and gradient flow for a block ending in a batchnorm layer
!-------------------------------------------------------------------------------
  subroutine test_batchnorm_last(success)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    logical, intent(inout) :: success
    type(block_layer_type) :: blk
    type(array_type) :: xin(1,1)
    type(array_type), pointer :: loss

    blk = block_layer_type( &
         layers = [ &
              make_container(full_layer_type( &
                   num_outputs=3, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )), &
              make_container(batchnorm0d_layer_type()) &
         ], &
         input_shape = [4] &
    )

    call xin(1,1)%allocate(array_shape=[4,2])
    xin(1,1)%val(:,1) = [0.1_real32, -0.2_real32, 0.3_real32, 0.4_real32]
    xin(1,1)%val(:,2) = [0.5_real32, 0.6_real32, -0.7_real32, 0.8_real32]
    xin(1,1)%is_temporary = .false.
    call xin(1,1)%set_requires_grad(.true.)

    call blk%forward(xin)
    if(any(shape(blk%output(1,1)%val).ne.[3,2]))then
       success = .false.
       write(0,*) 'batchnorm-last block has the wrong output shape'
    end if
    if(any(.not.ieee_is_finite(blk%output(1,1)%val)))then
       success = .false.
       write(0,*) 'batchnorm-last block produced non-finite values'
    end if

    loss => sum(squared(blk%output(1,1)))
    call loss%grad_reverse()
    if(.not.associated(xin(1,1)%grad))then
       success = .false.
       write(0,*) 'batchnorm-last block did not set the input gradient'
    elseif(any(.not.ieee_is_finite(xin(1,1)%grad%val)))then
       success = .false.
       write(0,*) 'batchnorm-last block produced a non-finite input gradient'
    elseif(.not.any(abs(xin(1,1)%grad%val).gt.1.E-8_real32))then
       success = .false.
       write(0,*) 'batchnorm-last block produced an all-zero input gradient'
    end if

    if(associated(loss)) call loss%release_expression()
    call xin(1,1)%deallocate()

  end subroutine test_batchnorm_last
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! Central finite-difference check of the block input gradient
!-------------------------------------------------------------------------------
  subroutine test_input_gradients(success)
    logical, intent(inout) :: success
    type(block_layer_type) :: blk
    type(array_type) :: xin(1,1)
    type(array_type), pointer :: loss
    real(real32), allocatable :: x0(:,:), ana_x(:,:), grads(:)
    real(real32) :: eps, num
    real(8) :: num_loss
    integer :: i, s, fails

    blk = block_layer_type( &
         layers = [ &
              make_container(full_layer_type( &
                   num_outputs=3, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )), &
              make_container(full_layer_type( &
                   num_outputs=2, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )) &
         ], &
         input_shape = [4] &
    )
    call xin(1,1)%allocate(array_shape=[4,2])
    xin(1,1)%val(:,1) = [0.1_real32, -0.2_real32, 0.3_real32, 0.4_real32]
    xin(1,1)%val(:,2) = [0.5_real32, 0.6_real32, -0.7_real32, 0.8_real32]
    xin(1,1)%is_temporary = .false.
    call xin(1,1)%set_requires_grad(.true.)

    call blk%forward(xin)
    loss => sum(squared(blk%output(1,1)))
    call loss%grad_reverse()
    if(.not.associated(xin(1,1)%grad))then
       success = .false.
       write(0,*) 'block layer did not set the input gradient'
       call loss%release_expression()
       call xin(1,1)%deallocate()
       return
    end if
    allocate(x0(4,2), ana_x(4,2))
    x0 = xin(1,1)%val
    ana_x = xin(1,1)%grad%val

    grads = blk%get_gradients()
    if(size(grads).ne.blk%num_params)then
       success = .false.
       write(0,*) 'block layer get_gradients returned the wrong size'
    elseif(.not.any(abs(grads).gt.1.E-8_real32))then
       success = .false.
       write(0,*) 'block layer get_gradients returned all zeros'
    end if

    call blk%set_gradients(0._real32)
    grads = blk%get_gradients()
    if(any(abs(grads).gt.tol))then
       success = .false.
       write(0,*) 'block layer set_gradients(0) did not zero gradients'
    end if

    call blk%set_gradients(1._real32)
    grads = blk%get_gradients()
    if(any(abs(grads - 1._real32).gt.tol))then
       success = .false.
       write(0,*) 'block layer set_gradients(1) did not set unit gradients'
    end if

    call loss%release_expression()

    eps = 1.E-2_real32
    fails = 0
    do s = 1, 2
       do i = 1, 4
          xin(1,1)%val(i,s) = x0(i,s) + eps
          num_loss = numeric_loss(blk, xin)
          xin(1,1)%val(i,s) = x0(i,s) - eps
          num_loss = num_loss - numeric_loss(blk, xin)
          xin(1,1)%val(i,s) = x0(i,s)
          num = real(num_loss / real(2._8 * eps, 8), real32)
          call check_value(ana_x(i,s), num, fails)
       end do
    end do
    if(fails.ne.0) success = .false.

    deallocate(x0, ana_x, grads)
    call xin(1,1)%deallocate()

  end subroutine test_input_gradients
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! Repeated forward with a temporary activation output at the end of the block
!-------------------------------------------------------------------------------
  subroutine test_repeated_forward_actv(success)
    logical, intent(inout) :: success
    type(block_layer_type) :: blk
    type(array_type) :: xin(1,1)
    type(array_type), pointer :: loss
    integer :: iter

    blk = block_layer_type( &
         layers = [ &
              make_container(full_layer_type( &
                   num_outputs=3, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )), &
              make_container(actv_layer_type(activation='relu')) &
         ], &
         input_shape = [4] &
    )
    call xin(1,1)%allocate(array_shape=[4,2])
    xin(1,1)%val(:,1) = [0.1_real32, 0.2_real32, 0.3_real32, 0.4_real32]
    xin(1,1)%val(:,2) = [0.5_real32, 0.6_real32, 0.7_real32, 0.8_real32]
    xin(1,1)%is_temporary = .false.
    call xin(1,1)%set_requires_grad(.true.)

    do iter = 1, 2
       call blk%forward(xin)
    end do
    loss => sum(squared(blk%output(1,1)))
    call loss%grad_reverse()

    if(.not.associated(xin(1,1)%grad))then
       success = .false.
       write(0,*) 'repeated forward did not set the input gradient'
    else
       if(any(xin(1,1)%grad%val.ne.xin(1,1)%grad%val))then
          success = .false.
          write(0,*) 'repeated forward produced a non-finite input gradient'
       end if
       if(.not.any(abs(xin(1,1)%grad%val).gt.1.E-8_real32))then
          success = .false.
          write(0,*) 'repeated forward produced a zero input gradient'
       end if
    end if

    call loss%release_expression()
    call xin(1,1)%deallocate()

  end subroutine test_repeated_forward_actv
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! Inference flag propagation (DropBlock must be the identity in inference mode)
!-------------------------------------------------------------------------------
  subroutine test_inference(success)
    logical, intent(inout) :: success
    type(block_layer_type) :: blk
    type(array_type) :: xin(1,1)
    integer :: i

    blk = block_layer_type( &
         layers = [ &
              make_container(dropblock2d_layer_type( &
                   rate=0.9_real32, &
                   block_size=1 &
              )) &
         ], &
         input_shape = [6,6,3] &
    )
    blk%inference = .true.

    call xin(1,1)%allocate(array_shape=[6,6,3,1])
    xin(1,1)%val(:,1) = [ (real(i,real32), i=1,product([6,6,3])) ]
    xin(1,1)%is_temporary = .false.

    call blk%forward(xin)
    if(any(abs(blk%output(1,1)%val - xin(1,1)%val).gt.tol))then
       success = .false.
       write(0,*) 'dropblock block is not the identity in inference mode'
    end if
    call xin(1,1)%deallocate()

  end subroutine test_inference
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! Print/read round-trip, including a rank-1 activation layer
!-------------------------------------------------------------------------------
  subroutine test_io_roundtrip(success)
    logical, intent(inout) :: success
    type(block_layer_type) :: blk
    type(block_layer_type), allocatable :: direct_block
    type(array_type) :: xin(1,1)
    real(real32), allocatable :: p_orig(:), p_read(:)
    real(real32), allocatable :: out_orig(:,:), out_read(:,:)
    integer :: unit, l
    character(20), dimension(3) :: layer_names

    blk = block_layer_type( &
         layers = [ &
              make_container(full_layer_type( &
                   num_outputs=3, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )), &
              make_container(actv_layer_type(activation='relu')), &
              make_container(full_layer_type( &
                   num_outputs=2, &
                   kernel_initialiser='ones', &
                   bias_initialiser='zeros' &
              )) &
         ], &
         input_shape = [4] &
    )
    call xin(1,1)%allocate(array_shape=[4,2])
    xin(1,1)%val(:,1) = [0.1_real32, 0.2_real32, 0.3_real32, 0.4_real32]
    xin(1,1)%val(:,2) = [0.5_real32, 0.6_real32, 0.7_real32, 0.8_real32]
    xin(1,1)%is_temporary = .false.

    call blk%forward(xin)
    p_orig = blk%get_params()
    out_orig = blk%output(1,1)%val
    layer_names = [character(20) :: 'full', 'actv', 'full']

    open(newunit=unit, file='test_block_layer.tmp', &
         status='replace', action='write')
    write(unit,'("BLOCK")')
    call blk%print_to_unit(unit)
    write(unit,'("END BLOCK")')
    close(unit)

    open(newunit=unit, file='test_block_layer.tmp', status='old', action='read')
    read(unit,*)
    read_layer = read_block_layer(unit)
    close(unit)

    select type(block_read => read_layer)
    type is(block_layer_type)
       if(block_read%num_layers.ne.3)then
          success = .false.
          write(0,*) 'read block layer has wrong number of layers'
       end if
       if(trim(block_read%name).ne.'block')then
          success = .false.
          write(0,*) 'read block layer has wrong name'
       end if
       if(any(block_read%input_shape.ne.[4]))then
          success = .false.
          write(0,*) 'read block layer has wrong input_shape'
       end if
       if(any(block_read%output_shape.ne.[2]))then
          success = .false.
          write(0,*) 'read block layer has wrong output_shape'
       end if
       do l = 1, 3
          if(trim(block_read%layers(l)%layer%name).ne. &
               trim(layer_names(l)))then
             success = .false.
             write(0,*) 'read block layer has layers in the wrong order'
          end if
       end do
       p_read = block_read%get_params()
       if(size(p_read).ne.size(p_orig))then
          success = .false.
          write(0,*) 'read block layer has the wrong number of parameters'
       elseif(any(abs(p_read - p_orig).gt.tol))then
          success = .false.
          write(0,*) 'read block layer has the wrong parameters'
       end if
       call block_read%forward(xin)
       out_read = block_read%output(1,1)%val
       if(any(abs(out_read - out_orig).gt.tol))then
          success = .false.
          write(0,*) 'read block layer has the wrong forward output'
       end if
    class default
       success = .false.
       write(0,*) 'read layer is not block_layer_type'
    end select

    ! Read directly into an allocated block with no placeholder layers
    open(newunit=unit, file='test_block_layer.tmp', status='old', action='read')
    read(unit,*)
    allocate(block_layer_type :: direct_block)
    call direct_block%read(unit)
    close(unit)

    if(direct_block%num_layers.ne.3)then
       success = .false.
       write(0,*) 'directly read block layer has wrong number of layers'
    end if
    if(any(direct_block%input_shape.ne.[4]))then
       success = .false.
       write(0,*) 'directly read block layer has wrong input_shape'
    end if
    if(any(direct_block%output_shape.ne.[2]))then
       success = .false.
       write(0,*) 'directly read block layer has wrong output_shape'
    end if
    deallocate(direct_block)

    open(newunit=unit, file='test_block_layer.tmp', status='old')
    close(unit, status='delete')

    call xin(1,1)%deallocate()

  end subroutine test_io_roundtrip
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
  real(8) function numeric_loss(layer, xin) result(l)
    type(block_layer_type), intent(inout) :: layer
    type(array_type), intent(in) :: xin(1,1)

    call layer%forward(xin)
    l = sum(real(layer%output(1,1)%val, 8)**2)

  end function numeric_loss
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
  subroutine check_value(analytic, numeric, fails)
    real(real32), intent(in) :: analytic, numeric
    integer, intent(inout) :: fails
    real(real32) :: rel

    rel = abs(analytic - numeric) / max(1.E-4_real32, abs(numeric))
    if(rel.gt.5.E-2_real32)then
       fails = fails + 1
       if(fails.le.5)then
          write(0,'("grad mismatch analytic=",E12.4," numeric=",E12.4)') &
               analytic, numeric
       end if
    end if

  end subroutine check_value
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
  function make_container(layer) result(container)
    class(base_layer_type), intent(in) :: layer
    type(container_layer_type) :: container

    allocate(container%layer, source=layer)

  end function make_container
!-------------------------------------------------------------------------------

end program test_block_layer
