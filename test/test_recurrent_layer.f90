program test_recurrent_layer
  use coreutils, only: real32
  use athena, only: &
       base_layer_type
  use athena__recurrent_layer, only: &
       recurrent_layer_type, &
       read_recurrent_layer
  use diffstruc, only: array_type
  implicit none

  class(base_layer_type), allocatable :: recu_layer1, recu_layer2, recu_layer3
  class(base_layer_type), allocatable :: read_layer
  type(array_type), dimension(1,1) :: input
  integer :: unit
  logical :: success = .true.


!-------------------------------------------------------------------------------
! set up layer
!-------------------------------------------------------------------------------
  recu_layer1 = recurrent_layer_type(input_size=5, hidden_size=10)

  !! check layer name
  if(.not. recu_layer1%name .eq. 'recu')then
     success = .false.
     write(0,*) 'recurrent layer has wrong name'
  end if

  if(any(recu_layer1%input_shape .ne. [5]))then
     success = .false.
     write(0,*) 'recurrent layer has wrong input_shape'
  end if

  if(any(recu_layer1%output_shape .ne. [10]))then
     success = .false.
     write(0,*) 'recurrent layer has wrong output shape'
  end if

  !! check layer type
  select type(recu_layer1)
  type is(recurrent_layer_type)
     !! check default layer activation function
     if(recu_layer1%activation%name .ne. 'tanh')then
        success = .false.
        write(0,*) 'recurrent layer has wrong activation: '//&
             recu_layer1%activation%name
     end if

     !! check hidden size
     if(recu_layer1%hidden_size .ne. 10)then
        success = .false.
        write(0,*) 'recurrent layer has wrong hidden_size'
     end if

     !! check input size
     if(recu_layer1%input_size .ne. 5)then
        success = .false.
        write(0,*) 'recurrent layer has wrong input_size'
     end if
  class default
     success = .false.
     write(0,*) 'recurrent layer has wrong type'
  end select

  recu_layer2 = recurrent_layer_type(hidden_size=20)
  call recu_layer2%init(recu_layer1%output_shape)

  if(any(recu_layer2%input_shape .ne. [10]))then
     success = .false.
     write(0,*) 'recurrent layer has wrong input_shape'
  end if

  if(any(recu_layer2%output_shape .ne. [20]))then
     success = .false.
     write(0,*) 'recurrent layer has wrong output_shape'
  end if


!-------------------------------------------------------------------------------
! test forward pass with sequential data
!-------------------------------------------------------------------------------
  write(*,*) "Testing forward pass with sequential data..."

  sequence_test: block
    type(recurrent_layer_type) :: rnn_layer
    type(array_type), dimension(1,1) :: input_seq, output_seq
    real(real32), allocatable, dimension(:,:) :: input_data
    integer :: time_step, batch_size, input_dim

    input_dim = 5
    batch_size = 2
    rnn_layer = recurrent_layer_type(&
         input_size=input_dim, &
         hidden_size=10, &
         activation='tanh', &
         kernel_initialiser='glorot_uniform' &
    )

    !! simulate 3 time steps
    do time_step = 1, 3
       !! create random input
       allocate(input_data(input_dim, batch_size))
       call random_number(input_data)
       call input_seq(1,1)%allocate(source=input_data)

       !! forward pass
       call rnn_layer%forward(input_seq)

       !! check output shape
       if(.not. allocated(rnn_layer%output))then
          success = .false.
          write(0,*) 'recurrent layer output not allocated'
          exit
       end if

       if(size(rnn_layer%output(1,1)%val, 1) .ne. 10)then
          success = .false.
          write(0,*) 'recurrent layer output has wrong first dimension'
       end if

       if(size(rnn_layer%output(1,1)%val, 2) .ne. batch_size)then
          success = .false.
          write(0,*) 'recurrent layer output has wrong batch dimension'
       end if

       deallocate(input_data)
       call input_seq(1,1)%deallocate()
    end do

    !! check that time_step counter was incremented
    if(rnn_layer%time_step .ne. 3)then
       success = .false.
       write(0,*) 'recurrent layer time_step not incremented correctly'
    end if

  end block sequence_test


!-------------------------------------------------------------------------------
! test layer with bias disabled
!-------------------------------------------------------------------------------
  write(*,*) "Testing layer with bias disabled..."

  no_bias_test: block
    type(recurrent_layer_type) :: rnn_no_bias

    rnn_no_bias = recurrent_layer_type(&
         input_size=3, &
         hidden_size=5, &
         use_bias=.false. &
    )

    if(rnn_no_bias%use_bias)then
       success = .false.
       write(0,*) 'recurrent layer should have use_bias=.false.'
    end if

    !! check number of parameters (should only have 2 weight matrices)
    if(size(rnn_no_bias%params) .ne. 2)then
       success = .false.
       write(0,*) 'recurrent layer with no bias has wrong number of params'
    end if

  end block no_bias_test


!-------------------------------------------------------------------------------
! test different activations
!-------------------------------------------------------------------------------
  write(*,*) "Testing different activation functions..."

  activation_test: block
    type(recurrent_layer_type) :: rnn_sigmoid, rnn_relu

    rnn_sigmoid = recurrent_layer_type(&
         input_size=4, &
         hidden_size=8, &
         activation='sigmoid' &
    )

    if(rnn_sigmoid%activation%name .ne. 'sigmoid')then
       success = .false.
       write(0,*) 'recurrent layer has wrong activation'
    end if

    rnn_relu = recurrent_layer_type(&
         input_size=4, &
         hidden_size=8, &
         activation='relu' &
    )

    if(rnn_relu%activation%name .ne. 'relu')then
       success = .false.
       write(0,*) 'recurrent layer has wrong activation'
    end if

  end block activation_test


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  io_test: block
    type(recurrent_layer_type) :: write_layer, check_layer

    write_layer = recurrent_layer_type(&
         input_size=3, &
         hidden_size=7, &
         use_bias=.true., &
         activation='tanh' &
    )

    ! Create a temporary file for testing
    open(newunit=unit, file='test_recurrent_layer.tmp', &
         status='replace', action='write')

    ! Write layer to file
    write(unit,'("RECU")')
    call write_layer%print_to_unit(unit)
    write(unit,'("END RECU")')
    close(unit)

    ! Read layer from file
    open(newunit=unit, file='test_recurrent_layer.tmp', &
         status='old', action='read')
    read(unit,*) ! Skip first line
    read_layer = read_recurrent_layer(unit)
    close(unit)

    ! Check that read layer has correct properties
    select type(read_layer)
    type is (recurrent_layer_type)
       if (.not. read_layer%name .eq. 'recu') then
          success = .false.
          write(0,*) 'read recurrent layer has wrong name'
       end if

       if(read_layer%hidden_size .ne. 7)then
          success = .false.
          write(0,*) 'read recurrent layer has wrong hidden_size'
       end if

       if(read_layer%input_size .ne. 3)then
          success = .false.
          write(0,*) 'read recurrent layer has wrong input_size'
       end if

       if(.not. read_layer%use_bias)then
          success = .false.
          write(0,*) 'read recurrent layer has wrong use_bias'
       end if

       if(read_layer%activation%name .ne. 'tanh')then
          success = .false.
          write(0,*) 'read recurrent layer has wrong activation'
       end if

    class default
       success = .false.
       write(0,*) 'read layer is not recurrent_layer_type'
    end select

    ! Clean up temporary file
    open(newunit=unit, file='test_recurrent_layer.tmp', status='old')
    close(unit, status='delete')

  end block io_test


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_recurrent_layer passed all tests'
  else
     write(0,*) 'test_recurrent_layer failed one or more tests'
     stop 1
  end if

end program test_recurrent_layer
