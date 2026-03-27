program test_gno_layer
  !! Tests for the Graph Neural Operator (GNO) layer
  use athena, only: &
       graph_nop_layer_type, &
       base_layer_type, &
       learnable_layer_type
  use athena__graph_nop_layer, only: read_graph_nop_layer
  use graphstruc, only: graph_type
  use diffstruc, only: array_type
  use coreutils, only: real32
  implicit none

  class(base_layer_type), allocatable :: layer1, layer2, layer3
  class(base_layer_type), allocatable :: read_layer
  type(graph_type), dimension(1) :: graph
  integer :: unit
  logical :: success = .true.
  real(real32), parameter :: tol = 1.0e-6_real32

  integer, parameter :: num_vertices = 6
  integer, parameter :: num_edges = 8
  integer, parameter :: F_in = 4         ! input features
  integer, parameter :: F_out = 3        ! output features
  integer, parameter :: d = 2            ! edge-feature dimension
  integer, parameter :: H = 5            ! kernel hidden width


!-------------------------------------------------------------------------------
! Test 1: basic layer construction with explicit num_inputs
!-------------------------------------------------------------------------------
  write(*,*) "Test 1: layer construction..."
  layer1 = graph_nop_layer_type( &
       num_inputs=F_in, num_outputs=F_out, coord_dim=d, kernel_hidden=H)

  if(layer1%name .ne. 'graph_nop')then
     success = .false.
     write(0,*) 'graph_nop layer has wrong name: '//trim(layer1%name)
  end if

  if(any(layer1%input_shape .ne. [F_in, 0]))then
     success = .false.
     write(0,*) 'graph_nop layer has wrong input_shape'
  end if

  if(any(layer1%output_shape .ne. [F_out, 0]))then
     success = .false.
     write(0,*) 'graph_nop layer has wrong output_shape'
  end if

  select type(layer1)
  type is(graph_nop_layer_type)
     if(layer1%num_vertex_features(0) .ne. F_in)then
        success = .false.
        write(0,*) 'graph_nop layer has wrong num_inputs'
     end if
     if(layer1%num_outputs .ne. F_out)then
        success = .false.
        write(0,*) 'graph_nop layer has wrong num_outputs'
     end if
     if(layer1%coord_dim .ne. d)then
        success = .false.
        write(0,*) 'graph_nop layer has wrong coord_dim'
     end if
     if(layer1%kernel_hidden .ne. H)then
        success = .false.
        write(0,*) 'graph_nop layer has wrong kernel_hidden'
     end if
     if(layer1%activation%name .ne. 'none')then
        success = .false.
        write(0,*) 'graph_nop layer has wrong default activation: '// &
             layer1%activation%name
     end if
     ! num_params = H*d + H + F_out*F_in*H + F_out*F_in + F_out*F_in + F_out
     !            = 5*2 + 5 + 3*4*5 + 3*4 + 3*4 + 3
     !            = 10 + 5 + 60 + 12 + 12 + 3 = 102
     if(layer1%num_params .ne. 102)then
        success = .false.
        write(0,'("graph_nop layer has wrong num_params: ",I0)') &
             layer1%num_params
     end if
  class default
     success = .false.
     write(0,*) 'layer1 has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 2: deferred initialisation (num_inputs provided later)
!-------------------------------------------------------------------------------
  write(*,*) "Test 2: deferred initialisation..."
  layer2 = graph_nop_layer_type(num_outputs=F_out, coord_dim=d, kernel_hidden=H)
  call layer2%init([F_in, 0])

  if(any(layer2%input_shape .ne. [F_in, 0]))then
     success = .false.
     write(0,*) 'deferred graph_nop layer has wrong input_shape'
  end if
  if(any(layer2%output_shape .ne. [F_out, 0]))then
     success = .false.
     write(0,*) 'deferred graph_nop layer has wrong output_shape'
  end if


!-------------------------------------------------------------------------------
! Test 3: layer without bias
!-------------------------------------------------------------------------------
  write(*,*) "Test 3: no-bias layer..."
  layer3 = graph_nop_layer_type( &
       num_inputs=F_in, num_outputs=F_out, coord_dim=d, &
       kernel_hidden=H, use_bias=.false.)

  select type(layer3)
  type is(graph_nop_layer_type)
     if(layer3%use_bias)then
        success = .false.
        write(0,*) 'no-bias graph_nop layer has use_bias = T'
     end if
     ! num_params = H*d + H + F_out*F_in*H + F_out*F_in + F_out*F_in
     !            = 10 + 5 + 60 + 12 + 12 = 99
     if(layer3%num_params .ne. 99)then
        success = .false.
        write(0,'("no-bias layer has wrong num_params: ",I0)') layer3%num_params
     end if
  class default
     success = .false.
     write(0,*) 'layer3 has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test 4: activation function
!-------------------------------------------------------------------------------
  write(*,*) "Test 4: activation function..."
  layer1 = graph_nop_layer_type( &
       num_inputs=F_in, num_outputs=F_out, coord_dim=d, &
       kernel_hidden=H, activation="relu")

  select type(layer1)
  type is(graph_nop_layer_type)
     if(layer1%activation%name .ne. 'relu')then
        success = .false.
        write(0,*) 'graph_nop layer has wrong activation: '// &
             layer1%activation%name
     end if
  class default
     success = .false.
     write(0,*) 'layer1 has wrong type after re-assignment'
  end select


!-------------------------------------------------------------------------------
! Test 5: forward pass produces correct output shape
!-------------------------------------------------------------------------------
  write(*,*) "Test 5: forward pass..."
  call setup_graph(graph)
  layer1 = graph_nop_layer_type( &
       num_inputs=F_in, num_outputs=F_out, coord_dim=d, &
       kernel_hidden=H, use_bias=.true., activation="none")
  call layer1%set_graph(graph)

  block
    type(array_type), allocatable, dimension(:,:) :: input
    real(real32) :: vertex_features(F_in, num_vertices)
    real(real32) :: edge_features(d, num_edges)
    integer :: i

    ! Set up input features and edge geometry
    do i = 1, num_vertices
       vertex_features(:, i) = real(i, real32) * 0.1_real32
    end do
    do i = 1, num_edges
       edge_features(1, i) = 0.05_real32 * real(i, real32)
       edge_features(2, i) = -0.03_real32 * real(i, real32)
    end do

    allocate(input(2, 1))
    call input(1,1)%allocate(source=vertex_features)
    call input(2,1)%allocate(source=edge_features)
    call layer1%forward(input)

    ! Output should have shape [F_out, num_vertices]
    if(.not.allocated(layer1%output))then
       success = .false.
       write(0,*) 'forward pass did not produce output'
    else
       if(size(layer1%output,1) .ne. 2 .or. &
            size(layer1%output,2) .ne. 1)then
          success = .false.
          write(0,*) 'forward pass produced wrong output container shape'
       end if
       if(size(layer1%output(1,1)%val, 1) .ne. F_out)then
          success = .false.
          write(0,'("forward pass wrong F_out: ",I0," expected ",I0)') &
               size(layer1%output(1,1)%val, 1), F_out
       end if
       if(size(layer1%output(1,1)%val, 2) .ne. num_vertices)then
          success = .false.
          write(0,'("forward pass wrong num_vertices: ",I0," expected ",I0)') &
               size(layer1%output(1,1)%val, 2), num_vertices
       end if
       if(any(shape(layer1%output(2,1)%val) .ne. [d, num_edges]))then
          success = .false.
          write(0,*) 'forward pass did not preserve edge features'
       end if
    end if
    call input(1,1)%deallocate()
    call input(2,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Test 6: deterministic output for fixed weights
!-------------------------------------------------------------------------------
  write(*,*) "Test 6: deterministic output..."

  block
    type(array_type), allocatable, dimension(:,:) :: input
    real(real32) :: vertex_features(F_in, num_vertices)
    real(real32) :: edge_features(d, num_edges)
    real(real32), allocatable :: params(:)
    real(real32) :: output1(F_out, num_vertices)
    real(real32) :: output2(F_out, num_vertices)
    integer :: i

    layer1 = graph_nop_layer_type( &
         num_inputs=F_in, num_outputs=F_out, coord_dim=d, &
         kernel_hidden=H, use_bias=.true., activation="none")
    call layer1%set_graph(graph)

    ! Set fixed weights
    select type(layer1)
    class is(learnable_layer_type)
       params = layer1%get_params()
       params = 0.01_real32
       call layer1%set_params(params)
    end select

    do i = 1, num_vertices
       vertex_features(:, i) = real(i, real32) * 0.1_real32
    end do
    do i = 1, num_edges
       edge_features(1, i) = 0.05_real32 * real(i, real32)
       edge_features(2, i) = -0.03_real32 * real(i, real32)
    end do

    allocate(input(2, 1))
    call input(1,1)%allocate(source=vertex_features)
    call input(2,1)%allocate(source=edge_features)

    ! First forward pass
    call layer1%forward(input)
    output1 = layer1%output(1,1)%val

    ! Second forward pass with same inputs
    call layer1%forward(input)
    output2 = layer1%output(1,1)%val

    if(any(abs(output1 - output2) .gt. tol))then
       success = .false.
       write(0,*) 'forward pass is not deterministic'
    end if

    call input(1,1)%deallocate()
    call input(2,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Test 7: parameter get/set round-trip
!-------------------------------------------------------------------------------
  write(*,*) "Test 7: parameter get/set..."

  layer1 = graph_nop_layer_type( &
       num_inputs=F_in, num_outputs=F_out, coord_dim=d, &
       kernel_hidden=H, use_bias=.true.)

  select type(layer1)
  class is(learnable_layer_type)
     block
       real(real32), allocatable :: params(:), params2(:)

       params = layer1%get_params()
       if(size(params) .ne. layer1%num_params)then
          success = .false.
          write(0,'("get_params returned wrong size: ",I0," expected ",I0)') &
               size(params), layer1%num_params
       end if

       params = 0.5_real32
       call layer1%set_params(params)
       params2 = layer1%get_params()
       if(any(abs(params2 - 0.5_real32) .gt. tol))then
          success = .false.
          write(0,*) 'set_params/get_params round-trip failed'
       end if
     end block
  end select


!-------------------------------------------------------------------------------
! Test 8: gradient get/set
!-------------------------------------------------------------------------------
  write(*,*) "Test 8: gradient get/set..."

  select type(layer1)
  class is(learnable_layer_type)
     block
       real(real32), allocatable :: grads(:)

       grads = layer1%get_gradients()
       if(size(grads) .ne. layer1%num_params)then
          success = .false.
          write(0,'("get_gradients returned wrong size: ",I0)') size(grads)
       end if

       call layer1%set_gradients(2.0_real32)
       grads = layer1%get_gradients()
       if(any(abs(grads - 2.0_real32) .gt. tol))then
          success = .false.
          write(0,*) 'set_gradients/get_gradients round-trip failed'
       end if
     end block
  end select


!-------------------------------------------------------------------------------
! Test 9: layer addition and reduction
!-------------------------------------------------------------------------------
  write(*,*) "Test 9: layer addition and reduction..."
  layer2 = graph_nop_layer_type( &
       num_inputs=F_in, num_outputs=F_out, coord_dim=d, &
       kernel_hidden=H, activation="sigmoid")
  layer3 = graph_nop_layer_type( &
       num_inputs=F_in, num_outputs=F_out, coord_dim=d, &
       kernel_hidden=H, activation="sigmoid")

  select type(l2 => layer2)
  class is(learnable_layer_type)
     select type(l3 => layer3)
     class is(learnable_layer_type)
        block
          class(learnable_layer_type), allocatable :: result_layer

          result_layer = l2 + l3
          select type(result_layer)
          type is(graph_nop_layer_type)
             if(result_layer%num_vertex_features(0) .ne. F_in .or. &
                  result_layer%num_outputs .ne. F_out)then
                success = .false.
                write(0,*) 'layer addition produced wrong shape'
             end if
             if(result_layer%activation%name .ne. 'sigmoid')then
                success = .false.
                write(0,*) 'layer addition produced wrong activation'
             end if
          class default
             success = .false.
             write(0,*) 'layer addition returned wrong type'
          end select

          call l2%reduce(l3)
          select type(l2)
          type is(graph_nop_layer_type)
             if(l2%num_vertex_features(0) .ne. F_in)then
                success = .false.
                write(0,*) 'layer reduce produced wrong shape'
             end if
          class default
             success = .false.
             write(0,*) 'layer reduce returned wrong type'
          end select
        end block
     end select
  end select


!-------------------------------------------------------------------------------
! Test 10: file I/O
!-------------------------------------------------------------------------------
  write(*,*) "Test 10: file I/O..."

  layer1 = graph_nop_layer_type( &
       num_inputs=F_in, num_outputs=F_out, coord_dim=d, &
       kernel_hidden=H, use_bias=.true., activation="relu")

  open(newunit=unit, file='test_graph_nop_layer.tmp', &
       status='replace', action='write')
  write(unit,'("GRAPH_NOP")')
  call layer1%print_to_unit(unit)
  write(unit,'("END GRAPH_NOP")')
  close(unit)

  open(newunit=unit, file='test_graph_nop_layer.tmp', &
       status='old', action='read')
  read(unit,*)   ! skip GRAPH_NOP header line
  read_layer = read_graph_nop_layer(unit)
  close(unit)

  select type(read_layer)
  type is(graph_nop_layer_type)
     if(read_layer%name .ne. 'graph_nop')then
        success = .false.
        write(0,*) 'read layer has wrong name'
     end if
     if(read_layer%num_vertex_features(0) .ne. F_in)then
        success = .false.
        write(0,*) 'read layer has wrong num_inputs'
     end if
     if(read_layer%num_outputs .ne. F_out)then
        success = .false.
        write(0,*) 'read layer has wrong num_outputs'
     end if
     if(read_layer%coord_dim .ne. d)then
        success = .false.
        write(0,*) 'read layer has wrong coord_dim'
     end if
     if(read_layer%kernel_hidden .ne. H)then
        success = .false.
        write(0,*) 'read layer has wrong kernel_hidden'
     end if
     if(read_layer%activation%name .ne. 'relu')then
        success = .false.
        write(0,*) 'read layer has wrong activation: '// &
             read_layer%activation%name
     end if
     select type(layer1)
     type is(graph_nop_layer_type)
        block
          integer :: p
          do p = 1, size(layer1%params)
             if(any(abs(read_layer%params(p)%val(:,1) - &
                  layer1%params(p)%val(:,1)) .gt. tol))then
                success = .false.
                write(0,'("read layer params(",I0,") differ from original")') p
             end if
          end do
        end block
     end select
  class default
     success = .false.
     write(0,*) 'read layer is not graph_nop_layer_type'
  end select

  open(newunit=unit, file='test_graph_nop_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! Test 11: numerical stability - large inputs
!-------------------------------------------------------------------------------
  write(*,*) "Test 11: numerical stability..."

  block
    type(array_type), allocatable, dimension(:,:) :: input
    real(real32) :: vertex_features(F_in, num_vertices)
    real(real32) :: edge_features(d, num_edges)
    integer :: i

    layer1 = graph_nop_layer_type( &
         num_inputs=F_in, num_outputs=F_out, coord_dim=d, &
         kernel_hidden=H, use_bias=.true., activation="sigmoid")
    call layer1%set_graph(graph)

    ! Use moderately large values
    do i = 1, num_vertices
       vertex_features(:, i) = real(i, real32) * 10.0_real32
    end do
    do i = 1, num_edges
       edge_features(1, i) = real(i, real32) * 5.0_real32
       edge_features(2, i) = real(i, real32) * 3.0_real32
    end do

    allocate(input(2, 1))
    call input(1,1)%allocate(source=vertex_features)
    call input(2,1)%allocate(source=edge_features)
    call layer1%forward(input)

    ! Check for NaN or Inf in output
    if(any(layer1%output(1,1)%val .ne. layer1%output(1,1)%val))then
       success = .false.
       write(0,*) 'forward pass produced NaN values'
    end if

    ! Sigmoid should produce values in (0, 1)
    if(any(layer1%output(1,1)%val .lt. 0.0_real32) .or. &
         any(layer1%output(1,1)%val .gt. 1.0_real32))then
       success = .false.
       write(0,*) 'sigmoid activation produced values outside (0,1)'
    end if

    call input(1,1)%deallocate()
    call input(2,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Test 12: forward with relu activation
!-------------------------------------------------------------------------------
  write(*,*) "Test 12: relu activation..."

  block
    type(array_type), allocatable, dimension(:,:) :: input
    real(real32) :: vertex_features(F_in, num_vertices)
    real(real32) :: edge_features(d, num_edges)
    real(real32), allocatable :: params(:)
    integer :: i

    layer1 = graph_nop_layer_type( &
         num_inputs=F_in, num_outputs=F_out, coord_dim=d, &
         kernel_hidden=H, use_bias=.true., activation="relu")
    call layer1%set_graph(graph)

    ! Set all weights to small negative values to test relu
    select type(layer1)
    class is(learnable_layer_type)
       params = layer1%get_params()
       params = -0.01_real32
       call layer1%set_params(params)
    end select

    do i = 1, num_vertices
       vertex_features(:, i) = 1.0_real32
    end do
    do i = 1, num_edges
       edge_features(1, i) = real(i, real32)
       edge_features(2, i) = 0.0_real32
    end do

    allocate(input(2, 1))
    call input(1,1)%allocate(source=vertex_features)
    call input(2,1)%allocate(source=edge_features)
    call layer1%forward(input)

    ! With relu, all outputs should be >= 0
    if(any(layer1%output(1,1)%val .lt. -tol))then
       success = .false.
       write(0,*) 'relu activation produced negative values'
    end if

    call input(1,1)%deallocate()
    call input(2,1)%deallocate()
  end block


!-------------------------------------------------------------------------------
! Result
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_gno_layer passed all tests'
  else
     write(0,*) 'test_gno_layer failed one or more tests'
     stop 1
  end if


contains

  subroutine setup_graph(graph)
    !! Set up a simple test graph
    type(graph_type), dimension(1), intent(inout) :: graph
    integer, allocatable :: index_list(:, :)

    call graph(1)%set_num_vertices(num_vertices, F_in)
    call graph(1)%set_num_edges(num_edges)

    graph(1)%is_sparse = .true.
    allocate(graph(1)%vertex_features(F_in, num_vertices))
    graph(1)%vertex_features = 1.0

    allocate(index_list(2, num_edges))
    index_list(:, 1) = [1, 2]
    index_list(:, 2) = [1, 3]
    index_list(:, 3) = [2, 3]
    index_list(:, 4) = [2, 4]
    index_list(:, 5) = [3, 5]
    index_list(:, 6) = [4, 5]
    index_list(:, 7) = [4, 6]
    index_list(:, 8) = [5, 6]

    call graph(1)%generate_adjacency(index_list)
    deallocate(index_list)

    allocate(graph(1)%edge_weights(num_edges))
    graph(1)%edge_weights = 1.0
    allocate(graph(1)%edge_features(1, num_edges), source = 0.0)

  end subroutine setup_graph

end program test_gno_layer
