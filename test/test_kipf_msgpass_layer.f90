program test_kipf_msgpass_layer
  use athena, only: &
       kipf_msgpass_layer_type, &
       base_layer_type, &
       learnable_layer_type
  use athena__kipf_msgpass_layer, only: read_kipf_msgpass_layer
  use graphstruc, only: graph_type
  use diffstruc, only: array_type
  implicit none

  class(base_layer_type), allocatable :: msgpass_layer
  class(base_layer_type), allocatable :: read_layer
  type(graph_type), dimension(1) :: graph
  integer, parameter :: num_features = 10, num_time_steps = 1, num_outputs = 5
  integer, parameter :: num_vertices = 6, num_edges = 8
  integer :: unit
  integer, parameter :: batch_size = 1
  real, allocatable, dimension(:,:) :: vertex_features, output
  real, parameter :: tol = 1.E-7
  logical :: success = .true.
  type(array_type), allocatable, dimension(:,:) :: input

  real, allocatable, dimension(:) :: params
  real, allocatable, dimension(:,:) :: outputs


!-------------------------------------------------------------------------------
! set up message passing layer
!-------------------------------------------------------------------------------
  msgpass_layer = kipf_msgpass_layer_type( &
       num_vertex_features = [num_features, num_outputs], &
       num_time_steps = num_time_steps &
  )

  !! check layer name
  if(.not. msgpass_layer%name .eq. 'kipf')then
     success = .false.
     write(0,*) 'kipf layer has wrong name'
  end if


!-------------------------------------------------------------------------------
! check layer type
!-------------------------------------------------------------------------------
  select type(msgpass_layer)
  type is(kipf_msgpass_layer_type)
     !! check number of vertex features
     if(any(msgpass_layer%num_vertex_features .ne. [ num_features, num_outputs ]))then
        success = .false.
        write(0,*) 'kipf layer has wrong input features'
     end if

     !! check number of outputs
     if(msgpass_layer%num_vertex_features(num_time_steps) .ne. num_outputs)then
        success = .false.
        write(0,*) 'kipf layer has wrong output features'
     end if

     !! check number of time steps
     if(msgpass_layer%num_time_steps .ne. num_time_steps)then
        success = .false.
        write(0,*) 'kipf layer has wrong number of time steps'
     end if
  end select


!-------------------------------------------------------------------------------
! set up a simple graph for testing
!-------------------------------------------------------------------------------
  ! Create a simple graph with 6 vertices and some edges
  call graph(1)%set_num_vertices(num_vertices, num_features)
  call graph(1)%set_num_edges(num_edges)

  ! Set up some dummy vertex features
  graph(1)%is_sparse = .true.
  allocate(graph(1)%vertex_features(num_features, num_vertices))
  graph(1)%vertex_features = 1.0

  ! Set up edge connectivity (creating a simple connected graph)
  block
    integer, allocatable :: index_list(:, :)
    allocate(index_list(2, num_edges))
    index_list(:, 1) = [1, 2]
    index_list(:, 2) = [1, 3]
    index_list(:, 3) = [2, 3]
    index_list(:, 4) = [2, 4]
    index_list(:, 5) = [3, 5]
    index_list(:, 6) = [4, 5]
    index_list(:, 7) = [4, 6]
    index_list(:, 8) = [5, 6]

    ! Generate adjacency matrix
    call graph(1)%generate_adjacency(index_list)
    deallocate(index_list)

    ! Set edge weights
    allocate(graph(1)%edge_weights(num_edges))
    graph(1)%edge_weights = 1.0

    allocate(graph(1)%edge_features(1, num_edges), source = 0.0)
  end block

  ! Set the graph for the layer
  call msgpass_layer%set_graph(graph)

  ! ! Initialise the layer
  ! call msgpass_layer%init([num_features, num_vertices])

!-------------------------------------------------------------------------------
! check handling of layer parameters, gradients, and outputs
!-------------------------------------------------------------------------------
  select type(msgpass_layer)
  class is(learnable_layer_type)
     !! check layer parameter handling
     params = msgpass_layer%get_params()
     if(size(params) .eq. 0)then
        success = .false.
        write(0,*) 'kipf layer has wrong number of parameters'
     end if
     params = 1.E0
     call msgpass_layer%set_params(params)
     params = msgpass_layer%get_params()
     if(any(abs(params - 1.E0).gt.tol))then
        success = .false.
        write(0,*) 'kipf layer has wrong parameters'
     end if

     !! check layer gradient handling
     params = msgpass_layer%get_gradients()
     if(size(params) .eq. 0)then
        success = .false.
        write(0,*) 'kipf layer has wrong number of gradients'
     end if
     params = 1.E0
     call msgpass_layer%set_gradients(params)
     params = msgpass_layer%get_gradients()
     if(any(abs(params - 1.E0).gt.tol))then
        success = .false.
        write(0,*) 'kipf layer has wrong gradients'
     end if
     call msgpass_layer%set_gradients(10.E0)
     params = msgpass_layer%get_gradients()
     if(any(abs(params - 10.E0).gt.tol))then
        success = .false.
        write(0,*) 'kipf layer has wrong gradients'
     end if

     allocate(input(2, batch_size))
     call input(1,1)%allocate(source = graph(1)%vertex_features)
     call input(2,1)%allocate(source = graph(1)%edge_features)
     call msgpass_layer%forward(input)
     !! check layer output handling
     if(any(shape(msgpass_layer%output) .ne. [1, batch_size]))then
        success = .false.
        write(0,*) 'kipf layer has wrong derived type output shape'
     end if
     if(size(msgpass_layer%output(1,1)%val,1) .ne. msgpass_layer%output_shape(1))then
        success = .false.
        write(0,*) 'kipf layer has wrong number of outputs'
     end if
     if(any(shape(msgpass_layer%output(1,1)%val) .ne. &
          [msgpass_layer%output_shape(1), num_vertices]))then
        success = .false.
        write(0,*) 'kipf layer has wrong number of outputs'
     end if
  end select


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_kipf_msgpass_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("KIPF")')
  call msgpass_layer%print_to_unit(unit)
  write(unit,'("END KIPF")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_kipf_msgpass_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_kipf_msgpass_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (kipf_msgpass_layer_type)
     if (.not. read_layer%name .eq. 'kipf') then
        success = .false.
        write(0,*) 'read kipf layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not kipf_msgpass_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_kipf_msgpass_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_kipf_msgpass_layer passed all tests'
  else
     write(0,*) 'test_kipf_msgpass_layer failed one or more tests'
     stop 1
  end if

end program test_kipf_msgpass_layer
