program test_onnx_duvenaud_gnn
  use coreutils, only: real32
  use athena

  implicit none

  logical :: success = .true.
  real(real32), parameter :: tol = 1.e-5_real32
  type(network_type) :: network, imported_network
  type(graph_type), allocatable :: graphs(:,:)
  real(real32), allocatable :: output_ref(:,:), output_new(:,:)


!-------------------------------------------------------------------------------
! Build a deterministic Duvenaud network and compute a reference output.
!-------------------------------------------------------------------------------
  call build_graph_batch(graphs)
  call random_setup(654, restart=.false.)
  call build_duvenaud_network(network)
  output_ref = evaluate_graph_network(network, graphs)


!-------------------------------------------------------------------------------
! Validate ATHENA-metadata and expanded-import paths.
!-------------------------------------------------------------------------------
  call write_onnx('test_duvenaud_model.json', network)
  imported_network = read_onnx('test_duvenaud_model.json', verbose=0)
  call compile_duvenaud_network(imported_network)
  output_new = evaluate_graph_network(imported_network, graphs)
  call require_close(output_new, output_ref, &
       'metadata Duvenaud ONNX round-trip changed output')

  call strip_metadata_from_json( &
       'test_duvenaud_model.json', 'test_duvenaud_model_no_meta.json' &
  )
  imported_network = read_onnx('test_duvenaud_model_no_meta.json', verbose=0)
  call compile_duvenaud_network(imported_network)
  output_new = evaluate_graph_network(imported_network, graphs)
  call require_close(output_new, output_ref, &
       'expanded Duvenaud ONNX round-trip changed output')


!-------------------------------------------------------------------------------
! Check for failures.
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_onnx_duvenaud_gnn passed all tests'
  else
     write(0,*) 'test_onnx_duvenaud_gnn failed one or more tests'
     stop 1
  end if


contains

!-------------------------------------------------------------------------------
  subroutine build_duvenaud_network(net)
    type(network_type), intent(out) :: net

    call net%add(duvenaud_msgpass_layer_type( &
         num_vertex_features = [2], &
         num_edge_features = [1], &
         num_time_steps = 2, &
         max_vertex_degree = 4, &
         num_outputs = 3 &
    ))

    call compile_duvenaud_network(net)
  end subroutine build_duvenaud_network

!-------------------------------------------------------------------------------
  subroutine compile_duvenaud_network(net)
    type(network_type), intent(inout) :: net

    call net%compile( &
         optimiser = adam_optimiser_type(learning_rate=1.e-2_real32), &
         loss_method = 'mse', &
         metrics = ['loss'], &
         accuracy_method = 'mse', &
         batch_size = 1, &
         verbose = 0 &
    )
    call net%set_batch_size(1)
    call net%set_inference_mode()
  end subroutine compile_duvenaud_network

!-------------------------------------------------------------------------------
  subroutine build_graph_batch(graph_batch)
    type(graph_type), allocatable, intent(out) :: graph_batch(:,:)
    integer, allocatable :: edge_index(:,:)

    allocate(graph_batch(1,1))

    graph_batch(1,1)%is_sparse = .true.
    call graph_batch(1,1)%set_num_vertices(4, 2)
    call graph_batch(1,1)%set_num_edges(5, 1)

    graph_batch(1,1)%vertex_features = reshape( &
         [ &
              1._real32, 2._real32, 3._real32, 4._real32, &
              0.5_real32, 1.5_real32, 2.5_real32, 3.5_real32 &
         ], &
         shape(graph_batch(1,1)%vertex_features) &
    )

    allocate(edge_index(2, 5))
    edge_index(:,1) = [1, 2]
    edge_index(:,2) = [2, 3]
    edge_index(:,3) = [3, 4]
    edge_index(:,4) = [4, 1]
    edge_index(:,5) = [1, 3]
    call graph_batch(1,1)%generate_adjacency(edge_index)
    deallocate(edge_index)

    graph_batch(1,1)%edge_weights = 1._real32
    graph_batch(1,1)%edge_features = 0._real32
  end subroutine build_graph_batch

!-------------------------------------------------------------------------------
  function evaluate_graph_network(net, graph_batch) result(output)
    type(network_type), intent(inout) :: net
    type(graph_type), intent(in) :: graph_batch(:,:)
    real(real32), allocatable :: output(:,:)
    integer :: leaf_id

    call net%forward(graph_batch)
    leaf_id = net%auto_graph%vertex(net%leaf_vertices(1))%id
    allocate(output, source=net%model(leaf_id)%layer%output(1,1)%val)
  end function evaluate_graph_network

!-------------------------------------------------------------------------------
  subroutine require_close(actual, expected, message)
    real(real32), intent(in) :: actual(:,:), expected(:,:)
    character(*), intent(in) :: message

    call require( &
         all(shape(actual).eq.shape(expected)), &
         trim(message) // ' (shape mismatch)' &
    )
    if(all(shape(actual).eq.shape(expected)))then
       call require(maxval(abs(actual - expected)).lt.tol, message)
    end if
  end subroutine require_close

!-------------------------------------------------------------------------------
  subroutine require(condition, message)
    logical, intent(in) :: condition
    character(*), intent(in) :: message

    if(.not.condition)then
       success = .false.
       write(0,*) trim(message)
    end if
  end subroutine require

!-------------------------------------------------------------------------------
  subroutine strip_metadata_from_json(infile, outfile)
    character(*), intent(in) :: infile, outfile

    integer :: uin, uout, stat
    character(131072) :: line
    logical :: skipping

    open(newunit=uin, file=infile, status='old', action='read', &
         iostat=stat)
    call require(stat.eq.0, 'failed to open Duvenaud ONNX input file')
    if(stat.ne.0) return

    open(newunit=uout, file=outfile, status='replace', action='write', &
         iostat=stat)
    call require(stat.eq.0, 'failed to create Duvenaud stripped ONNX file')
    if(stat.ne.0)then
       close(uin)
       return
    end if

    skipping = .false.
    do
       read(uin, '(A)', iostat=stat) line
       if(stat.ne.0) exit

       if(index(line, '"metadataProps"').gt.0)then
          write(uout, '(A)') '  "metadataProps": []'
          skipping = .true.
          cycle
       end if

       if(skipping)then
          if(index(line, ']') .gt. 0) skipping = .false.
          cycle
       end if

       write(uout, '(A)') trim(line)
    end do

    close(uin)
    close(uout)
  end subroutine strip_metadata_from_json

end program test_onnx_duvenaud_gnn
