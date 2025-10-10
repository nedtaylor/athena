module read_euler
  use constants_mnist, only: real32
  use misc_linalg, only: modu
  use athena, only: graph_type, edge_type
  implicit none

  private

  public :: read_graph

contains

!###############################################################################
  subroutine read_graph(vertex_file, edge_file, graph)
    implicit none
    character(len=*), intent(in) :: vertex_file, edge_file
    type(graph_type), intent(out) :: graph

    integer :: unit
    integer :: i, j
    integer :: num_vertices, num_vertex_features, num_edges
    integer, dimension(:, :), allocatable :: index_list


    graph%is_sparse = .true.

    open(newunit=unit, file=vertex_file, status='old', action='read')
    read(unit, *) num_vertices, num_vertex_features
    write(*,*) "Number of vertices: ", num_vertices
    write(*,*) "Number of vertex features: ", num_vertex_features

    call graph%set_num_vertices(num_vertices, num_vertex_features)

    do i = 1, num_vertices
       read(unit, *) graph%vertex_features(:,i)
    end do

    close(unit)

    open(newunit=unit, file=edge_file, status='old', action='read')
    rewind(unit)
    read(unit, *) num_edges

    call graph%set_num_edges(num_edges)

    allocate(index_list(2, num_edges))
    write(*,*) "Number of edges: ", num_edges
    do i = 1, num_edges
       read(unit, *) index_list(:, i)
    end do
    close(unit)
    graph%edge_weights = 1.0_real32

    call graph%generate_adjacency(index_list)
    deallocate(index_list)

  end subroutine read_graph
!###############################################################################

end module read_euler
