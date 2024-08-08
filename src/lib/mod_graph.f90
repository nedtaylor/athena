module graph_constructs
  use constants, only: real32
  implicit none

  private

  public :: vertex_type, edge_type, graph_type


!!!-----------------------------------------------------------------------------
!!! graph vertex type
!!!-----------------------------------------------------------------------------
  type :: vertex_type
     integer :: degree = 1
     real(real32), dimension(:), allocatable :: feature
  end type vertex_type

  type :: edge_type
     !! for directed graphs, index(1) is the source vertex, index(2) is the target vertex
     integer, dimension(2) :: index
     real(real32) :: weight = 1._real32
     real(real32), dimension(:), allocatable :: feature
  end type edge_type

  type :: graph_type
     logical :: directed = .false.
     integer :: num_vertices, num_edges
     integer :: num_vertex_features, num_edge_features
     character(len=128) :: name
     !! adjacency matrix
     !! NUM if edge exists, 0 otherwise (NUM = edge index)
     !! if -ve, then edge is directed from j to i
     !! if +ve, then edge is bidirectional
     integer, dimension(:,:), allocatable :: adjacency
     type(vertex_type), dimension(:), allocatable :: vertex
     type(edge_type), dimension(:), allocatable :: edge
   contains
     procedure, pass(this) :: calculate_degree
     procedure, pass(this) :: generate_adjacency
  end type graph_type

  interface edge_type
     module function edge_type_init(index, weight, feature) result(output)
       integer, dimension(2), intent(in) :: index
       real(real32), intent(in), optional :: weight
       real(real32), dimension(:), intent(in), optional :: feature
       type(edge_type) :: output
     end function edge_type_init
  end interface edge_type


contains
  
  subroutine calculate_degree(this)
    implicit none
    class(graph_type), intent(inout) :: this
    integer :: i, j
    !!! NEED TO ACCOUNT FOR DIRECTION
    this%vertex(:)%degree = 1
    do i = 1, this%num_vertices
      do j = i + 1, this%num_vertices, 1
        if(this%adjacency(i,j) .gt. 0) then
          this%vertex(i)%degree = this%vertex(i)%degree + 1
          this%vertex(j)%degree = this%vertex(j)%degree + 1
        elseif(this%adjacency(i,j) .lt. 0) then
          this%vertex(i)%degree = this%vertex(i)%degree + 1
        end if
      end do
    end do
  end subroutine calculate_degree

  module function edge_type_init(index, weight, feature) result(output)
    implicit none
    integer, dimension(2), intent(in) :: index
    real(real32), intent(in), optional :: weight
    real(real32), dimension(:), intent(in), optional :: feature
    type(edge_type) :: output
    output%index = index
    if(present(weight)) output%weight = weight
    if(present(feature)) output%feature = feature
  end function edge_type_init

  subroutine generate_adjacency(this)
    implicit none
    class(graph_type), intent(inout) :: this
    integer :: i, j, k
    allocate(this%adjacency(this%num_vertices, this%num_vertices))
    this%adjacency = 0
    do k = 1, this%num_edges
      i = this%edge(k)%index(1)
      j = this%edge(k)%index(2)
      if(this%directed) then
        this%adjacency(i,j) = k
        this%adjacency(j,i) = -k
      else
        this%adjacency(i,j) = k
        this%adjacency(j,i) = k
      end if
    end do
  end subroutine generate_adjacency

end module graph_constructs