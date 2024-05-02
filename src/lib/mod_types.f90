!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains custom derived types for the ATHENA library
!!! module contains the following derived types:
!!! - activation_type  - abstract type for activation functions
!!! - initialiser_type - abstract type for initialising weights and biases
!!!##################
!!! the activation_type contains the following deferred procedures:
!!! - activate_<N>d      - activation function for rank <N> input
!!! - differentiate_<N>d - derivative of activation function for rank <N> input
!!!##################
!!! the initialiser_type contains the following deferred procedures:
!!! - initialise - initialises weights and biases
!!!#############################################################################
module custom_types
  use constants, only: real12
  implicit none


  private

  public :: activation_type
  public :: initialiser_type
  public :: graph_type, vertex_type, edge_type


!!!-----------------------------------------------------------------------------
!!! activation (transfer) function base type
!!!-----------------------------------------------------------------------------
  type, abstract :: activation_type
     !! memory leak as allocatable character goes out of bounds
     !! change to defined length
     !character(:), allocatable :: name
     character(10) :: name
     real(real12) :: scale
     real(real12) :: threshold
   contains
     procedure (activation_function_1d), deferred, pass(this) :: activate_1d
     procedure (derivative_function_1d), deferred, pass(this) :: differentiate_1d
     procedure (activation_function_2d), deferred, pass(this) :: activate_2d
     procedure (derivative_function_2d), deferred, pass(this) :: differentiate_2d
     procedure (activation_function_3d), deferred, pass(this) :: activate_3d
     procedure (derivative_function_3d), deferred, pass(this) :: differentiate_3d
     procedure (activation_function_4d), deferred, pass(this) :: activate_4d
     procedure (derivative_function_4d), deferred, pass(this) :: differentiate_4d
     procedure (activation_function_5d), deferred, pass(this) :: activate_5d
     procedure (derivative_function_5d), deferred, pass(this) :: differentiate_5d
     generic :: activate => activate_1d, activate_2d, &
          activate_3d , activate_4d, activate_5d
     generic :: differentiate => differentiate_1d, differentiate_2d, &
          differentiate_3d, differentiate_4d, differentiate_5d
  end type activation_type
  

  !! interface for activation function
  !!----------------------------------------------------------------------------
  abstract interface
     pure function activation_function_1d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:), intent(in) :: val
       real(real12), dimension(size(val,1)) :: output
     end function activation_function_1d
     
     pure function activation_function_2d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:), intent(in) :: val
       real(real12), dimension(size(val,1),size(val,2)) :: output
     end function activation_function_2d

     pure function activation_function_3d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:), intent(in) :: val
       real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function activation_function_3d

     pure function activation_function_4d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:,:), intent(in) :: val
       real(real12), dimension(&
            size(val,1),size(val,2),size(val,3),size(val,4)) :: output
     end function activation_function_4d

     pure function activation_function_5d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:,:,:), intent(in) :: val
       real(real12), dimension(&
            size(val,1),size(val,2),size(val,3), &
            size(val,4),size(val,5)) :: output
     end function activation_function_5d
  end interface


  !! interface for derivative function
  !!----------------------------------------------------------------------------
  abstract interface
     pure function derivative_function_1d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:), intent(in) :: val
       real(real12), dimension(size(val,1)) :: output
     end function derivative_function_1d

     pure function derivative_function_2d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:), intent(in) :: val
       real(real12), dimension(size(val,1),size(val,2)) :: output
     end function derivative_function_2d

     pure function derivative_function_3d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:), intent(in) :: val
       real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function derivative_function_3d

     pure function derivative_function_4d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:,:), intent(in) :: val
       real(real12), dimension(&
            size(val,1),size(val,2),size(val,3),size(val,4)) :: output
     end function derivative_function_4d

     pure function derivative_function_5d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:,:,:), intent(in) :: val
       real(real12), dimension(&
            size(val,1),size(val,2),size(val,3), &
            size(val,4),size(val,5)) :: output
     end function derivative_function_5d
  end interface


!!!-----------------------------------------------------------------------------
!!! weights and biases initialiser base type
!!!-----------------------------------------------------------------------------
  type, abstract :: initialiser_type
     real(real12) :: scale = 1._real12, mean = 1._real12, std = 0.01_real12
   contains
     procedure (initialiser_subroutine), deferred, pass(this) :: initialise
  end type initialiser_type


  !! interface for initialiser function
  !!----------------------------------------------------------------------------
  abstract interface
     subroutine initialiser_subroutine(this, input, fan_in, fan_out)
       import initialiser_type, real12
       class(initialiser_type), intent(inout) :: this
       real(real12), dimension(..), intent(out) :: input
       integer, optional, intent(in) :: fan_in, fan_out
       real(real12) :: scale
     end subroutine initialiser_subroutine
  end interface


!!!-----------------------------------------------------------------------------
!!! graph vertex type
!!!-----------------------------------------------------------------------------
  type :: vertex_type
     integer :: degree = 1
     real(real12), dimension(:), allocatable :: feature
  end type vertex_type

  type :: edge_type
     !! for directed graphs, index(1) is the source vertex, index(2) is the target vertex
     integer, dimension(2) :: index
     real(real12) :: weight = 1._real12
     real(real12), dimension(:), allocatable :: feature
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
       real(real12), intent(in), optional :: weight
       real(real12), dimension(:), intent(in), optional :: feature
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
    real(real12), intent(in), optional :: weight
    real(real12), dimension(:), intent(in), optional :: feature
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


end module custom_types
!!!#############################################################################
