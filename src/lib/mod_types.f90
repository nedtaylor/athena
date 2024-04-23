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
     real(real12), dimension(:), allocatable :: feature
  end type vertex_type

  type :: edge_type
     integer :: source, target
     real(real12) :: weight
     real(real12), dimension(:), allocatable :: feature
  end type edge_type

  type :: graph_type
     logical :: directed = .false.
     integer :: num_vertices, num_edges
     integer :: num_vertex_features, num_edge_features
     !! adjacency matrix
     !! 1 if edge exists, 0 otherwise
     !! -1 if edge is outgoing, 1 if edge is incoming
     integer, dimension(:,:), allocatable :: adjacency
     type(vertex_type), dimension(:), allocatable :: vertex
     type(edge_type), dimension(:,:), allocatable :: edge
   contains
     procedure, pass(this) :: degree => get_degree
  end type graph_type

contains
  
  subroutine get_degree(this, degree)
    implicit none
    class(graph_type), intent(in) :: this
    integer, dimension(:), intent(out) :: degree
    integer :: i, j
    degree = 0
    !!! NEED TO ACCOUNT FOR DIRECTION
    do i = 1, this%num_vertices
      do j = 1, this%num_vertices
        if (this%adjacency(i,j) == 1) then
          degree(i) = degree(i) + 1
          degree(j) = degree(j) + 1
        end if
      end do
    end do
  end subroutine get_degree


end module custom_types
!!!#############################################################################
