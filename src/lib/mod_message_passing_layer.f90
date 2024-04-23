!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a message passing neural network
!!!#############################################################################
module mpnn_module
  use constants, only: real12
  use custom_types, only: graph_type
  implicit none
  

  private

  public :: mpnn_type, feature_type
  public :: state_method_type, message_method_type, readout_method_type
  public :: state_update, get_state_differential
  public :: message_update, get_message_differential
  public :: get_readout_output, get_readout_differential


  type :: mpnn_type
     integer :: num_features
     integer :: num_vertices
     integer :: num_time_steps
     integer :: batch_size
     !! state and message dimension is (time_step)
     class(state_method_type), dimension(:), allocatable :: state
     class(message_method_type), dimension(:), allocatable :: message
     class(readout_method_type), allocatable :: readout
     real(real12), dimension(:,:), allocatable :: output
     real(real12), dimension(:,:,:,:), allocatable :: di
   contains
     procedure, pass(this) :: forward
     procedure, pass(this) :: backward
  end type mpnn_type


  type :: feature_type
     real(real12), dimension(:,:), allocatable :: val
  end type feature_type


  type, abstract :: state_method_type
     integer :: batch_size
     integer :: num_features
     !! feature has dimensions (feature, vertex)
     type(feature_type), dimension(:), allocatable :: feature
     type(feature_type), dimension(:), allocatable :: di
   contains
     procedure(state_update), deferred, pass(this) :: update
     procedure(get_state_differential), deferred, pass(this) :: get_differential
  end type state_method_type

  type, abstract :: message_method_type
     integer :: batch_size
     integer :: num_features
     !! feature has dimensions (feature, vertex)
     type(feature_type), dimension(:), allocatable :: feature
     type(feature_type), dimension(:), allocatable :: di
   contains
     procedure(message_update), deferred, pass(this) :: update
     procedure(get_message_differential), deferred, pass(this) :: get_differential
  end type message_method_type

  type, abstract :: readout_method_type
     integer :: batch_size
     integer :: num_outputs
   contains
     procedure(get_readout_output), deferred, pass(this) :: get_output
     procedure(get_readout_differential), deferred, pass(this) :: get_differential
  end type readout_method_type


  abstract interface
     subroutine state_update(this, message, graph)
       import :: state_method_type, real12, graph_type
       class(state_method_type), intent(inout) :: this
       !! message has dimensions (feature, vertex, batch_size)
       real(real12), dimension(:,:), intent(in) :: message
       type(graph_type), dimension(:), intent(in) :: graph
     end subroutine state_update

     pure function get_state_differential(this, message, graph) result(output)
       import :: state_method_type, real12, graph_type
       class(state_method_type), intent(in) :: this
       !! message has dimensions (feature, vertex, batch_size)
       real(real12), dimension(:,:), intent(in) :: message
       type(graph_type), dimension(:), intent(in) :: graph
       real(real12), dimension(:,:), allocatable :: output
     end function get_state_differential


     subroutine message_update(this, hidden, graph)
       import :: message_method_type, real12, graph_type
       class(message_method_type), intent(inout) :: this
       !! hidden features has dimensions (feature, vertex, batch_size)
       real(real12), dimension(:,:), intent(in) :: hidden
       type(graph_type), dimension(:), intent(in) :: graph
     end subroutine message_update

     pure function get_message_differential(this, hidden, graph) result(output)
       import :: message_method_type, real12, graph_type
       class(message_method_type), intent(in) :: this
       !! hidden features has dimensions (feature, vertex, batch_size)
       real(real12), dimension(:,:), intent(in) :: hidden
       type(graph_type), dimension(:), intent(in) :: graph
       real(real12), dimension(:,:), allocatable :: output
     end function get_message_differential


     function get_readout_output(this, state) result(output)
       import :: readout_method_type, state_method_type, real12
       class(readout_method_type), intent(inout) :: this
       class(state_method_type), dimension(:), intent(in) :: state
       real(real12), dimension(:,:), allocatable :: output
    end function get_readout_output

     pure function get_readout_differential(this, state) result(output)
       import :: readout_method_type, state_method_type, real12
       class(readout_method_type), intent(in) :: this
       class(state_method_type), dimension(:), intent(in) :: state
       real(real12), dimension(:,:), allocatable :: output  
     end function get_readout_differential

  end interface
  

  interface mpnn_type
     module function layer_setup( &
          state_method, message_method, readout_method, &
          num_features, num_vertices, num_time_steps, batch_size &
      ) result(layer)
       !! MAKE THESE ASSUMED RANK
       class(state_method_type), intent(in) :: state_method
       class(message_method_type), intent(in) :: message_method
       class(readout_method_type), intent(in) :: readout_method
       integer, intent(in) :: num_features
       integer, intent(in) :: num_vertices
       integer, intent(in) :: num_time_steps
       integer, optional, intent(in) :: batch_size
       type(mpnn_type) :: layer
     end function layer_setup
  end interface mpnn_type


contains

!!!#############################################################################
!!! layer setup
!!!#############################################################################
  module function layer_setup( &
       state_method, message_method, readout_method, &
       num_features, num_vertices, num_time_steps, batch_size &
   ) result(layer)
    implicit none
    type(mpnn_type) :: layer
    class(state_method_type), intent(in) :: state_method
    class(message_method_type), intent(in) :: message_method
    class(readout_method_type), intent(in) :: readout_method
    integer, intent(in) :: num_features
    integer, intent(in) :: num_vertices
    integer, intent(in) :: num_time_steps
    integer, optional, intent(in) :: batch_size

    integer :: i

    layer%num_features = num_features
    layer%num_vertices = num_vertices
    layer%num_time_steps = num_time_steps
    if (present(batch_size)) then
       layer%batch_size = batch_size
    else
       layer%batch_size = 1
    end if

    layer%readout = readout_method
    allocate(layer%output(num_features * num_vertices, layer%batch_size))
    allocate(layer%di(num_features, num_vertices, num_time_steps, layer%batch_size))

    allocate(layer%state(num_time_steps))
    allocate(layer%message(num_time_steps))
    do i = 1, num_time_steps
       allocate(layer%state(i), source = state_method)
       allocate(layer%message(i), source = message_method)
    end do

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  subroutine forward(this, graph)
    implicit none
    class(mpnn_type), intent(inout) :: this
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: v, s, t

    do s = 1, this%batch_size
       do v = 1, this%num_vertices
          this%state(1)%feature(s)%val(:,v) = graph(s)%vertex(v)%feature
       end do
       do t = 1, this%num_time_steps
          call this%message(t)%update(this%state(t)%feature(s)%val(:,:), graph)
          call this%state(t)%update(this%message(t+1)%feature(s)%val(:,:), graph)
       end do
     end do

    this%output = this%readout%get_output(this%state)

  end subroutine forward
!!!#############################################################################


!!!#############################################################################
!!! backpropagation
!!!#############################################################################
  subroutine backward(this, graph, gradient)
    implicit none
    class(mpnn_type), intent(inout) :: this
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    real(real12), dimension( &
         this%readout%num_outputs, &
         this%batch_size &
    ), intent(in) :: gradient

    integer :: s, t

    !df/dv_c = h(M_c) * df/dM_y

    ! M_y = sum_c v_c * h(M_c)     message for output y
    ! h()                          hidden function

    this%state(this%num_time_steps)%di(s)%val(:,:) = gradient(:,:) * &
         this%readout%get_differential(this%state)

    do s = 1, this%batch_size
       do t = this%num_time_steps-1, 1, -1
         !! check if time_step t are all handled correctly here
         this%message(t+1)%di(s)%val(:,:) = this%state(t+1)%di(s)%val(:,:) * &
               this%state(t+1)%get_differential( &
                    this%message(t+1)%feature(s)%val(:,:), graph &
               )
         this%state(t)%di(s)%val(:,:) = this%message(t+1)%di(s)%val(:,:) * &
               this%message(t+1)%get_differential( &
                    this%state(t)%feature(s)%val(:,:), graph &
               )

         ! this%di(:,:,t,s) = this%di(:,:,t+1,s) * &
         !       this%state(t+1)%get_differential( &
         !            this%message(t+1)%feature(s)%val(:,:) &
         !       ) * &
         !       this%message(t+1)%get_differential( &
         !            this%state(t)%feature(s)%val(:,:), graph &
         !       )
         
         !! ! this is method dependent
         !! this%dw(:,:,t,s) = this%message(:,t+1,s) * this%v(:,t,s)
       end do
    end do

  end subroutine backward
!!!#############################################################################

end module mpnn_module
!!!#############################################################################
