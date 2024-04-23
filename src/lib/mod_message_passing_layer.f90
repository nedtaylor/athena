!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a 1D convolutional layer
!!!#############################################################################
module mpnn_module
  use constants, only: real12
  use custom_types, only: graph_type
  implicit none
  

  private

  public :: mpnn_type

  type :: feature_type
     real(real12), dimension(:), allocatable :: feature
  end type feature_type


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
     !! hidden features has dimensions (vertex, time step, batch_size)
     !type(feature_type), dimension(:,:,:), allocatable :: hidden
!     type(feature_type), dimension(:,:), allocatable :: hidden_vertex
!     type(feature_type), dimension(:,:), allocatable :: hidden_edge
     !! message has dimensions (vertex, time step, batch_size)
     !type(feature_type), dimension(:,:,:), allocatable :: message
     !type(feature_type), dimension(:,:,:), allocatable :: weight
     !! output has dimensions (num_outputs, batch_size)
     !! v and dw have dimensions (num_features_t, num_features_t, time_step, batch_size)
     !real(real12), dimension(:,:,:,:), allocatable :: v
     !real(real12), dimension(:,:,:,:), allocatable :: dw
     !! di has dimensions (feature, vertex, time_step, batch_size)
     !procedure(message_update), pass(this), pointer :: message_update => null()
     !procedure(state_update), pass(this), pointer :: state_update => null()
     !procedure(readout), pass(this), pointer :: readout => null()
   contains
     procedure, pass(this) :: forward
     procedure, pass(this) :: backward
  end type mpnn_type




  type, abstract :: state_method_type
     !! feature has dimensions (feature, vertex, time_step, batch_size)
     real(real12), dimension(:,:,:), allocatable :: feature     
   contains
     procedure(state_update), deferred, pass(this) :: update
     procedure(get_state_differential), deferred, pass(this) :: get_differential
  end type state_method_type

  type, abstract :: message_method_type
     !! feature has dimensions (feature, vertex, time_step, batch_size)
     real(real12), dimension(:,:,:), allocatable :: feature     
   contains
     procedure(message_update), deferred, pass(this) :: update
     procedure(get_message_differential), deferred, pass(this) :: get_differential
  end type message_method_type

  type, abstract :: readout_method_type    
   contains
     procedure(get_readout_output), deferred, pass(this) :: get_output
     procedure(get_readout_differential), deferred, pass(this) :: get_differential
  end type readout_method_type


  abstract interface
     subroutine message_update(this, hidden, graph)
       import :: message_method_type, real12, graph_type
       class(message_method_type), intent(inout) :: this
       !! hidden features has dimensions (feature, vertex, batch_size)
       real(real12), dimension(:,:,:), intent(in) :: hidden
       type(graph_type), dimension(:), intent(in) :: graph
     end subroutine message_update

     pure function get_message_differential(this, hidden, graph)
       import :: message_method_type, real12, graph_type
       class(message_method_type), intent(in) :: this
       !! hidden features has dimensions (feature, vertex, batch_size)
       real(real12), dimension(:,:,:), intent(in) :: hidden
       type(graph_type), dimension(:), intent(in) :: graph
     end function get_message_differential


     subroutine state_update(this, message)
       import :: state_method_type, real12
       class(state_method_type), intent(inout) :: this
       !! message has dimensions (feature, vertex, batch_size)
       real(real12), dimension(:,:,:), intent(in) :: message
     end subroutine state_update

     pure function get_state_differential(this, message)
       import :: state_method_type, real12
       class(state_method_type), intent(in) :: this
       !! message has dimensions (feature, vertex, batch_size)
       real(real12), dimension(:,:,:), intent(in) :: message
     end function get_state_differential


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
       real(real12), dimension(:,:,:), allocatable :: output  
     end function get_readout_differential

  end interface
  


contains

! !!!#############################################################################
! !!! message function
! !!!#############################################################################
!   pure subroutine convolutional_message_update(this, graph, time_step)
!     implicit none
!     class(mpnn_type), intent(inout) :: this
!     type(graph_type), intent(in) :: graph
!     integer, intent(in) :: time_step

!     integer :: i, j
!     integer :: num_features

!     !! assume all hidden vertices for one time_step have the same number of features
!     num_features = size(this%hidden(1,time_step)%feature)
!     do i = 1, graph%num_vertices
!        allocate(this%message(i,time_step+1)%feature(num_features), source=0._real12)
!        do j = 1, graph%num_vertices
!           this%message(i,time_step+1)%feature = &
!                this%message(i,time_step+1)%feature + &
!                [ this%hidden(j,time_step), graph%edge(i,j)%feature ]
!                ![ graph%vertex(j)%feature, graph%edge(i,j)%feature ]
!                ! at time step 1, set hidden to graph vertex features
!        end do
!     end do

!   end subroutine convolutional_message_update

!   pure subroutine convolutional_state_update(this, graph, time_step)
!     implicit none
!     class(mpnn_type), intent(inout) :: this
!     type(graph_type), intent(in) :: graph
!     integer, intent(in) :: time_step

!     integer :: i

!     do i = 1, graph%num_vertices
!        this%hidden(:,i,time_step+1) = sigmoid( &
!             matmul( this%update_matrix(graph%get_degree(i),time_step), &
!                     this%message(i,time_step+1) ) )
!     end do

!   end subroutine convolutional_state_update

!   pure function convolutional_readout(this) result(output)
!     implicit none
!     class(mpnn_type), intent(in) :: this

!     real(real12), dimension(:), allocatable :: output

!     integer :: i, t

!     allocate(output(size(this%hidden(1,num_time_steps)%feature)), source=0._real12)

!     do i = 1, this%num_vertices
!        do t = 1, this%num_time_steps
!           output = output + softmax( matmul( &
!                this%readout_matrix(:,:,t), this%hidden(i,t)%feature ) )
!        end do
!     end do

!   end function convolutional_readout

!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  subroutine forward(this, graph)
    implicit none
    class(mpnn_type), intent(inout) :: this
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: i, s, t

    do s = 1, this%batch_size
       do i = 1, this%num_vertices
          this%state(1)%feature(:,i,s) = graph(s)%vertex(i)%feature
       end do
    end do

    do t = 1, this%num_time_steps
       call this%message(t)%update(this%state(t)%feature(:,:,:), graph)
       call this%state(t)%update(this%message(t+1)%feature(:,:,:))
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
         this%num_features, &
         this%num_vertices, &
         this%batch_size &
    ), intent(in) :: gradient

    integer :: t

    !df/dv_c = h(M_c) * df/dM_y

    ! M_y = sum_c v_c * h(M_c)     message for output y
    ! h()                          hidden function

    this%di(:,:,this%num_time_steps,:) = gradient(:,:,:) * &
         this%readout%get_differential(this%state)

    do t = this%num_time_steps-1, 1, -1
      !! check if time_step t are all handled correctly here
      this%di(:,:,t,:) = this%di(:,:,t+1,:) * &
            this%state(t+1)%get_differential( &
                 this%message(t+1)%feature(:,:,:) &
            ) * &
            this%message(t+1)%get_differential( &
                 this%state(t)%feature(:,:,:), graph &
            )
      
      !! ! this is method dependent
      !! this%dw(:,:,t,s) = this%message(:,t+1,s) * this%v(:,t,s)
    end do

  end subroutine backward
!!!#############################################################################

end module mpnn_module
!!!#############################################################################
